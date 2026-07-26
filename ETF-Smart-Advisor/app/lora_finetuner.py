# app/lora_finetuner.py
# 需要安装：pip install peft trl transformers datasets swanlab

import os
import sys
import re
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
from .config import LLM_CONFIG

# ============================================================
# 配置
# ============================================================

class FinetuneConfig:
    """LoRA 微调配置"""
    def __init__(self):
        self.model_path = LLM_CONFIG.get("model_name", "./models/Qwen/mapfinben-qwen35-9b")
        self.output_dir = "./data/models/lora_etf_advisor"
        self.max_length = 1024
        self.batch_size = 1
        self.gradient_accumulation_steps = 8
        self.epochs = 3
        self.learning_rate = 2e-4
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.seed = 42
        self.device_map = "auto"
        self.bf16 = True
        self.fp16 = False
        self.save_steps = 50
        self.logging_steps = 10
        self.save_total_limit = 2
        self.gradient_checkpointing = True
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ============================================================
# ETF Advisor LoRA 微调器
# ============================================================

class ETFAdvisorLoRATuner:
    """
    基于 LoRA 的 ETF Advisor 模型微调器，专为 AMD ROCm 优化
    使用 ETF 历史数据进行技术分析训练
    """

    def __init__(
        self, 
        base_model_name: Optional[str] = None, 
        device_map: str = "auto",
        config: Optional[FinetuneConfig] = None
    ):
        if base_model_name is None:
            base_model_name = LLM_CONFIG.get("model_name", "./models/Qwen/mapfinben-qwen35-9b")
        self.base_model_name = base_model_name
        self.device_map = device_map
        self.config = config or FinetuneConfig()
        self.model = None
        self.tokenizer = None
        
        # 设置离线模式（如果模型已下载）
        if os.path.exists("./models/Qwen/mapfinben-qwen35-9b"):
            os.environ["HF_HUB_OFFLINE"] = "1"
        
        # ROCm 环境配置
        torch.cuda.empty_cache()
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        set_seed(self.config.seed)

    def load_model_and_tokenizer(self, use_4bit: bool = False):
        """加载基础模型和分词器"""
        print(f"📥 Loading base model: {self.base_model_name}...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, 
            trust_remote_code=True,
            use_fast=True
        )
        
        # 设置 padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 尝试加载模型
        try:
            if use_4bit:
                print("  📌 使用 4-bit 量化加载...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                )
            print("✅ Model and tokenizer loaded successfully.")
            
            # 打印模型信息
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            print("  尝试使用 CPU 加载...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            print("✅ CPU 加载成功")

    def _load_etf_data(self, data_dir: str) -> pd.DataFrame:
        """加载所有 ETF 历史数据"""
        data_path = Path(data_dir)
        all_data = []
        
        print(f"📊 Loading ETF data from: {data_dir}")
        
        # 支持 .txt 和 .csv 文件
        for file_path in list(data_path.glob("*.txt")) + list(data_path.glob("*.csv")):
            try:
                # 尝试不同格式
                if file_path.suffix == '.txt':
                    # TXT 格式：日期的格式可能为 YYYY/MM/DD
                    df = pd.read_csv(
                        file_path,
                        encoding='gb2312',
                        skipfooter=1,
                        names=['date', 'open', 'high', 'low', 'close', 'volume', 'money'],
                        dtype={'date': str, 'open': float, 'high': float, 'low': float, 'close': float},
                        engine='python'
                    )
                else:
                    # CSV 格式
                    df = pd.read_csv(file_path, encoding='gb2312')
                    # 尝试自动识别列名
                    if 'date' not in df.columns:
                        # 假设第一列是日期
                        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'money'][:len(df.columns)]
                
                df['symbol'] = file_path.stem
                df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y/%m/%d')
                df = df.sort_values('date')
                all_data.append(df)
                print(f"  ✅ 加载: {file_path.name} ({len(df)} 条记录)")
            except Exception as e:
                print(f"  ⚠️ 跳过 {file_path.name}: {e}")
        
        if not all_data:
            raise ValueError("没有加载到任何数据，请检查数据目录")
        
        result = pd.concat(all_data, ignore_index=True)
        print(f"✅ 共加载 {len(result)} 条记录，{len(result['symbol'].unique())} 个 ETF")
        return result

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标，带有除零保护"""
        df = df.copy()
        close = df['close']
        volume = df['volume']
        
        # 移动平均线
        df['ma5'] = close.rolling(5).mean()
        df['ma10'] = close.rolling(10).mean()
        df['ma20'] = close.rolling(20).mean()
        df['ma60'] = close.rolling(60).mean()
        
        # 价格变化率
        df['price_change_1d'] = close.pct_change()
        df['price_change_5d'] = close.pct_change(5)
        df['price_change_20d'] = close.pct_change(20)
        
        # RSI (相对强弱指标)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        # 防止除零
        rs = gain / loss.where(loss != 0, 1)  # 如果 loss 为 0，使用 1 代替
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # 用中性值填充 NaN
        
        # 布林带
        df['bb_middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        # 防止除零
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (close - df['bb_lower']) / bb_range.where(bb_range != 0, 1)
        df['bb_position'] = df['bb_position'].clip(0, 1)  # 限制在 0-1 范围
        df['bb_position'] = df['bb_position'].fillna(0.5)  # 用中性值填充 NaN
        
        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 成交量指标 - 防止除零
        df['volume_ma5'] = volume.rolling(5).mean()
        # 如果 volume_ma5 为 0，使用 1 代替，避免除零
        df['volume_ratio'] = volume / df['volume_ma5'].where(df['volume_ma5'] != 0, 1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1)  # 用 1 填充 NaN
        
        # 计算历史波动率（年化）
        df['volatility'] = df['price_change_1d'].rolling(30).std() * np.sqrt(252)
        df['volatility'] = df['volatility'].fillna(0)
        
        return df

    def _prepare_financial_dataset(self, df: pd.DataFrame) -> Dataset:
        """从 ETF 历史数据准备训练数据集"""
        print("🔄 生成训练样本...")
        training_samples = []
        
        # 按 ETF 分组处理
        for symbol, group in df.groupby('symbol'):
            print(f"  处理 {symbol}...")
            group = group.sort_values('date')
            group = self._calculate_technical_indicators(group)
            
            # 丢弃 NaN 行
            group = group.dropna()
            
            if len(group) < 100:
                print(f"  ⚠️ {symbol} 数据不足 ({len(group)} 条)，跳过")
                continue
            
            # 生成训练样本
            samples_per_etf = 0
            for i in range(60, len(group) - 30, 5):
                if i + 30 >= len(group):
                    break
                
                try:
                    window = group.iloc[i-60:i]  # 过去60天的数据
                    future = group.iloc[i:i+30]  # 未来30天的数据
                    
                    current_price = window['close'].iloc[-1]
                    future_price = future['close'].iloc[-1]
                    future_change = (future_price - current_price) / current_price if current_price != 0 else 0
                    
                    # 获取当前指标
                    ma5 = window['ma5'].iloc[-1]
                    ma20 = window['ma20'].iloc[-1]
                    ma60 = window['ma60'].iloc[-1]
                    rsi = window['rsi'].iloc[-1]
                    bb_pos = window['bb_position'].iloc[-1]
                    macd = window['macd'].iloc[-1]
                    macd_signal = window['macd_signal'].iloc[-1]
                    volume_ratio = window['volume_ratio'].iloc[-1]
                    
                    # 检查是否有 NaN 值
                    if pd.isna(rsi) or pd.isna(bb_pos):
                        continue
                    
                    # 计算历史波动率（年化）
                    hist_vol = window['price_change_1d'].std() * np.sqrt(252)
                    if pd.isna(hist_vol):
                        hist_vol = 0
                    
                    # 判断趋势
                    if ma5 > ma20 > ma60:
                        trend = "上涨"
                    elif ma5 < ma20 < ma60:
                        trend = "下跌"
                    else:
                        trend = "震荡"
                    
                    # RSI 状态
                    if rsi > 70:
                        rsi_status = "超买"
                    elif rsi < 30:
                        rsi_status = "超卖"
                    else:
                        rsi_status = "中性"
                    
                    # 布林带状态
                    if bb_pos > 0.8:
                        bb_status = "上轨附近"
                    elif bb_pos < 0.2:
                        bb_status = "下轨附近"
                    else:
                        bb_status = "中轨附近"
                    
                    # MACD 状态
                    if macd > macd_signal:
                        macd_status = "金叉"
                    elif macd < macd_signal:
                        macd_status = "死叉"
                    else:
                        macd_status = "持平"
                    
                    instruction = f"""分析 {symbol} ETF 的技术指标并预测未来走势。

当前价格: {current_price:.4f}
5日均线: {ma5:.4f} ({'高于' if ma5 > ma20 else '低于'}20日均线)
20日均线: {ma20:.4f}
60日均线: {ma60:.4f}
RSI(14): {rsi:.1f} ({rsi_status})
布林带位置: {bb_pos:.2f} ({bb_status})
MACD: {macd:.4f} ({macd_status})
成交量比率: {volume_ratio:.2f}
历史波动率(年化): {hist_vol:.2%}

请基于以上技术指标，判断当前趋势并给出投资建议。"""

                    # 确定信号
                    if future_change > 0.05:
                        signal, confidence = "强烈买入", "高"
                        target_mult = 1.08
                    elif future_change > 0.02:
                        signal, confidence = "买入", "中高"
                        target_mult = 1.05
                    elif future_change > -0.02:
                        signal, confidence = "持有", "中"
                        target_mult = 1.01
                    elif future_change > -0.05:
                        signal, confidence = "谨慎持有", "中低"
                        target_mult = 0.97
                    else:
                        signal, confidence = "卖出", "高"
                        target_mult = 0.92
                    
                    # 结合技术指标调整建议
                    if rsi > 70 and future_change > 0:
                        signal = "观望"  # 超买区域不建议追高
                        confidence = "中"
                    elif rsi < 30 and future_change < 0:
                        signal = "关注买入机会"  # 超卖区域可考虑买入
                        confidence = "中高"
                    
                    output = f"""投资建议: {signal}
信心程度: {confidence}
目标价: {current_price * target_mult:.4f} (预期收益: {future_change:.2%})
止损价: {current_price * 0.95:.4f}
技术面总结: 当前趋势{trend}，RSI{rsi:.1f}，{'建议等待回调' if rsi > 70 and future_change > 0 else '可逢低布局' if rsi < 30 else '维持现有仓位'}
风险提示: 投资有风险，请根据自身风险承受能力做出决策。"""

                    training_samples.append({
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    })
                    samples_per_etf += 1
                    
                except Exception as e:
                    continue
            
            print(f"    ✅ {symbol}: 生成 {samples_per_etf} 个样本")
        
        print(f"✅ 共生成 {len(training_samples)} 个训练样本")
        
        if len(training_samples) == 0:
            raise ValueError("没有生成任何训练样本，请检查数据质量")
        
        return Dataset.from_pandas(pd.DataFrame(training_samples))

    def _process_func_precise(self, example: Dict[str, str]) -> Dict[str, List[int]]:
        """
        精确处理：只对 assistant 回复部分计算 loss
        适配 mapfinben-qwen35-9b 模型
        """
        MAX_LENGTH = self.config.max_length
        
        instruction_text = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        
        # 构建 instruction 部分（不包含 assistant 回复）
        instruction_prefix = (
            "<|im_start|>system\n"
            "你是一个专业的 ETF 投资分析师，擅长技术分析和投资建议。<|im_end|>\n"
            f"<|im_start|>user\n{instruction_text + input_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # Tokenize instruction 部分
        instruction_tokens = self.tokenizer(
            instruction_prefix,
            add_special_tokens=False,
            return_tensors=None
        )
        
        # Tokenize response 部分
        response_tokens = self.tokenizer(
            output_text,
            add_special_tokens=False,
            return_tensors=None
        )
        
        # 拼接：instruction + response + eos
        input_ids = instruction_tokens["input_ids"] + response_tokens["input_ids"] + [self.tokenizer.eos_token_id]
        attention_mask = instruction_tokens["attention_mask"] + response_tokens["attention_mask"] + [1]
        
        # Labels: instruction 部分用 -100 忽略，response 部分正常
        labels = [-100] * len(instruction_tokens["input_ids"]) + response_tokens["input_ids"] + [self.tokenizer.eos_token_id]
        
        # 截断
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """准备数据集：应用处理函数"""
        print("🔄 Processing dataset...")
        
        # 添加 input 列（如果不存在）
        if 'input' not in dataset.column_names:
            dataset = dataset.map(lambda x: {**x, 'input': ''})
        
        # 应用处理函数
        processed = dataset.map(
            self._process_func_precise,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        print(f"✅ Dataset prepared: {len(processed)} samples")
        return processed

    def setup_lora(self):
        """配置 LoRA"""
        print("🔧 Setting up LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            inference_mode=False,
        )
        
        # 启用梯度检查点
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # 准备模型
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return lora_config

    def train_lora(
        self, 
        train_data: Union[pd.DataFrame, Dataset, str],
        output_dir: Optional[str] = None,
        use_swanlab: bool = True
    ):
        """
        执行 LoRA 微调
        
        Args:
            train_data: 训练数据（DataFrame、Dataset 或数据目录路径）
            output_dir: 输出目录
            use_swanlab: 是否使用 SwanLab 记录
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据集
        if isinstance(train_data, str):
            # 是数据目录路径
            df = self._load_etf_data(train_data)
            dataset = self._prepare_financial_dataset(df)
        elif isinstance(train_data, pd.DataFrame):
            # 检查是否是 ETF 数据（有 symbol 列）
            if 'symbol' in train_data.columns:
                dataset = self._prepare_financial_dataset(train_data)
            else:
                dataset = Dataset.from_pandas(train_data)
        else:
            dataset = train_data
        
        # 准备数据集
        if 'instruction' in dataset.column_names:
            train_dataset = self.prepare_dataset(dataset)
        else:
            raise ValueError("数据集缺少 'instruction' 列")
        
        # 设置 LoRA
        self.setup_lora()
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim="adamw_torch",  # ROCm 上使用 standard adamw
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            gradient_checkpointing=self.config.gradient_checkpointing,
            seed=self.config.seed,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # 创建 Trainer
        from transformers import Trainer
        
        # 集成 SwanLab
        callbacks = []
        if use_swanlab:
            try:
                from swanlab.integration.transformers import SwanLabCallback
                swanlab_callback = SwanLabCallback(
                    project="ETF-Advisor-Finetune",
                    experiment_name=f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                callbacks.append(swanlab_callback)
                print("✅ SwanLab callback enabled")
            except ImportError:
                print("⚠️ SwanLab not installed, skipping")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        print("🚀 Starting LoRA fine-tuning...")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"   Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   LoRA r: {self.config.lora_r}")
        print(f"   Max length: {self.config.max_length}")
        print(f"   Training samples: {len(train_dataset)}")
        
        # 训练
        trainer.train()
        
        # 保存模型
        print(f"💾 Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        config_path = os.path.join(output_dir, "finetune_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "base_model": self.base_model_name,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "max_length": self.config.max_length,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✅ LoRA adapters saved to {output_dir}")
        print(f"✅ Config saved to {config_path}")
        
        # 尝试保存合并模型
        try:
            merged_model = self.model.merge_and_unload()
            merged_path = f"{output_dir}_merged"
            merged_model.save_pretrained(merged_path)
            self.tokenizer.save_pretrained(merged_path)
            print(f"✅ Merged model saved to {merged_path}")
        except Exception as e:
            print(f"⚠️ 合并模型失败: {e}")
        
        return output_dir

    def train_with_etf_data(self, data_dir: str, output_dir: Optional[str] = None):
        """使用 ETF 历史数据训练"""
        print("\n📊 加载 ETF 历史数据...")
        df = self._load_etf_data(data_dir)
        print(f"✅ 共加载 {len(df)} 条记录")
        return self.train_lora(df, output_dir)


# ============================================================
# 便捷函数
# ============================================================

def finetune_etf_advisor(
    data_dir: str,
    model_path: Optional[str] = None,
    output_dir: str = "./data/models/lora_etf_advisor",
    epochs: int = 3,
    batch_size: int = 1,
    lora_r: int = 16,
    max_length: int = 1024,
    learning_rate: float = 2e-4,
    use_4bit: bool = False
):
    """
    快速微调 ETF Advisor 模型
    
    Args:
        data_dir: ETF 数据目录
        model_path: 模型路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        lora_r: LoRA 秩
        max_length: 最大序列长度
        learning_rate: 学习率
        use_4bit: 是否使用4bit量化
    """
    config = FinetuneConfig()
    if model_path:
        config.model_path = model_path
    config.output_dir = output_dir
    config.epochs = epochs
    config.batch_size = batch_size
    config.lora_r = lora_r
    config.max_length = max_length
    config.learning_rate = learning_rate
    
    tuner = ETFAdvisorLoRATuner(config=config)
    tuner.load_model_and_tokenizer(use_4bit=use_4bit)
    return tuner.train_with_etf_data(data_dir, output_dir)


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA 微调 ETF Advisor 模型")
    parser.add_argument("--data_dir", type=str, required=True, help="ETF 数据目录")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./data/models/lora_etf_advisor", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--use_4bit", action="store_true", help="使用 4-bit 量化加载")
    parser.add_argument("--no_swanlab", action="store_true", help="禁用 SwanLab")
    
    args = parser.parse_args()
    
    # 微调
    finetune_etf_advisor(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        use_4bit=args.use_4bit
    )


if __name__ == "__main__":
    main()