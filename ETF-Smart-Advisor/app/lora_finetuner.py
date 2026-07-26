# app/lora_finetuner.py
# 需要安装：pip install peft trl transformers datasets

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
from .config import LLM_CONFIG

# ============================================================
# 配置
# ============================================================

class FinetuneConfig:
    """LoRA 微调配置"""
    def __init__(self):
        self.model_path = LLM_CONFIG.get("model_name", "Qwen/Qwen3-30B-A3B-GPTQ-Int4")
        self.output_dir = "./data/models/lora_etf_advisor"
        self.max_length = 1024
        self.batch_size = 1
        self.gradient_accumulation_steps = 8
        self.epochs = 1
        self.learning_rate = 2e-4
        self.lora_r = 16
        self.lora_alpha = 32
        self.seed = 42
        self.device_map = "auto"
        self.bf16 = True
        self.fp16 = False
        self.save_steps = 50
        self.logging_steps = 10
        self.save_total_limit = 2


class ETFAdvisorLoRATuner:
    """基于 LoRA 的 ETF Advisor 模型微调器，专为 AMD ROCm 优化"""

    def __init__(
        self, 
        base_model_name: Optional[str] = None, 
        device_map: str = "auto",
        config: Optional[FinetuneConfig] = None
    ):
        if base_model_name is None:
            base_model_name = LLM_CONFIG.get("model_name", "Qwen/Qwen3-30B-A3B-GPTQ-Int4")
        self.base_model_name = base_model_name
        self.device_map = device_map
        self.config = config or FinetuneConfig()
        self.model = None
        self.tokenizer = None
        
        # ROCm 环境配置
        torch.cuda.empty_cache()
        print(f"PyTorch detected {torch.cuda.device_count()} GPU(s)")
        set_seed(self.config.seed)

    def load_model_and_tokenizer(self, use_4bit: bool = False):
        """加载基础模型和分词器"""
        print(f"📥 Loading base model: {self.base_model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, 
            trust_remote_code=True,
            use_fast=True
        )
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
                    quantization_config=bnb_config
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                )
            print("✅ Model and tokenizer loaded successfully.")
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
        
        for file_path in data_path.glob("*.txt"):
            try:
                df = pd.read_csv(
                    file_path,
                    encoding='gb2312',
                    skipfooter=1,
                    names=['date', 'open', 'high', 'low', 'close', 'volume', 'money'],
                    dtype={'date': str, 'open': float, 'high': float, 'low': float, 'close': float},
                    engine='python'
                )
                df['symbol'] = file_path.stem
                df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y/%m/%d')
                df = df.sort_values('date')
                all_data.append(df)
                print(f"  ✅ 加载: {file_path.name} ({len(df)} 条记录)")
            except Exception as e:
                print(f"  ⚠️ 跳过 {file_path.name}: {e}")
        
        if not all_data:
            raise ValueError("没有加载到任何数据")
        return pd.concat(all_data, ignore_index=True)

    def _prepare_financial_dataset(self, df: pd.DataFrame) -> Dataset:
        """从 ETF 历史数据准备训练数据集"""
        training_samples = []
        
        for symbol, group in df.groupby('symbol'):
            group = group.sort_values('date')
            close = group['close']
            
            # 计算技术指标
            group['ma5'] = close.rolling(5).mean()
            group['ma20'] = close.rolling(20).mean()
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            group['rsi'] = 100 - (100 / (1 + gain / loss))
            
            for i in range(60, len(group) - 20, 10):
                if i + 20 >= len(group):
                    break
                window = group.iloc[i-60:i]
                future = group.iloc[i:i+20]
                current_price = window['close'].iloc[-1]
                future_change = (future['close'].iloc[-1] - current_price) / current_price
                
                instruction = f"""分析 ETF {symbol} 的技术指标并预测未来走势。

当前价格: {current_price:.3f}
5日均线: {window['ma5'].iloc[-1]:.3f}
20日均线: {window['ma20'].iloc[-1]:.3f}
RSI: {window['rsi'].iloc[-1]:.1f}
成交量: {window['volume'].iloc[-1]:.0f}

请分析当前趋势并给出预测。"""

                if future_change > 0.03:
                    signal, reason = "强烈买入", f"预计上涨 {future_change:.2%}"
                elif future_change > 0.01:
                    signal, reason = "买入", f"预计上涨 {future_change:.2%}"
                elif future_change > -0.01:
                    signal, reason = "持有", f"预计平稳 ({future_change:.2%})"
                elif future_change > -0.03:
                    signal, reason = "谨慎持有", f"预计下跌 {abs(future_change):.2%}"
                else:
                    signal, reason = "卖出", f"预计下跌 {abs(future_change):.2%}"
                
                output = f"""预测建议: {signal}
理由: {reason}
目标价: {current_price * (1 + 0.05 if future_change > 0 else -0.03):.3f}
止损价: {current_price * (1 - 0.03):.3f}"""

                training_samples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                })
        
        print(f"✅ 生成 {len(training_samples)} 个训练样本")
        return Dataset.from_pandas(pd.DataFrame(training_samples))

    def _prepare_dataset_from_dataframe(self, data: pd.DataFrame) -> Dataset:
        """从 DataFrame 准备 SFT 数据集"""
        def format_example(example):
            return {"text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"}
        
        dataset = Dataset.from_pandas(data)
        dataset = dataset.map(format_example)
        return dataset

    def _process_func(self, example):
        """处理单个样本"""
        messages = [
            {"role": "system", "content": "你是一个专业的 ETF 投资分析师，擅长技术分析和投资建议。"},
            {"role": "user", "content": example['instruction'] + example['input']},
            {"role": "assistant", "content": example['output']}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
            return_tensors=None
        )
        return {
            "input_ids": tokenized['input_ids'],
            "attention_mask": tokenized.get('attention_mask', [1] * len(tokenized['input_ids'])),
            "labels": tokenized['input_ids'].copy()
        }

    def train_lora(self, train_data: pd.DataFrame, output_dir: Optional[str] = None):
        """执行 LoRA 微调"""
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()

        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 准备数据集
        if 'instruction' in train_data.columns:
            dataset = self._prepare_dataset_from_dataframe(train_data)
        else:
            # 假设是 ETF 历史数据
            dataset = self._prepare_financial_dataset(train_data)
            dataset = dataset.map(self._process_func, remove_columns=dataset.column_names)

        # LoRA 配置
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 准备模型
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim="adamw_8bit",
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            report_to="none",
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            gradient_checkpointing=True,
        )

        # 创建 Trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=self.config.max_length,
        )

        print("🚀 Starting LoRA fine-tuning on AMD GPU...")
        trainer.train()

        # 保存 LoRA 权重
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✅ LoRA adapters saved to {output_dir}")

        # 可选：合并权重并保存
        try:
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(f"{output_dir}_merged")
            print(f"✅ Merged model saved to {output_dir}_merged")
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
# 命令行入口
# ============================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA 微调 Qwen 模型")
    parser.add_argument("--data_dir", type=str, default="./data/1D", help="ETF 数据目录")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./data/models/lora_etf_advisor", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--use_4bit", action="store_true", help="使用 4-bit 量化加载")
    
    args = parser.parse_args()
    
    # 配置
    config = FinetuneConfig()
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.max_length = args.max_length
    if args.model_path:
        config.model_path = args.model_path
    
    # 创建调优器
    tuner = ETFAdvisorLoRATuner(config=config)
    
    # 加载模型
    tuner.load_model_and_tokenizer(use_4bit=args.use_4bit)
    
    # 训练
    tuner.train_with_etf_data(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()