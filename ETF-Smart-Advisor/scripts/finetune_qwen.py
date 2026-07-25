#!/usr/bin/env python
# scripts/finetune_qwen.py - LoRA 调优 Qwen 模型

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# ============================================================
# 配置
# ============================================================

MODEL_PATH = os.environ.get("FINETUNE_MODEL_PATH", "./models/Qwen/Qwen3-30B-A3B-GPTQ-Int4")
OUTPUT_DIR = os.environ.get("FINETUNE_OUTPUT_DIR", "./data/models/lora_etf_advisor")
MAX_LENGTH = 1024
BATCH_SIZE = 1  # 30B 模型用 1
GRADIENT_ACCUMULATION = 8
EPOCHS = 1
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
SEED = 42

set_seed(SEED)

# ============================================================
# 数据加载与处理
# ============================================================

def load_etf_data(data_dir: str) -> pd.DataFrame:
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


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """准备训练数据"""
    training_samples = []
    
    # 按股票分组处理
    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('date')
        
        # 计算技术指标
        close = group['close']
        group['ma5'] = close.rolling(5).mean()
        group['ma20'] = close.rolling(20).mean()
        group['rsi'] = calculate_rsi(close)
        group['return'] = close.pct_change()
        
        # 生成训练样本
        for i in range(60, len(group) - 20, 10):
            if i + 20 >= len(group):
                break
            
            window = group.iloc[i-60:i]
            future = group.iloc[i:i+20]
            
            # 当前价格和技术指标
            current_price = window['close'].iloc[-1]
            current_ma5 = window['ma5'].iloc[-1]
            current_ma20 = window['ma20'].iloc[-1]
            current_rsi = window['rsi'].iloc[-1]
            current_vol = window['volume'].iloc[-1]
            
            # 未来价格变化
            future_change = (future['close'].iloc[-1] - current_price) / current_price
            
            # 构建 instruction
            instruction = f"""分析 ETF {symbol} 的技术指标并预测未来走势。

当前价格: {current_price:.3f}
5日均线: {current_ma5:.3f}
20日均线: {current_ma20:.3f}
RSI: {current_rsi:.1f}
成交量: {current_vol:.0f}

请分析当前趋势并给出预测。"""

            # 构建 output
            if future_change > 0.03:
                signal = "强烈买入"
                reason = f"预计上涨 {future_change:.2%}"
            elif future_change > 0.01:
                signal = "买入"
                reason = f"预计上涨 {future_change:.2%}"
            elif future_change > -0.01:
                signal = "持有"
                reason = f"预计平稳 ({future_change:.2%})"
            elif future_change > -0.03:
                signal = "谨慎持有"
                reason = f"预计下跌 {abs(future_change):.2%}"
            else:
                signal = "卖出"
                reason = f"预计下跌 {abs(future_change):.2%}"
            
            output = f"""预测建议: {signal}
理由: {reason}
目标价: {current_price * (1 + 0.05 if future_change > 0 else -0.03):.3f}
止损价: {current_price * (1 - 0.03):.3f}"""

            training_samples.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "symbol": symbol
            })
    
    print(f"✅ 生成 {len(training_samples)} 个训练样本")
    return pd.DataFrame(training_samples)


def process_func(example, tokenizer, max_length: int = MAX_LENGTH):
    """处理单个样本"""
    # 使用 chat template
    messages = [
        {"role": "system", "content": "你是一个专业的 ETF 投资分析师，擅长技术分析和投资建议。"},
        {"role": "user", "content": example['instruction'] + example['input']},
        {"role": "assistant", "content": example['output']}
    ]
    
    # 应用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    # 创建 labels (所有 token 都参与 loss 计算)
    input_ids = tokenized['input_ids']
    attention_mask = tokenized.get('attention_mask', [1] * len(input_ids))
    labels = input_ids.copy()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("="*60)
    print("🚀 Qwen LoRA 调优 (基于 ETF 历史数据)")
    print("="*60)
    
    # 1. 确定数据路径
    script_path = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = f'{script_path}/../data/1D'
    print(f"\n📁 数据目录: {raw_data_path}")
    
    if not os.path.exists(raw_data_path):
        print(f"❌ 数据目录不存在: {raw_data_path}")
        return
    
    # 2. 加载数据
    print("\n📊 加载 ETF 历史数据...")
    df = load_etf_data(raw_data_path)
    print(f"✅ 共加载 {len(df)} 条记录")
    
    # 3. 准备训练数据
    print("\n📝 准备训练数据...")
    train_df = prepare_training_data(df)
    print(f"✅ 生成 {len(train_df)} 个训练样本")
    
    # 4. 检查模型是否存在
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print("请先运行 setup_env.sh 部署 Qwen 模型")
        return
    
    # 5. 加载 Tokenizer
    print(f"\n📥 加载 Tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 6. 处理数据集
    print("\n🔧 处理数据集...")
    dataset = Dataset.from_pandas(train_df)
    tokenized_dataset = dataset.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=dataset.column_names
    )
    print(f"✅ 处理完成，共 {len(tokenized_dataset)} 个样本")
    
    # 7. 创建 LoRA 配置
    print("\n⚙️ 配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        bias="none"
    )
    
    # 8. 加载模型 - 使用 `dtype` 替代 `torch_dtype`
    print(f"\n📥 加载模型权重 (GPTQ 量化)...")
    try:
        # 尝试使用 optimum 加载 GPTQ 模型
        from optimum.gptq import GPTQConfig
        from transformers import GPTQConfig as TransformersGPTQConfig
        
        # 检查模型配置中的量化信息
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,  # 使用 dtype 替代 torch_dtype
            trust_remote_code=True
        )
    except ImportError:
        print("⚠️ optimum 未安装，尝试直接加载...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ 加载失败: {e}")
        print("尝试使用 CPU 加载...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 9. 训练参数
    print("\n🏋️ 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        logging_steps=10,
        num_train_epochs=EPOCHS,
        save_steps=50,
        learning_rate=LEARNING_RATE,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        bf16=True,
        fp16=False,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    
    # 10. 创建 Trainer 并训练
    print("\n🚀 开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train()
    
    # 11. 保存模型
    print(f"\n💾 保存 LoRA 权重到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✅ Qwen LoRA 调优完成!")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()