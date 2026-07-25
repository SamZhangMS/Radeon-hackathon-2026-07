# app/lora_finetuner.py
# 需要安装：pip install peft trl transformers datasets

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
from .config import LLM_CONFIG

class ETFAdvisorLoRATuner:
    """基于 LoRA 的 ETF Advisor 模型微调器，专为 AMD ROCm 优化"""

    def __init__(self, base_model_name: str, device_map: str = "auto"):
        if base_model_name is None:
            base_model_name = LLM_CONFIG.get("model_name", "Qwen/Qwen3-30B-A3B-GPTQ-Int4")
        self.base_model_name = base_model_name
        self.device_map = device_map
        
        # ROCm 环境配置
        torch.cuda.empty_cache() # 根据显存调整
        print(f"PyTorch detected {torch.cuda.device_count()} GPU(s)")

    def load_model_and_tokenizer(self):
        """加载基础模型和分词器"""
        print(f"Loading base model: {self.base_model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=self.device_map, # 自动分配到 GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 # 推荐使用 bfloat16
        )
        print("Model and tokenizer loaded successfully.")

    def _prepare_dataset(self, data: pd.DataFrame) -> Dataset:
        """准备 SFT 数据集"""
        # 假设 data 有两列: 'instruction' 和 'output'
        def format_example(example):
            return {"text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"}
        
        dataset = Dataset.from_pandas(data)
        dataset = dataset.map(format_example)
        return dataset

    def train_lora(self, train_data: pd.DataFrame, output_dir: str = "./lora_etf_advisor"):
        """执行 LoRA 微调"""
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()

        # 1. LoRA 配置 (参考最佳实践)
        lora_config = LoraConfig(
            r=8, # 低秩矩阵的维度，常用 4-64 [citation:2]
            lora_alpha=16, # 缩放参数，通常设为 r 的 2 倍
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 作用于所有线性层效果更佳[citation:1]
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 2. 准备模型
        # 对于大模型，可以启用梯度检查点以节省显存
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters() # 查看可训练参数量

        # 3. 训练参数 (针对 ROCm 优化)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1, # 根据 GPU 显存调整 (MI300X 可设更大)[citation:2]
            gradient_accumulation_steps=4,
            optim="adamw_8bit", # 或 "adamw_torch"
            learning_rate=4e-4,
            fp16=False,
            bf16=True,          # ROCm 上推荐使用 bf16[citation:2]
            logging_steps=10,
            save_steps=50,
            report_to="none",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0, # ROCm 下设置为 0 可能更稳定[citation:2]
        )

        # 4. 创建 Trainer 并开始训练
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self._prepare_dataset(train_data),
            args=training_args,
            max_seq_length=1024, # 可根据需要调整
        )

        print("Starting LoRA fine-tuning on AMD GPU...")
        trainer.train()

        # 5. 保存 LoRA 权重
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"LoRA adapters saved to {output_dir}")

        # 6. 可选：合并权重并保存
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(f"{output_dir}_merged")