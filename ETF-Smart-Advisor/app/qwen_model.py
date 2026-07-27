# app/qwen_model.py
import torch
from typing import Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import QWEN_MODEL_PATH, LLM_CONFIG
import logging

logger = logging.getLogger(__name__)


class QwenModel:
    """Qwen 模型封装类 - 使用 transformers 直接加载"""
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_model(self, model_path: Optional[str] = None):
        """加载模型（单例）"""
        if self._model is not None:
            return self._model, self._tokenizer
        
        model_path = model_path or QWEN_MODEL_PATH
        logger.info(f"📥 Loading Qwen model from: {model_path}")
        
        try:
            # 加载分词器
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 设置 padding token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # 加载模型
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            )
            
            logger.info("✅ Qwen model loaded successfully")
            return self._model, self._tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen model: {e}")
            raise
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """生成回复"""
        model, tokenizer = self.load_model()
        
        # 应用 chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        # 解码
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def generate_with_system_prompt(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """带系统提示的生成"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_message})
        
        return self.generate_response(messages, max_new_tokens, **kwargs)
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """聊天接口"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.generate_response(messages, **kwargs)
            return {
                "success": True,
                "response": response,
                "messages": messages
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"生成失败: {e}"
            }


# 全局单例
def get_qwen_model() -> QwenModel:
    return QwenModel.get_instance()