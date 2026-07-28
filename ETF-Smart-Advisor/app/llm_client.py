# app/llm_client.py
"""
统一 LLM 客户端 - 支持 vLLM（优先）和 Transformers（降级）
合并了 qwen_model.py 和 vllm_client.py 的所有功能

设计原则：
1. 单一入口：所有 LLM 调用统一通过 LLMClient
2. 自动降级：vLLM 不可用时自动切换到 Transformers
3. 接口兼容：完全兼容 qwen_model.py 的接口
4. 单例模式：全局共享一个实例
5. 配置驱动：所有配置从 config.py 读取
"""

import asyncio
import httpx
import torch
import logging
from typing import Optional, List, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig,AutoConfig
import bitsandbytes

from .config import LLM_API_CONFIG

logger = logging.getLogger(__name__)


class LLMClient:
    """
    统一 LLM 客户端
    
    功能：
    - 优先使用 vLLM（高性能推理）
    - 降级使用 Transformers（直接加载）
    - 完全兼容 qwen_model.py 的接口
    
    使用方式：
        llm = get_llm_client()
        
        # 方式1：生成回复
        response = llm.generate_response(
            messages=[{"role": "user", "content": "你好"}],
            max_new_tokens=512
        )
        
        # 方式2：聊天
        result = llm.chat(
            user_message="你好",
            system_prompt="你是专业的ETF投资顾问"
        )
        
        # 方式3：带系统提示生成
        response = llm.generate_with_system_prompt(
            user_message="分析510300",
            system_prompt="你是专业的ETF投资顾问"
        )
    
    配置：
        - use_vllm: 是否启用 vLLM（默认 True）
        - vllm_host: vLLM 服务地址
        - vllm_port: vLLM 服务端口
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # 从配置读取
        vllm_config = LLM_API_CONFIG.get("vllm", {})
        self.model_path = LLM_API_CONFIG.get("model_path")
        self.enable_thinking = LLM_API_CONFIG.get("enable_thinking", False)
        
        # vLLM 配置
        self.use_vllm = vllm_config.get("enabled", True)
        self.vllm_host = vllm_config.get("host", "localhost")
        self.vllm_port = vllm_config.get("port", 8000)
        self.vllm_base_url = f"http://{self.vllm_host}:{self.vllm_port}"
        self.vllm_model_name = vllm_config.get("served_model_name", "qwen-model")
        self.vllm_timeout = 60.0
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 状态
        self._vllm_available = None
        self._transformers_loaded = False
        
        logger.info(f"✅ LLM 客户端初始化完成")
        logger.info(f"   vLLM: {'启用' if self.use_vllm else '禁用'}")
        logger.info(f"   vLLM 端点: {self.vllm_base_url}")
        logger.info(f"   模型路径: {self.model_path}")
        logger.info(f"   目标设备: {self.device}")
        logger.info(f"   思考模式: {'开启' if self.enable_thinking else '关闭'}")
    
    # ============================================================
    # vLLM 相关方法
    # ============================================================
    
    async def check_vllm_health(self) -> bool:
        """
        检查 vLLM 服务是否可用
        
        Returns:
            True: vLLM 服务正常
            False: vLLM 服务不可用
        """
        if self._vllm_available is not None:
            return self._vllm_available
        
        if not self.use_vllm:
            self._vllm_available = False
            return False
        
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.vllm_base_url}/health")
                self._vllm_available = response.status_code == 200
                if self._vllm_available:
                    logger.debug("vLLM 服务健康检查通过")
                else:
                    logger.warning(f"vLLM 服务返回异常状态码: {response.status_code}")
                return self._vllm_available
        except httpx.ConnectError:
            logger.warning(f"vLLM 服务连接失败: {self.vllm_base_url}")
            self._vllm_available = False
            return False
        except Exception as e:
            logger.warning(f"vLLM 健康检查异常: {e}")
            self._vllm_available = False
            return False
    
    async def _vllm_generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 vLLM 生成响应（异步）
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: top_p 采样参数
        
        Returns:
            {
                "success": bool,
                "response": str,
                "usage": dict,
                "model": str
            }
        """
        try:
            async with httpx.AsyncClient(timeout=self.vllm_timeout) as client:
                response = await client.post(
                    f"{self.vllm_base_url}/v1/chat/completions",
                    json={
                        "model": self.vllm_model_name,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stream": False,
                        **kwargs
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "response": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                        "usage": data.get("usage", {}),
                        "model": data.get("model", "vllm"),
                        "backend": "vllm"
                    }
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"vLLM 请求失败: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "response": "",
                        "backend": "vllm"
                    }
                    
        except httpx.TimeoutException:
            logger.error(f"vLLM 请求超时 ({self.vllm_timeout}s)")
            return {
                "success": False,
                "error": f"请求超时 ({self.vllm_timeout}s)",
                "response": "",
                "backend": "vllm"
            }
        except Exception as e:
            logger.error(f"vLLM 请求异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "backend": "vllm"
            }
    
        
    def _get_model_max_length(self) -> int:
        """获取模型最大序列长度"""
        try:

            
            # 从配置读取
            config_max_len = LLM_API_CONFIG.get("max_model_len", 4096)
            
            # 尝试从模型配置获取
            try:
                config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                
                # 检查各种可能的配置字段
                max_len = getattr(config, 'max_position_embeddings', None)
                if max_len is None:
                    max_len = getattr(config, 'max_sequence_length', None)
                if max_len is None:
                    max_len = getattr(config, 'n_positions', None)
                if max_len is None:
                    max_len = config_max_len
                
                # 取较小值，避免超出模型限制
                result = min(max_len, config_max_len)
                logger.info(f"   模型最大序列长度: {result}")
                return result
            except Exception as e:
                logger.warning(f"   无法从模型配置获取最大长度: {e}")
                return config_max_len
                
        except Exception as e:
            logger.warning(f"   获取模型最大长度失败: {e}")
            return 4096
        
    # ============================================================
    # Transformers 相关方法（降级方案）
    # ============================================================
    
    def _load_transformers_model(self):
        """
        加载 Transformers 模型（单例）
        
        Returns:
            (model, tokenizer): 模型和分词器实例
        """
        if self._model is not None:
            return self._model, self._tokenizer
        
        logger.info(f"📥 加载 Transformers 模型: {self.model_path}")
        
        try:
            # 加载分词器
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 设置 padding token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._max_seq_length = self._get_model_max_length()
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
                        
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0", # "auto",
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
            )
            
            self._transformers_loaded = True
            logger.info("✅ Transformers 模型加载完成")
            
            if hasattr(self._model, 'device'):
                logger.info(f"   模型主设备: {self._model.device}")
            
            # 检查是否有参数在 CPU 上
            cpu_params = 0
            gpu_params = 0
            meta_params = 0
            for name, param in self._model.named_parameters():
                if param.device.type == 'cpu':
                    cpu_params += param.numel()
                elif param.device.type == 'cuda':
                    gpu_params += param.numel()
                elif param.device.type == 'meta':
                    meta_params += param.numel()
            
            logger.info(f"   📊 参数分布:")
            logger.info(f"      GPU: {gpu_params/1e6:.2f}M")
            if cpu_params > 0:
                logger.warning(f"      CPU: {cpu_params/1e6:.2f}M ⚠️ 部分参数在 CPU 上")
            if meta_params > 0:
                logger.warning(f"      Meta: {meta_params/1e6:.2f}M ⚠️ 未分配")
            
            total_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"   总参数: {total_params:,}")
            
            return self._model, self._tokenizer
            
        except Exception as e:
            logger.error(f"❌ Transformers 模型加载失败: {e}")
            raise
    
    def _transformers_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 Transformers 生成（降级方案）
        
        Args:
            messages: 消息列表
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: top_p 采样参数
        
        Returns:
            {
                "success": bool,
                "response": str,
                "backend": str
            }
        """
        try:
            model, tokenizer = self._load_transformers_model()
            
            # 应用 chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            
            max_total_length = getattr(self, '_max_seq_length', 4096)
            
            # Tokenize
            inputs = tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=max_total_length - max_new_tokens  # 预留生成空间
            ).to(model.device)
                        
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, max_total_length - inputs['input_ids'].shape[1]),
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,  # 使用贪婪解码，减少显存使用
                    **kwargs
                )
            
            # 解码
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return {
                "success": True,
                "response": response,
                "backend": "transformers",
                "tokens_generated": outputs.shape[1] - inputs['input_ids'].shape[1]
            }
            
        except Exception as e:
            logger.error(f"Transformers 推理失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "backend": "transformers"
            }
    
    # ============================================================
    # 统一公共接口
    # ============================================================
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        生成回复 - 统一接口
        
        优先使用 vLLM，失败时降级到 Transformers
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: top_p 采样参数
            enable_thinking: 是否启用思考模式（None 则使用配置默认值）
        
        Returns:
            str: 生成的回复文本
        """
        # 覆盖思考模式设置
        if enable_thinking is not None:
            original_thinking = self.enable_thinking
            self.enable_thinking = enable_thinking
        
        try:
            max_new_tokens = min(max_new_tokens, 512)
            # 尝试 vLLM
            if self.use_vllm:
                # 同步调用异步方法
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 如果已在事件循环中，使用 run_in_executor
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self._vllm_generate(
                                    messages=messages,
                                    max_tokens=max_new_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    **kwargs
                                )
                            )
                            result = future.result(timeout=self.vllm_timeout + 5)
                    else:
                        result = asyncio.run(self._vllm_generate(
                            messages=messages,
                            max_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            **kwargs
                        ))
                except RuntimeError:
                    # 没有事件循环，创建新的
                    result = asyncio.run(self._vllm_generate(
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **kwargs
                    ))
                except Exception as e:
                    logger.warning(f"vLLM 调用异常: {e}")
                    result = {"success": False, "error": str(e)}
                
                if result.get('success'):
                    return result.get('response', '')
                
                logger.warning(f"vLLM 失败，降级到 Transformers: {result.get('error')}")
            
            # 降级到 Transformers
            result = self._transformers_generate(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            if result.get('success'):
                return result.get('response', '')
            else:
                return f"生成失败: {result.get('error', '未知错误')}"
                
        finally:
            # 恢复思考模式设置
            if enable_thinking is not None:
                self.enable_thinking = original_thinking
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        聊天接口 - 兼容 qwen_model.py 的 chat 方法
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示（可选）
            history: 历史对话（可选）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: top_p 采样参数
        
        Returns:
            {
                "success": bool,
                "response": str,
                "messages": List[Dict]  # 完整的消息列表
            }
        """
        # 构建消息列表
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.generate_response(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            return {
                "success": True,
                "response": response,
                "messages": messages
            }
        except Exception as e:
            logger.error(f"Chat 失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"生成失败: {e}",
                "messages": messages
            }
    
    def generate_with_system_prompt(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        带系统提示的生成 - 兼容 qwen_model.py
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示（可选）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: top_p 采样参数
        
        Returns:
            str: 生成的回复文本
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_message})
        
        return self.generate_response(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    # ============================================================
    # 模型状态方法
    # ============================================================
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        获取模型状态
        
        Returns:
            {
                "vllm_available": bool,
                "transformers_loaded": bool,
                "use_vllm": bool,
                "model_path": str,
                "device": str
            }
        """
        # 检查 vLLM 状态（同步方式）
        vllm_available = False
        if self.use_vllm:
            try:
                vllm_available = asyncio.run(self.check_vllm_health())
            except:
                vllm_available = False
        
        status = {
            "vllm_available": vllm_available,
            "transformers_loaded": self._transformers_loaded,
            "use_vllm": self.use_vllm,
            "model_path": self.model_path,
            "enable_thinking": self.enable_thinking,
            "vllm_endpoint": self.vllm_base_url if self.use_vllm else None,
        }
        
        # 如果 Transformers 已加载，添加设备信息
        if self._model is not None:
            status["device"] = str(self._model.device)
            status["device_type"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        return status
    
    def is_available(self) -> bool:
        """
        检查 LLM 是否可用
        
        Returns:
            bool: 至少有一种推理方式可用
        """
        if self.use_vllm:
            try:
                vllm_ok = asyncio.run(self.check_vllm_health())
                if vllm_ok:
                    return True
            except:
                pass
        
        # 检查 Transformers 是否可加载
        try:
            self._load_transformers_model()
            return True
        except:
            return False
    
    # ============================================================
    # 配置管理
    # ============================================================
    
    def set_vllm_enabled(self, enabled: bool):
        """
        启用/禁用 vLLM
        
        Args:
            enabled: True 启用 vLLM，False 禁用
        """
        self.use_vllm = enabled
        self._vllm_available = None  # 重置缓存
        logger.info(f"vLLM 已{'启用' if enabled else '禁用'}")
    
    def set_vllm_endpoint(self, host: str, port: int):
        """
        设置 vLLM 端点
        
        Args:
            host: vLLM 服务地址
            port: vLLM 服务端口
        """
        self.vllm_host = host
        self.vllm_port = port
        self.vllm_base_url = f"http://{host}:{port}"
        self._vllm_available = None  # 重置缓存
        logger.info(f"vLLM 端点已更新: {self.vllm_base_url}")
    
    # ============================================================
    # 便捷方法
    # ============================================================
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        简化的生成方法 - 直接输入文本
        
        Args:
            prompt: 提示文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: top_p 采样参数
        
        Returns:
            str: 生成的回复文本
        """
        return self.generate_response(
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    def generate_with_context(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        带上下文的生成
        
        Args:
            context: 上下文信息
            question: 问题
            max_new_tokens: 最大生成 token 数
        
        Returns:
            str: 生成的回复文本
        """
        prompt = f"""基于以下信息回答问题：

上下文:
{context}

问题: {question}

回答:"""
        
        return self.generate(prompt, max_new_tokens, **kwargs)


# ============================================================
# 全局单例获取函数
# ============================================================

def get_llm_client() -> LLMClient:
    """
    获取 LLM 客户端单例
    
    Returns:
        LLMClient: 全局唯一的 LLM 客户端实例
    """
    return LLMClient()


# ============================================================
# 兼容旧代码的别名（平滑过渡）
# ============================================================

# 兼容 qwen_model.py 的接口
get_qwen_model = get_llm_client
QwenModel = LLMClient

# 兼容 vllm_client.py 的接口
get_vllm_client = get_llm_client
VLLMClient = LLMClient


# ============================================================
# 导出列表
# ============================================================

__all__ = [
    'LLMClient',
    'get_llm_client',
    'QwenModel',
    'get_qwen_model',
    'VLLMClient',
    'get_vllm_client',
]