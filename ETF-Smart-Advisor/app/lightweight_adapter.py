import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LightweightConfig:
    """轻量化配置"""
    quantization: str = "int8"           # int8, fp16, fp32
    max_seq_length: int = 2048
    batch_size: int = 8
    enable_cache: bool = True
    cache_ttl: int = 3600
    enable_compression: bool = True
    max_memory_gb: float = 4.0
    max_cpu_percent: int = 50
    max_gpu_percent: int = 60


class LightweightAdapter:
    """轻量化适配器 - 针对资源受限环境优化"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.cache = {}
        self.logger = logger
        
        logger.info(f"📱 轻量化适配器初始化完成")
        logger.info(f"   🔧 量化: {self.config.quantization}")
        logger.info(f"   📏 最大序列长度: {self.config.max_seq_length}")
        logger.info(f"   💾 缓存: {'启用' if self.config.enable_cache else '禁用'}")
    
    def _load_config(self, config_path: Optional[str]) -> LightweightConfig:
        """加载配置"""
        config = LightweightConfig()
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
            except:
                pass
        
        return config
    
    def adapt_model_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """适配模型输入"""
        adapted = input_data.copy()
        
        # 限制序列长度
        if "prompt" in adapted:
            adapted["prompt"] = adapted["prompt"][:self.config.max_seq_length]
        
        # 应用量化设置
        if self.config.quantization == "int8":
            adapted["use_quantized"] = True
        
        # 缓存控制
        if self.config.enable_cache:
            cache_key = self._get_cache_key(adapted)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                adapted["cached"] = True
                adapted["response"] = cached_result
        
        return adapted
    
    def _get_cache_key(self, data: Dict) -> str:
        """生成缓存键"""
        import hashlib
        key_string = f"{data.get('prompt', '')}_{data.get('symbol', '')}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.config.cache_ttl:
                return entry["data"]
        return None
    
    def _add_to_cache(self, key: str, data: Any):
        """添加到缓存"""
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
        # 限制缓存大小
        if len(self.cache) > 1000:
            # 删除最旧的条目
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest]
    
    def optimize_workflow(self, workflow: Dict) -> Dict:
        """优化工作流"""
        # 根据资源情况调整工作流
        import psutil
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > self.config.max_cpu_percent:
            workflow["batch_size"] = min(workflow.get("batch_size", 8), 4)
            workflow["priority"] = "low"
        
        if memory_percent > 70:
            workflow["use_compression"] = True
        
        return workflow
    
    def get_resource_report(self) -> Dict:
        """获取资源报告"""
        import psutil
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_memory = torch.cuda.memory_allocated() / 1024**3 if gpu_available else 0
        except:
            gpu_available = False
            gpu_memory = 0
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
            "gpu_available": gpu_available,
            "gpu_memory_used_gb": gpu_memory,
            "cache_size": len(self.cache),
        }