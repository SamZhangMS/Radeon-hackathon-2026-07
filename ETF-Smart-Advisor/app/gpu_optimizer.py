import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ROCmGPUOptimizer:
    """AMD ROCm GPU 优化器 - 专为 AMD Radeon GPU 优化"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.memory_pool = None
        self.kernel_cache = {}
        self.optimization_level = self._detect_gpu_capability()
        
        logger.info(f"🚀 AMD GPU 优化器初始化完成")
        logger.info(f"   📊 GPU: {self._get_gpu_info()}")
        logger.info(f"   ⚡ 优化等级: {self.optimization_level}")
    
    def _setup_device(self):
        """配置 ROCm 设备"""
        # 设置 ROCm 可见设备
        os.environ["ROCM_VISIBLE_DEVICES"] = "0"
        
        # 启用 ROCm 内存池
        os.environ["PYTORCH_ROCM_MEMORY_POOL"] = "1"
        
        # 设置 GPU 内存分配策略
        os.environ["PYTORCH_ROCM_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # 启用 ROCm 图优化
        os.environ["TORCH_ROCM_GRAPH"] = "1"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            # ROCm 特定优化
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            
            # 尝试启用 Flash Attention
            self._try_enable_flash_attention()
        
        return device
    
    def _detect_gpu_capability(self) -> str:
        """检测 GPU 能力，返回优化等级"""
        if not torch.cuda.is_available():
            return "cpu"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if "MI" in gpu_name or "Instinct" in gpu_name:
            return "ultra"  # 数据中心 GPU
        elif gpu_memory >= 16:
            return "high"   # 高端消费级 GPU
        elif gpu_memory >= 8:
            return "medium" # 中端消费级 GPU
        else:
            return "low"    # 入门级 GPU
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取 GPU 信息"""
        if not torch.cuda.is_available():
            return {"name": "CPU", "memory": 0, "compute_capability": 0}
        
        return {
            "name": torch.cuda.get_device_name(0),
            "memory": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "compute_capability": torch.cuda.get_device_capability(0),
        }
    
    def _try_enable_flash_attention(self):
        """尝试启用 Flash Attention"""
        try:
            import flash_attn
            logger.info("✅ Flash Attention 已启用")
            return True
        except ImportError:
            logger.warning("⚠️ Flash Attention 未安装，使用标准 attention")
            return False
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细性能统计（供 benchmark.py 使用）"""
        stats = self.get_performance_stats()
        
        # ✅ 添加更多统计信息
        if torch.cuda.is_available():
            stats.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_free": (torch.cuda.get_device_properties(0).total_memory - 
                                   torch.cuda.memory_allocated()) / 1024**3,
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0,
            })
        
        return stats
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """优化模型推理速度"""
        logger.info("🔧 开始优化模型...")
        
        # 1. 根据优化等级调整模型
        if self.optimization_level in ["high", "ultra"]:
            # 混合精度训练/推理
            model = model.half()
            logger.info("   ✅ 混合精度 (FP16) 已启用")
        
        # 2. 模型量化（INT8）
        if self.optimization_level in ["medium", "low", "high"]:
            try:
                from torch.ao.quantization import quantize_dynamic
                model = quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.qint8
                )
                logger.info("   ✅ 动态量化 (INT8) 已启用")
            except Exception as e:
                logger.warning(f"   ⚠️ 量化失败: {e}")
        
        # 3. 使用 torch.jit 编译
        try:
            model = torch.jit.script(model)
            logger.info("   ✅ JIT 编译已启用")
        except Exception as e:
            logger.warning(f"   ⚠️ JIT 编译失败: {e}")
        
        # 4. 转移到 GPU
        model = model.to(self.device)
        logger.info(f"   ✅ 模型已转移到 {self.device}")
        
        return model
    
    def inference_optimized(self, model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """优化后的推理"""
        # 转移到 GPU
        input_tensor = input_tensor.to(self.device)
        
        # 使用 torch.no_grad 减少内存使用
        with torch.no_grad():
            # 混合精度推理
            if self.optimization_level in ["high", "ultra"]:
                with torch.cuda.amp.autocast():
                    output = model(input_tensor)
            else:
                output = model(input_tensor)
        
        # 返回 CPU 结果
        return output.cpu()
    
    def optimize_vllm_config(self) -> Dict[str, Any]:
        """生成优化的 vLLM 配置"""
        base_config = {
            "gpu_memory_utilization": 0.95,
            "max_num_seqs": 32,
            "max_num_batched_tokens": 8192,
            "dtype": "bfloat16",
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        }
        
        # 根据优化等级调整
        if self.optimization_level == "ultra":
            base_config.update({
                "max_num_seqs": 64,
                "max_num_batched_tokens": 16384,
                "enable_flash_attn": True,
            })
        elif self.optimization_level == "high":
            base_config.update({
                "max_num_seqs": 48,
                "max_num_batched_tokens": 12288,
            })
        elif self.optimization_level == "medium":
            base_config.update({
                "max_num_seqs": 24,
                "max_num_batched_tokens": 4096,
            })
        else:  # low
            base_config.update({
                "gpu_memory_utilization": 0.7,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 2048,
            })
        
        return base_config
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not torch.cuda.is_available():
            return {"status": "cpu_mode"}
        
        return {
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
            "gpu_memory_max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "optimization_level": self.optimization_level,
            "device": str(self.device),
        }