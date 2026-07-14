
import time
import threading
import psutil
import requests
from typing import Dict, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    name: str
    status: str  # healthy, degraded, unhealthy
    last_check: Optional[datetime]
    uptime: float
    error_count: int


class StabilityManager:
    """本地部署稳定性管理器"""
    
    def __init__(self):
        self.services = {}
        self.is_running = False
        self.check_interval = 30
        self.error_threshold = 3
        self.max_restarts = 5
        
        # 资源监控
        self.resource_thresholds = {
            "cpu": 90,      # CPU 使用率阈值
            "memory": 85,   # 内存使用率阈值
            "gpu": 90,      # GPU 使用率阈值
        }
        
        logger.info("🛡️ 稳定性管理器初始化完成")
    
    def register_service(self, name: str, check_func: Callable, 
                         recover_func: Optional[Callable] = None,
                         recover_delay: int = 5):
        """注册服务"""
        self.services[name] = {
            "check": check_func,
            "recover": recover_func,
            "recover_delay": recover_delay,
            "status": ServiceStatus(
                name=name,
                status="unknown",
                last_check=None,
                uptime=0,
                error_count=0
            ),
            "restart_count": 0,
        }
        logger.info(f"📋 服务已注册: {name}")
    
    def start(self):
        """启动稳定性监控"""
        self.is_running = True
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()
        logger.info("✅ 稳定性监控已启动")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            for name, service in self.services.items():
                self._check_service(name, service)
            
            # 检查资源使用
            self._check_resources()
            
            time.sleep(self.check_interval)
    
    def _check_service(self, name: str, service: Dict):
        """检查单个服务"""
        try:
            status = service["check"]()
            service["status"].last_check = datetime.now()
            
            if status:
                service["status"].status = "healthy"
                service["status"].error_count = 0
            else:
                service["status"].error_count += 1
                service["status"].status = "degraded"
                
                if service["status"].error_count >= self.error_threshold:
                    service["status"].status = "unhealthy"
                    self._try_recover(name, service)
        except Exception as e:
            logger.error(f"❌ 检查服务 {name} 失败: {e}")
            service["status"].error_count += 1
    
    def _try_recover(self, name: str, service: Dict):
        """尝试恢复服务"""
        if service["restart_count"] >= self.max_restarts:
            logger.error(f"❌ 服务 {name} 重启次数已达上限")
            return
        
        if service["recover"]:
            logger.warning(f"🔄 尝试恢复服务: {name}")
            try:
                service["recover"]()
                service["restart_count"] += 1
                time.sleep(service["recover_delay"])
                logger.info(f"✅ 服务 {name} 已恢复")
            except Exception as e:
                logger.error(f"❌ 恢复服务 {name} 失败: {e}")
    
    def _check_resources(self):
        """检查系统资源"""
        # CPU 使用率
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > self.resource_thresholds["cpu"]:
            logger.warning(f"⚠️ CPU 使用率高: {cpu_percent}%")
        
        # 内存使用率
        memory = psutil.virtual_memory()
        if memory.percent > self.resource_thresholds["memory"]:
            logger.warning(f"⚠️ 内存使用率高: {memory.percent}%")
        
        # GPU 使用率（如果有）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    if gpu.load * 100 > self.resource_thresholds["gpu"]:
                        logger.warning(f"⚠️ GPU 使用率高: {gpu.load * 100:.1f}%")
        except:
            pass
    
    def get_status_report(self) -> Dict:
        """获取状态报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "resources": {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent,
                "disk": psutil.disk_usage('/').percent,
            }
        }
        
        for name, service in self.services.items():
            report["services"][name] = {
                "status": service["status"].status,
                "last_check": service["status"].last_check.isoformat() if service["status"].last_check else None,
                "error_count": service["status"].error_count,
                "restart_count": service["restart_count"],
            }
        
        return report
    
    def stop(self):
        """停止稳定性监控"""
        self.is_running = False
        logger.info("⏹️ 稳定性监控已停止")