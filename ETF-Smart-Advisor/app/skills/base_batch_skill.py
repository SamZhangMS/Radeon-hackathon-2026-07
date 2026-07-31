# app/skills/base_batch_skill.py
"""
批量处理技能基类
"""

import threading
import time
import re
import json
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from abc import abstractmethod

from .base_skill import BaseSkill
from ..llm_client import get_llm_client


class BaseBatchSkill(BaseSkill):
    """
    批量处理技能基类
    抽象：多线程分批处理、进度条、错误处理、降级机制
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        batch_size: int = 15,
        max_workers: int = 4,
        timeout: int = 60,
        output_tokens: int = 500
    ):
        super().__init__(name, description)
        
        self.llm = get_llm_client()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.output_tokens = output_tokens
        
        # 线程安全
        self._lock = threading.Lock()
        self._results = []
        self._progress = 0
    
    # ============================================================
    # 公共执行方法
    # ============================================================
    
    def execute(self, items: List[Any], keep_count: int = None, **kwargs) -> List[Dict]:
        """
        批量执行 - 多线程版本
        """
        if not items:
            return []
        
        print(f"   📊 开始处理 {len(items)} 个项目 ({self.max_workers} 线程)...")
        start_time = time.time()
        
        # 预处理
        processed = self._preprocess(items, **kwargs)
        
        # 分批
        batches = self._create_batches(processed)
        print(f"   📦 共 {len(batches)} 批，每批 {self.batch_size} 个")
        
        # 多线程执行
        all_results = []
        all_results_lock = threading.Lock()
        
        with tqdm(total=len(batches), desc="处理批次", unit="批") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._process_batch, batch, **kwargs): i
                    for i, batch in enumerate(batches)
                }
                
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result(timeout=self.timeout)
                        if batch_results:
                            with all_results_lock:
                                all_results.extend(batch_results)
                    except Exception as e:
                        print(f"      ⚠️ 批次 {batch_idx+1} 失败: {e}")
                        fallback = self._fallback(batches[batch_idx], **kwargs)
                        with all_results_lock:
                            all_results.extend(fallback)
                    pbar.update(1)
        
        # 后处理
        results = self._postprocess(all_results, **kwargs)
        
        # 排序
        results = self._sort_results(results)
        
        if keep_count is not None:
            results = results[:keep_count]
        
        elapsed = time.time() - start_time
        print(f"      ✅ 完成，耗时 {elapsed:.2f}s，保留 {len(results)} 个")
        
        return results
    
    # ============================================================
    # 子类必须实现的方法
    # ============================================================
    
    @abstractmethod
    def _process_batch(self, batch: List[Any], **kwargs) -> List[Dict]:
        """处理单批数据"""
        pass
    
    @abstractmethod
    def _fallback(self, batch: List[Any], **kwargs) -> List[Dict]:
        """降级处理"""
        pass
    
    # ============================================================
    # 子类可重写的方法
    # ============================================================
    
    def _preprocess(self, items: List[Any], **kwargs) -> List[Any]:
        """预处理"""
        return items
    
    def _postprocess(self, results: List[Dict], **kwargs) -> List[Dict]:
        """后处理"""
        return results
    
    def _sort_results(self, results: List[Dict]) -> List[Dict]:
        """排序"""
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    def _get_symbol(self, item: Any) -> str:
        """获取symbol"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('symbol', '')
        return ''
    
    # ============================================================
    # 公共工具方法
    # ============================================================
    
    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """创建批次"""
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i+self.batch_size])
        return batches
    
    def _extract_json(self, response: str) -> Optional[Dict]:
        """从响应中提取JSON"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None
    
    # ============================================================
    # 异步版本
    # ============================================================
    
    async def execute_async(self, items: List[Any], keep_count: int = None, **kwargs) -> List[Dict]:
        """异步执行"""
        import asyncio
        
        if not items:
            return []
        
        processed = self._preprocess(items, **kwargs)
        batches = self._create_batches(processed)
        
        tasks = [
            asyncio.get_event_loop().run_in_executor(
                None, self._process_batch, batch, **kwargs
            )
            for batch in batches
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                fallback = self._fallback(batches[i], **kwargs)
                all_results.extend(fallback)
            elif result:
                all_results.extend(result)
        
        results = self._postprocess(all_results, **kwargs)
        results = self._sort_results(results)
        
        if keep_count is not None:
            results = results[:keep_count]
        
        return results