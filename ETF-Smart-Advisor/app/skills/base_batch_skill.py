# app/skills/base_batch_skill.py - 修复数据保留逻辑

import threading
import time
import re
import json
import gc
from typing import Dict, List, Any, Optional, Callable, Iterator, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from abc import abstractmethod

from .base_skill import BaseSkill
from ..llm_client import get_llm_client
from ..config import LLM_API_CONFIG


class BaseBatchSkill(BaseSkill):
    """
    Batch processing skill base class with memory optimization
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        batch_size: int = 15,
        max_workers: int = 4,
        timeout: int = 60,
        output_tokens: int = 400,
        safety_margin: float = 0.75
    ):
        super().__init__(name, description)
        
        self.llm = get_llm_client()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.output_tokens = output_tokens
        self.safety_margin = safety_margin
        
        # ✅ 从 config 读取 max_model_len
        self.max_model_len = LLM_API_CONFIG.get("max_model_len", 65536)
        vllm_config = LLM_API_CONFIG.get("vllm", {})
        self.vllm_max_model_len = vllm_config.get("max_model_len", self.max_model_len)
        self.max_context_tokens = min(self.max_model_len, self.vllm_max_model_len)
        
        # ✅ 计算安全可用 tokens
        self.available_tokens = int(self.max_context_tokens * self.safety_margin) - self.output_tokens
        if self.available_tokens < 1000:
            self.available_tokens = 1000
        
        # ✅ 实际限制（更保守）
        self.actual_limit = int(self.available_tokens * 0.85)
        
        print(f"   📊 Max context: {self.max_context_tokens}, Available: {self.available_tokens}")
        print(f"   📊 Actual limit: {self.actual_limit}")
        
        # Initialize tokenizer
        try:
            import tiktoken
            self._token_counter = tiktoken.get_encoding("cl100k_base")
        except:
            self._token_counter = None
        
        self._lock = threading.Lock()
        self._results = []
        self._progress = 0
        
        # ✅ 内存优化：数据缓存管理
        self._data_cache = {}  # 缓存已加载的数据
        self._cache_lock = threading.Lock()
        self._max_cache_size = 100  # 最大缓存数量
        
        # ✅ 存储所有加载的数据（用于后续过滤）
        self._all_loaded_data = {}
        self._keep_symbols = set()  # 需要保留的symbol集合
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self._token_counter:
            return len(self._token_counter.encode(text))
        return len(text) // 2
    
    def _get_base_prompt(self) -> str:
        """Get base prompt template (subclass can override)"""
        return ""
    
    def _get_item_data_str(self, item: Any, **kwargs) -> str:
        """Get item data string (subclass can override)"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('full_data', item.get('summary', str(item)))
        return str(item)
    
    def _extract_json(self, response: str) -> Optional[Dict]:
        """Extract JSON from response"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None
    
    # ============================================================
    # 流式数据处理核心
    # ============================================================
    
    def _streaming_process(
        self, 
        items: List[Any], 
        data_loader: Callable,  # 加载数据的函数
        **kwargs
    ) -> List[Dict]:
        """
        流式处理：边加载边生成prompt，达到限制时立即处理
        ✅ 每批处理完成后，立即记录选中的基金
        """
        all_results = []
        all_loaded_data = {}  # 本地存储所有加载的数据
        selected_symbols = set()  # ✅ 记录所有被选中的基金
        
        # 1. 获取基础prompt用于token计算
        base_prompt = self._get_base_prompt()
        base_tokens = self._count_tokens(base_prompt)
        
        # 2. 初始化当前批次
        current_batch = []
        current_data = {}  # symbol -> data
        current_tokens = base_tokens + self.output_tokens
        
        # 3. 分批处理进度条
        total_items = len(items)
        
        with tqdm(total=total_items, desc="Streaming processing", unit="items") as pbar:
            for item in items:
                # 加载数据
                symbol = self._get_symbol(item)
                if not symbol:
                    pbar.update(1)
                    continue
                
                # 检查缓存
                if symbol in self._data_cache:
                    data = self._data_cache[symbol]
                else:
                    data = data_loader(symbol)
                    if data is None:
                        pbar.update(1)
                        continue
                    # 缓存数据
                    with self._cache_lock:
                        if len(self._data_cache) < self._max_cache_size:
                            self._data_cache[symbol] = data
                
                # 保存所有加载的数据（用于后续过滤）
                all_loaded_data[symbol] = data
                
                # 生成该项的prompt片段
                item_str = self._get_item_data_str_for_item(item, data, **kwargs)
                item_tokens = self._count_tokens(item_str)
                
                # 检查是否超出限制
                if current_tokens + item_tokens > self.actual_limit:
                    # ✅ 达到限制，处理当前批次
                    if current_batch:
                        batch_results, batch_selected = self._process_batch_with_data_and_return_selected(
                            current_batch, current_data, **kwargs
                        )
                        if batch_results:
                            all_results.extend(batch_results)
                            # ✅ 记录本批选中的基金
                            selected_symbols.update(batch_selected)
                            # ✅ 只保留选中基金的数据在缓存中
                            self._update_cache_with_selected(current_batch, batch_selected)
                    
                    # 开始新批次
                    current_batch = [symbol]
                    current_data = {symbol: data}
                    current_tokens = base_tokens + self.output_tokens + item_tokens
                else:
                    current_batch.append(symbol)
                    current_data[symbol] = data
                    current_tokens += item_tokens
                
                pbar.update(1)
            
            # 处理最后一批
            if current_batch:
                batch_results, batch_selected = self._process_batch_with_data_and_return_selected(
                    current_batch, current_data, **kwargs
                )
                if batch_results:
                    all_results.extend(batch_results)
                    selected_symbols.update(batch_selected)
                    self._update_cache_with_selected(current_batch, batch_selected)
        
        # ✅ 根据所有选中的基金，过滤保存的数据
        self._all_loaded_data = {
            symbol: data for symbol, data in all_loaded_data.items()
            if symbol in selected_symbols
        }
        self._keep_symbols = selected_symbols
        
        print(f"   📊 Selected {len(selected_symbols)} symbols, kept {len(self._all_loaded_data)} data items")
        
        # 强制垃圾回收
        gc.collect()
        
        return all_results
    
    def _process_batch_with_data_and_return_selected(
        self, 
        batch: List[str], 
        batch_data: Dict[str, Any], 
        **kwargs
    ) -> tuple[List[Dict], Set[str]]:
        """
        使用已加载的数据处理批次，返回结果和选中的symbol集合
        """
        selected_symbols = set()
        
        # 构建批次数据
        batch_items = []
        for symbol in batch:
            data = batch_data.get(symbol)
            if data is not None:
                item = self._create_item_from_data(symbol, data)
                if item:
                    batch_items.append(item)
        
        if not batch_items:
            return [], set()
        
        # 调用_process_batch处理
        results = self._process_batch(batch_items, **kwargs)
        
        # ✅ 提取选中的symbol
        for r in results:
            symbol = r.get('symbol')
            if symbol:
                selected_symbols.add(symbol)
        
        return results, selected_symbols
    
    def _update_cache_with_selected(self, batch: List[str], selected_symbols: Set[str]):
        """
        更新缓存：只保留选中基金的数据
        """
        with self._cache_lock:
            for sym in batch:
                if sym not in selected_symbols:
                    # ✅ 未选中的基金，从缓存中删除（释放内存）
                    if sym in self._data_cache:
                        del self._data_cache[sym]
                # 选中的基金保留在缓存中
    
    def _get_item_data_str_for_item(self, item: Any, data: Any, **kwargs) -> str:
        """获取单个项目的prompt片段（子类可重写）"""
        return self._get_item_data_str(item, **kwargs)
    
    def _process_batch_with_data(
        self, 
        batch: List[str], 
        batch_data: Dict[str, Any], 
        **kwargs
    ) -> List[Dict]:
        """兼容旧接口"""
        results, _ = self._process_batch_with_data_and_return_selected(batch, batch_data, **kwargs)
        return results
    
    def _create_item_from_data(self, symbol: str, data: Any) -> Optional[Dict]:
        """
        从数据创建item（子类可重写）
        """
        if isinstance(data, dict):
            data['symbol'] = symbol
            return data
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return {
                    'symbol': symbol,
                    'data': data,
                    'full_data': self._get_summary_from_data(data),
                    'summary': self._get_summary_from_data(data, days=20)
                }
        except:
            pass
        return {'symbol': symbol, 'data': data}
    
    def _get_summary_from_data(self, data: Any, days: int = 60) -> str:
        """从数据获取摘要（子类可重写）"""
        if hasattr(data, 'get_summary'):
            return data.get_summary(days)
        return str(data)[:500]
    
    # ============================================================
    # 内存管理方法
    # ============================================================
    
    def clear_cache(self):
        """清除数据缓存"""
        with self._cache_lock:
            self._data_cache.clear()
        self._all_loaded_data = {}
        self._keep_symbols = set()
        gc.collect()
        print("   ✅ Data cache cleared")
    
    def get_loaded_data(self, symbol: str) -> Optional[Any]:
        """获取已加载的数据（只返回保留的数据）"""
        return self._all_loaded_data.get(symbol) if hasattr(self, '_all_loaded_data') else None
    
    def get_all_loaded_data(self) -> Dict[str, Any]:
        """获取所有已加载的数据（只返回保留的数据）"""
        return getattr(self, '_all_loaded_data', {})
    
    def get_keep_symbols(self) -> Set[str]:
        """获取保留的symbol集合"""
        return getattr(self, '_keep_symbols', set())
    
    # ============================================================
    # Abstract methods (subclasses must implement)
    # ============================================================
    
    @abstractmethod
    def _process_batch(self, batch: List[Any], **kwargs) -> List[Dict]:
        """Process a single batch"""
        pass
    
    @abstractmethod
    def _fallback(self, batch: List[Any], **kwargs) -> List[Dict]:
        """Fallback processing"""
        pass
    
    # ============================================================
    # Optional override methods
    # ============================================================
    
    def _preprocess(self, items: List[Any], **kwargs) -> List[Any]:
        return items
    
    def _postprocess(self, results: List[Dict], **kwargs) -> List[Dict]:
        return results
    
    def _sort_results(self, results: List[Dict]) -> List[Dict]:
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    def _get_symbol(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('symbol', '')
        return ''
    
    # ============================================================
    # Main execution method
    # ============================================================
    
    def execute(self, items: List[Any], keep_count: int = None, **kwargs) -> List[Dict]:
        """
        Batch execution with streaming processing
        ✅ 执行完成后，只保留大模型选中基金的数据
        """
        if not items:
            return []
        
        print(f"   📊 Streaming processing {len(items)} items...")
        start_time = time.time()
        
        # Preprocess
        processed = self._preprocess(items, **kwargs)
        
        # 定义数据加载函数
        def data_loader(symbol: str):
            return self._load_item_data(symbol, **kwargs)
        
        # 流式处理（内部会记录选中的基金并过滤数据）
        results = self._streaming_process(processed, data_loader, **kwargs)
        
        # Postprocess
        results = self._postprocess(results, **kwargs)
        results = self._sort_results(results)
        
        if keep_count is not None:
            results = results[:keep_count]
        
        # ✅ 最终过滤：只保留最终结果中symbol的数据
        final_keep_symbols = {r.get('symbol') for r in results if r.get('symbol')}
        self._all_loaded_data = {
            sym: data for sym, data in self._all_loaded_data.items()
            if sym in final_keep_symbols
        }
        self._keep_symbols = final_keep_symbols
        
        elapsed = time.time() - start_time
        print(f"      ✅ Completed, time used {elapsed:.2f}s, retained {len(results)} items")
        print(f"      📊 Final data cache size: {len(self._all_loaded_data)} items")
        
        return results
    
    def _load_item_data(self, symbol: str, **kwargs) -> Optional[Any]:
        """
        加载单个项目的数据（子类可重写）
        """
        return None
    
    # ============================================================
    # Async version
    # ============================================================
    
    async def execute_async(self, items: List[Any], keep_count: int = None, **kwargs) -> List[Dict]:
        import asyncio
        if not items:
            return []
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.execute, items, keep_count, **kwargs
        )