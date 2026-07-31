# app/skills/base_batch_skill.py

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
from ..config import LLM_API_CONFIG


class BaseBatchSkill(BaseSkill):
    """
    Batch processing skill base class
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        batch_size: int = 15,
        max_workers: int = 4,
        timeout: int = 60,
        output_tokens: int = 400,
        safety_margin: float = 0.75  # 75% safety margin
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
        # vLLM 配置中的 max_model_len
        vllm_config = LLM_API_CONFIG.get("vllm", {})
        self.vllm_max_model_len = vllm_config.get("max_model_len", self.max_model_len)
        # 使用较小的值作为安全上限
        self.max_context_tokens = min(self.max_model_len, self.vllm_max_model_len)
        
        # ✅ 计算安全可用 tokens（预留输出 tokens 和安全裕度）
        self.available_tokens = int(self.max_context_tokens * self.safety_margin) - self.output_tokens
        
        print(f"   📊 Max context tokens: {self.max_context_tokens}")
        print(f"   📊 Available tokens (with safety): {self.available_tokens}")
        print(f"   📊 Output tokens reserved: {self.output_tokens}")
        print(f"   📊 Safety margin: {self.safety_margin * 100}%")
        
        # Initialize tokenizer
        try:
            import tiktoken
            self._token_counter = tiktoken.get_encoding("cl100k_base")
        except:
            self._token_counter = None
        
        self._lock = threading.Lock()
        self._results = []
        self._progress = 0
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self._token_counter:
            return len(self._token_counter.encode(text))
        # Fallback: rough estimate
        return len(text) // 3
    
    def _get_base_prompt(self) -> str:
        """Get base prompt template (subclass can override)"""
        return ""
    
    def _build_sample_prompt(self, sample_items: List[Any], **kwargs) -> str:
        """Build sample prompt for token estimation (subclass can override)"""
        return ""
    
    def _get_item_data_str(self, item: Any, **kwargs) -> str:
        """Get item data string (subclass can override)"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('full_data', item.get('summary', str(item)))
        return str(item)
    
    def _build_batch_dynamically(self, items: List[Any], **kwargs) -> List[List[Any]]:
        """
        Dynamically build batches based on token count
        """
        base_prompt = self._get_base_prompt()
        base_tokens = self._count_tokens(base_prompt)
        
        batches = []
        current_batch = []
        current_tokens = base_tokens + self.output_tokens
        
        for item in items:
            # Get item data string
            item_str = self._get_item_data_str(item, **kwargs)
            item_tokens = self._count_tokens(item_str)
            
            # Check if adding this item exceeds limit
            if current_tokens + item_tokens > self.available_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_tokens = base_tokens + self.output_tokens + item_tokens
            else:
                current_batch.append(item)
                current_tokens += item_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Create batches - can be overridden for dynamic batching"""
        # Try dynamic batching first
        try:
            batches = self._build_batch_dynamically(items)
            if batches:
                return batches
        except Exception as e:
            print(f"   ⚠️ Dynamic batching failed, using fixed batch size: {e}")
        
        # Fallback to fixed batch size
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i+self.batch_size])
        return batches
    
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
        """Preprocess items"""
        return items
    
    def _postprocess(self, results: List[Dict], **kwargs) -> List[Dict]:
        """Postprocess results"""
        return results
    
    def _sort_results(self, results: List[Dict]) -> List[Dict]:
        """Sort results"""
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    def _get_symbol(self, item: Any) -> str:
        """Get symbol from item"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('symbol', '')
        return ''
    
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
    # Main execution method
    # ============================================================
    
    def execute(self, items: List[Any], keep_count: int = None, **kwargs) -> List[Dict]:
        """
        Batch execution - with dynamic batch sizing
        """
        if not items:
            return []
        
        print(f"   📊 Processing {len(items)} items ({self.max_workers} threads)...")
        start_time = time.time()
        
        # Preprocess
        processed = self._preprocess(items, **kwargs)
        
        # Create batches (with dynamic sizing)
        batches = self._create_batches(processed)
        print(f"   📦 {len(batches)} batches, avg {len(batches[0]) if batches else 0} items/batch")
        
        # Multi-threaded execution
        all_results = []
        all_results_lock = threading.Lock()
        
        with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
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
                        print(f"      ⚠️ Batch {batch_idx+1} failed: {e}")
                        fallback = self._fallback(batches[batch_idx], **kwargs)
                        with all_results_lock:
                            all_results.extend(fallback)
                    pbar.update(1)
        
        # Postprocess
        results = self._postprocess(all_results, **kwargs)
        results = self._sort_results(results)
        
        if keep_count is not None:
            results = results[:keep_count]
        
        elapsed = time.time() - start_time
        print(f"      ✅ Completed, time used {elapsed:.2f}s, retained {len(results)} items")
        
        return results
    
    # ============================================================
    # Async version
    # ============================================================
    
    async def execute_async(self, items: List[Any], keep_count: int = None, **kwargs) -> List[Dict]:
        """Async execution"""
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