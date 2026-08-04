# app/skills/base_batch_skill.py
"""
Base Batch Skill with Milvus caching support
"""

import threading
import time
import re
import json
import gc
from typing import Dict, List, Any, Optional, Callable, Iterator, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from abc import abstractmethod
from datetime import datetime
import pandas as pd

from .base_skill import BaseSkill
from ..llm_client import get_llm_client
from ..config import LLM_API_CONFIG
from ..milvus_client import get_milvus_client
from pymilvus import DataType
from ..utils import format_exception


class BaseBatchSkill(BaseSkill):
    """
    Batch processing skill base class with memory optimization and Milvus caching
    """
    _execution_lock = threading.Lock()
    def __init__(
        self,
        name: str,
        description: str,
        batch_size: int = 15,
        max_workers: int = 4,
        timeout: int = 60,
        output_tokens: int = 400*2,
        safety_margin: float = 0.75,
        enable_cache: bool = True  ,
        parallel_batches: int = 2
        
    ):
        super().__init__(name, description)
        
        self.llm = get_llm_client()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.output_tokens = output_tokens
        self.safety_margin = safety_margin
        self.enable_cache = enable_cache
        self.parallel_batches = parallel_batches
        
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
        print(f"   🚀 Parallel batches: {self.parallel_batches}")
        
        # Initialize tokenizer
        self._init_token_counter()
        
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
        
        # ✅ Milvus 缓存
        if self.enable_cache:
            try:
                self._milvus = get_milvus_client()
                self._cache_available = self._milvus.recommendation is not None
                if self._cache_available:
                    print(f"   ✅ Milvus cache enabled")
                else:
                    print(f"   ⚠️ Milvus cache not available")
                    self.enable_cache = False
            except Exception as e:
                print(f"   ⚠️ Milvus init failed: {e}, cache disabled")
                self.enable_cache = False
                self._milvus = None

    def _init_token_counter(self):
        """初始化 token 计数器"""
        self._token_counter = None
        try:
            import tiktoken
            self._token_counter = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            try:
                from transformers import AutoTokenizer
                self._token_counter = AutoTokenizer.from_pretrained("gpt2")
            except:
                pass
        except:
            pass
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text - 使用多种方法"""
        if not text:
            return 0
        
        # 方法1: 使用 tiktoken
        if self._token_counter is not None:
            try:
                if hasattr(self._token_counter, 'encode'):
                    return len(self._token_counter.encode(text))
            except:
                pass
        
        # 方法2: 简单估算（约 4 字符 = 1 token）
        return len(text) // 4 + 1
        
    def _extract_json(self, response: str) -> Optional[Dict]:
        """从响应中提取 JSON - 支持不完整JSON"""
        if not response:
            return None
        
        # ✅ 方法1: 尝试直接解析完整JSON
        try:
            return json.loads(response.strip())
        except:
            pass
        
        # ✅ 方法2: 尝试提取到最后一个完整的对象
        try:
            # 找到第一个 {
            start = response.find('{')
            if start == -1:
                return None
            
            # 从前往后扫描，找到最后一个完整的 }
            brace_count = 0
            last_complete_end = -1
            for i in range(start, len(response)):
                char = response[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_complete_end = i
            
            if last_complete_end != -1:
                # 提取到最后一个完整的 } 为止
                json_str = response[start:last_complete_end+1]
                try:
                    return json.loads(json_str)
                except:
                    pass
        except:
            pass
        
        # ✅ 方法3: 尝试补全不完整的JSON
        try:
            # 找到第一个 {
            start = response.find('{')
            if start == -1:
                return None
            
            # 获取从 { 到末尾的内容
            partial = response[start:]
            
            # 检查是否以 { 开头
            if partial.startswith('{'):
                # 尝试补全
                # 1. 补全缺失的 ]
                if partial.count('[') > partial.count(']'):
                    partial += ']'
                # 2. 补全缺失的 }
                if partial.count('{') > partial.count('}'):
                    partial += '}'
                # 3. 补全缺失的 "
                if partial.count('"') % 2 != 0:
                    partial += '"'
                
                try:
                    return json.loads(partial)
                except:
                    pass
        except:
            pass
        
        return None
    
    def _extract_json_array(self, response: str) -> Optional[List]:
        """从响应中提取 JSON 数组"""
        if not response:
            return None
        try:
            # 尝试找到 JSON 数组
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None
    
    def _parse_json_response(self, response: str, key: str = None) -> Optional[Any]:
        """
        解析 JSON 响应，支持多种格式
        - 如果是对象，返回对象
        - 如果指定 key，返回该 key 的值
        """
        if not response:
            return None
        
        # 清理响应
        cleaned = response.strip()
        
        # 尝试解析
        try:
            data = json.loads(cleaned)
            if key:
                return data.get(key)
            return data
        except:
            pass
        
        # 尝试提取 JSON
        try:
            data = self._extract_json(cleaned)
            if data:
                if key:
                    return data.get(key)
                return data
        except:
            pass
        
        # 尝试提取 JSON 数组
        try:
            data = self._extract_json_array(cleaned)
            if data:
                return data
        except:
            pass
        
        return None
    
    def _ensure_cache_collection(self):
        """确保 Milvus 缓存集合存在"""
        if not self.enable_cache or self._milvus is None:
            print(f'[BaseBatchSkill._ensure_cache_collection] Cache disabled or Milvus not available')
            return
        try:
            # ✅ 安全检查：确保 _client 存在
            if not hasattr(self._milvus, '_client') or self._milvus._client is None:
                print(f"   ⚠️ Milvus client not available")
                self.enable_cache = False
                return
            
            if not self._milvus._client.has_collection(self._cache_collection):
                schema = self._milvus._client.create_schema(
                    auto_id=True,
                    enable_dynamic_field=True
                )
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
                schema.add_field(field_name="symbol", datatype=DataType.VARCHAR, max_length=20)
                schema.add_field(field_name="analysis_type", datatype=DataType.VARCHAR, max_length=50)
                schema.add_field(field_name="latest_date", datatype=DataType.VARCHAR, max_length=20)
                schema.add_field(field_name="result", datatype=DataType.JSON)
                schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=30)
                schema.add_field(field_name="updated_at", datatype=DataType.VARCHAR, max_length=30)
                
                index_params = self._milvus._client.prepare_index_params()
                index_params.add_index(field_name="symbol", index_type="INVERTED")
                index_params.add_index(field_name="analysis_type", index_type="INVERTED")
                
                self._milvus._client.create_collection(
                    collection_name=self._cache_collection,
                    schema=schema,
                    index_params=index_params
                )
                print(f"   ✅ Created Milvus cache collection: {self._cache_collection}")
        except Exception as e:
            print(f"   ⚠️ Milvus cache not available: {e}\nTrackback:{format_exception(e)}")
            self.enable_cache = False
    
    def _is_milvus_available(self) -> bool:
        """检查 Milvus 是否可用"""
        return (self.enable_cache and 
            self._milvus is not None and 
            hasattr(self._milvus, '_client') and 
            self._milvus._client is not None)
        # return (self.enable_cache and 
        #         self._milvus is not None and 
        #         hasattr(self._milvus, '_client') and 
        #         self._milvus._client is not None and
        #         self._milvus._client.has_collection(self._cache_collection))
   
    # ============================================================
    # Milvus 缓存方法
    # ============================================================
    
    def _get_cache_key(self) -> str:
        """获取缓存类型标识，子类可重写"""
        return self.name
    
    def _get_cached_result(self, symbol: str) -> Optional[Dict]:
        """从 Milvus 获取缓存结果"""
        if not self.enable_cache or self._milvus is None:
            print(f'[BaseBatchSkill._get_cached_result] self.enable_cache not available or self._milvus is None')
            return None
        
        try:
            cache_type = self._get_cache_key()
            return self._milvus.recommendation.get(symbol, cache_type)
            print(f'[BaseBatchSkill._get_cached_result] 获取缓存成功: {symbol} {cache_type}')
        except Exception as e:
            print(f'[BaseBatchSkill._get_cached_result] 获取缓存失败: {e}\nTrackback:{format_exception(e)}')
            return None
    
    def _save_cached_result(self, symbol: str, latest_date: str, result: Dict):
        """保存结果到 Milvus"""
        if not self.enable_cache or self._milvus is None:
            print(f'[BaseBatchSkill._save_cached_result] self.enable_cache not available or self._milvus is None')
            return
        
        try:
            cache_type = self._get_cache_key()
            self._milvus.recommendation.save(symbol, cache_type, latest_date, result)
            print(f'[BaseBatchSkill._save_cached_result] 保存缓存成功: {symbol} {cache_type} {latest_date}')
        except Exception as e:
            print(f'[BaseBatchSkill._save_cached_result] 保存缓存失败: {e}\nTrackback:{format_exception(e)}')
    
    def _clear_cache(self, symbol: Optional[str] = None):
        """清除缓存"""
        if not self.enable_cache or self._milvus is None:
            print(f'[BaseBatchSkill._clear_cache] self.enable_cache not available or self._milvus is None')
            return
        
        try:
            cache_type = self._get_cache_key()
            if symbol:
                self._milvus.recommendation.clear(symbol, cache_type)
            else:
                self._milvus.recommendation.clear()
            
            print(f'[BaseBatchSkill._clear_cache] 清除缓存成功: {cache_type}')
        except Exception as e:
            print(f'[BaseBatchSkill._clear_cache] 清除缓存失败: {e}\nTrackback:{format_exception(e)}')
            pass
    
    # ============================================================
    # 缓存检查与处理
    # ============================================================
    
    def _check_cache_for_item(self, symbol: str, data: Any) -> tuple[Optional[Dict], bool]:
        """
        检查单个项目的缓存
        返回: (缓存结果, 是否命中)
        """
        if not self._is_milvus_available():
            return None, False
        
        # 获取数据最新日期
        latest_date = self._get_data_latest_date(data)
        if not latest_date:
            return None, False
        
        # 查询缓存
        cached = self._get_cached_result(symbol)
        if cached and cached.get("latest_date") == latest_date:
            print(f"[BaseBatchSkill._check_cache_for_item] 命中缓存: {symbol} {cached.get('result')} {latest_date}")
            return cached.get("result"), True
        
        print(f"[BaseBatchSkill._check_cache_for_item] 未命中缓存: {symbol} {latest_date}")
        return None, False
    
    def _get_data_latest_date(self, data: Any) -> Optional[str]:
        """获取数据的最新日期，子类可重写"""
        if data is None:
            return None
        
        try:
            if isinstance(data, pd.DataFrame) and not data.empty:
                # ✅ 检查索引是否为日期类型
                if isinstance(data.index, pd.DatetimeIndex):
                    return data.index[-1].strftime('%Y-%m-%d')
                
                # ✅ 检查 'date' 列
                if 'date' in data.columns:
                    return data['date'].iloc[-1].strftime('%Y-%m-%d')
                
                # ✅ 检查其他可能的日期列
                for col in data.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            return pd.to_datetime(data[col].iloc[-1]).strftime('%Y-%m-%d')
                        except:
                            continue
        except Exception as e:
            print(f"      ⚠️ _get_data_latest_date error: {e}\nTrackback:{format_exception(e)}")
        
        return None
    
    def _process_batch_with_cache(
        self, 
        batch: List[Dict], 
        cache_check_func: Optional[Callable] = None,
        **kwargs
    ) -> List[Dict]:
        """
        处理批次，自动处理缓存
        cache_check_func: 可选的自定义缓存检查函数
        """
        if not batch:
            return []
        
        # 分离需要分析的和可以从缓存获取的
        to_analyze = []
        cached_results = []
        symbol_to_item = {}
        
        for item in batch:
            symbol = self._get_symbol(item)
            if not symbol:
                continue
            
            # 获取数据
            data = self.get_loaded_data(symbol)
            if data is None:
                data = self._data_cache.get(symbol)
                if data is None:
                    data = self._load_item_data(symbol, **kwargs)
                    if data is not None:
                        with self._cache_lock:
                            self._data_cache[symbol] = data
            
            print(f"[DEBUG] Checking cache for {symbol}, data: {data is not None}")
            # 检查缓存
            if cache_check_func:
                result, hit = cache_check_func(item, data)
            else:
                result, hit = self._check_cache_for_item(symbol, data)
            
            if hit and result:
                print(f"[DEBUG] Cache HIT for {symbol}")
                cached_results.append(result)
                continue
            
            print(f"[DEBUG] Cache MISS for {symbol}")
            # 需要分析
            item_data = self._create_item_from_data(symbol, data) if data is not None else item
            if item_data:
                to_analyze.append(item_data)
        
        print(f"[BaseBatchSkill._process_batch_with_cache] To analyze: len(to_analyze): {len(to_analyze)}, Cached: {len(cached_results)}")
        # 如果全部命中缓存
        if not to_analyze and cached_results:
            print(f"      ✅ All {len(cached_results)} results from cache")
            return cached_results
        
        # 分析需要处理的
        if to_analyze:
            if len(to_analyze) <= self.parallel_batches * self.batch_size:
                return self._process_batches_sequential(to_analyze, cached_results, symbol_to_item, **kwargs)
        
        # ✅ 并发处理多个批次
            return self._process_batches_parallel(to_analyze, cached_results, symbol_to_item, **kwargs)
        
    def _process_batches_sequential(
        self, 
        to_analyze: List[Dict], 
        cached_results: List[Dict], 
        symbol_to_item: Dict[str, Dict],
        **kwargs
    ) -> List[Dict]:
        results = self._process_batch(to_analyze, **kwargs)
        print(f'[_process_batches_sequential] results:{results}')
        if results:
            symbol_to_item = {item.get('symbol'): item for item in to_analyze if item.get('symbol')}
            print(f'[_process_batches_sequential] symbol_to_item:{symbol_to_item}')
            # 保存到缓存
            for r in results:
                print(f'[_process_batches_sequential] Saving cache for {r}')
                symbol = r.get('symbol')
                if not symbol:
                    # 尝试从 to_analyze 中匹配
                    idx = results.index(r)
                    if idx < len(to_analyze):
                        symbol = to_analyze[idx].get('symbol')
                        if symbol:
                            r['symbol'] = symbol
                
                if symbol:
                    # ✅ 从 symbol_to_item 获取数据
                    item_data = symbol_to_item.get(symbol)
                    if item_data:
                        # 尝试从 item_data 获取日期
                        latest_date = item_data.get('_latest_date')
                        
                        if not latest_date:
                            data = self.get_loaded_data(symbol)
                            if data is None:
                                data = self._data_cache.get(symbol)
                            latest_date = self._get_data_latest_date(data)
                    else:
                        data = self.get_loaded_data(symbol)
                        if data is None:
                            data = self._data_cache.get(symbol)
                        latest_date = self._get_data_latest_date(data)
                    
                    print(f'[_process_batches_sequential] latest_date:{latest_date}')
                    if latest_date:
                        self._save_cached_result(symbol, latest_date, r)
            
            results.extend(cached_results)
            return results
        else:
            # ✅ 分析失败，返回缓存结果
            return cached_results
    
    def _process_batches_parallel(
        self, 
        to_analyze: List[Dict], 
        cached_results: List[Dict], 
        symbol_to_item: Dict[str, Dict],
        **kwargs
    ) -> List[Dict]:
        """
        ✅ 并发处理多个批次
        """
        # 将 to_analyze 分割成多个批次
        batches = []
        batch_size = self.batch_size
        for i in range(0, len(to_analyze), batch_size):
            batch = to_analyze[i:i+batch_size]
            batches.append(batch)
        
        print(f"   🚀 Processing {len(to_analyze)} items in {len(batches)} batches, "
              f"parallel limit: {self.parallel_batches}")
        
        all_results = []
        results_lock = threading.Lock()
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=self.parallel_batches) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self._process_single_batch, batch, kwargs): batch
                for batch in batches
            }
            
            # 使用 tqdm 显示进度
            with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results = future.result(timeout=self.timeout)
                        if batch_results:
                            with results_lock:
                                all_results.extend(batch_results)
                    except TimeoutError:
                        print(f"   ⚠️ Batch processing timeout")
                    except Exception as e:
                        print(f"   ⚠️ Batch processing failed: {e}")
                    pbar.update(1)
        
        # 保存缓存
        if all_results:
            for r in all_results:
                symbol = r.get('symbol')
                if symbol:
                    item_data = symbol_to_item.get(symbol)
                    if item_data:
                        latest_date = item_data.get('_latest_date')
                        if not latest_date:
                            data = self.get_loaded_data(symbol)
                            if data is None:
                                data = self._data_cache.get(symbol)
                            latest_date = self._get_data_latest_date(data)
                    else:
                        data = self.get_loaded_data(symbol)
                        if data is None:
                            data = self._data_cache.get(symbol)
                        latest_date = self._get_data_latest_date(data)
                    
                    if latest_date:
                        self._save_cached_result(symbol, latest_date, r)
            
            all_results.extend(cached_results)
            return all_results
        
        return cached_results
    
    def _process_single_batch(
        self, 
        batch: List[Dict], 
        kwargs: Dict
    ) -> List[Dict]:
        """
        ✅ 处理单个批次（线程安全）
        """
        try:
            return self._process_batch(batch, **kwargs)
        except Exception as e:
            print(f"   ⚠️ Batch processing error: {e}")
            return []
    # ============================================================
    # 流式数据处理核心
    # ============================================================
    
    def _streaming_process(
        self, 
        items: List[Any], 
        data_loader: Callable,
        **kwargs
    ) -> List[Dict]:
        """
        流式处理：边加载边生成prompt，达到限制时立即处理
        """
        all_results = []
        all_loaded_data = {}
        selected_symbols = set()
        
        base_prompt = self._get_base_prompt()
        base_tokens = self._count_tokens(base_prompt)
        
        current_batch = []
        current_data = {}
        current_tokens = base_tokens + self.output_tokens
        
        total_items = len(items)
        
        with tqdm(total=total_items, desc="Streaming processing", unit="items") as pbar:
            for item in items:
                symbol = self._get_symbol(item)
                if not symbol:
                    pbar.update(1)
                    continue
                
                if symbol in self._data_cache:
                    data = self._data_cache[symbol]
                else:
                    data = data_loader(symbol)
                    if data is None:
                        pbar.update(1)
                        continue
                    with self._cache_lock:
                        if len(self._data_cache) < self._max_cache_size:
                            self._data_cache[symbol] = data
                
                all_loaded_data[symbol] = data
                
                item_str = self._get_item_data_str_for_item(item, data, **kwargs)
                item_tokens = self._count_tokens(item_str)
                
                if current_tokens + item_tokens > self.actual_limit:
                    if current_batch:
                        batch_results, batch_selected = self._process_batch_with_data_and_return_selected(
                            current_batch, current_data, **kwargs
                        )
                        if batch_results:
                            all_results.extend(batch_results)
                            selected_symbols.update(batch_selected)
                            self._update_cache_with_selected(current_batch, batch_selected)
                    
                    current_batch = [symbol]
                    current_data = {symbol: data}
                    current_tokens = base_tokens + self.output_tokens + item_tokens
                else:
                    current_batch.append(symbol)
                    current_data[symbol] = data
                    current_tokens += item_tokens
                
                pbar.update(1)
            
            if current_batch:
                batch_results, batch_selected = self._process_batch_with_data_and_return_selected(
                    current_batch, current_data, **kwargs
                )
                if batch_results:
                    all_results.extend(batch_results)
                    selected_symbols.update(batch_selected)
                    self._update_cache_with_selected(current_batch, batch_selected)
        
        self._all_loaded_data = {
            symbol: data for symbol, data in all_loaded_data.items()
            if symbol in selected_symbols
        }
        self._keep_symbols = selected_symbols
        
        print(f"   📊 Selected {len(selected_symbols)} symbols, kept {len(self._all_loaded_data)} data items")
        
        gc.collect()
        
        return all_results
    
    def _process_batch_with_data_and_return_selected(
        self, 
        batch: List[str], 
        batch_data: Dict[str, Any], 
        **kwargs
    ) -> tuple[List[Dict], Set[str]]:
        """使用已加载的数据处理批次，返回结果和选中的symbol集合"""
        selected_symbols = set()
        
        batch_items = []
        for symbol in batch:
            data = batch_data.get(symbol)
            if data is not None:
                item = self._create_item_from_data(symbol, data)
                if item:
                    batch_items.append(item)
        
        if not batch_items:
            return [], set()
        
        results = self._process_batch_with_cache(batch_items, **kwargs)
        
        for r in results:
            symbol = r.get('symbol')
            if symbol:
                selected_symbols.add(symbol)
        
        return results, selected_symbols
    
    def _update_cache_with_selected(self, batch: List[str], selected_symbols: Set[str]):
        """更新缓存：只保留选中基金的数据"""
        
        with self._cache_lock:
            for sym in batch:
                if sym not in selected_symbols:
                    if sym in self._data_cache:
                        del self._data_cache[sym]
    
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
        """从数据创建item（子类可重写）"""
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
        def get_score(item):
            score = item.get('score', 0)
            # ✅ 确保 score 是数值类型
            try:
                return int(score)
            except (ValueError, TypeError):
                return 0
        return sorted(results, key=get_score, reverse=True)
        # return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
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
        
        with BaseBatchSkill._execution_lock:
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