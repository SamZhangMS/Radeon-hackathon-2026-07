# app/skills/etf_skills.py
"""
ETF related skills implementation
"""

import pandas as pd
import numpy as np
import json
import re
import time
import threading
import gc
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base_skill import BaseSkill
from .base_batch_skill import BaseBatchSkill
from ..data_fetcher import ETFDataFetcher
from ..llm_client import get_llm_client
from ..utils import format_exception


class ETFDataLoader:
    """ETF数据加载器 - 支持流式加载和内存管理"""
    
    def __init__(self):
        self.fetcher = ETFDataFetcher()
        self._cache = {}
        self._max_cache_size = 200
    
    def load_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """加载单个ETF数据"""
        cache_key = f"{symbol}_{days}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            df = self.fetcher.get_history(symbol, f"{days}d")
            if not df.empty:
                if len(self._cache) >= self._max_cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[cache_key] = df
                return df
        except Exception as e:
            print(f"      ⚠️ Failed to load {symbol}: {e}")
        return None
    
    def get_latest_date(self, df: pd.DataFrame) -> Optional[str]:
        """获取数据的最新日期"""
        if df is None or df.empty:
            return None
        
        # ✅ 检查索引是否为日期类型
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index[-1].strftime('%Y-%m-%d')
        
        # ✅ 检查 'date' 列
        if 'date' in df.columns:
            return df['date'].iloc[-1].strftime('%Y-%m-%d')
        
        # ✅ 检查其他可能的日期列
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    return pd.to_datetime(df[col].iloc[-1]).strftime('%Y-%m-%d')
                except:
                    continue
        
        return None
    
    def get_summary(self, df: pd.DataFrame, days: int = 20) -> str:
        """获取数据摘要"""
        if df.empty:
            return ""
        
        df = df.tail(days)
        if 'date' in df.columns:
            df = df.set_index('date')
        
        cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in cols if c in df.columns]
        
        if not available_cols:
            return ""
        
        csv_str = df[available_cols].round(4).to_csv(
            header=False,
            float_format='%.4f',
            date_format='%Y-%m-%d'
        )
        
        return csv_str.strip().replace('\n', '|')
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        import gc
        gc.collect()



class ETFDataSkill(BaseSkill):
    """Skill: ETF Data Fetcher"""
    
    def __init__(self):
        super().__init__(
            name="etf_data_fetcher",
            description="Get ETF historical data, supports single and batch"
        )
        self.fetcher = ETFDataFetcher()
        self.cache = {}
        self.max_workers = 8
    
    def execute(self, symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """Get ETF data - multi-threaded version"""
        if not symbols:
            return {}
        
        print(f"📊 Start loading data of {len(symbols)} stocks ({self.max_workers} threads)...")
        start_time = time.time()
        
        result = {}
        result_lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._get_single, symbol, days): symbol
                for symbol in symbols
            }
            
            with tqdm(total=len(symbols), desc="Loading data", unit="stocks") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result(timeout=30)
                        if df is not None and not df.empty:
                            with result_lock:
                                result[symbol] = df
                    except Exception as e:
                        print(f"      ⚠️ Failed to load data of {symbol}: {e}")
                    pbar.update(1)
                    
        elapsed = time.time() - start_time
        print(f"      ✅ Completed, loaded {len(result)} stocks, time used {elapsed:.2f}s")
        
        return result
    
    def _get_single(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Get single ETF data"""
        try:
            cache_key = f"{symbol}_{days}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            df = self.fetcher.get_history(symbol, f"{days}d")
            if not df.empty:
                self.cache[cache_key] = df
                return df
        except Exception as e:
            print(f"      ⚠️ Failed to load data of {symbol}: {e}")
        return None
    
    def get_summary(self, df: pd.DataFrame, days: int = 20) -> str:
        """
        Get data summary - only raw OHLCV data
        Format: Date|O|H|L|C|V
        """
        if df.empty:
            return ""
        
        df = df.tail(days)
        if 'date' in df.columns:
            df = df.set_index('date')
        
        cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in cols if c in df.columns]
        
        if not available_cols:
            return ""
        
        csv_str = df[available_cols].round(4).to_csv(
            header=False,
            float_format='%.4f',
            date_format='%Y-%m-%d'
        )
        
        return csv_str.strip().replace('\n', '|')



class ETFAnalyzeSkill(BaseBatchSkill):
    """
    Skill: ETF Quick Analysis (Stage 1)
    Function: Quick scoring and analysis for ETFs
    """
    
    def __init__(self):
        super().__init__(
            name="etf_quick_analyze",
            description="Quick scoring for ETFs for initial screening",
            batch_size=3, #5,
            max_workers=2,
            timeout=180,
            output_tokens=800*2,
            safety_margin=0.75,
            enable_cache=True  # ✅ 启用缓存
        )
        self.data_loader = ETFDataLoader()
        self.MIN_KEEP_COUNT = 50
    
    def _get_cache_key(self) -> str:
        """缓存类型标识"""
        return "quick"
    
    def _get_base_prompt(self) -> str:
        return """You are an experienced quantitative analyst. Please analyze the raw OHLCV data of ETFs.

Data format: Symbol|YYYY-MM-DD|O|H|L|C|V
Legend: O=Open, H=High, L=Low, C=Close, V=Volume

Tasks:
1. Calculate technical indicators from raw data (MA5, MA20, MA60, RSI, MACD, etc.)
2. Design your own scoring system
3. Give each ETF a comprehensive score (0-100)
4. Provide signal (buy/hold/sell)

Output JSON:
{
    "scoring_system": {"dimensions": [{"name": "Trend", "weight": 0.35}, ...]},
    "scores": [{"symbol": "Symbol", "score": 0-100 integer, "signal": "buy/hold/sell", "reason": "Reason"}]
}"""
    
    def _load_item_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        days = kwargs.get('data_days', 30)
        return self.data_loader.load_data(symbol, days)
    
    def _get_data_latest_date(self, data: Any) -> Optional[str]:
        """获取数据最新日期"""
        if isinstance(data, pd.DataFrame):
            return self.data_loader.get_latest_date(data)
        return None
    
    def _get_item_data_str_for_item(self, item: Any, data: Any, **kwargs) -> str:
        symbol = self._get_symbol(item)
        if isinstance(data, pd.DataFrame):
            summary = self.data_loader.get_summary(data, days=20)
            return f"{symbol}|{summary}"
        return symbol
    
    def _create_item_from_data(self, symbol: str, data: Any) -> Optional[Dict]:
        if isinstance(data, pd.DataFrame):
            latest_date = self._get_data_latest_date(data)
            return {
                'symbol': symbol,
                'full_data': self.data_loader.get_summary(data, 20),
                '_latest_date': latest_date  # ✅ 保存日期到 item
            }
        return {'symbol': symbol}
    
    def _process_batch(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Analyze a batch of ETFs"""
        if not batch:
            return []
        
        batch_data = []
        for item in batch:
            symbol = item.get('symbol', '')
            full_data = item.get('full_data', '')
            if full_data:
                batch_data.append(f"{symbol}|{full_data}")
        
        if not batch_data:
            return []
        
        batch_symbols = [item.get('symbol') for item in batch if item.get('symbol')]
        data_text = "\n".join(batch_data)
        prompt = f"""Analyze the following {len(batch_data)} ETFs' raw OHLCV data.

Data format: Symbol|Date|O|H|L|C|V

Data:
{data_text}

Output JSON ONLY (no other text):
{{
    "scores": [
        {{"symbol": "Symbol", "score": 0-100 integer, "signal": "buy/hold/sell", "reason": "short reason"}}
    ]
}}"""

        try:
            # print(f'ETFAnalyzeSkill._process_batch\n{prompt}')
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=self.output_tokens,
                temperature=0.3,
                enable_thinking=False
            )
            print(f'\nETFAnalyzeSkill._process_batch response: \n{response}')
            results = self._parse_response(response, batch_symbols)
            return results 
        except Exception as e:
            print(f"      ⚠️ LLM analysis failed: {e}")
            return []    
    def _parse_response(self, response: str, batch_symbols: List[str] = None) -> List[Dict]:
        """Parse LLM response with symbol matching - 支持不完整JSON"""
        results = []
        
        print(f"[DEBUG] _parse_response: response length={len(response)}")
        print(f"[DEBUG] _parse_response: batch_symbols={batch_symbols[:5] if batch_symbols else None}...")
        
        # ✅ 方法1: 尝试提取完整的JSON
        try:
            data = self._extract_json(response)
            print(f"[DEBUG] _parse_response: extracted data={data is not None}")
            if data:
                scores = data.get('scores', [])
                if scores:
                    for idx, item in enumerate(scores):
                        if not item.get('symbol') and batch_symbols and idx < len(batch_symbols):
                            item['symbol'] = batch_symbols[idx]
                        if item.get('symbol'):
                            if 'score' in item:
                                try:
                                    item['score'] = int(item['score'])
                                except (ValueError, TypeError):
                                    item['score'] = 50
                            results.append(item)
                    if results:
                        return results
        except Exception as e:
            print(f"      ⚠️ JSON extraction error: {e}")
        
        # ✅ 方法2: 从不完整的JSON中提取已完成的条目
        # 查找所有完整的 {"symbol": "...", "score": ..., ...} 对象
        try:
            # 匹配完整的对象（以 { 开始，以 } 结束）
            object_pattern = r'\{[^{}]*"symbol"[^{}]*"score"[^{}]*\}'
            matches = re.findall(object_pattern, response)
            
            print(f"[DEBUG] _parse_response: found {len(matches)} complete objects")
            
            for idx, match in enumerate(matches):
                try:
                    item = json.loads(match)
                    if item.get('symbol') and 'score' in item:
                        try:
                            item['score'] = int(item['score'])
                        except (ValueError, TypeError):
                            item['score'] = 50
                        results.append(item)
                        print(f"[DEBUG] _parse_response: added {item.get('symbol')} score={item.get('score')}")
                except:
                    continue
            
            if results:
                print(f"[DEBUG] _parse_response: returning {len(results)} results from partial JSON")
                return results
        except Exception as e:
            print(f"      ⚠️ Partial JSON extraction error: {e}")
        
        # ✅ 方法3: 使用正则提取 symbol 和 score（原有的fallback）
        if batch_symbols:
            symbol_pattern = r'"symbol"\s*:\s*"([^"]+)"'
            score_pattern = r'"score"\s*:\s*([-+]?\d+)'
            signal_pattern = r'"signal"\s*:\s*"([^"]+)"'
            
            symbols = re.findall(symbol_pattern, response)
            scores = re.findall(score_pattern, response)
            signals = re.findall(signal_pattern, response)
            
            print(f"[DEBUG] Fallback: found {len(symbols)} symbols, {len(scores)} scores")
            
            # ✅ 如果提取到了 symbols，即使数量不匹配也使用
            if symbols:
                # 取 min(len(symbols), len(batch_symbols)) 个
                count = min(len(symbols), len(batch_symbols))
                
                for i in range(count):
                    symbol = symbols[i]
                    # 验证 symbol 是否有效（至少4个字符）
                    if len(symbol) >= 4:
                        try:
                            score_val = int(scores[i]) if i < len(scores) else 50
                        except (ValueError, TypeError):
                            score_val = 50
                        
                        results.append({
                            'symbol': symbol,
                            'score': score_val,
                            'signal': signals[i] if i < len(signals) else 'hold',
                            'reason': 'Parsed from response'
                        })
                
                if results:
                    print(f"[DEBUG] Fallback: returning {len(results)} results from partial match")
                    return results
            
            # ✅ 如果 symbols 提取失败，尝试从 batch_symbols 匹配
            # 从响应中提取分数列表
            score_pattern_all = r'([-+]?\d+)\s*[,}\]]'
            all_scores = re.findall(score_pattern_all, response)
            
            # 过滤出合理的分数（-100 到 100 之间）
            valid_scores = []
            for s in all_scores:
                try:
                    val = int(s)
                    if -100 <= val <= 100:
                        valid_scores.append(val)
                except:
                    continue
            
            print(f"[DEBUG] Fallback: found {len(valid_scores)} valid scores")
            
            if valid_scores:
                # 使用 batch_symbols 和提取到的分数
                count = min(len(valid_scores), len(batch_symbols))
                
                for i in range(count):
                    symbol = batch_symbols[i]
                    score_val = valid_scores[i] if i < len(valid_scores) else 50
                    
                    results.append({
                        'symbol': symbol,
                        'score': min(100, max(-100, score_val)),
                        'signal': 'hold',
                        'reason': 'Parsed from response (fallback)'
                    })
                
                if results:
                    print(f"[DEBUG] Fallback: returning {len(results)} results from score fallback")
                    return results
        
        return results
    
    
    def _fallback(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Fallback analysis - simple rules"""
        results = []
        for item in batch:
            symbol = item.get('symbol', '')
            if not symbol:
                continue
            
            # 尝试从已加载数据获取
            data = self.get_loaded_data(symbol)
            if data is None:
                data = self.data_loader.load_data(symbol, 30)
            
            if data is None or data.empty:
                # 没有数据，给默认评分
                results.append({
                    'symbol': symbol,
                    'score': 50,
                    'signal': 'hold',
                    'reason': 'No data available'
                })
                continue
            
            try:
                close = data['close']
                current = close.iloc[-1]
                ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else current
                change = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
                
                score = 50
                if current > ma20:
                    score += 20
                if change > 0:
                    score += 15
                if len(close) > 10 and close.iloc[-1] > close.iloc[-10]:
                    score += 15
                
                signal = 'buy' if score > 70 else 'hold' if score > 45 else 'sell'
                
                results.append({
                    'symbol': symbol,
                    'score': min(100, max(0, score)),
                    'signal': signal,
                    'reason': 'Rule engine fallback scoring'
                })
            except Exception as e:
                print(f"      ⚠️ Fallback error for {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'score': 50,
                    'signal': 'hold',
                    'reason': f'Fallback error: {str(e)[:50]}'
                })
        
        return results
    
    def execute(self, symbols: List[str], keep_count: int = 700, **kwargs) -> List[Dict]:
        """Batch analyze ETFs"""
        if not symbols:
            print("   ⚠️ No symbols to analyze")
            return []
        
        print(f"   📊 ETFAnalyzeSkill: processing {len(symbols)} symbols...")
        
        # 设置数据加载参数
        kwargs['data_days'] = 30
        
        # 调用父类流式处理
        results = super().execute(symbols, keep_count, **kwargs)
        
        # 检查结果数量
        if len(results) < self.MIN_KEEP_COUNT:
            print(f"   ⚠️ LLM analysis results insufficient ({len(results)} < {self.MIN_KEEP_COUNT}), using fallback...")
            
            # 获取所有已加载的数据
            all_loaded = self.get_all_loaded_data()
            available_symbols = list(all_loaded.keys())
            
            if available_symbols:
                existing_symbols = {r.get('symbol') for r in results}
                remaining = [s for s in available_symbols if s not in existing_symbols]
                if remaining:
                    # 限制补充数量
                    max_fallback = min(len(remaining), keep_count - len(results))
                    fallback_items = [{'symbol': s} for s in remaining[:max_fallback]]
                    fallback_results = self._fallback(fallback_items)
                    results.extend(fallback_results)
                    print(f"   ✅ Added {len(fallback_results)} fallback results")
        
        # 如果结果仍然太少，从原始符号列表补充
        if len(results) < self.MIN_KEEP_COUNT:
            print(f"   ⚠️ Still insufficient ({len(results)}), generating more fallback...")
            existing_symbols = {r.get('symbol') for r in results}
            remaining = [s for s in symbols if s not in existing_symbols]
            if remaining:
                max_fallback = min(len(remaining), keep_count - len(results))
                fallback_items = [{'symbol': s} for s in remaining[:max_fallback]]
                fallback_results = self._fallback(fallback_items)
                results.extend(fallback_results)
                print(f"   ✅ Added {len(fallback_results)} more fallback results")
        
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = results[:keep_count]
        print(f"   ✅ Final retained {len(final_results)} candidates")
        
        # 清理缓存
        # self.data_loader.clear_cache()
        # self.clear_cache()
        
        return final_results


class ETFRankingSkill(BaseBatchSkill):
    """
    Skill: ETF Fine Ranking (Stage 2)
    Function: Fine comparison and ranking of candidate ETFs
    """
    
    def __init__(self):
        super().__init__(
            name="etf_ranking",
            description="Fine comparison and ranking of candidate ETFs",
            batch_size=10, #20,
            max_workers=4,
            timeout=60,
            output_tokens=400*2,
            safety_margin=0.75
        )
        self.data_loader = ETFDataLoader()
        self.MIN_KEEP_COUNT = 10
    
    def _get_base_prompt(self) -> str:
        return """
Please perform fine comparative ranking of the following ETFs.

Data format: Symbol|Init Score|Signal|YYYY-MM-DD|O|H|L|C|V

Task: Based on raw OHLCV data and initial scores, re-evaluate the relative strength.

Output JSON: {"rankings": [{"symbol": "Symbol", "rank_score": Score, "signal": "buy/hold/sell", "reason": "Reason"}]}"""
    
    def _load_item_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """加载单个ETF数据"""
        days = kwargs.get('data_days', 60)
        return self.data_loader.load_data(symbol, days)
    
    def _get_item_data_str_for_item(self, item: Any, data: Any, **kwargs) -> str:
        """获取单个项目的prompt片段"""
        if isinstance(item, dict):
            symbol = item.get('symbol', '')
            score = item.get('score', 0)
            signal = item.get('signal', 'hold')
            days = kwargs.get('summary_days', 60)
            
            if isinstance(data, pd.DataFrame):
                summary = self.data_loader.get_summary(data, days)
            else:
                summary = ''
            
            return f"{symbol}|Init:{score}|Signal:{signal}|{summary}"
        return str(item)
    
    def _create_item_from_data(self, symbol: str, data: Any) -> Optional[Dict]:
        if isinstance(data, pd.DataFrame):
            latest_date = self._get_data_latest_date(data)
            return {
                'symbol': symbol,
                'full_data': self.data_loader.get_summary(data, 60),
                '_latest_date': latest_date  # ✅ 保存日期到 item
            }
        return {'symbol': symbol}
    
    def _process_batch(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Rank a batch of candidates"""
        data_text = []
        for item in batch:
            symbol = item.get('symbol', '')
            score = item.get('score', 0)
            signal = item.get('signal', 'hold')
            full_data = item.get('full_data', '')
            
            if full_data:
                data_text.append(f"{symbol}|Init:{score}|Signal:{signal}|{full_data}")
            else:
                data_text.append(f"{symbol}|Init:{score}|Signal:{signal}|No data")
        
        if not data_text:
            return []
        
        batch_symbols = [item.get('symbol') for item in batch if item.get('symbol')]
        prompt = f"""
Please perform fine comparative ranking of the following {len(batch)} ETFs.

Data:
{chr(10).join(data_text)}

Output JSON:
{{
    "rankings": [
        {{"symbol": "Symbol", "rank_score": Score, "signal": "buy/hold/sell", "reason": "Reason"}}
    ]
}}"""

        try:
            # print(f'ETFRankingSkill._process_batch\n{prompt}')
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=self.output_tokens,
                temperature=0.3,
                enable_thinking=False
            )
            
            print(f'ETFRankingSkill._process_batch\n{response}')
            data = self._extract_json(response)
            rankings = data.get('rankings', []) if data else []
        
            # ✅ 验证并补全 symbol
            for idx, item in enumerate(rankings):
                if not item.get('symbol') and batch_symbols and idx < len(batch_symbols):
                    item['symbol'] = batch_symbols[idx]
            
            return [r for r in rankings if r.get('symbol')]
                
        except Exception as e:
            print(f"   ⚠️ Ranking failed: {e}\nTrackback:{format_exception(e)}")
            return []
    
    def _fallback(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Fallback ranking - keep original order"""
        return batch
    
    def execute(self, candidates: List[Dict], keep_count: int = 100, **kwargs) -> List[Dict]:
        """Execute ranking"""
        if not candidates:
            print("   ⚠️ No candidates to rank")
            return []
        
        # 设置数据加载参数
        kwargs['data_days'] = 60
        kwargs['summary_days'] = 60
        
        # 调用父类流式处理
        results = super().execute(candidates, keep_count, **kwargs)
        
        # 检查结果数量
        if len(results) < self.MIN_KEEP_COUNT:
            print(f"   ⚠️ Insufficient results ({len(results)}), using fallback...")
            existing_symbols = {r.get('symbol') for r in results}
            remaining = [c for c in candidates if c.get('symbol') not in existing_symbols]
            if remaining:
                remaining.sort(key=lambda x: x.get('score', 0), reverse=True)
                needed = keep_count - len(results)
                for item in remaining[:needed]:
                    item['rank_score'] = item.get('score', 0)
                    results.append(item)
                print(f"   ✅ Added {len(remaining[:needed])} fallback results")
        
        results.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
        final_results = results[:keep_count]
        print(f"   ✅ Final retained {len(final_results)} candidates")
        
        # 清理缓存
        # self.data_loader.clear_cache()
        # self.clear_cache()
        
        return final_results


class ETFDeepAnalyzeSkill(BaseBatchSkill):
    """
    Skill: ETF Deep Analysis (Stage 3)
    Function: Deep analysis of final candidates
    Only provide raw OHLCV data, all analysis done by LLM
    """
    
    def __init__(self):
        super().__init__(
            name="etf_deep_analyze",
            description="Deep technical analysis of ETFs",
            batch_size=1,
            max_workers=1,
            timeout=120,
            output_tokens=600*2,
            safety_margin=0.75
        )
        self.data_loader = ETFDataLoader()
        self.MIN_KEEP_COUNT = 1
    
    def _get_base_prompt(self) -> str:
        return """
# Task
Perform deep technical analysis on ETF.

# Raw OHLCV Data
Format: YYYY-MM-DD|O|H|L|C|V

# Analysis Requirements
Please calculate indicators from raw data:
1. Moving Average System (MA5, MA20, MA60)
2. RSI (14-day)
3. MACD
4. Bollinger Bands
5. Price-Volume Relationship
6. Trend Analysis
7. Support and Resistance Levels
8. Candlestick Patterns

# Output JSON
{
    "deep_score": 0-100 Score,
    "recommendation": "buy/hold/sell",
    "signal": "Specific suggestion",
    "confidence": 0.0-1.0,
    "risk_level": "low/medium/high",
    "analysis": "Detailed analysis",
    "target_price": Target price,
    "stop_loss": Stop loss price
}"""
    
    def _load_item_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """加载单个ETF数据"""
        days = kwargs.get('data_days', 60)
        return self.data_loader.load_data(symbol, days)
    
    def _get_item_data_str_for_item(self, item: Any, data: Any, **kwargs) -> str:
        """获取单个项目的prompt片段"""
        if isinstance(item, dict):
            symbol = item.get('symbol', '')
            if isinstance(data, pd.DataFrame):
                summary = self.data_loader.get_summary(data, days=30)
                return f"{symbol}|{summary}"
        return str(item)
    
    def _create_item_from_data(self, symbol: str, data: Any) -> Optional[Dict]:
        """从数据创建item"""
        if isinstance(data, pd.DataFrame):
            return {
                'symbol': symbol,
                'full_data': self.data_loader.get_summary(data, 30)
            }
        return {'symbol': symbol}
    
    def _process_batch(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Deep analysis of single ETF"""
        if not batch:
            return []
        
        item = batch[0]
        symbol = item.get('symbol')
        if not symbol:
            return []
        
        print(f"     Deep analyzing {symbol}...")
        
        full_data = item.get('full_data', '')
        if not full_data:
            print(f"     ⚠️ No data for {symbol}")
            return []
        
        prompt = f"""
# Task
Perform deep technical analysis on {symbol}.

# Raw OHLCV Data
Format: YYYY-MM-DD|O|H|L|C|V

{full_data}

# Output JSON
{{
    "deep_score": 0-100 Score,
    "recommendation": "buy/hold/sell",
    "signal": "Specific suggestion",
    "confidence": 0.0-1.0,
    "risk_level": "low/medium/high",
    "analysis": "Detailed analysis",
    "target_price": Target price,
    "stop_loss": Stop loss price
}}"""

        try:
            # print(f'ETFDeepAnalyzeSkill._process_batch\n{prompt}')
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=self.output_tokens,
                temperature=0.3,
                enable_thinking=False
            )
            
            print(f'ETFDeepAnalyzeSkill._process_batch\n{response}')
            data = self._extract_json(response)
            if data:
                data['symbol'] = symbol
                data['quick_score'] = item.get('score', 50)
                data['rank_score'] = item.get('rank_score', 50)
                data['final_score'] = (
                    data.get('deep_score', 0) * 0.5 +
                    data.get('rank_score', 0) * 0.25 +
                    data.get('quick_score', 0) * 0.25
                )
                
                rec = data.get('recommendation', '').lower()
                if rec not in ['buy', 'sell', 'hold']:
                    if 'buy' in rec or 'bullish' in rec:
                        data['recommendation'] = 'buy'
                    elif 'sell' in rec or 'bearish' in rec:
                        data['recommendation'] = 'sell'
                    else:
                        data['recommendation'] = 'hold'
                        
                return [data]
                
        except Exception as e:
            print(f"      ⚠️ Analysis of {symbol} failed: {e}")
        
        return []
    
    def _fallback(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Fallback analysis"""
        results = []
        for item in batch:
            symbol = item.get('symbol')
            if not symbol:
                continue
            
            results.append({
                'symbol': symbol,
                'deep_score': 50,
                'recommendation': 'hold',
                'signal': 'Hold',
                'confidence': 0.3,
                'risk_level': 'medium',
                'analysis': 'Deep analysis failed, downgraded to neutral suggestion',
                'target_price': 0,
                'stop_loss': 0,
                'quick_score': item.get('score', 50),
                'rank_score': item.get('rank_score', 50),
                'final_score': 50
            })
        
        return results
    
    def execute(self, items: List[Dict], top_k: int = 3, **kwargs) -> List[Dict]:
        """Execute deep analysis"""
        if not items:
            print("   ⚠️ No items for deep analysis")
            return []
        
        total = min(len(items), kwargs.get('max_items', 50))
        print(f"   📊 Starting deep analysis of {total} ETFs...")
        
        # 设置数据加载参数
        kwargs['data_days'] = 60
        kwargs['summary_days'] = 30
        
        # 调用父类流式处理
        results = super().execute(items[:total], None, **kwargs)
        
        # 如果没有结果，使用降级
        if len(results) < top_k:
            print(f"   ⚠️ Insufficient results ({len(results)} < {top_k}), using fallback...")
            remaining = [item for item in items[:total] if item.get('symbol') not in {r.get('symbol') for r in results}]
            if remaining:
                fallback_results = self._fallback(remaining[:top_k * 2])
                results.extend(fallback_results)
                print(f"   ✅ Added {len(fallback_results)} fallback results")
        
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        final_results = results[:top_k * 3]
        
        print(f"   ✅ Final returned {len(final_results)} deep analysis results")
        
        # 清理缓存
        # self.data_loader.clear_cache()
        # self.clear_cache()
        
        return final_results