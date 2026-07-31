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
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base_skill import BaseSkill
from .base_batch_skill import BaseBatchSkill
from ..data_fetcher import ETFDataFetcher
from ..llm_client import get_llm_client


class ETFDataSkill(BaseSkill):
    """
    Skill: ETF Data Fetcher
    Function: Get historical data for single or multiple ETFs
    """
    
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
        Use full date format YYYY-MM-DD to avoid cross-year confusion
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
    Skill: ETF Quick Analysis
    Function: Quick scoring and analysis for ETFs (Stage 1)
    Only provide raw OHLCV data, all indicators calculated by LLM itself
    """
    
    def __init__(self):
        super().__init__(
            name="etf_quick_analyze",
            description="Quick scoring for ETFs for initial screening",
            batch_size=15,
            max_workers=4,
            timeout=60,
            output_tokens=500,
            max_context_tokens=30000  # Safe threshold for DeepSeek-R1-Distill-Qwen-1.5B
        )
        self.data_skill = ETFDataSkill()
        self.MIN_KEEP_COUNT = 50
        self._token_counter = None
        
        # Initialize tokenizer
        try:
            import tiktoken
            self._token_counter = tiktoken.get_encoding("cl100k_base")
        except:
            self._token_counter = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self._token_counter:
            return len(self._token_counter.encode(text))
        # Fallback: rough estimate (~3 chars per token)
        return len(text) // 3
    
    def _build_batch_dynamically(self, items: List[str], base_prompt: str, **kwargs) -> List[List[str]]:
        """
        Dynamically build batches based on token count
        Each ETF data is loaded and token count is checked before adding to batch
        """
        batches = []
        current_batch = []
        current_tokens = self._count_tokens(base_prompt) + self.output_tokens
        
        for symbol in items:
            # Load data for this symbol
            df = self.data_skill._get_single(symbol, days=30)
            if df is None or df.empty:
                continue
            
            # Generate summary and calculate token count
            summary = self.data_skill.get_summary(df, days=20)
            item_str = f"{symbol}|{summary}"
            item_tokens = self._count_tokens(item_str)
            
            # Check if adding this item exceeds limit
            if current_tokens + item_tokens > self.max_context_tokens:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [symbol]
                current_tokens = self._count_tokens(base_prompt) + self.output_tokens + item_tokens
            else:
                current_batch.append(symbol)
                current_tokens += item_tokens
        
        # Add last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _preprocess(self, items: List[str], **kwargs) -> List[str]:
        return [s for s in items if s]
    
    def _process_batch(self, batch: List[str], **kwargs) -> List[Dict]:
        """Analyze a batch of ETFs - only provide raw OHLCV data"""
        if not batch:
            return []
        
        # Get raw data
        data = self.data_skill.execute(batch, days=30)
        if not data:
            return []
        
        # Build batch data - only raw OHLCV
        batch_data = []
        for symbol, df in data.items():
            summary = self.data_skill.get_summary(df, days=20)
            batch_data.append(f"{symbol}|{summary}")
        
        data_text = "\n".join(batch_data)
        
        prompt = f"""You are an experienced quantitative analyst. Please analyze the raw OHLCV data of the following {len(batch_data)} ETFs.

Data format: Symbol|YYYY-MM-DD|O|H|L|C|V
Legend: O=Open, H=High, L=Low, C=Close, V=Volume

Data:
{data_text}

Tasks:
1. Calculate technical indicators from raw data (MA5, MA20, MA60, RSI, MACD, etc.)
2. Design your own scoring system (trend, momentum, technical signals, etc.)
3. Give each ETF a comprehensive score (0-100)
4. Provide signal (buy/hold/sell)

Output JSON:
{{
    "scoring_system": {{
        "dimensions": [
            {{"name": "Trend", "weight": 0.35}},
            {{"name": "Momentum", "weight": 0.30}},
            {{"name": "Technical Signals", "weight": 0.35}}
        ]
    }},
    "scores": [
        {{
            "symbol": "Symbol",
            "score": Score,
            "signal": "buy/hold/sell",
            "reason": "Reason"
        }}
    ]
}}"""

        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=self.output_tokens,
                temperature=0.3,
                enable_thinking=False
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            print(f"      ⚠️ LLM analysis failed: {e}")
            return []
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse LLM response"""
        data = self._extract_json(response)
        if data:
            system = data.get('scoring_system', {})
            if system.get('dimensions'):
                dims = system['dimensions']
                print(f"     📋 Scoring system: {len(dims)} dimensions")
                for d in dims[:3]:
                    print(f"        - {d.get('name')} ({d.get('weight', 0)})")
            return data.get('scores', [])
        return []
    
    def _fallback(self, batch: List[str], **kwargs) -> List[Dict]:
        """Fallback analysis - simple rules"""
        results = []
        data = self.data_skill.execute(batch, days=30)
        
        for symbol, df in data.items():
            try:
                close = df['close']
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
            except:
                continue
        
        return results
    
    def execute(self, symbols: List[str], keep_count: int = 700) -> List[Dict]:
        """
        Batch analyze ETFs - ensure enough quantity for next stage
        """
        if not symbols:
            print("   ⚠️ No symbols to analyze")
            return []
        
        # Call parent execute
        results = super().execute(symbols, keep_count)
        
        # Check result count, if too few, use fallback to supplement
        if len(results) < self.MIN_KEEP_COUNT:
            print(f"   ⚠️ LLM analysis results insufficient ({len(results)} < {self.MIN_KEEP_COUNT}), using rule engine to supplement...")
            
            # Get all symbols with data
            data = self.data_skill.execute(symbols, days=30)
            available_symbols = list(data.keys())
            
            if available_symbols:
                # Supplement from remaining symbols
                existing_symbols = {r.get('symbol') for r in results}
                remaining = [s for s in available_symbols if s not in existing_symbols]
                
                if remaining:
                    fallback_results = self._fallback(remaining[:keep_count])
                    results.extend(fallback_results)
                    print(f"   ✅ Added {len(fallback_results)} rule engine scoring results")
        
        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Ensure returned count doesn't exceed keep_count
        final_results = results[:keep_count]
        print(f"   ✅ Final retained {len(final_results)} candidates")
        
        return final_results


class ETFRankingSkill(BaseBatchSkill):
    """
    Skill: ETF Fine Ranking
    Function: Fine comparison and ranking of candidate ETFs (Stage 2)
    Based on Stage 1 LLM scoring results, with reference to raw data
    """
    
    def __init__(self):
        super().__init__(
            name="etf_ranking",
            description="Fine comparison and ranking of candidate ETFs",
            batch_size=20,
            max_workers=4,
            timeout=60,
            output_tokens=400,
            max_context_tokens=30000  # Safe threshold
        )
        self.data_skill = ETFDataSkill()
        self.MIN_KEEP_COUNT = 10
        self._token_counter = None
        
        try:
            import tiktoken
            self._token_counter = tiktoken.get_encoding("cl100k_base")
        except:
            self._token_counter = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self._token_counter:
            return len(self._token_counter.encode(text))
        return len(text) // 3
    
    def _build_batch_dynamically(self, items: List[Dict], base_prompt: str, **kwargs) -> List[List[Dict]]:
        """
        Dynamically build batches based on token count
        Each ETF data is loaded and token count is checked before adding to batch
        """
        batches = []
        current_batch = []
        current_tokens = self._count_tokens(base_prompt) + self.output_tokens
        
        for item in items:
            symbol = item.get('symbol')
            if not symbol:
                continue
            
            # Get full data
            df = self.data_skill._get_single(symbol, days=60)
            if df is None or df.empty:
                full_data = ''
            else:
                full_data = self.data_skill.get_summary(df, days=60)
            
            # Build item string and count tokens
            score = item.get('score', 0)
            signal = item.get('signal', 'hold')
            item_str = f"{symbol}|Init:{score}|Signal:{signal}|{full_data}"
            item_tokens = self._count_tokens(item_str)
            
            # Check if adding this item exceeds limit
            if current_tokens + item_tokens > self.max_context_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_tokens = self._count_tokens(base_prompt) + self.output_tokens + item_tokens
            else:
                current_batch.append(item)
                current_tokens += item_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _preprocess(self, candidates: List[Dict], **kwargs) -> List[Dict]:
        """Enrich with full data for ranking"""
        enriched = []
        symbols_to_fetch = [item.get('symbol') for item in candidates if item.get('symbol')]
        
        if not symbols_to_fetch:
            return []
        
        # Batch get data
        data_map = self.data_skill.execute(symbols_to_fetch, days=60)
        
        for item in candidates:
            symbol = item.get('symbol')
            if not symbol:
                continue
            
            df = data_map.get(symbol)
            if df is not None and not df.empty:
                item['full_data'] = self.data_skill.get_summary(df, days=60)
                enriched.append(item)
            else:
                item['full_data'] = ''
                enriched.append(item)
        
        return enriched
    
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
        
        prompt = f"""
Please perform fine comparative ranking of the following {len(batch)} ETFs.

Data format: Symbol|Init Score|Signal|YYYY-MM-DD|O|H|L|C|V

Data:
{chr(10).join(data_text)}

Task: Based on raw OHLCV data and initial scores, re-evaluate the relative strength of each ETF, give refined scores (0-100).

Output JSON:
{{
    "rankings": [
        {{"symbol": "Symbol", "rank_score": Score, "signal": "buy/hold/sell", "reason": "Reason"}}
    ]
}}"""

        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=self.output_tokens,
                temperature=0.3,
                enable_thinking=False
            )
            
            data = self._extract_json(response)
            return data.get('rankings', []) if data else []
                
        except Exception as e:
            print(f"   ⚠️ Ranking failed: {e}")
            return []
    
    def _fallback(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Fallback ranking - keep original order"""
        return batch
    
    def execute(self, candidates: List[Dict], keep_count: int = 100) -> List[Dict]:
        """
        Fine ranking - ensure enough quantity for next stage
        """
        if not candidates:
            print("   ⚠️ No candidates to rank")
            return []
        
        # Call parent execute
        results = super().execute(candidates, keep_count)
        
        # Check result count
        if len(results) < self.MIN_KEEP_COUNT:
            print(f"   ⚠️ LLM ranking results insufficient ({len(results)} < {self.MIN_KEEP_COUNT}), using fallback...")
            
            existing_symbols = {r.get('symbol') for r in results}
            remaining = [c for c in candidates if c.get('symbol') not in existing_symbols]
            
            if remaining:
                remaining.sort(key=lambda x: x.get('score', 0), reverse=True)
                needed = keep_count - len(results)
                fallback_results = remaining[:needed]
                
                for item in fallback_results:
                    item['rank_score'] = item.get('score', 0)
                
                results.extend(fallback_results)
                print(f"   ✅ Added {len(fallback_results)} fallback ranking results")
        
        # Sort by rank_score
        results.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
        
        final_results = results[:keep_count]
        print(f"   ✅ Final retained {len(final_results)} candidates")
        
        return final_results


class ETFDeepAnalyzeSkill(BaseBatchSkill):
    """
    Skill: ETF Deep Analysis
    Function: Deep analysis of final candidates (Stage 3)
    Only provide raw OHLCV data, all analysis done by LLM
    """
    
    def __init__(self):
        super().__init__(
            name="etf_deep_analyze",
            description="Deep technical analysis of ETFs",
            batch_size=1,
            max_workers=1,
            timeout=120,
            output_tokens=600,
            max_context_tokens=30000
        )
        self.data_skill = ETFDataSkill()
        self.MIN_KEEP_COUNT = 1
    
    def _preprocess(self, candidates: List[Dict], **kwargs) -> List[Dict]:
        max_items = kwargs.get('max_items', 50)
        sorted_candidates = sorted(candidates, key=lambda x: x.get('rank_score', 0), reverse=True)
        return sorted_candidates[:max_items]
    
    def _process_batch(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """Deep analysis of single ETF - only provide raw data"""
        if not batch:
            return []
        
        item = batch[0]
        symbol = item.get('symbol')
        if not symbol:
            return []
        
        print(f"     Deep analyzing {symbol}...")
        
        # Get raw data (no indicators calculated)
        df = self.data_skill.execute([symbol], days=60).get(symbol)
        if df is None or df.empty:
            print(f"     ⚠️ Cannot get raw data for {symbol}")
            return []
        
        data_str = self.data_skill.get_summary(df, days=30)
        
        prompt = f"""
# Task
Perform deep technical analysis on {symbol}.

# Raw OHLCV Data
Format: YYYY-MM-DD|O|H|L|C|V

{data_str}

# Analysis Requirements
Please calculate the following indicators from raw data:
1. Moving Average System (MA5, MA20, MA60)
2. RSI (14-day)
3. MACD
4. Bollinger Bands
5. Price-Volume Relationship
6. Trend Analysis
7. Support and Resistance Levels
8. Candlestick Patterns

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
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=self.output_tokens,
                temperature=0.3,
                enable_thinking=False
            )
            
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
                return [data]
                
        except Exception as e:
            print(f"      ⚠️ Analysis of {symbol} failed: {e}")
        
        return []
    
    def _fallback(self, batch: List[Dict], **kwargs) -> List[Dict]:
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
        """
        Deep analysis - ensure at least top_k results returned
        """
        if not items:
            print("   ⚠️ No items for deep analysis")
            return []
        
        total = min(len(items), kwargs.get('max_items', 50))
        print(f"   📊 Starting deep analysis of {total} ETFs...")
        start_time = time.time()
        
        results = []
        failed_symbols = []
        
        with tqdm(total=total, desc="Deep Analysis", unit="stocks") as pbar:
            for item in items[:total]:
                symbol = item.get('symbol')
                batch_result = self._process_batch([item], **kwargs)
                if batch_result:
                    results.extend(batch_result)
                else:
                    if symbol:
                        failed_symbols.append(symbol)
                pbar.update(1)
        
        if len(results) < top_k and failed_symbols:
            print(f"   ⚠️ Successfully analyzed {len(results)}, failed {len(failed_symbols)}, using fallback...")
            
            fallback_items = [item for item in items if item.get('symbol') in failed_symbols[:top_k * 2]]
            if fallback_items:
                fallback_results = self._fallback(fallback_items)
                results.extend(fallback_results)
                print(f"   ✅ Added {len(fallback_results)} fallback analysis results")
        
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        elapsed = time.time() - start_time
        print(f"      ✅ Completed, time used {elapsed:.2f}s")
        
        final_results = results[:top_k * 3]
        if len(final_results) < top_k:
            print(f"   ⚠️ Insufficient results ({len(final_results)} < {top_k}), trying to get more...")
            remaining = [item for item in items[:total] if item.get('symbol') not in {r.get('symbol') for r in results}]
            if remaining:
                extra_fallback = self._fallback(remaining[:top_k * 2])
                results.extend(extra_fallback)
                results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
                final_results = results[:top_k * 3]
        
        print(f"   ✅ Final returned {len(final_results)} deep analysis results")
        return final_results