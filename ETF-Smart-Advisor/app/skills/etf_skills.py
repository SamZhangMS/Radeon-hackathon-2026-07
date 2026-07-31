# app/skills/etf_skills.py
"""
ETF相关技能实现
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
    技能: 获取ETF数据
    功能: 获取单只或多只ETF的历史数据
    """
    
    def __init__(self):
        super().__init__(
            name="etf_data_fetcher",
            description="获取ETF历史数据，支持单只和批量"
        )
        self.fetcher = ETFDataFetcher()
        self.cache = {}
        self.max_workers = 8
    
    def execute(self, symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """获取ETF数据 - 多线程版本"""
        if not symbols:
            return {}
        
        print(f"📊 开始获取 {len(symbols)} 个ETF数据 ({self.max_workers} 线程)...")
        start_time = time.time()
        
        result = {}
        result_lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._get_single, symbol, days): symbol
                for symbol in symbols
            }
            
            with tqdm(total=len(symbols), desc="读取数据", unit="个") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result(timeout=30)
                        if df is not None and not df.empty:
                            with result_lock:
                                result[symbol] = df
                    except Exception as e:
                        print(f"      ⚠️ 获取 {symbol} 失败: {e}")
                    pbar.update(1)
                    
        elapsed = time.time() - start_time
        print(f"      ✅ 完成，获取 {len(result)} 个ETF，耗时 {elapsed:.2f}s")
        
        return result
    
    def _get_single(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """获取单个ETF数据"""
        try:
            cache_key = f"{symbol}_{days}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            df = self.fetcher.get_history(symbol, f"{days}d")
            if not df.empty:
                self.cache[cache_key] = df
                return df
        except Exception as e:
            print(f"      ⚠️ 获取 {symbol} 失败: {e}")
        return None
    
    def get_summary(self, df: pd.DataFrame, days: int = 20) -> str:
        """获取数据摘要 - 只包含原始OHLCV"""
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
    技能: ETF快速分析
    功能: 对ETF进行快速评分和分析（阶段1）
    只提供原始OHLCV数据，所有指标由大模型自己计算
    """
    
    def __init__(self):
        super().__init__(
            name="etf_quick_analyze",
            description="对ETF进行快速评分，用于初步筛选",
            batch_size=15,
            max_workers=4,
            timeout=60,
            output_tokens=500
        )
        self.data_skill = ETFDataSkill()
    
    def _preprocess(self, items: List[str], **kwargs) -> List[str]:
        return [s for s in items if s]
    
    def _process_batch(self, batch: List[str], **kwargs) -> List[Dict]:
        """分析一批ETF - 只提供原始OHLCV数据"""
        if not batch:
            return []
        
        # 获取原始数据
        data = self.data_skill.execute(batch, days=30)
        if not data:
            return []
        
        # 构建批次数据 - 只传原始OHLCV
        batch_data = []
        for symbol, df in data.items():
            summary = self.data_skill.get_summary(df, days=20)
            batch_data.append(f"{symbol}|{summary}")
        
        data_text = "\n".join(batch_data)
        
        prompt = f"""你是有经验的量化分析师。请分析以下 {len(batch_data)} 只ETF的原始OHLCV数据。

数据格式: 代码|YYMMDD|O|H|L|C|V
说明: O=开盘价, H=最高价, L=最低价, C=收盘价, V=成交量

数据:
{data_text}

任务:
1. 根据原始数据自行计算技术指标（MA5, MA20, MA60, RSI, MACD等）
2. 自主设计评分体系（趋势、动量、技术信号等维度）
3. 给每只ETF综合打分(0-100分)
4. 给出信号(buy/hold/sell)

输出JSON:
{{
    "scoring_system": {{
        "dimensions": [
            {{"name": "趋势", "weight": 0.35}},
            {{"name": "动量", "weight": 0.30}},
            {{"name": "技术信号", "weight": 0.35}}
        ]
    }},
    "scores": [
        {{
            "symbol": "代码",
            "score": 分数,
            "signal": "buy/hold/sell",
            "reason": "理由"
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
            print(f"      ⚠️ LLM分析失败: {e}")
            return []
    
    def _parse_response(self, response: str) -> List[Dict]:
        """解析响应"""
        data = self._extract_json(response)
        if data:
            system = data.get('scoring_system', {})
            if system.get('dimensions'):
                dims = system['dimensions']
                print(f"     📋 评分体系: {len(dims)} 个维度")
                for d in dims[:3]:
                    print(f"        - {d.get('name')} ({d.get('weight', 0)})")
            return data.get('scores', [])
        return []
    
    def _fallback(self, batch: List[str], **kwargs) -> List[Dict]:
        """降级分析 - 简单规则"""
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
                    'reason': '规则引擎降级评分'
                })
            except:
                continue
        
        return results


class ETFRankingSkill(BaseBatchSkill):
    """
    技能: ETF精细排名
    功能: 对候选ETF进行精细比较和排名（阶段2）
    基于第一阶段大模型的评分结果，不再需要原始数据
    """
    
    def __init__(self):
        super().__init__(
            name="etf_ranking",
            description="对候选ETF进行精细比较和排名",
            batch_size=20,
            max_workers=4,
            timeout=60,
            output_tokens=400
        )
        self.data_skill = ETFDataSkill()
    
    def _preprocess(self, candidates: List[Dict], **kwargs) -> List[Dict]:
        """补充原始数据用于排名"""
        enriched = []
        for item in candidates:
            symbol = item.get('symbol')
            if not symbol:
                continue
            
            df = self.data_skill.execute([symbol], days=60).get(symbol)
            if df is not None and not df.empty:
                item['full_data'] = self.data_skill.get_summary(df, days=60)
                enriched.append(item)
        
        return enriched
    
    def _process_batch(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """排名一批候选"""
        data_text = []
        for item in batch:
            data_text.append(
                f"{item['symbol']}|初评:{item.get('score', 0)}|"
                f"信号:{item.get('signal', 'hold')}|{item.get('full_data', '')}"
            )
        
        prompt = f"""
请对以下 {len(batch)} 只ETF进行精细比较排名。

数据格式: 代码|初评分|信号|YYMMDD|O|H|L|C|V

数据:
{chr(10).join(data_text)}

任务: 根据原始OHLCV数据和初评分，重新评估各ETF的相对强弱，给出精细评分(0-100分)。

输出JSON:
{{
    "rankings": [
        {{"symbol": "代码", "rank_score": 分数, "signal": "buy/hold/sell", "reason": "理由"}}
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
            print(f"   ⚠️ 排名失败: {e}")
            return []
    
    def _fallback(self, batch: List[Dict], **kwargs) -> List[Dict]:
        return batch


class ETFDeepAnalyzeSkill(BaseBatchSkill):
    """
    技能: ETF深度分析
    功能: 对最终候选进行深度分析（阶段3）
    只提供原始OHLCV数据，所有分析由大模型完成
    """
    
    def __init__(self):
        super().__init__(
            name="etf_deep_analyze",
            description="对ETF进行深度技术分析",
            batch_size=1,
            max_workers=1,
            timeout=120,
            output_tokens=600
        )
        self.data_skill = ETFDataSkill()
    
    def _preprocess(self, candidates: List[Dict], **kwargs) -> List[Dict]:
        max_items = kwargs.get('max_items', 50)
        return candidates[:max_items]
    
    def _process_batch(self, batch: List[Dict], **kwargs) -> List[Dict]:
        """深度分析单只ETF - 只提供原始数据"""
        if not batch:
            return []
        
        item = batch[0]
        symbol = item.get('symbol')
        if not symbol:
            return []
        
        print(f"     深度分析 {symbol}...")
        
        # 获取原始数据（不计算指标）
        df = self.data_skill.execute([symbol], days=60).get(symbol)
        if df is None or df.empty:
            return []
        
        data_str = self.data_skill.get_summary(df, days=30)
        
        prompt = f"""
# 任务
对 {symbol} 进行深度技术分析。

# 原始OHLCV数据
格式: 日期|O|H|L|C|V

{data_str}

# 分析要求
请根据原始数据自行计算以下指标：
1. 均线系统 (MA5, MA20, MA60)
2. RSI (14日)
3. MACD
4. 布林带
5. 量价关系
6. 趋势判断
7. 支撑压力位
8. K线形态

# 输出JSON
{{
    "deep_score": 0-100评分,
    "recommendation": "buy/hold/sell",
    "signal": "具体建议",
    "confidence": 0.0-1.0,
    "risk_level": "low/medium/high",
    "analysis": "详细分析",
    "target_price": 目标价,
    "stop_loss": 止损价
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
            print(f"      ⚠️ 分析失败: {e}")
        
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
                'signal': '持有观望',
                'confidence': 0.3,
                'risk_level': 'medium',
                'analysis': '深度分析失败，降级到中性建议',
                'target_price': 0,
                'stop_loss': 0,
                'quick_score': item.get('score', 50),
                'rank_score': item.get('rank_score', 50),
                'final_score': 50
            })
        
        return results
    
    def execute(self, items: List[Dict], top_k: int = 3, **kwargs) -> List[Dict]:
        """深度分析 - 重写以显示进度"""
        if not items:
            return []
        
        total = min(len(items), kwargs.get('max_items', 50))
        print(f"   📊 开始深度分析 {total} 个ETF...")
        start_time = time.time()
        
        results = []
        with tqdm(total=total, desc="深度分析", unit="个") as pbar:
            for item in items[:total]:
                batch_result = self._process_batch([item], **kwargs)
                if batch_result:
                    results.extend(batch_result)
                pbar.update(1)
        
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        elapsed = time.time() - start_time
        print(f"      ✅ 完成，耗时 {elapsed:.2f}s")
        
        return results[:top_k * 3]