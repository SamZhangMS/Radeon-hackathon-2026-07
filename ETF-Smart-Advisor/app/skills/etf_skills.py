# app/skills/etf_skills.py
"""
ETF相关技能实现
"""

import pandas as pd
import numpy as np
import json
import re
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_skill import BaseSkill
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
    
    def execute(self, symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """
        获取ETF数据
        """
        result = {}
        for symbol in symbols:
            try:
                df = self.fetcher.get_history(symbol, f"{days}d")
                if not df.empty:
                    result[symbol] = df
            except Exception as e:
                print(f"⚠️ 获取 {symbol} 数据失败: {e}")
        return result
    
    def get_summary(self, df: pd.DataFrame, days: int = 20) -> str:
        """获取数据摘要"""
        df = df.tail(days)
        parts = []
        for idx, row in df.iterrows():
            date_str = idx.strftime('%m-%d') if hasattr(idx, 'strftime') else str(idx)
            parts.append(
                f"{date_str} {row['open']:.3f} {row['high']:.3f} "
                f"{row['low']:.3f} {row['close']:.3f} {row['volume']:.0f}"
            )
        return "|".join(parts)
    
    def get_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        close = df['close']
        current = close.iloc[-1]
        
        # 均线
        ma5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else current
        ma10 = close.rolling(10).mean().iloc[-1] if len(close) >= 10 else current
        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else current
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 50
        if not pd.isna(rs.iloc[-1]) and rs.iloc[-1] != 0:
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # 涨跌幅
        change_1w = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
        change_1m = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
        
        # 成交量
        avg_volume = df['volume'].tail(5).mean()
        volume_ratio = df['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
        
        # 波动率
        volatility = close.pct_change().std() * np.sqrt(252)
        
        return {
            'price': float(current),
            'ma5': float(ma5),
            'ma10': float(ma10),
            'ma20': float(ma20),
            'ma60': float(ma60),
            'rsi': float(rsi),
            'change_1w': float(change_1w),
            'change_1m': float(change_1m),
            'volume_ratio': float(volume_ratio),
            'volatility': float(volatility)
        }


class ETFAnalyzeSkill(BaseSkill):
    """
    技能: ETF快速分析
    功能: 对ETF进行快速评分和分析（阶段1）
    """
    
    def __init__(self):
        super().__init__(
            name="etf_quick_analyze",
            description="对ETF进行快速评分，用于初步筛选"
        )
        self.llm = get_llm_client()
        self.data_skill = ETFDataSkill()
        self.batch_size = 50
    
    def execute(self, symbols: List[str], keep_count: int = 700) -> List[Dict]:
        """
        批量分析ETF
        """
        all_scores = []
        total = len(symbols)
        
        for i in range(0, total, self.batch_size):
            batch = symbols[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            
            print(f"  批次 {batch_num}/{total_batches}: 分析 {len(batch)} 个...")
            
            try:
                batch_scores = self._analyze_batch(batch)
                if batch_scores:
                    all_scores.extend(batch_scores)
            except Exception as e:
                print(f"    ⚠️ 批次 {batch_num} 失败: {e}")
                # 降级：规则评分
                batch_scores = self._fallback_analyze(batch)
                all_scores.extend(batch_scores)
        
        # 按分数排序
        all_scores.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_scores[:keep_count]
    
    def _analyze_batch(self, batch: List[str]) -> List[Dict]:
        """分析一批ETF"""
        # 获取数据
        data = self.data_skill.execute(batch, days=30)
        
        if not data:
            return []
        
        # 构建批次数据
        batch_data = []
        for symbol, df in data.items():
            summary = self.data_skill.get_summary(df, days=20)
            indicators = self.data_skill.get_indicators(df)
            batch_data.append(
                f"{symbol}|{indicators['price']:.3f}|"
                f"MA20:{indicators['ma20']:.3f}|RSI:{indicators['rsi']:.1f}|"
                f"CHG:{indicators['change_1m']:.2%}|{summary}"
            )
        
        data_text = "\n".join(batch_data)
        
        prompt = f"""
你是有经验的量化分析师。请分析以下 {len(batch_data)} 只ETF。

数据格式: 代码|价格|MA20|RSI|月涨幅|数据

{data_text}

请自主设计评分体系，给每只ETF打分(0-100分)。

输出JSON:
{{
    "scoring_system": {{
        "dimensions": [{{"name": "维度", "weight": 权重}}]
    }},
    "scores": [
        {{"symbol": "代码", "score": 分数, "signal": "buy/hold/sell", "reason": "理由"}}
    ]
}}"""

        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=400,
                temperature=0.3,
                enable_thinking=False
            )
            
            return self._parse_response(response, batch)
            
        except Exception as e:
            print(f"   ⚠️ LLM分析失败: {e}")
            return []
    
    def _parse_response(self, response: str, batch: List[str]) -> List[Dict]:
        """解析LLM响应"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                # 显示评分体系
                system = data.get('scoring_system', {})
                if system.get('dimensions'):
                    dims = system['dimensions']
                    print(f"     📋 评分体系: {len(dims)} 个维度")
                    for d in dims[:3]:
                        print(f"        - {d.get('name')} ({d.get('weight', 0)})")
                
                return data.get('scores', [])
        except:
            pass
        
        return []
    
    def _fallback_analyze(self, batch: List[str]) -> List[Dict]:
        """降级分析"""
        results = []
        data = self.data_skill.execute(batch, days=30)
        
        for symbol, df in data.items():
            indicators = self.data_skill.get_indicators(df)
            score = 50
            if indicators['price'] > indicators['ma20']:
                score += 20
            if indicators['rsi'] < 70 and indicators['rsi'] > 30:
                score += 15
            if indicators['change_1m'] > 0:
                score += 15
            
            signal = 'buy' if score > 70 else 'hold' if score > 45 else 'sell'
            
            results.append({
                'symbol': symbol,
                'score': min(100, max(0, score)),
                'signal': signal,
                'reason': '规则引擎降级评分'
            })
        
        return results


class ETFRankingSkill(BaseSkill):
    """
    技能: ETF精细排名
    功能: 对候选ETF进行精细比较和排名（阶段2）
    """
    
    def __init__(self):
        super().__init__(
            name="etf_ranking",
            description="对候选ETF进行精细比较和排名"
        )
        self.llm = get_llm_client()
        self.data_skill = ETFDataSkill()
        self.batch_size = 30
    
    def execute(self, candidates: List[Dict], keep_count: int = 100) -> List[Dict]:
        """
        精细排名
        """
        all_results = []
        total = len(candidates)
        
        # 补充完整数据
        enriched = self._enrich_candidates(candidates)
        
        for i in range(0, len(enriched), self.batch_size):
            batch = enriched[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(enriched) + self.batch_size - 1) // self.batch_size
            
            print(f"  批次 {batch_num}/{total_batches}: 精细排名 {len(batch)} 个...")
            
            try:
                batch_results = self._rank_batch(batch)
                if batch_results:
                    all_results.extend(batch_results)
            except Exception as e:
                print(f"   ⚠️ 批次 {batch_num} 失败: {e}")
                all_results.extend(batch)
        
        all_results.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
        return all_results[:keep_count]
    
    def _enrich_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """补充完整数据"""
        enriched = []
        for item in candidates:
            symbol = item.get('symbol')
            if not symbol:
                continue
            
            df = self.data_skill.execute([symbol], days=60).get(symbol)
            if df is not None and not df.empty:
                item['full_data'] = self.data_skill.get_summary(df, days=60)
                item['indicators'] = self.data_skill.get_indicators(df)
                enriched.append(item)
        
        return enriched
    
    def _rank_batch(self, batch: List[Dict]) -> List[Dict]:
        """排名一批候选"""
        data_text = []
        for item in batch:
            data_text.append(
                f"{item['symbol']}|初评:{item.get('score', 0)}|"
                f"信号:{item.get('signal', 'hold')}|{item.get('full_data', '')}"
            )
        
        prompt = f"""
请对以下 {len(batch)} 只ETF进行精细比较排名。

数据格式: 代码|初评分|信号|数据

{chr(10).join(data_text)}

任务: 根据相对强弱重新排名，给出精细评分(0-100分)。

输出JSON:
{{
    "rankings": [
        {{"symbol": "代码", "rank_score": 分数, "signal": "buy/hold/sell", "reason": "理由"}}
    ]
}}"""

        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=400,
                temperature=0.3,
                enable_thinking=False
            )
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('rankings', [])
                
        except Exception as e:
            print(f"   ⚠️ 排名失败: {e}")
        
        return batch


class ETFDeepAnalyzeSkill(BaseSkill):
    """
    技能: ETF深度分析
    功能: 对最终候选进行深度分析（阶段3）
    """
    
    def __init__(self):
        super().__init__(
            name="etf_deep_analyze",
            description="对ETF进行深度技术分析"
        )
        self.llm = get_llm_client()
        self.data_skill = ETFDataSkill()
    
    def execute(self, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        深度分析
        """
        results = []
        total = min(len(candidates), 50)
        
        for i, item in enumerate(candidates[:total], 1):
            symbol = item.get('symbol')
            if not symbol:
                continue
            
            print(f"  {i}/{total}: 深度分析 {symbol}...")
            
            try:
                analysis = self._analyze_single(symbol, item)
                if analysis:
                    results.append(analysis)
            except Exception as e:
                print(f"    ⚠️ 分析 {symbol} 失败: {e}")
        
        # 综合评分
        for item in results:
            item['final_score'] = (
                item.get('deep_score', 0) * 0.5 +
                item.get('rank_score', 0) * 0.25 +
                item.get('quick_score', 0) * 0.25
            )
        
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return results[:top_k * 3]
    
    def _analyze_single(self, symbol: str, item: Dict) -> Optional[Dict]:
        """分析单只ETF"""
        # 获取数据
        df = self.data_skill.execute([symbol], days=60).get(symbol)
        if df is None or df.empty:
            return None
        
        indicators = self.data_skill.get_indicators(df)
        data_str = self.data_skill.get_summary(df, days=30)
        
        prompt = f"""
# 任务
对 {symbol} 进行深度技术分析。

# 技术指标
价格: {indicators['price']:.4f}
MA20: {indicators['ma20']:.4f}
MA60: {indicators['ma60']:.4f}
RSI: {indicators['rsi']:.1f}
波动率: {indicators['volatility']:.2%}

# 数据
{data_str}

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
                max_new_tokens=600,
                temperature=0.3,
                enable_thinking=False
            )
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                result['symbol'] = symbol
                result['current_price'] = indicators['price']
                result['ma20'] = indicators['ma20']
                result['rsi'] = indicators['rsi']
                result['quick_score'] = item.get('score', 50)
                result['rank_score'] = item.get('rank_score', 50)
                return result
                
        except Exception as e:
            print(f"    ⚠️ 分析失败: {e}")
        
        return None