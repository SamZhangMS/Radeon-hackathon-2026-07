# app/advisor.py
"""
投资顾问引擎 - 整合预测和推荐功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

from .predictor import ETFPricePredictor
from .data_fetcher import ETFDataFetcher
from .utils import get_last_date, format_exception
from .config import AGENT_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT_EXTENDED


class InvestmentAdvisor:
    """投资顾问引擎 - 整合预测和推荐"""
    
    def __init__(self):
        self.predictor = ETFPricePredictor()
        self.fetcher = ETFDataFetcher()
    
    # ============================================================
    # 主入口
    # ============================================================
    
    def get_recommendation(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """获取投资建议 - 优先使用LLM，降级到规则引擎"""
        try:
            if df.empty or len(df) < 60:
                return self._empty_response(symbol)
            
            df_clean = df.copy()
            
            # 1. 计算技术指标
            indicators = self._calculate_indicators(df_clean)
            indicators = self._clean_indicators(indicators)
            
            # 2. 获取价格预测
            prediction = self.predictor.predict(df_clean)
            pred = prediction if prediction.get('success', False) else None
            
            # 3. 尝试LLM分析
            llm_result = self._call_llm_for_analysis(symbol, df_clean, indicators, pred)
            if llm_result.get('success'):
                return self._format_llm_response(llm_result, indicators, pred, df_clean)
            
            # 4. 降级到规则引擎
            return self._get_rule_based_recommendation(symbol, df_clean, indicators, pred)
            
        except Exception as e:
            print(f"get_recommendation error: {e}\n{format_exception(e)}")
            return self._error_response(symbol, str(e))
    
    # ============================================================
    # LLM 分析（统一入口）
    # ============================================================
    
    def _call_llm_for_analysis(self, symbol: str, df: pd.DataFrame,
                               indicators: Dict, prediction: Dict) -> Dict:
        """调用LLM进行分析 - 统一入口"""
        try:
            from .llm_client import get_llm_client
            llm = get_llm_client()
            
            # 构建prompt（使用DeepSeek-R1-Distill-Qwen-1.5B格式）
            prompt = self._build_analysis_prompt(symbol, df, indicators, prediction)
            system_prompt = AGENT_SYSTEM_PROMPT + "\n" + AGENT_SYSTEM_PROMPT_EXTENDED
            
            response = llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_new_tokens=600,
                temperature=0.3,
                enable_thinking=False
            )
            
            return self._parse_llm_response(response)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _build_analysis_prompt(self, symbol: str, df: pd.DataFrame,
                               indicators: Dict, prediction: Dict) -> str:
        """构建分析prompt - DeepSeek-R1-Distill-Qwen-1.5B格式"""
        current_price = indicators['price']
        latest_date = get_last_date(df)
        pred_change = prediction.get('predicted_change', 0) if prediction else 0
        
        # 获取最近10天数据
        recent_df = df.tail(10)
        data_str = ""
        for idx, row in recent_df.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            data_str += f"{date_str} O:{row['open']:.4f} H:{row['high']:.4f} L:{row['low']:.4f} C:{row['close']:.4f}\n"
        
        return f"""# 任务
你是一个股票分析师，请根据输入的股票数据，运用技术分析指标，做出简洁的股票涨跌预测。

# 核心分析要点
- **趋势判断**：MA均线排列、MACD动能、RSI强弱
- **关键信号**：金叉死叉、背离现象、突破确认
- **风险提示**：超买超卖、支撑压力位

# 输入数据
标的: {symbol}
日期: {latest_date}
当前价格: {current_price:.4f}
趋势: {indicators['trend']}
RSI(14): {indicators['rsi']:.1f}
MACD柱: {indicators['macd_hist']:.6f}
MA5: {indicators['ma5']:.4f}
MA20: {indicators['ma20']:.4f}
MA60: {indicators['ma60']:.4f}
布林上轨: {indicators['bb_upper']:.4f}
布林下轨: {indicators['bb_lower']:.4f}
波动率: {indicators['volatility']:.2%}
预测变化: {pred_change:.2%}

最近10天数据:
{data_str}

# 输出格式
<content>简要分析（不多于300字）</content>
<recommendation>buy/hold/sell/neutral</recommendation>
<signal>强烈买入/买入/持有/谨慎持有/卖出/观望</signal>
<score>整数(-8到8)</score>
<confidence>0.0-1.0</confidence>
<risk_level>low/medium/high</risk_level>
<target_price>目标价</target_price>
<stop_loss>止损价</stop_loss>"""
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            # 尝试JSON解析（兼容旧格式）
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                result['success'] = True
                return result
            
            # 尝试XML标签解析（新格式）
            tags = ['content', 'recommendation', 'signal', 'score', 
                   'confidence', 'risk_level', 'target_price', 'stop_loss']
            result = {'success': True}
            
            for tag in tags:
                match = re.search(f'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    if tag == 'score':
                        result[tag] = int(value) if value else 0
                    elif tag in ['confidence', 'target_price', 'stop_loss']:
                        result[tag] = float(value) if value else 0
                    else:
                        result[tag] = value
            
            # 确保必要字段存在
            if 'recommendation' not in result:
                result['recommendation'] = 'neutral'
            if 'signal' not in result:
                result['signal'] = '观望'
            if 'score' not in result:
                result['score'] = 0
            if 'confidence' not in result:
                result['confidence'] = 0.5
            if 'risk_level' not in result:
                result['risk_level'] = 'medium'
            
            # 转换recommendation为小写
            if result.get('recommendation'):
                result['recommendation'] = result['recommendation'].lower()
            
            result['reasons'] = [result.get('content', '基于技术分析预测')[:200]]
            result['analysis'] = result.get('content', '')
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _format_llm_response(self, llm_result: Dict, indicators: Dict,
                             prediction: Dict, df: pd.DataFrame) -> Dict:
        """格式化LLM响应"""
        return {
            'recommendation': llm_result.get('recommendation', 'neutral'),
            'signal': llm_result.get('signal', '持有'),
            'score': llm_result.get('score', 0),
            'reasons': llm_result.get('reasons', []),
            'confidence': llm_result.get('confidence', 0.5),
            'current_price': indicators['price'],
            'technical': indicators,
            'prediction': prediction,
            'risk_level': llm_result.get('risk_level', 'medium'),
            'target_price': llm_result.get('target_price', indicators['price'] * 1.05),
            'stop_loss': llm_result.get('stop_loss', indicators['price'] * 0.95),
            'llm_analysis': llm_result.get('analysis', ''),
            'latest_date': self._get_latest_date(df),
            'generatedby': 'LLM'
        }
    
    # ============================================================
    # 规则引擎（降级方案）
    # ============================================================
    
    def _get_rule_based_recommendation(self, symbol: str, df: pd.DataFrame,
                                       indicators: Dict, prediction: Dict) -> Dict:
        """规则引擎降级方案"""
        pred_trend = prediction.get('predicted_change', 0) if prediction else 0
        score = 0
        reasons = []
        
        # 趋势判断
        if indicators['trend'] == 'up':
            score += 2
            reasons.append('📈 上升趋势')
        elif indicators['trend'] == 'down':
            score -= 2
            reasons.append('📉 下降趋势')
        else:
            reasons.append('➡️ 横盘整理')
        
        # RSI判断
        if indicators['rsi'] < 30:
            score += 2
            reasons.append(f'🟢 RSI超卖 ({indicators["rsi"]:.1f})')
        elif indicators['rsi'] > 70:
            score -= 2
            reasons.append(f'🔴 RSI超买 ({indicators["rsi"]:.1f})')
        else:
            reasons.append(f'⚪ RSI中性 ({indicators["rsi"]:.1f})')
        
        # MACD判断
        if indicators['macd_hist'] > 0:
            score += 1
            reasons.append('🟢 MACD金叉')
        else:
            score -= 1
            reasons.append('🔴 MACD死叉')
        
        # 均线判断
        if indicators['price'] > indicators['ma20']:
            score += 1
            reasons.append(f'✅ 价格在MA20({indicators["ma20"]:.3f})上方')
        else:
            score -= 1
            reasons.append(f'❌ 价格在MA20({indicators["ma20"]:.3f})下方')
        
        # 预测判断
        if pred_trend > 0.03:
            score += 2
            reasons.append(f'🔮 预测上涨 {pred_trend:.1%}')
        elif pred_trend < -0.03:
            score -= 2
            reasons.append(f'🔮 预测下跌 {abs(pred_trend):.1%}')
        else:
            reasons.append(f'🔮 预测平稳 ({pred_trend:.1%})')
        
        # 生成建议
        if score >= 4:
            recommendation, signal = 'buy', '强烈买入'
        elif score >= 2:
            recommendation, signal = 'buy', '买入'
        elif score >= 0:
            recommendation, signal = 'hold', '持有'
        elif score >= -2:
            recommendation, signal = 'hold', '谨慎持有'
        else:
            recommendation, signal = 'sell', '卖出'
        
        risk = self._assess_risk(df)
        
        return {
            'recommendation': recommendation,
            'signal': signal,
            'score': score,
            'reasons': reasons,
            'confidence': min(abs(score) / 8, 1.0),
            'current_price': indicators['price'],
            'technical': indicators,
            'prediction': prediction,
            'risk_level': risk,
            'target_price': indicators['price'] * (1 + 0.05 * (score / 4)),
            'stop_loss': indicators['price'] * (1 - 0.03 * abs(score) / 4),
            'latest_date': self._get_latest_date(df),
            'generatedby': 'rule_based engine'
        }
    
    # ============================================================
    # 技术指标计算
    # ============================================================
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        close = df['close']
        close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        if close.isna().all():
            return self._empty_indicators()
        
        # 均线
        ma5 = float(close.rolling(5).mean().iloc[-1]) if not pd.isna(close.rolling(5).mean().iloc[-1]) else 0.0
        ma10 = float(close.rolling(10).mean().iloc[-1]) if not pd.isna(close.rolling(10).mean().iloc[-1]) else 0.0
        ma20 = float(close.rolling(20).mean().iloc[-1]) if not pd.isna(close.rolling(20).mean().iloc[-1]) else 0.0
        ma60 = float(close.rolling(60).mean().iloc[-1]) if not pd.isna(close.rolling(60).mean().iloc[-1]) else 0.0
        
        # 趋势
        if ma5 > ma20 > ma60:
            trend = 'up'
        elif ma5 < ma20 < ma60:
            trend = 'down'
        else:
            trend = 'sideways'
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if rs.iloc[-1] != 0 else 50.0
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd - signal).iloc[-1])
        
        # 布林带
        bb_middle = float(close.rolling(20).mean().iloc[-1])
        bb_std = float(close.rolling(20).std().iloc[-1])
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        return {
            'price': float(close.iloc[-1]),
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'ma60': ma60,
            'trend': trend,
            'rsi': rsi,
            'macd': float(macd.iloc[-1]),
            'macd_signal': float(signal.iloc[-1]),
            'macd_hist': macd_hist,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'volatility': float(close.pct_change().std() * np.sqrt(252)),
        }
    
    def _clean_indicators(self, indicators: Dict) -> Dict:
        """清理指标中的NaN和Inf"""
        for key, value in indicators.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                indicators[key] = 0.0
        return indicators
    
    def _empty_indicators(self) -> Dict:
        """返回空指标"""
        return {
            'price': 0.0, 'ma5': 0.0, 'ma10': 0.0, 'ma20': 0.0, 'ma60': 0.0,
            'trend': 'sideways', 'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0,
            'macd_hist': 0.0, 'bb_upper': 0.0, 'bb_middle': 0.0, 'bb_lower': 0.0,
            'volatility': 0.0,
        }
    
    # ============================================================
    # 风险评估
    # ============================================================
    
    def _assess_risk(self, df: pd.DataFrame) -> str:
        """评估风险等级"""
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        cummax = df['close'].cummax()
        drawdown = (cummax - df['close']) / cummax
        max_drawdown = drawdown.max()
        
        if volatility < 0.2 and abs(max_drawdown) < 0.15:
            return 'low'
        elif volatility < 0.35 and abs(max_drawdown) < 0.3:
            return 'medium'
        else:
            return 'high'
    
    # ============================================================
    # Top 推荐
    # ============================================================
    
    def get_top_recommendations(self, symbols: List[str] = None) -> Dict[str, List]:
        """获取 Top 3 推荐"""
        if symbols is None:
            symbols = self.fetcher.get_etf_list()
        
        results = {'buy': [], 'hold': [], 'sell': [], 'all_recommendations': []}
        
        for symbol in symbols:
            df = self.fetcher.get_history(symbol, "6mo")
            if df.empty:
                continue
            
            advice = self.get_recommendation(symbol, df)
            result = {
                'symbol': symbol,
                'name': advice.get('name', symbol),
                'recommendation': advice['recommendation'],
                'signal': advice['signal'],
                'score': advice['score'],
                'confidence': advice['confidence'],
                'current_price': advice['current_price'],
                'target_price': advice.get('target_price', 0),
                'risk_level': advice['risk_level'],
                'reasons': advice['reasons'],
                'latest_date': self._get_latest_date(df)
            }
            results['all_recommendations'].append(result)
            
            if advice['recommendation'] == 'buy':
                results['buy'].append(result)
            elif advice['recommendation'] == 'hold':
                results['hold'].append(result)
            else:
                results['sell'].append(result)
        
        for key in ['buy', 'hold', 'sell']:
            results[key] = sorted(
                results[key],
                key=lambda x: x['score'],
                reverse=(key == 'buy')
            )[:3]
        
        results['latest_date'] = datetime.now().strftime('%Y-%m-%d')
        return results
    
    # ============================================================
    # LLM 预测（新增）
    # ============================================================
    
    def predict_with_llm(self, symbol: str, df: pd.DataFrame, 
                         llm_type: str = 'qwen_local') -> Dict:
        """使用LLM进行价格预测"""
        try:
            from .llm_client import get_llm_client
            
            if len(df) < 30:
                return {'success': False, 'error': '数据不足，需要至少30个交易日'}
            
            llm = get_llm_client()
            
            # 准备数据
            n_days = min(100, len(df))
            recent_df = df.tail(n_days).copy()
            data_str = self._df_to_text(recent_df)
            
            current_price = float(df['close'].iloc[-1])
            
            prompt = f"""# 任务
基于以下ETF数据预测未来20个交易日的收盘价。

# 输入数据
标的: {symbol}
当前价格: {current_price:.4f}
最近{n_days}天数据:
{data_str}

# 输出格式
<analysis>简要分析（不多于100字）</analysis>
<prediction>[价格数组，20个数值]</prediction>"""

            response = llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=800,
                temperature=0.2,
                enable_thinking=False
            )
            
            return self._parse_prediction_response(response, df, symbol)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _df_to_text(self, df: pd.DataFrame) -> str:
        """将DataFrame转为文本格式"""
        lines = []
        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            lines.append(
                f"{date_str} O:{row['open']:.4f} H:{row['high']:.4f} "
                f"L:{row['low']:.4f} C:{row['close']:.4f} V:{row['volume']:.0f}"
            )
        return "\n".join(lines)
    
    def _parse_prediction_response(self, response: str, df: pd.DataFrame, 
                                   symbol: str) -> Dict:
        """解析预测响应"""
        import re
        
        # 提取分析
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else ""
        
        # 提取价格数组
        price_match = re.search(r'<prediction>\s*\[([\d.,\s]+)\]\s*</prediction>', response, re.DOTALL)
        if price_match:
            try:
                pred_prices = [float(x.strip()) for x in price_match.group(1).split(',')]
                if len(pred_prices) >= 20:
                    pred_prices = pred_prices[:20]
                    last_date = get_last_date(df)
                    future_dates = self._generate_future_dates(last_date, len(pred_prices))
                    
                    return {
                        'success': True,
                        'symbol': symbol,
                        'model': 'LLM',
                        'dates': future_dates,
                        'close': pred_prices,
                        'analysis': analysis,
                        'predicted_change': (pred_prices[-1] - pred_prices[0]) / pred_prices[0],
                        'confidence': 0.6
                    }
            except:
                pass
        
        return {'success': False, 'error': '无法解析预测结果'}
    
    def _generate_future_dates(self, last_date, n_days: int) -> List[str]:
        """生成未来日期列表"""
        from .utils import generate_future_daily_dates
        dates = generate_future_daily_dates(last_date, n_days)
        return [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates]
    
    # ============================================================
    # 工具方法
    # ============================================================
    
    def _get_latest_date(self, df: pd.DataFrame) -> str:
        """获取最新日期"""
        return get_last_date(df).strftime('%Y/%m/%d')
    
    def _empty_response(self, symbol: str) -> Dict:
        """空响应"""
        return {
            'recommendation': 'neutral',
            'confidence': 0,
            'reasons': ['数据不足'],
            'signal': 'wait',
            'latest_date': None
        }
    
    def _error_response(self, symbol: str, error: str) -> Dict:
        """错误响应"""
        return {
            'recommendation': 'neutral',
            'confidence': 0,
            'reasons': [f'分析失败: {error}'],
            'signal': 'wait',
            'latest_date': None,
            'error': error
        }