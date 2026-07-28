import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime, timedelta

from .predictor import ETFPricePredictor
from .data_fetcher import ETFDataFetcher
from .utils import get_latest_date


class InvestmentAdvisor:
    """投资顾问引擎"""
    
    def __init__(self):
        self.predictor = ETFPricePredictor()
        self.fetcher = ETFDataFetcher()
    
    def get_recommendation(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """获取投资建议 - 使用大模型生成完整建议"""
        if df.empty or len(df) < 60:
            return {
                'recommendation': 'neutral',
                'confidence': 0,
                'reasons': ['数据不足'],
                'signal': 'wait',
                'latest_date': None
            }
        
        # 1. 计算技术指标（用于上下文）
        indicators = self._calculate_indicators(df)
        
        # 2. 获取价格预测
        prediction = self.predictor.predict(df)
        if not prediction.get('success', False):
            pred_trend = 0
            pred = None
        else:
            pred_trend = prediction.get('predicted_change', 0)
            pred = prediction
        
        # 3. 使用大模型生成完整投资建议
        llm_analysis = self._get_llm_analysis(symbol, df, indicators, pred)
        
        # 4. 如果大模型分析失败，使用规则引擎作为降级方案
        if llm_analysis.get('success', False):
            return {
                'recommendation': llm_analysis.get('recommendation', 'neutral'),
                'signal': llm_analysis.get('signal', '持有'),
                'score': llm_analysis.get('score', 0),
                'reasons': llm_analysis.get('reasons', []),
                'confidence': llm_analysis.get('confidence', 0.5),
                'current_price': indicators['price'],
                'technical': indicators,
                'prediction': pred,
                'risk_level': llm_analysis.get('risk_level', 'medium'),
                'target_price': llm_analysis.get('target_price', indicators['price'] * 1.05),
                'stop_loss': llm_analysis.get('stop_loss', indicators['price'] * 0.95),
                'llm_analysis': llm_analysis.get('analysis', ''),
                'latest_date': self._get_latest_date(df),
                'generatedby': 'LLM'
            }
        else:
            # 降级方案：使用规则引擎
            return self._get_rule_based_recommendation(symbol, df, indicators, pred)
            
            
    def _get_llm_analysis(self, symbol: str, df: pd.DataFrame, 
                          indicators: Dict, prediction: Dict) -> Dict:
        """调用大模型进行投资分析"""
        try:
            from .llm_client import get_llm_client
            
            llm = get_llm_client()
            
            # 准备数据摘要
            current_price = indicators['price']
            ma5 = indicators['ma5']
            ma20 = indicators['ma20']
            ma60 = indicators['ma60']
            rsi = indicators['rsi']
            macd_hist = indicators['macd_hist']
            trend = indicators['trend']
            volatility = indicators['volatility']
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            
            # 获取预测变化
            pred_change = prediction.get('predicted_change', 0) if prediction else 0
            
            # 获取最新日期
            latest_date = self._get_latest_date(df)
            
            prompt = f"""你是一个专业的 ETF 投资顾问。请基于以下技术指标和预测数据，对 {symbol} 进行完整的投资分析。

【当前数据】（日期: {latest_date}）
- 当前价格: {current_price:.4f}
- 5日均线: {ma5:.4f}
- 20日均线: {ma20:.4f}
- 60日均线: {ma60:.4f}
- 趋势: {trend}
- RSI(14): {rsi:.1f}
- MACD柱: {macd_hist:.6f}
- 布林上轨: {bb_upper:.4f}
- 布林下轨: {bb_lower:.4f}
- 波动率: {volatility:.2%}
- 预测变化: {pred_change:.2%}

请给出以下投资建议：
1. 建议类型: buy(买入), hold(持有), sell(卖出), neutral(观望)
2. 建议信号: 强烈买入/买入/持有/谨慎持有/卖出/观望
3. 综合评分: -8 到 8 之间的整数
4. 分析依据: 3-5条理由
5. 置信度: 0-1 之间的数值
6. 风险等级: low/medium/high
7. 目标价: 合理的目标价格
8. 止损价: 合理的止损价格
9. 详细分析: 一段完整的投资分析建议

请以 JSON 格式返回，格式如下：
{{
    "recommendation": "buy",
    "signal": "买入",
    "score": 6,
    "reasons": ["理由1", "理由2", "理由3"],
    "confidence": 0.85,
    "risk_level": "medium",
    "target_price": 5.20,
    "stop_loss": 4.35,
    "analysis": "详细的分析建议..."
}}"""

            response = llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=600,
                temperature=0.3,
                enable_thinking=False
            )
            
            # 解析 JSON 响应
            import re
            import json
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    result['success'] = True
                    return result
                except:
                    pass
            
            return {'success': False, 'error': '无法解析大模型响应'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_rule_based_recommendation(self, symbol: str, df: pd.DataFrame,
                                       indicators: Dict, prediction: Dict) -> Dict:
        """规则引擎降级方案"""
        pred_trend = prediction.get('predicted_change', 0) if prediction else 0
        
        score = 0
        reasons = []
        
        if indicators['trend'] == 'up':
            score += 2
            reasons.append('📈 上升趋势')
        elif indicators['trend'] == 'down':
            score -= 2
            reasons.append('📉 下降趋势')
        else:
            reasons.append('➡️ 横盘整理')
        
        if indicators['rsi'] < 30:
            score += 2
            reasons.append(f'🟢 RSI超卖 ({indicators["rsi"]:.1f})')
        elif indicators['rsi'] > 70:
            score -= 2
            reasons.append(f'🔴 RSI超买 ({indicators["rsi"]:.1f})')
        else:
            reasons.append(f'⚪ RSI中性 ({indicators["rsi"]:.1f})')
        
        if indicators['macd_hist'] > 0:
            score += 1
            reasons.append('🟢 MACD金叉')
        else:
            score -= 1
            reasons.append('🔴 MACD死叉')
        
        if indicators['price'] > indicators['ma20']:
            score += 1
            reasons.append(f'✅ 价格在MA20({indicators["ma20"]:.3f})上方')
        else:
            score -= 1
            reasons.append(f'❌ 价格在MA20({indicators["ma20"]:.3f})下方')
        
        if pred_trend > 0.03:
            score += 2
            reasons.append(f'🔮 预测上涨 {pred_trend:.1%}')
        elif pred_trend < -0.03:
            score -= 2
            reasons.append(f'🔮 预测下跌 {abs(pred_trend):.1%}')
        else:
            reasons.append(f'🔮 预测平稳 ({pred_trend:.1%})')
        
        if score >= 4:
            recommendation = 'buy'
            signal = '强烈买入'
        elif score >= 2:
            recommendation = 'buy'
            signal = '买入'
        elif score >= 0:
            recommendation = 'hold'
            signal = '持有'
        elif score >= -2:
            recommendation = 'hold'
            signal = '谨慎持有'
        else:
            recommendation = 'sell'
            signal = '卖出'
        
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
            'generatedby':'rule_based engine'
        }
    
    def _get_latest_date(self, df: pd.DataFrame) -> str:
        """获取最新日期"""
        from .utils import get_latest_date
        return get_latest_date(df)
    
    def get_top_recommendations(self, symbols: List[str] = None) -> Dict[str, List]:
        """获取 Top 3 推荐"""
        if symbols is None:
            symbols = self.fetcher.get_etf_list()
        
        results = {
            'buy': [],
            'hold': [],
            'sell': [],
            'all_recommendations': []
        }
        
        for symbol in symbols:
            df = self.fetcher.get_history(symbol, "6mo")
            if df.empty:
                continue
            
            advice = self.get_recommendation(symbol, df)
            
            latest_date = self._get_latest_date(df)
            
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
                'latest_date': latest_date
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

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        close = df['close']
        
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma10 = float(close.rolling(10).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        
        if ma5 > ma20 > ma60:
            trend = 'up'
        elif ma5 < ma20 < ma60:
            trend = 'down'
        else:
            trend = 'sideways'
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])
        
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd - signal).iloc[-1])
        
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
        
    def _get_qwen_analysis(self, symbol: str, advice: Dict) -> str:
        """调用 Qwen 进行增强分析"""
        try:
            from .llm_client import get_llm_client
        
            llm = get_llm_client()
            
            # 构建分析上下文
            prompt = f"""请对 ETF {symbol} 进行专业分析：

    当前价格: {advice['current_price']:.3f}
    技术趋势: {advice['technical']['trend']}
    RSI: {advice['technical']['rsi']:.1f}
    MACD柱: {advice['technical']['macd_hist']:.4f}
    投资建议: {advice['signal']}
    风险等级: {advice['risk_level']}

    请用简洁专业的语言给出分析评语和投资建议（50字以内）。"""

            messages = [{"role": "user", "content": prompt}]
            response = llm.generate_response(
                messages=messages,
                max_new_tokens=100,
                enable_thinking=False
            )
            
            return f"🤖 Qwen 分析: {response}"
        except Exception as e:
            return f"Qwen 分析暂时不可用: {e}"
    

        """评估风险等级"""
        returns = df['close'].pct_change().dropna()
        
        # 波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 最大回撤
        cummax = df['close'].cummax()
        drawdown = (cummax - df['close']) / cummax
        max_drawdown = drawdown.max()
        
        if volatility < 0.2 and abs(max_drawdown) < 0.15:
            return 'low'
        elif volatility < 0.35 and abs(max_drawdown) < 0.3:
            return 'medium'
        else:
            return 'high'