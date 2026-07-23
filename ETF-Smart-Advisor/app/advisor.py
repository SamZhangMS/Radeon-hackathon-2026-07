import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .predictor import ETFPricePredictor
from .data_fetcher import ETFDataFetcher


class InvestmentAdvisor:
    """投资顾问引擎"""
    
    def __init__(self):
        self.predictor = ETFPricePredictor()
        self.fetcher = ETFDataFetcher()
    
    def get_recommendation(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """获取投资建议"""
        if df.empty or len(df) < 60:
            return {
                'recommendation': 'neutral',
                'confidence': 0,
                'reasons': ['数据不足'],
                'signal': 'wait'
            }
        
        # 1. 计算技术指标
        indicators = self._calculate_indicators(df)
        
        # 2. 预测未来价格
        prediction = self.predictor.predict(df)
        if not prediction.get('success', False):
            pred_trend = 0
        else:
            pred_trend = prediction.get('predicted_change', 0)
        
        # 3. 综合评分
        score = 0
        reasons = []
        
        # 趋势分析
        if indicators['trend'] == 'up':
            score += 2
            reasons.append('📈 上升趋势')
        elif indicators['trend'] == 'down':
            score -= 2
            reasons.append('📉 下降趋势')
        else:
            reasons.append('➡️ 横盘整理')
        
        # RSI分析
        if indicators['rsi'] < 30:
            score += 2
            reasons.append(f'🟢 RSI超卖 ({indicators["rsi"]:.1f})')
        elif indicators['rsi'] > 70:
            score -= 2
            reasons.append(f'🔴 RSI超买 ({indicators["rsi"]:.1f})')
        else:
            reasons.append(f'⚪ RSI中性 ({indicators["rsi"]:.1f})')
        
        # MACD分析
        if indicators['macd_hist'] > 0:
            score += 1
            reasons.append('🟢 MACD金叉')
        else:
            score -= 1
            reasons.append('🔴 MACD死叉')
        
        # 均线分析
        if indicators['price'] > indicators['ma20']:
            score += 1
            reasons.append(f'✅ 价格在MA20({indicators["ma20"]:.3f})上方')
        else:
            score -= 1
            reasons.append(f'❌ 价格在MA20({indicators["ma20"]:.3f})下方')
        
        # 预测趋势
        if pred_trend > 0.03:
            score += 2
            reasons.append(f'🔮 预测上涨 {pred_trend:.1%}')
        elif pred_trend < -0.03:
            score -= 2
            reasons.append(f'🔮 预测下跌 {abs(pred_trend):.1%}')
        else:
            reasons.append(f'🔮 预测平稳 ({pred_trend:.1%})')
        
        # 4. 确定建议
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
        
        # 5. 风险等级
        risk = self._assess_risk(df)
        
        return {
            'recommendation': recommendation,
            'signal': signal,
            'score': score,
            'reasons': reasons,
            'confidence': min(abs(score) / 8, 1.0),
            'current_price': indicators['price'],
            'technical': indicators,
            'prediction': prediction if prediction.get('success', False) else None,
            'risk_level': risk,
            'target_price': indicators['price'] * (1 + 0.05 * (score / 4)),
            'stop_loss': indicators['price'] * (1 - 0.03 * abs(score) / 4),
        }
    
    # advisor.py - 在 InvestmentAdvisor 类中添加

    def get_top_recommendations(self, symbols: List[str] = None) -> Dict[str, List]:
        """✅ 获取 Top 3 买入/卖出/持有推荐（新增）
        
        Args:
            symbols: ETF 代码列表，默认使用配置中的 DEFAULT_ETF_POOL
        """
        from .config import DEFAULT_ETF_POOL
        
        if symbols is None:
            symbols = DEFAULT_ETF_POOL
        
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
            
            # 获取 Qwen 增强分析（通过 Agent）
            qwen_analysis = self._get_qwen_analysis(symbol, advice)
            
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
                'qwen_analysis': qwen_analysis,  # Qwen 分析结果
            }
            results['all_recommendations'].append(result)
            
            # 分类
            if advice['recommendation'] == 'buy':
                results['buy'].append(result)
            elif advice['recommendation'] == 'hold':
                results['hold'].append(result)
            else:
                results['sell'].append(result)
        
        # 按评分排序，取 Top 3
        for key in ['buy', 'hold', 'sell']:
            results[key] = sorted(
                results[key],
                key=lambda x: x['score'],
                reverse=(key == 'buy')
            )[:3]
        
        return results

    def _get_qwen_analysis(self, symbol: str, advice: Dict) -> str:
        """✅ 调用 Qwen 进行增强分析（新增）"""
        try:
            # 构建分析上下文
            context = f"""
            请对 ETF {symbol} 进行专业分析：
            - 当前价格: {advice['current_price']}
            - 技术趋势: {advice['technical']['trend']}
            - RSI: {advice['technical']['rsi']:.1f}
            - 投资建议: {advice['signal']}
            - 风险等级: {advice['risk_level']}
            
            请给出简短的分析评语和投资建议。
            """
            
            # 调用 Qwen（通过 Agent 的 chat 方法）
            # 这里简化处理，实际可通过 Agent 的 LLM 调用
            return f"Qwen 分析: {symbol} 当前处于{advice['technical']['trend']}趋势，RSI={advice['technical']['rsi']:.1f}，建议{advice['signal']}。"
        except Exception as e:
            return f"Qwen 分析暂时不可用: {e}"
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        close = df['Close']
        
        # 移动平均线
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        
        # 趋势判断
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
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - signal).iloc[-1]
        
        # 布林带
        bb_middle = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        return {
            'price': close.iloc[-1],
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'ma60': ma60,
            'trend': trend,
            'rsi': rsi,
            'macd': macd.iloc[-1],
            'macd_signal': signal.iloc[-1],
            'macd_hist': macd_hist,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'volatility': close.pct_change().std() * np.sqrt(252),
        }
    
    def _assess_risk(self, df: pd.DataFrame) -> str:
        """评估风险等级"""
        returns = df['Close'].pct_change().dropna()
        
        # 波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 最大回撤
        cummax = df['Close'].cummax()
        drawdown = (cummax - df['Close']) / cummax
        max_drawdown = drawdown.max()
        
        if volatility < 0.2 and abs(max_drawdown) < 0.15:
            return 'low'
        elif volatility < 0.35 and abs(max_drawdown) < 0.3:
            return 'medium'
        else:
            return 'high'