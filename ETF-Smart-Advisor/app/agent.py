from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import Dict, Any

from .gpu_optimizer import ROCmGPUOptimizer
from .stability_manager import StabilityManager
from .feedback_learning import FeedbackLearning
from .lightweight_adapter import LightweightAdapter

from .config import LLM_CONFIG, AGENT_SYSTEM_PROMPT, DEFAULT_ETF_POOL
from .data_fetcher import ETFDataFetcher
from .advisor import InvestmentAdvisor
from .predictor import ETFPricePredictor


class ETFAdvisorAgent:
    """ETF智能投顾Agent"""
    
    def __init__(self):
        # 初始化 GPU 优化器
        self.gpu_optimizer = ROCmGPUOptimizer()
        
        # 初始化稳定性管理器
        self.stability_manager = StabilityManager()
        self.stability_manager.start()
        
        # 初始化反馈学习
        self.feedback_learning = FeedbackLearning()
        
        # 初始化轻量化适配器
        self.lightweight_adapter = LightweightAdapter()
 
        # 初始化LLM
        provider = OpenAIProvider(
            base_url=LLM_CONFIG["api_base"],
            api_key=LLM_CONFIG["api_key"],
        )
        model = OpenAIModel(
            LLM_CONFIG["model_name"],
            provider=provider
        )
        
        # 初始化服务
        self.fetcher = ETFDataFetcher()
        self.advisor = InvestmentAdvisor()
        self.predictor = ETFPricePredictor()
        
        if hasattr(self.predictor, 'model'):
            self.predictor.model = self.gpu_optimizer.optimize_model(self.predictor.model )

        # 创建Agent
        self.agent = Agent(
            model=model,
            system_prompt=AGENT_SYSTEM_PROMPT,
            tools=self._get_tools(),
        )
        
    def _get_tools(self):
        """获取工具列表"""
        return [
            self._get_quote,
            self._get_analysis,
            self._get_prediction,
            self._get_recommendation,
            self._compare_etfs,
            self._record_feedback,  
            self._get_optimization_status,  
        ]
        
    @Tool
    async def _record_feedback(self, symbol: str, recommendation: str, 
                               actual_result: str, rating: int) -> str:
        """记录用户反馈以改进推荐
        
        Args:
            symbol: ETF代码
            recommendation: 推荐结果 (buy/sell/hold)
            actual_result: 实际结果 (up/down/sideways)
            rating: 用户评分 (1-5)
        """
        self.feedback_learning.record_feedback(
            symbol, recommendation, actual_result, rating
        )
        return f"✅ 反馈已记录，感谢您的反馈！"
    
    @Tool
    async def _get_optimization_status(self) -> str:
        """获取系统优化状态"""
        gpu_stats = self.gpu_optimizer.get_performance_stats()
        stability_report = self.stability_manager.get_status_report()
        resource_report = self.lightweight_adapter.get_resource_report()
        feedback_report = self.feedback_learning.get_accuracy_report()
        
        return (
            f"📊 系统状态报告\n"
            f"{'='*50}\n\n"
            f"🚀 GPU 状态:\n"
            f"  - 设备: {gpu_stats.get('device', 'N/A')}\n"
            f"  - 显存使用: {gpu_stats.get('gpu_memory_allocated', 0):.2f} GB\n"
            f"  - 优化等级: {gpu_stats.get('optimization_level', 'N/A')}\n\n"
            f"🛡️ 服务状态:\n"
            f"  - 健康服务: {sum(1 for s in stability_report['services'].values() if s['status'] == 'healthy')}/{len(stability_report['services'])}\n"
            f"  - 资源使用: CPU {resource_report['cpu_percent']}%, 内存 {resource_report['memory_percent']}%\n\n"
            f"📚 反馈学习:\n"
            f"  - 总反馈数: {feedback_report.get('total_feedback', 0)}\n"
            f"  - 准确率: {feedback_report.get('overall_accuracy', 0)*100:.1f}%\n"
            f"  - 平均评分: {feedback_report.get('avg_rating', 0):.1f}/5"
        )
    
    async def chat(self, message: str, symbol: str = None) -> Dict[str, Any]:
        """处理用户消息 - 优化版"""
        # 检查资源状态
        resource_status = self.lightweight_adapter.get_resource_report()
        if resource_status.get('cpu_percent', 0) > 80:
            return {
                "response": "⚠️ 系统负载较高，建议稍后再试",
                "success": False
            }
        
        # 使用 GPU 优化推理
        try:
            result = await self.agent.run(message)
            response = result.data
            
            # 如果有推荐，应用反馈学习优化
            if symbol and "recommendation" in str(response).lower():
                # 提取推荐内容进行优化
                pass
            
            return {
                "response": response,
                "success": True
            }
        except Exception as e:
            return {
                "response": f"处理失败: {str(e)}",
                "success": False
            }
    
    @Tool
    async def _get_quote(self, symbol: str) -> str:
        """获取ETF实时行情
        
        Args:
            symbol: ETF代码
        """
        quote = self.fetcher.get_etf_quote(symbol)
        if not quote:
            return f"未找到ETF: {symbol}"
        
        return (
            f"📊 {quote['name']} ({symbol})\n"
            f"价格: {quote['price']:.3f}\n"
            f"涨跌幅: {quote['change']:+.2f}%\n"
            f"成交量: {quote['volume']:,.0f}\n"
            f"最高: {quote['high']:.3f}\n"
            f"最低: {quote['low']:.3f}\n"
            f"今开: {quote['open']:.3f}"
        )
    
    @Tool
    async def _get_recommendation(self, symbol: str) -> str:
        """获取买入/卖出/持有建议
        
        Args:
            symbol: ETF代码
        """
        # 获取数据
        df = self.fetcher.get_history(symbol)
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        # 获取建议
        advice = self.advisor.get_recommendation(symbol, df)
        
        # 格式化输出
        emoji = {
            'buy': '🟢',
            'hold': '🟡',
            'sell': '🔴',
            'neutral': '⚪'
        }
        
        risk_emoji = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🔴'
        }
        
        report = f"📈 {symbol} 投资建议\n"
        report += "="*50 + "\n\n"
        report += f"{emoji.get(advice['recommendation'], '⚪')} 建议: {advice['signal']}\n"
        report += f"评分: {advice['score']:.1f} (满分8分)\n"
        report += f"置信度: {advice['confidence']:.0%}\n"
        report += f"风险等级: {risk_emoji.get(advice['risk_level'], '⚪')} {advice['risk_level']}\n\n"
        
        report += "📊 分析依据:\n"
        for reason in advice['reasons']:
            report += f"  {reason}\n"
        
        if advice.get('target_price'):
            report += f"\n🎯 目标价: {advice['target_price']:.3f}\n"
        if advice.get('stop_loss'):
            report += f"🛑 止损价: {advice['stop_loss']:.3f}\n"
        
        # 预测信息
        if advice.get('prediction'):
            pred = advice['prediction']
            report += f"\n🔮 未来{len(pred['close'])}周期预测:\n"
            report += f"  预测变化: {pred['predicted_change']:.2%}\n"
            report += f"  置信区间: ±{pred['confidence']:.2%}\n"
        
        report += f"\n⚠️ 风险提示: 投资有风险，决策需谨慎。"
        
        return report
    
    @Tool
    async def _get_prediction(self, symbol: str) -> str:
        """获取未来价格预测
        
        Args:
            symbol: ETF代码
        """
        df = self.fetcher.get_history(symbol)
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        pred = self.predictor.predict(df)
        if not pred.get('success', False):
            return f"预测失败: {pred.get('error', '未知错误')}"
        
        report = f"🔮 {symbol} 未来价格预测\n"
        report += "="*50 + "\n\n"
        report += f"预测周期: {len(pred['close'])} 个交易日\n"
        report += f"预测变化: {pred['predicted_change']:.2%}\n"
        report += f"置信区间: ±{pred['confidence']:.2%}\n\n"
        
        report += "📅 预测价格表:\n"
        report += "日期\t\t开盘\t最高\t最低\t收盘\n"
        for i in range(min(10, len(pred['dates']))):
            report += f"{pred['dates'][i]}\t{pred['open'][i]:.3f}\t{pred['high'][i]:.3f}\t{pred['low'][i]:.3f}\t{pred['close'][i]:.3f}\n"
        
        if len(pred['dates']) > 10:
            report += f"... 还有 {len(pred['dates']) - 10} 个周期\n"
        
        return report
    
    @Tool
    async def _get_analysis(self, symbol: str) -> str:
        """获取完整技术分析
        
        Args:
            symbol: ETF代码
        """
        df = self.fetcher.get_history(symbol)
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        advice = self.advisor.get_recommendation(symbol, df)
        
        report = f"📊 {symbol} 技术分析\n"
        report += "="*50 + "\n\n"
        
        tech = advice['technical']
        report += f"当前价格: {tech['price']:.3f}\n"
        report += f"趋势: {tech['trend']}\n"
        report += f"RSI: {tech['rsi']:.1f}\n"
        report += f"MACD: {tech['macd']:.4f}\n"
        report += f"MACD信号: {tech['macd_signal']:.4f}\n"
        report += f"MACD柱: {tech['macd_hist']:.4f}\n\n"
        
        report += "📈 均线:\n"
        report += f"  MA5: {tech['ma5']:.3f}\n"
        report += f"  MA10: {tech['ma10']:.3f}\n"
        report += f"  MA20: {tech['ma20']:.3f}\n"
        report += f"  MA60: {tech['ma60']:.3f}\n"
        
        return report
    
    @Tool
    async def _compare_etfs(self, symbols: str) -> str:
        """比较多个ETF
        
        Args:
            symbols: 逗号分隔的ETF代码
        """
        symbol_list = [s.strip() for s in symbols.split(',')]
        
        report = "📊 ETF对比\n"
        report += "="*50 + "\n\n"
        report += "代码\t名称\t价格\t涨跌幅\t建议\n"
        
        for symbol in symbol_list[:5]:
            quote = self.fetcher.get_etf_quote(symbol)
            if quote:
                df = self.fetcher.get_history(symbol)
                if not df.empty:
                    advice = self.advisor.get_recommendation(symbol, df)
                    adv_emoji = {
                        'buy': '🟢买入',
                        'hold': '🟡持有',
                        'sell': '🔴卖出'
                    }.get(advice['recommendation'], '⚪观望')
                    report += f"{symbol}\t{quote['name'][:8]}\t{quote['price']:.3f}\t{quote['change']:+.2f}%\t{adv_emoji}\n"
                else:
                    report += f"{symbol}\t{quote['name'][:8]}\t{quote['price']:.3f}\t{quote['change']:+.2f}%\t数据不足\n"
        
        return report