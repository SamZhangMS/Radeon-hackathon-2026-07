import json
import hashlib
from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path

from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIChatModel

from pydantic_ai.providers.openai import OpenAIProvider
from typing import Dict, Any, List, Optional

from .gpu_optimizer import ROCmGPUOptimizer
from .stability_manager import StabilityManager
from .feedback_learning import FeedbackLearning
from .lightweight_adapter import LightweightAdapter

from .config import LLM_CONFIG, AGENT_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT_EXTENDED, DEFAULT_ETF_POOL, RAG_CONFIG, MEMORY_CONFIG,MODELS_DIR
from .data_fetcher import ETFDataFetcher
from .advisor import InvestmentAdvisor
from .predictor import ETFPricePredictor

logger = logging.getLogger(__name__)

class MemoryManager:
    """本地多轮记忆管理（集成到 Agent 中）"""
    
    def __init__(self, memory_path: str = None):
        if memory_path is None:
            memory_path = MEMORY_CONFIG.get("memory_path", "data/memory.json")
        self.memory_path = memory_path
        self.memories = defaultdict(list)
        self._load_memories()
    
    def _load_memories(self):
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.memories.update(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save_memories(self):
        try:
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.memories), f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def add(self, session_id: str, query: str, response: str):
        memory = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response[:500],
            "hash": hashlib.md5(query.encode()).hexdigest()
        }
        self.memories[session_id].append(memory)
        self._save_memories()
    
    def get_context(self, session_id: str, limit: int = None) -> str:
        if limit is None:
            limit = MEMORY_CONFIG.get("max_history", 10)
        memories = self.memories.get(session_id, [])[-limit:]
        if not memories:
            return ""
        context = ""
        for m in memories:
            context += f"用户: {m['query']}\n助手: {m['response']}\n"
        return context



class TaskPlanner:
    """多步骤任务规划器（集成到 Agent 中）"""
    
    def __init__(self):
        self.step_templates = {
            "quick": [
                {"name": "get_quote", "params": {"symbol": "{symbol}"}, "desc": "获取实时行情"},
                {"name": "get_recommendation", "params": {"symbol": "{symbol}"}, "desc": "获取投资建议"},
            ],
            "full": [
                {"name": "get_quote", "params": {"symbol": "{symbol}"}, "desc": "获取实时行情"},
                {"name": "get_history", "params": {"symbol": "{symbol}", "period": "6mo"}, "desc": "获取历史数据"},
                {"name": "analyze_technical", "params": {"symbol": "{symbol}"}, "desc": "技术分析"},
                {"name": "predict_price", "params": {"symbol": "{symbol}"}, "desc": "价格预测"},
                {"name": "get_recommendation", "params": {"symbol": "{symbol}"}, "desc": "投资建议"},
                {"name": "generate_report", "params": {"symbol": "{symbol}"}, "desc": "生成报告"},
            ],
        }
    
    def plan(self, symbol: str, depth: str = "full") -> List[Dict]:
        steps = self.step_templates.get(depth, self.step_templates["full"])
        result = []
        for step in steps:
            params = {}
            for k, v in step["params"].items():
                if isinstance(v, str):
                    params[k] = v.format(symbol=symbol)
                else:
                    params[k] = v
            result.append({"name": step["name"], "params": params, "desc": step["desc"]})
        return result
    
class ETFAdvisorAgent:
    """ETF智能投顾Agent"""
    
    def __init__(self):
        self.logger = logger
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
        model = OpenAIChatModel(
            LLM_CONFIG["model_name"],
            provider=provider
        )
        
        # 初始化服务
        self.fetcher = ETFDataFetcher()
        self.advisor = InvestmentAdvisor()
        self.predictor = ETFPricePredictor()
        
        lora_path = MODELS_DIR / "lora_etf_advisor"
        if lora_path.exists():
            try:
                self.predictor.load_lora_adapter(str(lora_path))
                self.logger.info(f"✅ LoRA 适配器已加载: {lora_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ LoRA 加载失败: {e}")
                
        if hasattr(self.predictor, 'model'):
            self.predictor.model = self.gpu_optimizer.optimize_model(self.predictor.model )

        self.memory = MemoryManager() if MEMORY_CONFIG.get("enabled", True) else None
        self.planner = TaskPlanner()
        full_system_prompt = AGENT_SYSTEM_PROMPT + "\n" + AGENT_SYSTEM_PROMPT_EXTENDED

        # 创建Agent
        self.agent = Agent(
            model=model,
            system_prompt=full_system_prompt,
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
            self._get_history,
            self._analyze_technical,
            self._search_knowledge,
            self._generate_report,
            self._analyze_complete,
            self._get_top_recommendations,
            self._get_ensemble_prediction,
        ]
        
    @Tool
    async def _get_history(self, symbol: str, period: str = "6mo") -> str:
        """获取ETF历史数据
        
        Args:
            symbol: ETF代码
            period: 周期 (1mo, 3mo, 6mo, 1y, 2y)
        """
        df = self.fetcher.get_history(symbol, period)
        if df.empty:
            return f"无法获取 {symbol} 的历史数据"
        
        return (
            f"📈 {symbol} 历史数据 ({period})\n"
            f"交易日: {len(df)} 天\n"
            f"最新价: {df['Close'].iloc[-1]:.3f}\n"
            f"最高价: {df['High'].max():.3f}\n"
            f"最低价: {df['Low'].min():.3f}\n"
            f"均价: {df['Close'].mean():.3f}"
        )
    
    @Tool
    async def _analyze_technical(self, symbol: str) -> str:
        return await self._analyze_technical_impl(symbol)
    async def _analyze_technical_impl(self, symbol: str) -> str:
        """完整技术分析
        
        Args:
            symbol: ETF代码
        """
        df = self.fetcher.get_history(symbol, "6mo")
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        advice = self.advisor.get_recommendation(symbol, df)
        tech = advice['technical']
        
        return (
            f"📊 {symbol} 技术分析\n"
            f"{'='*40}\n"
            f"当前价格: {tech['price']:.3f}\n"
            f"趋势: {tech['trend']}\n"
            f"RSI: {tech['rsi']:.1f}\n"
            f"MACD: {tech['macd']:.4f}\n"
            f"MACD信号: {tech['macd_signal']:.4f}\n"
            f"MACD柱: {tech['macd_hist']:.4f}\n"
            f"MA5: {tech['ma5']:.3f}\n"
            f"MA20: {tech['ma20']:.3f}\n"
            f"MA60: {tech['ma60']:.3f}\n"
            f"布林上轨: {tech['bb_upper']:.3f}\n"
            f"布林中轨: {tech['bb_middle']:.3f}\n"
            f"布林下轨: {tech['bb_lower']:.3f}\n"
        )
    
    @Tool
    async def _search_knowledge(self, query: str) -> str:
        """搜索ETF知识库
        
        Args:
            query: 搜索问题
        """
        # 模拟RAG检索（实际可集成ChromaDB）
        knowledge_base = {
            "ETF": "ETF（交易型开放式指数基金）是一种在交易所上市交易的基金，可以像股票一样买卖。",
            "网格交易": "网格交易是一种在设定的价格区间内，通过分批买入和卖出获取收益的策略。",
            "定投": "定期定额投资是一种长期投资策略，通过固定时间投入固定金额来平均成本。",
            "沪深300": "沪深300指数由沪深两市规模最大、流动性最好的300只股票组成。",
        }
        
        results = []
        for key, value in knowledge_base.items():
            if key in query or any(k in query for k in key):
                results.append(value)
        
        if not results:
            return "未找到相关知识"
        
        return f"📚 知识检索结果\n{'='*40}\n\n" + "\n".join(results[:3])
    
    @Tool
    async def _generate_report(self, symbol: str) -> str:
        """生成完整分析报告
        
        Args:
            symbol: ETF代码
        """
        # 获取数据
        df = self.fetcher.get_history(symbol, "1y")
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        quote = self.fetcher.get_etf_quote(symbol)
        advice = self.advisor.get_recommendation(symbol, df)
        pred = self.predictor.predict(df)
        
        report = (
            f"📊 {symbol} 完整分析报告\n"
            f"{'='*50}\n\n"
            f"📈 行情信息\n"
            f"  名称: {quote['name'] if quote else symbol}\n"
            f"  价格: {quote['price']:.3f}\n"
            f"  涨跌幅: {quote['change']:+.2f}%\n\n"
            f"📊 技术指标\n"
            f"  趋势: {advice['technical']['trend']}\n"
            f"  RSI: {advice['technical']['rsi']:.1f}\n"
            f"  MACD柱: {advice['technical']['macd_hist']:.4f}\n\n"
            f"🎯 投资建议\n"
            f"  建议: {advice['signal']}\n"
            f"  评分: {advice['score']:.1f}/8\n"
            f"  风险: {advice['risk_level']}\n\n"
            f"🔮 价格预测\n"
            f"  预测变化: {pred.get('predicted_change', 0):.2%}\n"
            f"  目标价: {advice.get('target_price', 0):.3f}\n\n"
            f"⚠️ 风险提示: 投资有风险，决策需谨慎。"
        )
        return report
    
    @Tool
    async def _analyze_complete(self, symbol: str) -> str:
        """多步骤完整分析（任务规划与执行）
        
        Args:
            symbol: ETF代码
        """
        # 1. 规划步骤
        steps = self.planner.plan(symbol, "full")
        
        results = []
        for step in steps:
            # 执行每个步骤
            if step["name"] == "get_quote":
                result = await self._get_quote(symbol)
            elif step["name"] == "get_history":
                result = await self._get_history(symbol, "6mo")
            elif step["name"] == "analyze_technical":
                result = await self._analyze_technical(symbol)
            elif step["name"] == "predict_price":
                result = await self._get_prediction(symbol)
            elif step["name"] == "get_recommendation":
                result = await self._get_recommendation(symbol)
            elif step["name"] == "generate_report":
                result = await self._generate_report(symbol)
            else:
                result = f"未知步骤: {step['name']}"
            
            results.append(f"**{step['desc']}**\n{result}")
        
        # 2. 生成综合报告
        report = (
            f"📊 {symbol} 综合分析报告\n"
            f"{'='*60}\n\n"
        )
        
        # 提取关键信息
        df = self.fetcher.get_history(symbol, "6mo")
        if not df.empty:
            advice = self.advisor.get_recommendation(symbol, df)
            quote = self.fetcher.get_etf_quote(symbol)
            
            report += f"📈 当前行情: {quote['price']:.3f} ({quote['change']:+.2f}%)\n"
            report += f"🎯 投资建议: {advice['signal']} (评分: {advice['score']:.1f}/8)\n"
            report += f"📊 趋势: {advice['technical']['trend']}\n"
            report += f"🛡️ 风险等级: {advice['risk_level']}\n\n"
        
        # 添加分步详情
        report += "📋 分析详情\n" + "="*40 + "\n"
        report += "\n\n".join(results)
        
        report += f"\n\n⚠️ 风险提示: 投资有风险，决策需谨慎。"
        return report
    
    
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
    
    async def chat(self, message: str, symbol: str = None, session_id: str = None) -> Dict[str, Any]:
        """处理用户消息 - 增强版，自动识别意图"""
        
        # 检查资源状态
        resource_status = self.lightweight_adapter.get_resource_report()
        if resource_status.get('cpu_percent', 0) > 80:
            return {
                "response": "⚠️ 系统负载较高，建议稍后再试",
                "success": False
            }
        
        # ✅ 智能意图识别
        message_lower = message.lower()
        
        # 检测是否请求 Top 推荐
        if any(keyword in message_lower for keyword in ['推荐', 'top', '排名', '最好', '最佳']):
            top_results = await self._get_top_recommendations()
            return {
                "response": top_results,
                "success": True,
                "session_id": session_id,
                "intent": "top_recommendations"
            }
        
        # 检测是否请求预测
        if any(keyword in message_lower for keyword in ['预测', '未来', '走势', '涨跌']):
            if symbol:
                pred_result = await self._get_ensemble_prediction(symbol)
                return {
                    "response": pred_result,
                    "success": True,
                    "session_id": session_id,
                    "intent": "ensemble_prediction"
                }
        
        # 检测是否请求分析
        if any(keyword in message_lower for keyword in ['分析', '评估', '怎么看', '建议']):
            if symbol:
                # 获取技术分析
                tech_analysis = await self._analyze_technical(symbol)
                # 获取预测
                pred_result = await self._get_ensemble_prediction(symbol)
                # 获取推荐
                rec_result = await self._get_recommendation(symbol)
                
                combined = f"{tech_analysis}\n\n{pred_result}\n\n{rec_result}"
                return {
                    "response": combined,
                    "success": True,
                    "session_id": session_id,
                    "intent": "full_analysis"
                }
        
        # 原有逻辑：使用 Agent 处理
        context = ""
        if self.memory and session_id:
            context = self.memory.get_context(session_id)
        
        full_message = message
        if context:
            full_message = f"历史上下文:\n{context}\n\n当前问题:\n{message}"
        
        try:
            result = await self.agent.run(full_message)
            response = result.data
            
            if self.memory and session_id:
                self.memory.add(session_id, message, response)
            
            return {
                "response": response,
                "success": True,
                "session_id": session_id,
                "intent": "general"
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
        return await self._get_recommendation_impl(symbol)
    async def _get_recommendation_impl(self, symbol: str) -> str:
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
    
    @Tool
    async def _get_top_recommendations(self) -> str:
        """获取每日 Top 3 买入/卖出/持有 ETF 推荐
        
        Returns:
            格式化的推荐列表
        """
        top_results = self.advisor.get_top_recommendations()
        
        report = "📊 今日 Top 3 ETF 推荐\n"
        report += "="*60 + "\n\n"
        
        # 买入推荐
        report += "🟢 Top 3 买入推荐\n"
        report += "-"*40 + "\n"
        for i, rec in enumerate(top_results['buy'], 1):
            report += f"{i}. {rec['symbol']} ({rec['name']})\n"
            report += f"   价格: {rec['current_price']:.3f} | 目标: {rec['target_price']:.3f}\n"
            report += f"   评分: {rec['score']:.1f}/8 | 风险: {rec['risk_level']}\n"
            report += f"   {rec['reasons'][0] if rec['reasons'] else ''}\n"
            if rec.get('qwen_analysis'):
                report += f"   🤖 Qwen: {rec['qwen_analysis'][:50]}...\n"
            report += "\n"
        
        # 持有推荐
        report += "🟡 Top 3 持有推荐\n"
        report += "-"*40 + "\n"
        for i, rec in enumerate(top_results['hold'], 1):
            report += f"{i}. {rec['symbol']} ({rec['name']})\n"
            report += f"   价格: {rec['current_price']:.3f} | 目标: {rec['target_price']:.3f}\n"
            report += f"   评分: {rec['score']:.1f}/8 | 风险: {rec['risk_level']}\n"
            report += "\n"
        
        # 卖出推荐
        report += "🔴 Top 3 卖出推荐\n"
        report += "-"*40 + "\n"
        for i, rec in enumerate(top_results['sell'], 1):
            report += f"{i}. {rec['symbol']} ({rec['name']})\n"
            report += f"   价格: {rec['current_price']:.3f} | 目标: {rec['target_price']:.3f}\n"
            report += f"   评分: {rec['score']:.1f}/8 | 风险: {rec['risk_level']}\n"
            report += "\n"
        
        report += "⚠️ 风险提示: 投资有风险，决策需谨慎。以上推荐仅供参考。"
        return report

    # ✅ 新增：集成预测工具
    @Tool
    async def _get_ensemble_prediction(self, symbol: str) -> str:
        return await self._get_ensemble_prediction_impl(symbol)
    async def _get_ensemble_prediction_impl(self, symbol: str) -> str:
        """获取双模型集成预测（Transformer + LSTM）
        
        Args:
            symbol: ETF 代码
        """
        df = self.fetcher.get_history(symbol)
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        pred = self.predictor.predict(df, use_ensemble=True)
        if not pred.get('success', False):
            return f"预测失败: {pred.get('error', '未知错误')}"
        
        report = f"🔮 {symbol} 双模型集成预测\n"
        report += "="*60 + "\n\n"
        
        report += f"📊 集成预测 (Transformer {pred.get('ensemble_weight', 0.6)*100:.0f}% + LSTM {(1-pred.get('ensemble_weight', 0.6))*100:.0f}%)\n"
        report += f"   预测周期: {len(pred['close'])} 个交易日\n"
        report += f"   预测变化: {pred['predicted_change']:.2%}\n"
        report += f"   置信区间: ±{pred['confidence']:.2%}\n\n"
        
        # 各模型独立结果
        if 'transformer_prediction' in pred:
            report += "📈 Transformer 模型:\n"
            report += f"   预测变化: {pred['transformer_prediction']['change']:.2%}\n"
        
        if 'lstm_prediction' in pred:
            report += "📉 LSTM 模型:\n"
            report += f"   预测变化: {pred['lstm_prediction']['change']:.2%}\n"
        
        report += "\n📅 预测价格表:\n"
        report += "日期\t\t开盘\t最高\t最低\t收盘\n"
        for i in range(min(10, len(pred['dates']))):
            report += f"{pred['dates'][i]}\t{pred['open'][i]:.3f}\t{pred['high'][i]:.3f}\t{pred['low'][i]:.3f}\t{pred['close'][i]:.3f}\n"
        
        if len(pred['dates']) > 10:
            report += f"... 还有 {len(pred['dates']) - 10} 个周期\n"
        
        return report