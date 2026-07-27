# app/agent.py
import json
import hashlib
from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .qwen_model import get_qwen_model
from .gpu_optimizer import ROCmGPUOptimizer
from .stability_manager import StabilityManager
from .feedback_learning import FeedbackLearning
from .lightweight_adapter import LightweightAdapter
from .config import (
    AGENT_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT_EXTENDED,
    DEFAULT_ETF_POOL, MEMORY_CONFIG, MODELS_DIR
)
from .data_fetcher import ETFDataFetcher
from .advisor import InvestmentAdvisor
from .predictor import ETFPricePredictor

logger = logging.getLogger(__name__)


class MemoryManager:
    """本地多轮记忆管理"""
    
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
    """多步骤任务规划器"""
    
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
    """ETF智能投顾Agent - 使用 Qwen 模型"""
    
    def __init__(self):
        self.logger = logger
        
        # 初始化 Qwen 模型
        self.qwen = get_qwen_model()
        try:
            self.qwen.load_model()
        except Exception as e:
            logger.warning(f"Qwen 模型加载失败: {e}")
        
        # 初始化 GPU 优化器
        self.gpu_optimizer = ROCmGPUOptimizer()
        
        # 初始化稳定性管理器
        self.stability_manager = StabilityManager()
        self.stability_manager.start()
        
        # 初始化反馈学习
        self.feedback_learning = FeedbackLearning()
        
        # 初始化轻量化适配器
        self.lightweight_adapter = LightweightAdapter()
        
        # 初始化服务
        self.fetcher = ETFDataFetcher()
        self.advisor = InvestmentAdvisor()
        self.predictor = ETFPricePredictor()
        
        # 加载 LoRA 适配器
        lora_path = MODELS_DIR / "lora_etf_advisor"
        if lora_path.exists():
            try:
                self.predictor.load_lora_adapter(str(lora_path))
                self.logger.info(f"✅ LoRA 适配器已加载: {lora_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ LoRA 加载失败: {e}")
        
        # 记忆管理
        self.memory = MemoryManager() if MEMORY_CONFIG.get("enabled", True) else None
        self.planner = TaskPlanner()
        
        # 系统提示
        self.system_prompt = AGENT_SYSTEM_PROMPT + "\n" + AGENT_SYSTEM_PROMPT_EXTENDED
        
        logger.info("✅ ETFAdvisorAgent 初始化完成")
    
    def _call_qwen(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """调用 Qwen 模型"""
        system = system_prompt or self.system_prompt
        
        result = self.qwen.chat(
            user_message=user_message,
            system_prompt=system,
            history=history,
            **kwargs
        )
        
        return result
    
    async def chat(self, message: str, symbol: str = None, session_id: str = None) -> Dict[str, Any]:
        """处理用户消息"""
        
        # 检查资源状态
        resource_status = self.lightweight_adapter.get_resource_report()
        if resource_status.get('cpu_percent', 0) > 80:
            return {
                "response": "⚠️ 系统负载较高，建议稍后再试",
                "success": False
            }
        
        # 智能意图识别
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
                pred_result = await self._get_prediction(symbol)
                return {
                    "response": pred_result,
                    "success": True,
                    "session_id": session_id,
                    "intent": "prediction"
                }
        
        # 检测是否请求分析
        if any(keyword in message_lower for keyword in ['分析', '评估', '怎么看', '建议']):
            if symbol:
                tech_analysis = await self._analyze_technical(symbol)
                rec_result = await self._get_recommendation(symbol)
                
                combined = f"{tech_analysis}\n\n{rec_result}"
                return {
                    "response": combined,
                    "success": True,
                    "session_id": session_id,
                    "intent": "analysis"
                }
        
        # 使用 Qwen 处理一般对话
        context = ""
        if self.memory and session_id:
            context = self.memory.get_context(session_id)
        
        # 构建完整消息
        if context:
            full_message = f"历史上下文:\n{context}\n\n用户问题:\n{message}"
        else:
            full_message = message
        
        # 如果有 symbol，添加到上下文中
        if symbol:
            full_message = f"当前分析标的: {symbol}\n{full_message}"
        
        try:
            result = self._call_qwen(
                user_message=full_message,
                max_new_tokens=512,
                temperature=0.7
            )
            
            if result.get('success', False):
                response = result['response']
                if self.memory and session_id:
                    self.memory.add(session_id, message, response)
                
                return {
                    "response": response,
                    "success": True,
                    "session_id": session_id,
                    "intent": "general"
                }
            else:
                return {
                    "response": result.get('response', '处理失败'),
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "response": f"处理失败: {str(e)}",
                "success": False
            }
    
    # ============================================================
    # 工具方法
    # ============================================================
    
    async def _get_quote(self, symbol: str) -> str:
        """获取ETF实时行情"""
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
    
    async def _get_recommendation(self, symbol: str) -> str:
        """获取投资建议"""
        df = self.fetcher.get_history(symbol)
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        advice = self.advisor.get_recommendation(symbol, df)
        
        emoji = {'buy': '🟢', 'hold': '🟡', 'sell': '🔴', 'neutral': '⚪'}
        risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
        
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
        
        report += f"\n⚠️ 风险提示: 投资有风险，决策需谨慎。"
        return report
    
    async def _get_prediction(self, symbol: str) -> str:
        """获取价格预测"""
        df = self.fetcher.get_history(symbol)
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        pred = self.predictor.predict(df, use_ensemble=True)
        if not pred.get('success', False):
            return f"预测失败: {pred.get('error', '未知错误')}"
        
        report = f"🔮 {symbol} 价格预测\n"
        report += "="*50 + "\n\n"
        report += f"预测变化: {pred.get('predicted_change', 0):.2%}\n"
        report += f"置信区间: ±{pred.get('confidence', 0):.2%}\n\n"
        
        report += "📅 预测价格表 (前10天):\n"
        dates = pred.get('dates', [])[:10]
        closes = pred.get('close', [])[:10]
        for d, c in zip(dates, closes):
            report += f"  {d}: {c:.3f}\n"
        
        return report
    
    async def _analyze_technical(self, symbol: str) -> str:
        """技术分析"""
        df = self.fetcher.get_history(symbol, "6mo")
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        advice = self.advisor.get_recommendation(symbol, df)
        tech = advice['technical']
        
        report = f"📊 {symbol} 技术分析\n"
        report += "="*40 + "\n"
        report += f"当前价格: {tech['price']:.3f}\n"
        report += f"趋势: {tech['trend']}\n"
        report += f"RSI: {tech['rsi']:.1f}\n"
        report += f"MACD: {tech['macd']:.4f}\n"
        report += f"MACD信号: {tech['macd_signal']:.4f}\n"
        report += f"MACD柱: {tech['macd_hist']:.4f}\n"
        report += f"MA5: {tech['ma5']:.3f}\n"
        report += f"MA20: {tech['ma20']:.3f}\n"
        report += f"MA60: {tech['ma60']:.3f}\n"
        report += f"布林上轨: {tech['bb_upper']:.3f}\n"
        report += f"布林中轨: {tech['bb_middle']:.3f}\n"
        report += f"布林下轨: {tech['bb_lower']:.3f}\n"
        
        return report
    
    async def _get_top_recommendations(self) -> str:
        """获取 Top 3 推荐"""
        top_results = self.advisor.get_top_recommendations()
        
        report = "📊 今日 Top 3 ETF 推荐\n"
        report += "="*60 + "\n\n"
        
        # 买入推荐
        report += "🟢 Top 3 买入推荐\n"
        report += "-"*40 + "\n"
        for i, rec in enumerate(top_results['buy'], 1):
            report += f"{i}. {rec['symbol']}\n"
            report += f"   价格: {rec['current_price']:.3f} | 目标: {rec['target_price']:.3f}\n"
            report += f"   评分: {rec['score']:.1f}/8 | 风险: {rec['risk_level']}\n"
            report += f"   {rec['reasons'][0] if rec['reasons'] else ''}\n\n"
        
        # 持有推荐
        report += "🟡 Top 3 持有推荐\n"
        report += "-"*40 + "\n"
        for i, rec in enumerate(top_results['hold'], 1):
            report += f"{i}. {rec['symbol']}\n"
            report += f"   价格: {rec['current_price']:.3f} | 目标: {rec['target_price']:.3f}\n"
            report += f"   评分: {rec['score']:.1f}/8 | 风险: {rec['risk_level']}\n\n"
        
        # 卖出推荐
        report += "🔴 Top 3 卖出推荐\n"
        report += "-"*40 + "\n"
        for i, rec in enumerate(top_results['sell'], 1):
            report += f"{i}. {rec['symbol']}\n"
            report += f"   价格: {rec['current_price']:.3f} | 目标: {rec['target_price']:.3f}\n"
            report += f"   评分: {rec['score']:.1f}/8 | 风险: {rec['risk_level']}\n\n"
        
        report += "⚠️ 风险提示: 投资有风险，决策需谨慎。"
        return report
    
    async def _compare_etfs(self, symbols: str) -> str:
        """比较多个ETF"""
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
        
        return report
    
    async def _search_knowledge(self, query: str) -> str:
        """搜索知识库"""
        knowledge_base = {
            "ETF": "ETF（交易型开放式指数基金）是一种在交易所上市交易的基金。",
            "网格交易": "网格交易是一种在设定的价格区间内分批买卖的策略。",
            "定投": "定期定额投资是一种长期投资策略。",
            "沪深300": "沪深300指数由沪深两市规模最大的300只股票组成。",
        }
        
        results = []
        for key, value in knowledge_base.items():
            if key in query or any(k in query for k in key):
                results.append(value)
        
        if not results:
            return "未找到相关知识"
        
        return f"📚 知识检索结果\n{'='*40}\n\n" + "\n".join(results[:3])
    
    async def _generate_report(self, symbol: str) -> str:
        """生成完整分析报告"""
        df = self.fetcher.get_history(symbol, "1y")
        if df.empty:
            return f"无法获取 {symbol} 的数据"
        
        quote = self.fetcher.get_etf_quote(symbol)
        advice = self.advisor.get_recommendation(symbol, df)
        pred = self.predictor.predict(df, use_ensemble=True)
        
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
    
    async def _analyze_complete(self, symbol: str) -> str:
        """完整分析"""
        steps = self.planner.plan(symbol, "full")
        results = []
        
        for step in steps:
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
        
        report = f"📊 {symbol} 综合分析报告\n{'='*60}\n\n" + "\n\n".join(results)
        report += f"\n\n⚠️ 风险提示: 投资有风险，决策需谨慎。"
        return report
    
    async def _get_history(self, symbol: str, period: str = "6mo") -> str:
        """获取历史数据"""
        df = self.fetcher.get_history(symbol, period)
        if df.empty:
            return f"无法获取 {symbol} 的历史数据"
        
        return (
            f"📈 {symbol} 历史数据 ({period})\n"
            f"交易日: {len(df)} 天\n"
            f"最新价: {df['close'].iloc[-1]:.3f}\n"
            f"最高价: {df['high'].max():.3f}\n"
            f"最低价: {df['low'].min():.3f}\n"
            f"均价: {df['close'].mean():.3f}"
        )
    
    async def _record_feedback(self, symbol: str, recommendation: str, actual_result: str, rating: int) -> str:
        """记录反馈"""
        self.feedback_learning.record_feedback(symbol, recommendation, actual_result, rating)
        return f"✅ 反馈已记录，感谢您的反馈！"