import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FeedbackLearning:
    """用户反馈学习与自定义迭代"""
    
    def __init__(self, feedback_path: str = "data/feedback.json"):
        self.feedback_path = feedback_path
        self.feedback_data = self._load_feedback()
        self.learning_rate = 0.1
        self.confidence_threshold = 0.6
        
        # 策略权重
        self.strategy_weights = defaultdict(lambda: 1.0)
        self._load_strategy_weights()
        
        logger.info(f"📚 反馈学习系统初始化完成")
        logger.info(f"   📊 已收集反馈: {len(self.feedback_data)} 条")
    
    def _load_feedback(self) -> List[Dict]:
        """加载反馈数据"""
        try:
            with open(self.feedback_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_feedback(self):
        """保存反馈数据"""
        try:
            # 只保留最近的 10000 条
            if len(self.feedback_data) > 10000:
                self.feedback_data = self.feedback_data[-10000:]
            
            with open(self.feedback_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存反馈失败: {e}")
    
    def _load_strategy_weights(self):
        """加载策略权重"""
        try:
            weight_path = self.feedback_path.replace('feedback', 'weights')
            with open(weight_path, 'r') as f:
                weights = json.load(f)
                self.strategy_weights.update(weights)
        except:
            pass
    
    def record_feedback(self, symbol: str, recommendation: str, 
                       actual_result: str, user_rating: int,
                       user_comment: str = ""):
        """记录用户反馈"""
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "recommendation": recommendation,
            "actual_result": actual_result,
            "user_rating": user_rating,
            "user_comment": user_comment,
            "accuracy": self._calculate_accuracy(recommendation, actual_result)
        }
        self.feedback_data.append(feedback)
        self._save_feedback()
        
        # 更新策略权重
        self._update_weights(feedback)
        
        logger.info(f"✅ 反馈已记录: {symbol} - {recommendation} - 评分: {user_rating}")
    
    def _calculate_accuracy(self, recommendation: str, actual_result: str) -> float:
        """计算准确性"""
        accuracy_map = {
            ("buy", "up"): 1.0,
            ("sell", "down"): 1.0,
            ("hold", "sideways"): 0.8,
            ("buy", "up_strong"): 1.0,
            ("sell", "down_strong"): 1.0,
        }
        
        # 简化匹配
        for (rec, actual), acc in accuracy_map.items():
            if rec in recommendation and actual in actual_result:
                return acc
        
        return 0.3  # 默认准确率
    
    def _update_weights(self, feedback: Dict):
        """更新策略权重（强化学习）"""
        symbol = feedback["symbol"]
        recommendation = feedback["recommendation"]
        accuracy = feedback["accuracy"]
        rating = feedback["user_rating"]
        
        # 计算调整量
        adjustment = self.learning_rate * (accuracy * rating / 5 - 0.5)
        
        # 更新权重
        key = f"{symbol}_{recommendation}"
        self.strategy_weights[key] += adjustment
        self.strategy_weights[key] = max(0.1, min(2.0, self.strategy_weights[key]))
        
        # 保存权重
        self._save_weights()
    
    def _save_weights(self):
        """保存策略权重"""
        try:
            weight_path = self.feedback_path.replace('feedback', 'weights')
            with open(weight_path, 'w') as f:
                json.dump(dict(self.strategy_weights), f)
        except:
            pass
    
    def get_improved_recommendation(self, symbol: str, base_recommendation: Dict) -> Dict:
        """基于反馈改进推荐"""
        # 获取该股票的历史反馈
        symbol_feedback = [f for f in self.feedback_data if f["symbol"] == symbol]
        
        if not symbol_feedback:
            return base_recommendation
        
        # 计算准确率
        accurate = sum(1 for f in symbol_feedback if f["accuracy"] > 0.5)
        accuracy_rate = accurate / len(symbol_feedback) if symbol_feedback else 0
        
        # 获取权重
        key = f"{symbol}_{base_recommendation.get('recommendation', 'unknown')}"
        weight = self.strategy_weights.get(key, 1.0)
        
        # 调整建议
        if accuracy_rate > 0.7 and weight > 1.2:
            # 高准确率，增强建议
            base_recommendation["confidence"] = min(1.0, base_recommendation.get("confidence", 0.5) * 1.2)
            base_recommendation["enhanced"] = True
            
        elif accuracy_rate < 0.4 and weight < 0.8:
            # 低准确率，减弱建议
            base_recommendation["confidence"] = max(0.1, base_recommendation.get("confidence", 0.5) * 0.8)
            base_recommendation["warning"] = "历史准确率较低，请谨慎参考"
        
        # 添加学习信息
        base_recommendation["feedback_learning"] = {
            "total_feedback": len(symbol_feedback),
            "accuracy_rate": accuracy_rate,
            "strategy_weight": weight,
        }
        
        return base_recommendation
    
    def get_accuracy_report(self) -> Dict:
        """生成准确率报告"""
        if not self.feedback_data:
            return {"status": "no_data", "message": "暂无反馈数据"}
        
        total = len(self.feedback_data)
        accurate = sum(1 for f in self.feedback_data if f["accuracy"] > 0.5)
        avg_rating = sum(f["user_rating"] for f in self.feedback_data) / total
        
        # 按股票统计
        symbol_stats = defaultdict(lambda: {"total": 0, "accurate": 0, "ratings": []})
        for f in self.feedback_data:
            symbol_stats[f["symbol"]]["total"] += 1
            if f["accuracy"] > 0.5:
                symbol_stats[f["symbol"]]["accurate"] += 1
            symbol_stats[f["symbol"]]["ratings"].append(f["user_rating"])
        
        symbol_accuracy = {
            symbol: {
                "total": stats["total"],
                "accuracy": stats["accurate"] / stats["total"],
                "avg_rating": sum(stats["ratings"]) / len(stats["ratings"])
            }
            for symbol, stats in symbol_stats.items()
        }
        
        return {
            "total_feedback": total,
            "overall_accuracy": accurate / total if total > 0 else 0,
            "avg_rating": avg_rating,
            "symbol_accuracy": symbol_accuracy,
            "top_symbols": sorted(
                symbol_accuracy.items(),
                key=lambda x: x[1]["accuracy"],
                reverse=True
            )[:10]
        }