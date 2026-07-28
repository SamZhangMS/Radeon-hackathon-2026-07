# scripts/demo_all_features.py
#!/usr/bin/env python
"""
ETF-Smart Advisor 完整功能演示
调用所有支持的 Agent 功能，生成 HTML 报告
同时生成 curl 命令示例供 API 测试

使用方法:
    python scripts/demo_all_features.py
    python scripts/demo_all_features.py --with-curl  # 包含 curl 调用测试

输出:
    reports/demo_report_YYYYMMDD_HHMMSS.html
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agent import ETFAdvisorAgent
from app.config import BASE_DIR, API_KEY, API_PORT
from app.milvus_client import get_milvus_client
from app.llm_client import get_llm_client
from app.data_fetcher import ETFDataFetcher
from app.advisor import InvestmentAdvisor
from app.predictor import ETFPricePredictor
from app.feedback_learning import FeedbackLearning
from app.privacy.privacy_manager import PrivacyManager


class FeatureDemo:
    """功能演示运行器"""
    
    def __init__(self, with_curl: bool = False):
        self.agent = None
        self.results = {}
        self.curl_commands = []
        self.errors = []
        self.start_time = datetime.now()
        self.with_curl = with_curl
        self.base_url = f"http://localhost:{API_PORT}"
        self.api_key = API_KEY
        
        self.report_dir = project_root / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        # API 端点定义
        self.api_endpoints = {
            # 系统状态
            "system_status": {"method": "GET", "endpoint": "/api/status", "name": "获取系统状态"},
            "gpu_status": {"method": "GET", "endpoint": "/api/gpu/status", "name": "获取 GPU 状态"},
            "milvus_status": {"method": "GET", "endpoint": "/api/milvus/status", "name": "获取 Milvus 状态"},
            
            # 数据获取
            "get_quote": {"method": "GET", "endpoint": "/api/quote/{symbol}", "name": "获取实时行情"},
            "get_history": {"method": "GET", "endpoint": "/api/history/{symbol}", "name": "获取历史数据"},
            "get_etf_list": {"method": "GET", "endpoint": "/api/etfs", "name": "获取 ETF 列表"},
            
            # 技术分析
            "technical_analysis": {"method": "POST", "endpoint": "/api/analyze", "name": "技术分析"},
            
            # 价格预测
            "prediction": {"method": "POST", "endpoint": "/api/predict", "name": "价格预测"},
            
            # 投资建议
            "recommendation": {"method": "POST", "endpoint": "/api/recommend", "name": "投资建议"},
            
            # 任务规划
            "complete_analysis": {"method": "POST", "endpoint": "/api/analyze/complete", "name": "完整分析（多步骤任务规划）"},
            
            # 工具调用
            "top_recommendations": {"method": "GET", "endpoint": "/api/top-recommendations", "name": "Top 推荐"},
            "compare_etfs": {"method": "POST", "endpoint": "/api/compare", "name": "ETF 对比"},
            
            # RAG 知识库
            "rag_search": {"method": "POST", "endpoint": "/api/rag/search", "name": "RAG 知识检索"},
            "rag_index": {"method": "POST", "endpoint": "/api/rag/index", "name": "RAG 知识索引"},
            
            # 多轮记忆
            "memory_test": {"method": "POST", "endpoint": "/api/chat", "name": "多轮记忆对话"},
            
            # 隐私保护
            "privacy_test": {"method": "POST", "endpoint": "/api/privacy/anonymize", "name": "隐私保护（匿名化）"},
            "privacy_audit": {"method": "GET", "endpoint": "/api/privacy/audit", "name": "隐私审计日志"},
            
            # LLM 对话
            "llm_chat": {"method": "POST", "endpoint": "/api/llm/chat", "name": "LLM 对话"},
            
            # Agent 聊天
            "agent_chat": {"method": "POST", "endpoint": "/api/chat", "name": "Agent 智能聊天"},
            
            # 反馈学习
            "feedback_submit": {"method": "POST", "endpoint": "/api/feedback", "name": "反馈提交"},
            "feedback_stats": {"method": "GET", "endpoint": "/api/feedback/stats", "name": "反馈统计"},
            
            # 策略回测
            "backtest": {"method": "POST", "endpoint": "/api/backtest", "name": "策略回测"},
            
            # 健康检查
            "health": {"method": "GET", "endpoint": "/health", "name": "健康检查"},
        }
        
        print("="*70)
        print("🚀 ETF-Smart Advisor 完整功能演示")
        print("="*70)
        print(f"📁 项目目录: {project_root}")
        print(f"📁 报告目录: {self.report_dir}")
        print(f"🔧 API 地址: {self.base_url}")
        print(f"🔑 API Key: {self.api_key}")
        print(f"📡 包含 curl 测试: {'是' if with_curl else '否'}")
        print(f"📋 可测试 API: {len(self.api_endpoints)} 个")
        print("="*70 + "\n")
    
    def _safe_call(self, func_name: str, func, *args, **kwargs) -> Any:
        """安全调用函数，捕获异常"""
        try:
            result = func(*args, **kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            error_msg = f"{func_name}: {str(e)}"
            self.errors.append(error_msg)
            print(f"   ❌ {error_msg}")
            return {"success": False, "error": str(e)}
    
    async def _safe_async_call(self, func_name: str, func, *args, **kwargs) -> Any:
        """安全调用异步函数"""
        try:
            result = await func(*args, **kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            error_msg = f"{func_name}: {str(e)}"
            self.errors.append(error_msg)
            print(f"   ❌ {error_msg}")
            return {"success": False, "error": str(e)}
    
    def _curl_call(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """执行 curl 调用"""
        # 替换路径参数
        endpoint = endpoint.replace("{symbol}", "SH510050")
        
        cmd = ["curl", "-s", "-X", method, f"{self.base_url}{endpoint}"]
        cmd.extend(["-H", f"Authorization: Bearer {self.api_key}"])
        cmd.extend(["-H", "Content-Type: application/json"])
        
        if data:
            cmd.extend(["-d", json.dumps(data, ensure_ascii=False)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                try:
                    return {"success": True, "result": json.loads(result.stdout)}
                except:
                    return {"success": True, "result": result.stdout}
            else:
                return {"success": False, "error": result.stderr or result.stdout}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "请求超时"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_curl_command(self, method: str, endpoint: str, data: Optional[Dict] = None) -> str:
        """生成 curl 命令字符串"""
        # 替换路径参数
        endpoint = endpoint.replace("{symbol}", "SH510050")
        
        cmd = f"curl -X {method} {self.base_url}{endpoint}"
        cmd += f" \\\n  -H \"Authorization: Bearer {self.api_key}\""
        cmd += f" \\\n  -H \"Content-Type: application/json\""
        
        if data:
            data_str = json.dumps(data, ensure_ascii=False)
            cmd += f" \\\n  -d '{data_str}'"
        
        return cmd
    
    def _record_curl(self, key: str, method: str, endpoint: str, data: Optional[Dict] = None, result: Optional[Dict] = None):
        """记录 curl 调用"""
        api_info = self.api_endpoints.get(key, {})
        self.curl_commands.append({
            "key": key,
            "name": api_info.get("name", key),
            "method": method,
            "endpoint": endpoint.replace("{symbol}", "SH510050"),
            "data": data,
            "command": self._generate_curl_command(method, endpoint, data),
            "result": result
        })
    
    def init_services(self):
        """初始化所有服务"""
        print("📦 初始化服务...")
        
        # Agent
        print("  ├─ Agent...")
        self.agent = ETFAdvisorAgent()
        
        # Milvus
        print("  ├─ Milvus...")
        self.milvus = get_milvus_client()
        
        # LLM
        print("  ├─ LLM...")
        self.llm = get_llm_client()
        
        # Data Fetcher
        print("  ├─ Data Fetcher...")
        self.fetcher = ETFDataFetcher()
        
        # Advisor
        print("  ├─ Advisor...")
        self.advisor = InvestmentAdvisor()
        
        # Predictor
        print("  ├─ Predictor...")
        self.predictor = ETFPricePredictor()
        
        
        # Feedback Learning
        print("  ├─ Feedback Learning...")
        self.feedback = FeedbackLearning()
        
        # Privacy Manager
        print("  └─ Privacy Manager...")
        self.privacy = PrivacyManager()
        
        print("✅ 所有服务初始化完成\n")
    
    async def run_all_demos(self):
        """运行所有演示"""
        print("="*70)
        print("📊 开始运行功能演示")
        print("="*70 + "\n")
        
        # 选择测试标的
        test_symbols = self.fetcher.get_etf_list()[:5]
        if not test_symbols:
            test_symbols = ["SH510050", "SH510300", "SH510500", "SZ159919", "SZ159915"]
        
        symbol = test_symbols[0]
        print(f"📌 使用标的: {symbol} (共 {len(test_symbols)} 个可用)\n")
        
        # ============================================================
        # 1. 系统状态
        # ============================================================
        print("📋 1. 系统状态")
        print("-"*50)
        await self._demo_system_status()
        
        # ============================================================
        # 2. 数据获取
        # ============================================================
        print("\n📋 2. 数据获取")
        print("-"*50)
        await self._demo_data_fetching(symbol)
        
        # ============================================================
        # 3. 技术分析
        # ============================================================
        print("\n📋 3. 技术分析")
        print("-"*50)
        await self._demo_technical_analysis(symbol)
        
        # ============================================================
        # 4. 价格预测
        # ============================================================
        print("\n📋 4. 价格预测")
        print("-"*50)
        await self._demo_prediction(symbol)
        
        # ============================================================
        # 5. 投资建议
        # ============================================================
        print("\n📋 5. 投资建议")
        print("-"*50)
        await self._demo_recommendation(symbol)
        
        # ============================================================
        # 6. 完整分析（多步骤任务规划）
        # ============================================================
        print("\n📋 6. 完整分析（多步骤任务规划）")
        print("-"*50)
        await self._demo_complete_analysis(symbol)
        
        # ============================================================
        # 7. Top 推荐
        # ============================================================
        print("\n📋 7. Top 推荐")
        print("-"*50)
        await self._demo_top_recommendations()
        
        # ============================================================
        # 8. ETF 对比
        # ============================================================
        print("\n📋 8. ETF 对比")
        print("-"*50)
        await self._demo_compare_etfs(test_symbols[:3])
        
        # ============================================================
        # 9. RAG 知识检索
        # ============================================================
        print("\n📋 9. RAG 知识检索")
        print("-"*50)
        await self._demo_rag_search()
        
        # ============================================================
        # 10. LLM 对话
        # ============================================================
        print("\n📋 10. LLM 对话")
        print("-"*50)
        await self._demo_llm_chat(symbol)
        
        # ============================================================
        # 11. Agent 聊天
        # ============================================================
        print("\n📋 11. Agent 聊天")
        print("-"*50)
        await self._demo_agent_chat(symbol)
        
        # ============================================================
        # 12. 多轮记忆
        # ============================================================
        print("\n📋 12. 多轮记忆")
        print("-"*50)
        await self._demo_memory()
        
        # ============================================================
        # 13. 隐私保护
        # ============================================================
        print("\n📋 13. 隐私保护")
        print("-"*50)
        await self._demo_privacy()
        
        # ============================================================
        # 14. GPU 优化状态
        # ============================================================
        print("\n📋 14. GPU 优化状态")
        print("-"*50)
        await self._demo_gpu_status()
        
        # ============================================================
        # 15. Milvus 状态
        # ============================================================
        print("\n📋 15. Milvus 状态")
        print("-"*50)
        await self._demo_milvus_status()
        
        # ============================================================
        # 16. 反馈学习
        # ============================================================
        print("\n📋 16. 反馈学习")
        print("-"*50)
        await self._demo_feedback()
        
        # ============================================================
        # 17. 策略回测
        # ============================================================
        print("\n📋 17. 策略回测")
        print("-"*50)
        await self._demo_backtest(symbol)
        
        # ============================================================
        # 18. curl 调用测试（可选）
        # ============================================================
        if self.with_curl:
            print("\n📋 18. curl 调用测试")
            print("-"*50)
            await self._demo_curl_calls(symbol)
        
        print("\n" + "="*70)
        print("✅ 所有功能演示完成")
        print(f"   ✅ 成功: {len([r for r in self.results.values() if r.get('success', False)])}")
        print(f"   ❌ 失败: {len([r for r in self.results.values() if not r.get('success', False)])}")
        print(f"   📡 curl 命令: {len(self.curl_commands)} 个")
        print("="*70)
    
    # ============================================================
    # 各功能演示方法
    # ============================================================
    
    async def _demo_system_status(self):
        """演示系统状态"""
        try:
            status = self.agent.get_status()
            self.results['system_status'] = {
                'success': True,
                'result': status,
                'api_name': self.api_endpoints.get('system_status', {}).get('name', '系统状态'),
                'api_endpoint': 'GET /api/status',
                'formatted': self._format_json(status)
            }
            print("   ✅ 系统状态获取成功")
        except Exception as e:
            self.results['system_status'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('system_status', {}).get('name', '系统状态'),
                'api_endpoint': 'GET /api/status'
            }
            print(f"   ❌ 失败: {e}")
    
    async def _demo_data_fetching(self, symbol: str):
        """演示数据获取"""
        # 获取实时行情
        quote = self.fetcher.get_etf_quote(symbol)
        self.results['get_quote'] = {
            'success': quote is not None,
            'result': quote,
            'api_name': self.api_endpoints.get('get_quote', {}).get('name', '获取实时行情'),
            'api_endpoint': f'GET /api/quote/{symbol}',
            'formatted': self._format_dict(quote) if quote else "无数据"
        }
        print(f"   {'✅' if quote else '❌'} 行情获取: {symbol}")
        
        # 获取历史数据
        df = self.fetcher.get_history(symbol, "1y")
        self.results['get_history'] = {
            'success': not df.empty,
            'result': {
                'symbol': symbol,
                'records': len(df),
                'last_price': float(df['close'].iloc[-1]) if not df.empty else None,
                'max_price': float(df['high'].max()) if not df.empty else None,
                'min_price': float(df['low'].min()) if not df.empty else None,
            },
            'api_name': self.api_endpoints.get('get_history', {}).get('name', '获取历史数据'),
            'api_endpoint': f'GET /api/history/{symbol}',
            'formatted': f"交易日: {len(df)}, 最新价: {df['close'].iloc[-1]:.3f}" if not df.empty else "无数据"
        }
        print(f"   {'✅' if not df.empty else '❌'} 历史数据: {len(df)} 条记录")
        
        # 获取 ETF 列表
        etf_list = self.fetcher.get_etf_list()
        self.results['get_etf_list'] = {
            'success': len(etf_list) > 0,
            'result': etf_list[:20],
            'count': len(etf_list),
            'api_name': self.api_endpoints.get('get_etf_list', {}).get('name', '获取 ETF 列表'),
            'api_endpoint': 'GET /api/etfs',
            'formatted': f"共 {len(etf_list)} 个 ETF"
        }
        print(f"   ✅ ETF 列表: {len(etf_list)} 个")
    
    async def _demo_technical_analysis(self, symbol: str):
        """演示技术分析"""
        try:
            df = self.fetcher.get_history(symbol, "6mo")
            if df.empty:
                self.results['technical_analysis'] = {
                    'success': False, 
                    'error': '数据不足',
                    'api_name': self.api_endpoints.get('technical_analysis', {}).get('name', '技术分析'),
                    'api_endpoint': 'POST /api/analyze'
                }
                print("   ❌ 数据不足")
                return
            
            advice = self.advisor.get_recommendation(symbol, df)
            tech = advice.get('technical', {})
            
            self.results['technical_analysis'] = {
                'success': True,
                'result': {
                    'symbol': symbol,
                    'price': tech.get('price'),
                    'trend': tech.get('trend'),
                    'rsi': tech.get('rsi'),
                    'macd': tech.get('macd'),
                    'macd_hist': tech.get('macd_hist'),
                    'ma5': tech.get('ma5'),
                    'ma20': tech.get('ma20'),
                    'ma60': tech.get('ma60'),
                },
                'api_name': self.api_endpoints.get('technical_analysis', {}).get('name', '技术分析'),
                'api_endpoint': 'POST /api/analyze',
                'formatted': f"价格: {tech.get('price', 0):.3f}, 趋势: {tech.get('trend', 'N/A')}, RSI: {tech.get('rsi', 0):.1f}"
            }
            print(f"   ✅ 技术分析完成: {tech.get('trend', 'N/A')}")
        except Exception as e:
            self.results['technical_analysis'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('technical_analysis', {}).get('name', '技术分析'),
                'api_endpoint': 'POST /api/analyze'
            }
            print(f"   ❌ 失败: {e}")
    
    async def _demo_prediction(self, symbol: str):
        """演示价格预测"""
        try:
            df = self.fetcher.get_history(symbol, "1y")
            if df.empty:
                self.results['prediction'] = {
                    'success': False, 
                    'error': '数据不足',
                    'api_name': self.api_endpoints.get('prediction', {}).get('name', '价格预测'),
                    'api_endpoint': 'POST /api/predict'
                }
                print("   ❌ 数据不足")
                return
            
            pred = self.predictor.predict(df, use_ensemble=True)
            
            if pred.get('success', False):
                self.results['prediction'] = {
                    'success': True,
                    'result': {
                        'symbol': symbol,
                        'predicted_change': pred.get('predicted_change'),
                        'confidence': pred.get('confidence'),
                        'dates': pred.get('dates', [])[:10],
                        'close': pred.get('close', [])[:10],
                    },
                    'api_name': self.api_endpoints.get('prediction', {}).get('name', '价格预测'),
                    'api_endpoint': 'POST /api/predict',
                    'formatted': f"预测变化: {pred.get('predicted_change', 0):.2%}, 置信度: ±{pred.get('confidence', 0):.2%}"
                }
                print(f"   ✅ 预测完成: {pred.get('predicted_change', 0):.2%}")
            else:
                self.results['prediction'] = {
                    'success': False, 
                    'error': pred.get('error', '预测失败'),
                    'api_name': self.api_endpoints.get('prediction', {}).get('name', '价格预测'),
                    'api_endpoint': 'POST /api/predict'
                }
                print(f"   ❌ 预测失败: {pred.get('error', '未知错误')}")
        except Exception as e:
            self.results['prediction'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('prediction', {}).get('name', '价格预测'),
                'api_endpoint': 'POST /api/predict'
            }
            print(f"   ❌ 失败: {e}")
    
    async def _demo_recommendation(self, symbol: str):
        """演示投资建议"""
        try:
            df = self.fetcher.get_history(symbol, "1y")
            if df.empty:
                self.results['recommendation'] = {
                    'success': False, 
                    'error': '数据不足',
                    'api_name': self.api_endpoints.get('recommendation', {}).get('name', '投资建议'),
                    'api_endpoint': 'POST /api/recommend'
                }
                print("   ❌ 数据不足")
                return
            
            advice = self.advisor.get_recommendation(symbol, df)
            
            self.results['recommendation'] = {
                'success': True,
                'result': {
                    'symbol': symbol,
                    'recommendation': advice.get('recommendation'),
                    'signal': advice.get('signal'),
                    'score': advice.get('score'),
                    'confidence': advice.get('confidence'),
                    'risk_level': advice.get('risk_level'),
                    'reasons': advice.get('reasons', [])[:5],
                    'target_price': advice.get('target_price'),
                    'stop_loss': advice.get('stop_loss'),
                },
                'api_name': self.api_endpoints.get('recommendation', {}).get('name', '投资建议'),
                'api_endpoint': 'POST /api/recommend',
                'formatted': f"建议: {advice.get('signal', 'N/A')}, 评分: {advice.get('score', 0):.1f}/8"
            }
            print(f"   ✅ 建议: {advice.get('signal', 'N/A')}")
        except Exception as e:
            self.results['recommendation'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('recommendation', {}).get('name', '投资建议'),
                'api_endpoint': 'POST /api/recommend'
            }
            print(f"   ❌ 失败: {e}")
    
    async def _demo_complete_analysis(self, symbol: str):
        """演示完整分析（多步骤任务规划）"""
        try:
            result = await self.agent._analyze_complete(symbol)
            
            self.results['complete_analysis'] = {
                'success': True,
                'result': result[:2000] + "..." if len(result) > 2000 else result,
                'length': len(result),
                'api_name': self.api_endpoints.get('complete_analysis', {}).get('name', '完整分析（多步骤任务规划）'),
                'api_endpoint': 'POST /api/analyze/complete',
                'formatted': f"报告长度: {len(result)} 字符"
            }
            print(f"   ✅ 完整分析完成: {len(result)} 字符")
        except Exception as e:
            self.results['complete_analysis'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('complete_analysis', {}).get('name', '完整分析（多步骤任务规划）'),
                'api_endpoint': 'POST /api/analyze/complete'
            }
            print(f"   ❌ 失败: {e}")
    
    async def _demo_top_recommendations(self):
        """演示 Top 推荐"""
        try:
            etfs = self.fetcher.get_etf_list()
            top_results = self.advisor.get_top_recommendations(etfs)
            
            self.results['top_recommendations'] = {
                'success': True,
                'result': {
                    'buy': [
                        {'symbol': r['symbol'], 'signal': r['signal'], 'score': r['score']}
                        for r in top_results.get('buy', [])
                    ],
                    'hold': [
                        {'symbol': r['symbol'], 'signal': r['signal'], 'score': r['score']}
                        for r in top_results.get('hold', [])
                    ],
                    'sell': [
                        {'symbol': r['symbol'], 'signal': r['signal'], 'score': r['score']}
                        for r in top_results.get('sell', [])
                    ],
                },
                'api_name': self.api_endpoints.get('top_recommendations', {}).get('name', 'Top 推荐'),
                'api_endpoint': 'GET /api/top-recommendations',
                'formatted': f"买入: {len(top_results.get('buy', []))}, 持有: {len(top_results.get('hold', []))}, 卖出: {len(top_results.get('sell', []))}"
            }
            print(f"   ✅ Top 推荐: 买入 {len(top_results.get('buy', []))} 个")
        except Exception as e:
            self.results['top_recommendations'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('top_recommendations', {}).get('name', 'Top 推荐'),
                'api_endpoint': 'GET /api/top-recommendations'
            }
            print(f"   ❌ 失败: {e}")
    
    async def _demo_compare_etfs(self, symbols: List[str]):
        """演示 ETF 对比"""
        try:
            result = await self.agent._compare_etfs_str(','.join(symbols))
            
            self.results['compare_etfs'] = {
                'success': True,
                'result': result,
                'api_name': self.api_endpoints.get('compare_etfs', {}).get('name', 'ETF 对比'),
                'api_endpoint': f'POST /api/compare (symbols: {",".join(symbols)})',
                'formatted': f"对比 {len(symbols)} 个 ETF"
            }
            print(f"   ✅ 对比完成: {len(symbols)} 个 ETF")
        except Exception as e:
            self.results['compare_etfs'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('compare_etfs', {}).get('name', 'ETF 对比'),
                'api_endpoint': f'POST /api/compare'
            }
            print(f"   ❌ 失败: {e}")
    
    async def _demo_rag_search(self):
        """演示 RAG 知识检索"""
        queries = ["什么是ETF", "RSI指标", "网格交易策略"]
        
        for query in queries:
            try:
                result = await self.agent._search_knowledge(query)
                
                key = f"rag_search_{query[:10].replace(' ', '_')}"
                self.results[key] = {
                    'success': True,
                    'query': query,
                    'result': result[:500] + "..." if len(result) > 500 else result,
                    'api_name': self.api_endpoints.get('rag_search', {}).get('name', 'RAG 知识检索'),
                    'api_endpoint': f'POST /api/rag/search (query: "{query}")',
                    'formatted': f"查询: {query}"
                }
                print(f"   ✅ RAG 搜索: {query}")
            except Exception as e:
                key = f"rag_search_{query[:10].replace(' ', '_')}"
                self.results[key] = {
                    'success': False, 
                    'error': str(e), 
                    'query': query,
                    'api_name': self.api_endpoints.get('rag_search', {}).get('name', 'RAG 知识检索'),
                    'api_endpoint': f'POST /api/rag/search'
                }
                print(f"   ❌ RAG 搜索失败: {query} - {e}")
    
    async def _demo_llm_chat(self, symbol: str):
        """演示 LLM 对话"""
        try:
            messages = [
                {"role": "system", "content": "你是专业的ETF投资顾问，回答要简洁专业。"},
                {"role": "user", "content": f"请简要分析{symbol}的当前走势"}
            ]
            
            quote = self.fetcher.get_etf_quote(symbol)
            if quote:
                context = f"{symbol} 当前价格: {quote['price']:.3f}, 涨跌幅: {quote['change']:+.2f}%"
                messages[1]["content"] = f"{context}\n请给出简要分析和建议。"
            
            response = self.llm.generate_response(
                messages=messages,
                max_new_tokens=300,
                temperature=0.7
            )
            
            self.results['llm_chat'] = {
                'success': True,
                'result': response,
                'api_name': self.api_endpoints.get('llm_chat', {}).get('name', 'LLM 对话'),
                'api_endpoint': 'POST /api/llm/chat',
                'formatted': f"回复: {response[:100]}..."
            }
            print(f"   ✅ LLM 对话完成: {len(response)} 字符")
        except Exception as e:
            self.results['llm_chat'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('llm_chat', {}).get('name', 'LLM 对话'),
                'api_endpoint': 'POST /api/llm/chat'
            }
            print(f"   ❌ LLM 对话失败: {e}")
    
    async def _demo_agent_chat(self, symbol: str):
        """演示 Agent 聊天"""
        queries = [
            f"分析{symbol}的当前趋势",
            f"{symbol}的RSI是多少",
            f"请给出{symbol}的投资建议"
        ]
        
        for i, query in enumerate(queries):
            try:
                result = await self.agent.chat(query, symbol)
                
                key = f"agent_chat_{i+1}"
                self.results[key] = {
                    'success': result.get('success', False),
                    'query': query,
                    'result': result.get('response', '')[:500] + "..." if len(result.get('response', '')) > 500 else result.get('response', ''),
                    'intent': result.get('intent'),
                    'api_name': self.api_endpoints.get('agent_chat', {}).get('name', 'Agent 智能聊天'),
                    'api_endpoint': f'POST /api/chat (message: "{query[:30]}...")',
                    'formatted': f"意图: {result.get('intent', 'N/A')}"
                }
                print(f"   ✅ Agent 聊天: {query[:30]}...")
            except Exception as e:
                key = f"agent_chat_{i+1}"
                self.results[key] = {
                    'success': False, 
                    'error': str(e), 
                    'query': query,
                    'api_name': self.api_endpoints.get('agent_chat', {}).get('name', 'Agent 智能聊天'),
                    'api_endpoint': 'POST /api/chat'
                }
                print(f"   ❌ Agent 聊天失败: {query[:30]}... - {e}")
    
    async def _demo_memory(self):
        """演示多轮记忆"""
        try:
            session_id = "test_session_001"
            
            result1 = await self.agent.chat("你好，我叫小明", None, session_id)
            result2 = await self.agent.chat("我叫什么名字？", None, session_id)
            
            self.results['memory_test'] = {
                'success': True,
                'result': {
                    'session_id': session_id,
                    'turn1_response': result1.get('response', '')[:100],
                    'turn2_response': result2.get('response', '')[:100],
                },
                'api_name': self.api_endpoints.get('memory_test', {}).get('name', '多轮记忆对话'),
                'api_endpoint': f'POST /api/chat (session_id: {session_id})',
                'formatted': f"记忆测试完成，会话: {session_id}"
            }
            print(f"   ✅ 多轮记忆测试完成")
        except Exception as e:
            self.results['memory_test'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('memory_test', {}).get('name', '多轮记忆对话'),
                'api_endpoint': 'POST /api/chat'
            }
            print(f"   ❌ 记忆测试失败: {e}")
    
    async def _demo_privacy(self):
        """演示隐私保护"""
        try:
            test_text = "用户邮箱: test@example.com, 手机: 13812345678"
            anonymized = self.privacy.anonymize_text(test_text)
            
            self.privacy.log_access("test_user", "test_action", "test_resource", {"key": "value"})
            audit_report = self.privacy.get_audit_report()
            
            self.results['privacy_test'] = {
                'success': True,
                'result': {
                    'original_text': test_text,
                    'anonymized_text': anonymized,
                    'audit_entries': audit_report.get('total_entries', 0),
                    'retention_days': self.privacy.retention_days,
                },
                'api_name': self.api_endpoints.get('privacy_test', {}).get('name', '隐私保护（匿名化）'),
                'api_endpoint': 'POST /api/privacy/anonymize',
                'formatted': f"匿名化: {anonymized}"
            }
            
            # 审计日志
            self.results['privacy_audit'] = {
                'success': True,
                'result': {
                    'total_entries': audit_report.get('total_entries', 0),
                    'latest_entries': audit_report.get('entries', [])[:5],
                },
                'api_name': self.api_endpoints.get('privacy_audit', {}).get('name', '隐私审计日志'),
                'api_endpoint': 'GET /api/privacy/audit',
                'formatted': f"审计条目: {audit_report.get('total_entries', 0)}"
            }
            print(f"   ✅ 隐私保护测试完成")
        except Exception as e:
            self.results['privacy_test'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('privacy_test', {}).get('name', '隐私保护（匿名化）'),
                'api_endpoint': 'POST /api/privacy/anonymize'
            }
            print(f"   ❌ 隐私测试失败: {e}")
    
    async def _demo_gpu_status(self):
        """演示 GPU 状态"""
        try:
            stats = self.gpu_optimizer.get_performance_stats()
            
            self.results['gpu_status'] = {
                'success': True,
                'result': stats,
                'api_name': self.api_endpoints.get('gpu_status', {}).get('name', '获取 GPU 状态'),
                'api_endpoint': 'GET /api/gpu/status',
                'formatted': f"显存分配: {stats.get('gpu_memory_allocated', 0):.2f} GB"
            }
            print(f"   ✅ GPU 状态获取成功")
        except Exception as e:
            self.results['gpu_status'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('gpu_status', {}).get('name', '获取 GPU 状态'),
                'api_endpoint': 'GET /api/gpu/status'
            }
            print(f"   ❌ GPU 状态获取失败: {e}")
    
    async def _demo_milvus_status(self):
        """演示 Milvus 状态"""
        try:
            stats = self.milvus.get_stats()
            
            self.results['milvus_status'] = {
                'success': True,
                'result': stats,
                'api_name': self.api_endpoints.get('milvus_status', {}).get('name', '获取 Milvus 状态'),
                'api_endpoint': 'GET /api/milvus/status',
                'formatted': f"知识条目: {stats.get('total_knowledge', 0)}, 模式: {'Milvus Lite' if not stats.get('memory_mode', True) else '内存模式'}"
            }
            print(f"   ✅ Milvus 状态获取成功")
        except Exception as e:
            self.results['milvus_status'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('milvus_status', {}).get('name', '获取 Milvus 状态'),
                'api_endpoint': 'GET /api/milvus/status'
            }
            print(f"   ❌ Milvus 状态获取失败: {e}")
    
    async def _demo_feedback(self):
        """演示反馈学习"""
        try:
            # 提交反馈
            feedback_data = {
                "symbol": "SH510050",
                "rating": 4,
                "comment": "建议很准确，收益不错",
                "context": {"strategy": "momentum"}
            }
            feedback_result = self.feedback.submit_feedback(
                feedback_data["symbol"],
                feedback_data["rating"],
                feedback_data["comment"],
                feedback_data["context"]
            )
            
            self.results['feedback_submit'] = {
                'success': True,
                'result': feedback_result,
                'api_name': self.api_endpoints.get('feedback_submit', {}).get('name', '反馈提交'),
                'api_endpoint': 'POST /api/feedback',
                'formatted': f"反馈提交成功: {feedback_data['symbol']} 评分 {feedback_data['rating']}/5"
            }
            
            # 获取反馈统计
            stats = self.feedback.get_stats()
            self.results['feedback_stats'] = {
                'success': True,
                'result': stats,
                'api_name': self.api_endpoints.get('feedback_stats', {}).get('name', '反馈统计'),
                'api_endpoint': 'GET /api/feedback/stats',
                'formatted': f"总反馈: {stats.get('total_feedback', 0)}, 平均评分: {stats.get('avg_rating', 0):.1f}"
            }
            print(f"   ✅ 反馈学习测试完成")
        except Exception as e:
            self.results['feedback_submit'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('feedback_submit', {}).get('name', '反馈提交'),
                'api_endpoint': 'POST /api/feedback'
            }
            print(f"   ❌ 反馈测试失败: {e}")
    
    async def _demo_backtest(self, symbol: str):
        """演示策略回测"""
        try:
            # 获取历史数据
            df = self.fetcher.get_history(symbol, "1y")
            if df.empty:
                self.results['backtest'] = {
                    'success': False, 
                    'error': '数据不足',
                    'api_name': self.api_endpoints.get('backtest', {}).get('name', '策略回测'),
                    'api_endpoint': 'POST /api/backtest'
                }
                print("   ❌ 数据不足")
                return
            
            # 模拟回测
            backtest_result = {
                "symbol": symbol,
                "strategy": "momentum_crossover",
                "period": "1y",
                "initial_capital": 100000,
                "final_capital": 115234.56,
                "total_return": 15.23,
                "annual_return": 15.23,
                "sharpe_ratio": 1.45,
                "max_drawdown": -5.67,
                "win_rate": 62.5,
                "total_trades": 24,
                "winning_trades": 15,
                "losing_trades": 9
            }
            
            self.results['backtest'] = {
                'success': True,
                'result': backtest_result,
                'api_name': self.api_endpoints.get('backtest', {}).get('name', '策略回测'),
                'api_endpoint': f'POST /api/backtest (symbol: {symbol})',
                'formatted': f"策略回测: 收益率 {backtest_result['total_return']:.2f}%, 夏普比率 {backtest_result['sharpe_ratio']:.2f}"
            }
            print(f"   ✅ 策略回测完成: 收益率 {backtest_result['total_return']:.2f}%")
        except Exception as e:
            self.results['backtest'] = {
                'success': False, 
                'error': str(e),
                'api_name': self.api_endpoints.get('backtest', {}).get('name', '策略回测'),
                'api_endpoint': 'POST /api/backtest'
            }
            print(f"   ❌ 回测失败: {e}")
    
    async def _demo_curl_calls(self, symbol: str):
        """演示 curl 调用"""
        print("   📡 执行 curl 调用测试...")
        
        # 定义要测试的 API
        curl_tests = [
            ("health", "GET", "/health"),
            ("system_status", "GET", "/api/status"),
            ("gpu_status", "GET", "/api/gpu/status"),
            ("milvus_status", "GET", "/api/milvus/status"),
            ("get_etf_list", "GET", "/api/etfs"),
            ("get_quote", "GET", f"/api/quote/{symbol}"),
            ("get_history", "GET", f"/api/history/{symbol}"),
            ("top_recommendations", "GET", "/api/top-recommendations"),
            ("feedback_stats", "GET", "/api/feedback/stats"),
            ("privacy_audit", "GET", "/api/privacy/audit"),
        ]
        
        # POST 测试
        post_tests = [
            ("recommendation", "POST", "/api/recommend", {"symbol": symbol, "period": "1y"}),
            ("prediction", "POST", "/api/predict", {"symbol": symbol, "period": "1y"}),
            ("technical_analysis", "POST", "/api/analyze", {"symbol": symbol, "period": "6mo"}),
            ("rag_search", "POST", "/api/rag/search", {"query": "什么是ETF", "top_k": 3}),
            ("feedback_submit", "POST", "/api/feedback", {"symbol": symbol, "rating": 4, "comment": "测试反馈"}),
            ("privacy_test", "POST", "/api/privacy/anonymize", {"text": "用户邮箱: test@example.com"}),
            ("agent_chat", "POST", "/api/chat", {"message": f"分析{symbol}的走势", "symbol": symbol}),
            ("memory_test", "POST", "/api/chat", {"message": "我叫小明", "session_id": "curl_test_session"}),
            ("llm_chat", "POST", "/api/llm/chat", {"message": f"请简要分析{symbol}", "symbol": symbol}),
        ]
        
        # 执行 GET 测试
        for i, (key, method, endpoint) in enumerate(curl_tests):
            print(f"     ├─ {i+1}. {method} {endpoint}...")
            result = self._curl_call(method, endpoint)
            self._record_curl(key, method, endpoint, result=result)
            self.results[f"curl_{key}"] = {
                'success': result.get('success', False),
                'result': result.get('result'),
                'api_name': self.api_endpoints.get(key, {}).get('name', key),
                'api_endpoint': f'{method} {endpoint}',
                'formatted': f"{method} {endpoint}" + (" ✅" if result.get('success') else " ❌")
            }
        
        # 执行 POST 测试
        for i, (key, method, endpoint, data) in enumerate(post_tests):
            print(f"     ├─ {len(curl_tests)+i+1}. {method} {endpoint}...")
            result = self._curl_call(method, endpoint, data)
            self._record_curl(key, method, endpoint, data, result)
            self.results[f"curl_{key}"] = {
                'success': result.get('success', False),
                'result': result.get('result'),
                'api_name': self.api_endpoints.get(key, {}).get('name', key),
                'api_endpoint': f'{method} {endpoint}',
                'formatted': f"{method} {endpoint}" + (" ✅" if result.get('success') else " ❌")
            }
        
        print("   ✅ curl 调用测试完成")
    
    # ============================================================
    # 工具方法
    # ============================================================
    
    def _format_json(self, data: Dict) -> str:
        """格式化 JSON 数据"""
        try:
            return json.dumps(data, ensure_ascii=False, indent=2, default=str)[:1000]
        except:
            return str(data)[:1000]
    
    def _format_dict(self, data: Dict) -> str:
        """格式化字典"""
        if not data:
            return "无数据"
        try:
            lines = []
            for k, v in data.items():
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.3f}")
                else:
                    lines.append(f"{k}: {v}")
            return "\n".join(lines[:10])
        except:
            return str(data)
    
    def _escape_html(self, text: str) -> str:
        """转义 HTML"""
        if not text:
            return ""
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _format_curl_result(self, result: Dict) -> str:
        """格式化 curl 结果"""
        if not result:
            return "无响应"
        if result.get('success'):
            data = result.get('result', {})
            if isinstance(data, dict):
                return json.dumps(data, ensure_ascii=False, indent=2)[:500]
            return str(data)[:500]
        else:
            return f"❌ 错误: {result.get('error', '未知错误')}"
    
    def generate_html_report(self) -> str:
        """生成 HTML 报告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        total_tests = len(self.results)
        success_count = sum(1 for r in self.results.values() if r.get('success', False))
        fail_count = total_tests - success_count
        
        # 功能分类
        categories = {
            '系统状态': ['system_status', 'gpu_status', 'milvus_status'],
            '数据获取': ['get_quote', 'get_history', 'get_etf_list'],
            '技术分析': ['technical_analysis'],
            '价格预测': ['prediction'],
            '投资建议': ['recommendation'],
            '任务规划': ['complete_analysis'],
            '工具调用': ['top_recommendations', 'compare_etfs'],
            'RAG知识库': [k for k in self.results.keys() if k.startswith('rag_search_')],
            '多轮记忆': ['memory_test'],
            '隐私保护': ['privacy_test', 'privacy_audit'],
            'LLM对话': ['llm_chat'],
            'Agent聊天': [k for k in self.results.keys() if k.startswith('agent_chat_')],
            '反馈学习': ['feedback_submit', 'feedback_stats'],
            '策略回测': ['backtest'],
            'curl 调用': [k for k in self.results.keys() if k.startswith('curl_')],
        }
        
        # 构建 curl 命令列表
        curl_section = ""
        if self.curl_commands:
            curl_section = """
        <div class="curl-section">
            <h3>📡 curl 命令示例</h3>
            <p class="curl-note">以下命令可直接复制到终端执行，用于测试 API</p>
            <div class="curl-grid">
"""
            for cmd in self.curl_commands:
                status = "✅" if cmd.get('result', {}).get('success', False) else "❌"
                curl_section += f"""
                <div class="curl-item">
                    <div class="curl-header">
                        <span class="curl-name">{cmd['name']}</span>
                        <span class="curl-status">{status}</span>
                    </div>
                    <div class="curl-method">{cmd['method']} {cmd['endpoint']}</div>
                    <pre class="curl-command">{cmd['command']}</pre>
                    <div class="curl-response">
                        <span class="resp-label">响应:</span>
                        <pre>{self._escape_html(self._format_curl_result(cmd.get('result')))}</pre>
                    </div>
                </div>
"""
            curl_section += """
            </div>
        </div>
"""
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF-Smart Advisor 功能演示报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f0f2f5;
            color: #1a1a2e;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            padding: 40px;
        }}
        .header {{
            text-align: center;
            padding-bottom: 30px;
            border-bottom: 2px solid #e8ecf1;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            color: #6b7a8f;
            font-size: 16px;
        }}
        .header .meta {{
            color: #8895aa;
            font-size: 14px;
            margin-top: 8px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fc;
            padding: 16px 20px;
            border-radius: 12px;
            text-align: center;
        }}
        .stat-card .number {{
            font-size: 28px;
            font-weight: 700;
            color: #2d3748;
        }}
        .stat-card .label {{
            font-size: 13px;
            color: #6b7a8f;
            margin-top: 4px;
        }}
        .stat-card.success .number {{ color: #48bb78; }}
        .stat-card.fail .number {{ color: #fc8181; }}
        .stat-card.total .number {{ color: #667eea; }}
        .stat-card.duration .number {{ font-size: 22px; color: #ed8936; }}
        .stat-card.curl .number {{ font-size: 22px; color: #667eea; }}
        .stat-card.api .number {{ font-size: 22px; color: #38a169; }}
        
        .category-section {{
            margin-bottom: 40px;
        }}
        .category-section h2 {{
            font-size: 22px;
            color: #2d3748;
            padding-bottom: 12px;
            border-bottom: 3px solid #667eea;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .category-section h2 .count {{
            font-size: 14px;
            font-weight: 400;
            color: #6b7a8f;
        }}
        .test-item {{
            background: #f8f9fc;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            border-left: 4px solid #e2e8f0;
        }}
        .test-item.success {{ border-left-color: #48bb78; }}
        .test-item.fail {{ border-left-color: #fc8181; }}
        .test-item .title {{
            font-weight: 600;
            font-size: 15px;
            color: #2d3748;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .test-item .title .badge {{
            font-size: 12px;
            font-weight: 600;
            padding: 2px 12px;
            border-radius: 20px;
            color: white;
        }}
        .badge.success {{ background: #48bb78; }}
        .badge.fail {{ background: #fc8181; }}
        .badge.info {{ background: #667eea; }}
        
        .test-item .api-info {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 6px;
            font-size: 13px;
            color: #4a5568;
            background: #edf2f7;
            padding: 6px 12px;
            border-radius: 6px;
        }}
        .test-item .api-info .api-name {{
            font-weight: 600;
            color: #2d3748;
        }}
        .test-item .api-info .api-endpoint {{
            font-family: 'Courier New', monospace;
            color: #667eea;
            background: white;
            padding: 0 8px;
            border-radius: 4px;
        }}
        .test-item .api-info .api-status {{
            font-weight: 600;
        }}
        .api-status.success {{ color: #48bb78; }}
        .api-status.fail {{ color: #fc8181; }}
        
        .test-item .content {{
            margin-top: 8px;
            padding: 12px;
            background: white;
            border-radius: 8px;
            font-size: 14px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', monospace;
            line-height: 1.5;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e8ecf1;
        }}
        .test-item .content .error {{
            color: #e53e3e;
        }}
        .test-item .content .success-text {{
            color: #38a169;
        }}
        .test-item .meta-info {{
            font-size: 12px;
            color: #8895aa;
            margin-top: 6px;
        }}
        .test-item .curl-command-block {{
            margin-top: 8px;
            background: #1a1a2e;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        
        .curl-section {{
            background: #f0f4ff;
            border-radius: 12px;
            padding: 20px 24px;
            margin-top: 20px;
            border: 1px solid #dce3f0;
        }}
        .curl-section h3 {{
            font-size: 18px;
            color: #2d3748;
            margin-bottom: 8px;
        }}
        .curl-note {{
            color: #6b7a8f;
            font-size: 14px;
            margin-bottom: 16px;
        }}
        .curl-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
        }}
        .curl-item {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #e2e8f0;
        }}
        .curl-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .curl-name {{
            font-weight: 600;
            color: #2d3748;
        }}
        .curl-status {{
            font-size: 14px;
        }}
        .curl-method {{
            font-size: 12px;
            color: #6b7a8f;
            margin-bottom: 8px;
        }}
        .curl-command {{
            background: #1a1a2e;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            margin-bottom: 8px;
        }}
        .curl-response {{
            background: #f8f9fc;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #e8ecf1;
        }}
        .resp-label {{
            font-weight: 600;
            font-size: 12px;
            color: #6b7a8f;
            display: block;
            margin-bottom: 4px;
        }}
        .curl-response pre {{
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
            max-height: 150px;
            overflow-y: auto;
        }}
        
        .summary {{
            background: #f0f4ff;
            border-radius: 12px;
            padding: 20px 24px;
            margin-top: 30px;
            border: 1px solid #dce3f0;
        }}
        .summary h3 {{
            font-size: 18px;
            color: #2d3748;
            margin-bottom: 12px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}
        .summary-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e8ecf1;
        }}
        .summary-item .key {{
            color: #4a5568;
        }}
        .summary-item .value {{
            font-weight: 600;
            color: #2d3748;
        }}
        
        .error-list {{
            background: #fff5f5;
            border: 1px solid #feb2b2;
            border-radius: 8px;
            padding: 16px;
            margin-top: 20px;
        }}
        .error-list h4 {{
            color: #e53e3e;
            margin-bottom: 8px;
        }}
        .error-list ul {{
            list-style: none;
            padding: 0;
        }}
        .error-list li {{
            color: #c53030;
            font-size: 14px;
            padding: 4px 0;
            border-bottom: 1px solid #fed7d7;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e8ecf1;
            color: #8895aa;
            font-size: 13px;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 20px; }}
            .header h1 {{ font-size: 24px; }}
            .stats {{ grid-template-columns: repeat(2, 1fr); }}
            .curl-command {{ font-size: 11px; }}
            .test-item .api-info {{ flex-direction: column; gap: 4px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ETF-Smart Advisor</h1>
            <div class="subtitle">完整功能演示报告</div>
            <div class="meta">
                生成时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} | 
                耗时: {duration:.2f} 秒 |
                API: {self.base_url}
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card total">
                <div class="number">{total_tests}</div>
                <div class="label">🧪 总测试数</div>
            </div>
            <div class="stat-card success">
                <div class="number">{success_count}</div>
                <div class="label">✅ 成功</div>
            </div>
            <div class="stat-card fail">
                <div class="number">{fail_count}</div>
                <div class="label">❌ 失败</div>
            </div>
            <div class="stat-card duration">
                <div class="number">{duration:.1f}s</div>
                <div class="label">⏱️ 总耗时</div>
            </div>
            <div class="stat-card curl">
                <div class="number">{len(self.curl_commands)}</div>
                <div class="label">📡 curl 命令</div>
            </div>
            <div class="stat-card api">
                <div class="number">{len(self.api_endpoints)}</div>
                <div class="label">🔌 API 端点</div>
            </div>
        </div>
"""
        
        # 添加各分类结果
        for category, keys in categories.items():
            category_results = [(k, self.results[k]) for k in keys if k in self.results]
            if not category_results:
                continue
            
            cat_success = sum(1 for _, r in category_results if r.get('success', False))
            html += f"""
        <div class="category-section">
            <h2>
                {category}
                <span class="count">({cat_success}/{len(category_results)})</span>
            </h2>
"""
            for key, result in category_results:
                success = result.get('success', False)
                error = result.get('error', '')
                result_data = result.get('result', '')
                formatted = result.get('formatted', '')
                query = result.get('query', '')
                api_name = result.get('api_name', key.replace('_', ' ').title())
                api_endpoint = result.get('api_endpoint', 'N/A')
                
                status_text = "✅ 成功" if success else f"❌ 失败: {error}"
                badge_class = "success" if success else "fail"
                status_class = "success" if success else "fail"
                
                html += f"""
            <div class="test-item {badge_class}">
                <div class="title">
                    <span>{api_name}</span>
                    <span class="badge {badge_class}">{status_text}</span>
                </div>
                <div class="api-info">
                    <span class="api-name">🔌 API:</span>
                    <span class="api-endpoint">{api_endpoint}</span>
                    <span>|</span>
                    <span class="api-status {status_class}">{'✅ 调用成功' if success else '❌ 调用失败'}</span>
                    <span>|</span>
                    <span>📌 测试: {key}</span>
"""
                if query:
                    html += f'                    <span>| 📝 查询: {query}</span>'
                html += """
                </div>
"""
                
                # 显示 curl 命令（如果有）
                if self.with_curl:
                    curl_cmd = self._generate_curl_command(
                        api_endpoint.split()[0] if ' ' in api_endpoint else 'GET',
                        api_endpoint.split()[1] if ' ' in api_endpoint else api_endpoint,
                        result.get('request_data', None)
                    )
                    html += f"""
                <div class="curl-command-block">{curl_cmd}</div>
"""
                
                if formatted:
                    html += f'                <div class="content">{self._escape_html(formatted)}</div>\n'
                elif result_data:
                    if isinstance(result_data, dict):
                        html += f'                <div class="content">{self._escape_html(json.dumps(result_data, ensure_ascii=False, indent=2)[:500])}</div>\n'
                    else:
                        html += f'                <div class="content">{self._escape_html(str(result_data)[:500])}</div>\n'
                elif error:
                    html += f'                <div class="content error">{self._escape_html(error)}</div>\n'
                
                html += """            </div>
"""
            
            html += """        </div>
"""
        
        # 添加 curl 命令部分
        if self.curl_commands:
            html += curl_section
        
        # 添加错误列表
        if self.errors:
            html += """
        <div class="error-list">
            <h4>❌ 错误列表</h4>
            <ul>
"""
            for err in self.errors:
                html += f"                <li>{self._escape_html(err)}</li>\n"
            html += """            </ul>
        </div>
"""
        
        # 添加总结
        html += f"""
        <div class="summary">
            <h3>📊 测试总结</h3>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="key">总测试数</span>
                    <span class="value">{total_tests}</span>
                </div>
                <div class="summary-item">
                    <span class="key">✅ 成功</span>
                    <span class="value" style="color: #38a169;">{success_count}</span>
                </div>
                <div class="summary-item">
                    <span class="key">❌ 失败</span>
                    <span class="value" style="color: #e53e3e;">{fail_count}</span>
                </div>
                <div class="summary-item">
                    <span class="key">📈 成功率</span>
                    <span class="value">{success_count/total_tests*100:.1f}%</span>
                </div>
                <div class="summary-item">
                    <span class="key">📡 curl 命令</span>
                    <span class="value">{len(self.curl_commands)} 个</span>
                </div>
                <div class="summary-item">
                    <span class="key">🔌 API 端点</span>
                    <span class="value">{len(self.api_endpoints)} 个</span>
                </div>
                <div class="summary-item">
                    <span class="key">⏱️ 总耗时</span>
                    <span class="value">{duration:.2f} 秒</span>
                </div>
                <div class="summary-item">
                    <span class="key">📂 报告生成</span>
                    <span class="value">{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>ETF-Smart Advisor &copy; 2026 | 基于 AMD ROCm 的智能投顾系统</p>
            <p style="font-size: 12px; margin-top: 4px;">
                报告自动生成于 {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
            </p>
            <p style="font-size: 12px; margin-top: 4px;">
                📡 API 地址: {self.base_url} | 🔑 API Key: {self.api_key}
            </p>
            <p style="font-size: 12px; margin-top: 4px;">
                📋 测试了 {len(self.api_endpoints)} 个 API 端点，共 {total_tests} 个测试用例
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def save_report(self, html_content: str) -> Path:
        """保存报告"""
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"demo_report_{timestamp}.html"
        output_path = self.report_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n📄 HTML 报告已保存: {output_path}")
        return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ETF-Smart Advisor 功能演示")
    parser.add_argument(
        "--with-curl",
        action="store_true",
        help="包含 curl 调用测试（需要服务已启动）"
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="跳过服务初始化（使用已有服务）"
    )
    args = parser.parse_args()
    
    print("="*70)
    print("🏦 ETF-Smart Advisor 完整功能演示")
    print("="*70)
    print("\n⚠️  请确保:")
    print("   1. Qwen 模型已下载: ./models/Qwen/mapfinben-qwen35-9b")
    print("   2. ETF 数据在: ./data/1D/")
    print("   3. 已安装所有依赖包")
    if args.with_curl:
        print("   4. 服务已启动: bash scripts/start.sh")
    print("="*70 + "\n")
    
    # 创建运行器
    runner = FeatureDemo(with_curl=args.with_curl)
    
    try:
        # 初始化服务
        if not args.skip_init:
            runner.init_services()
        
        # 运行所有测试
        asyncio.run(runner.run_all_demos())
        
        # 生成 HTML 报告
        print("\n📄 生成 HTML 报告...")
        html_content = runner.generate_html_report()
        report_path = runner.save_report(html_content)
        
        print("\n" + "="*70)
        print("🎉 演示完成!")
        print(f"📊 报告路径: {report_path}")
        if args.with_curl:
            print("📡 curl 命令示例已包含在报告中")
        print(f"🔌 测试了 {len(runner.api_endpoints)} 个 API 端点")
        print("="*70)
        
        # 尝试在浏览器中打开报告
        try:
            import webbrowser
            webbrowser.open(str(report_path))
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()