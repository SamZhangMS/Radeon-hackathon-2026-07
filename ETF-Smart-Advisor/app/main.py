# app/main.py
"""
主应用入口 - 只负责路由和 HTTP 层
所有业务逻辑委托给 agent.py
"""

from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from fastapi.encoders import jsonable_encoder
import numpy as np

from .utils import (
    to_python, 
    get_last_date, 
    ensure_date_fields, 
    get_quote_with_date,
    format_response
)
from .agent import ETFAdvisorAgent
from .config import BASE_DIR, DATA_DIR, API_KEY, API_PORT, LLM_API_CONFIG
from .llm_client import get_llm_client
from .milvus_client import get_milvus_client
from .privacy.privacy_manager import PrivacyManager

# ============================================================
# Pydantic 请求模型
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    symbol: Optional[str] = Field(None, description="ETF 代码")
    session_id: Optional[str] = Field(None, description="会话 ID")

class SymbolRequest(BaseModel):
    symbol: str = Field(..., description="ETF 代码")
    period: str = Field("1y", description="数据周期")

class CompareRequest(BaseModel):
    symbols: List[str] = Field(..., description="ETF 代码列表")

class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="ETF 代码")
    depth: str = Field("full", description="分析深度: quick, full")

class FineTuneRequest(BaseModel):
    data_path: str = Field(..., description="训练数据路径")
    output_dir: str = Field("./data/models/lora_etf_advisor", description="输出目录")

class AddKnowledgeRequest(BaseModel):
    title: str = Field(..., description="知识标题")
    content: str = Field(..., description="知识内容")
    category: str = Field("自定义", description="分类")

class SearchKnowledgeRequest(BaseModel):
    query: str = Field(..., description="搜索关键词")
    top_k: int = Field(5, description="返回数量")
    category: Optional[str] = Field(None, description="分类过滤")

class BatchAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., description="ETF 代码列表")

class LoRAFineTuneRequest(BaseModel):
    symbol: str = Field("510300", description="ETF 代码")
    period: str = Field("3y", description="数据周期")
    

# ============================================================
# 创建应用
# ============================================================

app = FastAPI(
    title="ETF-Smart Advisor",
    description="基于 AMD ROCm + vLLM + Milvus 的 ETF 智能投顾系统",
    version="2.0.0"
)

# CORS 配置
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 安全认证
# ============================================================

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """验证 API Token"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ============================================================
# 初始化服务
# ============================================================

# Agent（包含所有业务逻辑）
agent = ETFAdvisorAgent()

# LLM 客户端（用于直接调用，但通常通过 agent）
llm_client = get_llm_client()

# Milvus 客户端（用于 RAG 管理）
milvus_client = get_milvus_client()

# 隐私管理器
privacy_manager = PrivacyManager()

# ============================================================
# 静态文件
# ============================================================

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================
# 首页
# ============================================================

@app.get("/")
async def root():
    """首页（公开）"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("""
    <html>
        <head>
            <title>ETF-Smart Advisor</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 40px; text-align: center; }
                h1 { color: #1a1a2e; }
                .status { margin: 20px 0; padding: 20px; background: #f5f7fa; border-radius: 8px; }
                .links a { display: inline-block; margin: 10px 20px; color: #667eea; text-decoration: none; }
            </style>
        </head>
        <body>
            <h1>🚀 ETF-Smart Advisor</h1>
            <p>基于 AMD ROCm + vLLM + Milvus 的 ETF 智能投顾系统</p>
            <div class="status">
                <p>🤖 Agent: 已就绪</p>
                <p>💻 后端: AMD ROCm GPU</p>
                <p>📚 知识库: Milvus</p>
            </div>
            <div class="links">
                <a href="/docs">📚 API 文档</a>
                <a href="/health">❤️ 健康检查</a>
            </div>
        </body>
    </html>
    """)

# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
async def health():
    """健康检查（公开）"""
    status = agent.get_status()
    return to_python(ensure_date_fields({
        "status": "healthy",
        "service": "ETF-Smart Advisor",
        "details": status
    }))

# ============================================================
# 核心 API
# ============================================================

@app.post("/api/chat", dependencies=[Depends(verify_token)])
async def chat(request: ChatRequest):
    """聊天接口 - 支持多轮对话"""
    result = await agent.chat(
        message=request.message,
        symbol=request.symbol,
        session_id=request.session_id
    )
    return to_python(ensure_date_fields(result))

@app.post("/api/chat/session", dependencies=[Depends(verify_token)])
async def chat_with_session(request: ChatRequest):
    """带会话记忆的聊天接口"""
    session_id = request.session_id or f"session_{datetime.now().timestamp()}"
    result = await agent.chat(
        message=request.message,
        symbol=request.symbol,
        session_id=session_id
    )
    return to_python(ensure_date_fields(result))

@app.post("/api/recommend", dependencies=[Depends(verify_token)])
async def get_recommendation(request: SymbolRequest):
    """获取投资建议"""
    result = agent.get_recommendation_sync(request.symbol, request.period)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "获取建议失败"))
    
    # ✅ 使用公共函数添加日期
    data = result.get("data", {})
    df = agent.fetcher.get_history(request.symbol, request.period)
    ensure_date_fields(data, df)
    
    return to_python({
        "status": "success",
        "data": data
    })

@app.post("/api/predict", dependencies=[Depends(verify_token)])
async def get_prediction(request: SymbolRequest):
    """获取价格预测"""
    result = agent.get_prediction_sync(request.symbol, request.period)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "预测失败"))
    
    # ✅ 使用公共函数添加日期
    data = result.get("data", {})
    df = agent.fetcher.get_history(request.symbol, request.period)
    ensure_date_fields(data, df)
    
    return to_python({
        "status": "success",
        "data": data
    })

@app.post("/api/predict/ensemble", dependencies=[Depends(verify_token)])
async def get_ensemble_prediction(request: SymbolRequest):
    """获取双模型集成预测"""
    result = agent.get_prediction_sync(request.symbol, request.period)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "预测失败"))
    
    # ✅ 使用公共函数添加日期
    data = result.get("data", {})
    df = agent.fetcher.get_history(request.symbol, request.period)
    ensure_date_fields(data, df)
    
    return to_python({
        "status": "success",
        "data": data
    })

@app.get("/api/quote/{symbol}", dependencies=[Depends(verify_token)])
async def get_quote(symbol: str):
    """获取实时行情"""
    result = agent.get_quote_sync(symbol)
    if not result.get("success"):
        raise HTTPException(404, result.get("error", f"未找到 {symbol}"))
    
    # ✅ 使用公共函数添加日期
    data = result.get("data", {})
    data = get_quote_with_date(data)
    
    return to_python({
        "status": "success",
        "data": data
    })

@app.get("/api/etfs", dependencies=[Depends(verify_token)])
async def list_etfs():
    """列出可用 ETF"""
    try:
        etfs = agent.fetcher.get_etf_list()
        return to_python(ensure_date_fields({
            "status": "success",
            "etfs": etfs,
            "count": len(etfs)
        }))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/top-recommendations", dependencies=[Depends(verify_token)])
async def get_top_recommendations():
    """获取 Top 3 买入/卖出/持有 ETF 推荐"""
    result = agent.get_top_recommendations_sync()
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "获取推荐失败"))
    
    # ✅ 使用公共函数添加日期
    data = result.get("data", {})
    ensure_date_fields(data)
    
    return to_python({
        "status": "success",
        "data": data,
        "timestamp": datetime.now().isoformat()
    })

@app.post("/api/analyze", dependencies=[Depends(verify_token)])
async def analyze_complete(request: AnalyzeRequest):
    """完整分析接口（多步骤任务规划）"""
    if request.depth == "quick":
        result = agent.get_recommendation_sync(request.symbol)
        if not result.get("success"):
            raise HTTPException(400, result.get("error", "分析失败"))
        
        # ✅ 使用公共函数添加日期
        data = result.get("data", {})
        df = agent.fetcher.get_history(request.symbol, "1y")
        ensure_date_fields(data, df)
        
        return to_python({
            "status": "success",
            "symbol": request.symbol,
            "depth": "quick",
            "data": data
        })
    else:
        result = await agent._analyze_complete(request.symbol)
        return to_python(ensure_date_fields({
            "status": "success",
            "symbol": request.symbol,
            "depth": "full",
            "report": result
        }))

@app.post("/api/compare", dependencies=[Depends(verify_token)])
async def compare_etfs(request: CompareRequest):
    """对比多个 ETF"""
    result = await agent._compare_etfs_str(','.join(request.symbols[:5]))
    return to_python(ensure_date_fields({
        "status": "success",
        "result": result,
        "symbols": request.symbols[:5]
    }))

@app.post("/api/batch-analysis", dependencies=[Depends(verify_token)])
async def batch_analysis(request: BatchAnalysisRequest):
    """批量分析多个 ETF"""
    results = []
    for symbol in request.symbols[:10]:
        try:
            df = agent.fetcher.get_history(symbol, "6mo")
            if not df.empty:
                advice = agent.advisor.get_recommendation(symbol, df)
                pred = agent.predictor.predict(df, use_ensemble=True)
                results.append({
                    "symbol": symbol,
                    "recommendation": advice,
                    "prediction": pred if pred.get('success') else None,
                    "latest_date": get_last_date(df).strftime('%Y/%m/%d')
                })
        except Exception as e:
            results.append({
                "symbol": symbol,
                "error": str(e),
                "latest_date": datetime.now().strftime('%Y-%m-%d')
            })
    
    return to_python(ensure_date_fields({
        "status": "success",
        "results": results,
        "count": len(results)
    }))

# ============================================================
# 记忆管理 API
# ============================================================

@app.get("/api/memory/{session_id}", dependencies=[Depends(verify_token)])
async def get_memory(session_id: str):
    """获取会话记忆"""
    if agent.memory:
        context = agent.memory.get_context(session_id)
        return to_python(ensure_date_fields({
            "status": "success",
            "session_id": session_id,
            "context": context
        }))
    return to_python(ensure_date_fields({"status": "error", "message": "记忆功能未启用"}))

# ============================================================
# RAG / 知识库 API
# ============================================================

@app.post("/api/rag/search", dependencies=[Depends(verify_token)])
async def rag_search(request: SearchKnowledgeRequest):
    """搜索知识库"""
    try:
        result = agent.search_knowledge_sync(request.query, request.top_k)
        if not result.get("success"):
            raise HTTPException(400, result.get("error", "搜索失败"))
        return to_python(ensure_date_fields({
            "status": "success",
            "query": request.query,
            "results": result.get("results", []),
            "count": result.get("count", 0)
        }))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/rag/add", dependencies=[Depends(verify_token)])
async def rag_add(request: AddKnowledgeRequest):
    """添加知识"""
    try:
        item_id = milvus_client.insert(
            title=request.title,
            content=request.content,
            category=request.category
        )
        return to_python(ensure_date_fields({
            "status": "success",
            "id": item_id,
            "message": "知识已添加"
        }))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/rag/delete/{item_id}", dependencies=[Depends(verify_token)])
async def rag_delete(item_id: str):
    """删除知识"""
    try:
        success = milvus_client.delete(item_id)
        if success:
            return to_python(ensure_date_fields({"status": "success", "message": "知识已删除"}))
        else:
            raise HTTPException(404, "知识不存在")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/rag/stats", dependencies=[Depends(verify_token)])
async def rag_stats():
    """获取 RAG 统计信息"""
    try:
        stats = milvus_client.get_stats()
        return to_python(ensure_date_fields({"status": "success", "stats": stats}))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/rag/clear", dependencies=[Depends(verify_token)])
async def rag_clear():
    """清空知识库"""
    try:
        milvus_client.delete_all()
        return to_python(ensure_date_fields({"status": "success", "message": "知识库已清空"}))
    except Exception as e:
        raise HTTPException(500, str(e))

# ============================================================
# 隐私保护 API
# ============================================================

@app.get("/api/privacy/audit", dependencies=[Depends(verify_token)])
async def get_audit_log():
    """获取审计日志（仅管理员）"""
    report = privacy_manager.get_audit_report()
    return to_python(ensure_date_fields(report))

@app.post("/api/privacy/cleanup", dependencies=[Depends(verify_token)])
async def cleanup_data():
    """清理过期数据"""
    privacy_manager.cleanup_old_data()
    return to_python(ensure_date_fields({"status": "success", "message": "数据清理完成"}))

@app.get("/api/privacy/status", dependencies=[Depends(verify_token)])
async def get_privacy_status():
    """获取隐私保护状态"""
    return to_python(ensure_date_fields({
        "status": "success",
        "enabled": privacy_manager.enabled,
        "retention_days": privacy_manager.retention_days,
        "anonymize": privacy_manager.anonymize,
        "local_only": privacy_manager.local_only,
        "audit_entries": len(privacy_manager.audit_log)
    }))

# ============================================================
# LoRA 微调 API
# ============================================================

@app.post("/api/finetune", dependencies=[Depends(verify_token)])
async def start_finetune(request: FineTuneRequest):
    """启动 LoRA 微调"""
    try:
        df = pd.read_csv(request.data_path)
        
        result = agent.predictor.fine_tune(df, request.output_dir)
        
        agent.predictor.load_lora_adapter(request.output_dir)
        
        return to_python(ensure_date_fields({
            "status": "success",
            "message": "微调完成并已加载",
            "output_dir": request.output_dir
        }))
    except Exception as e:
        return to_python(ensure_date_fields({"status": "error", "message": str(e)}))

@app.post("/api/finetune/upload", dependencies=[Depends(verify_token)])
async def upload_finetune_data(file: UploadFile = File(...)):
    """上传微调数据文件"""
    try:
        data_path = DATA_DIR / "finetune_data.csv"
        content = await file.read()
        with open(data_path, 'wb') as f:
            f.write(content)
        return to_python(ensure_date_fields({
            "status": "success",
            "data_path": str(data_path)
        }))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/lora/finetune", dependencies=[Depends(verify_token)])
async def lora_finetune(request: LoRAFineTuneRequest):
    """使用历史基金数据微调 Qwen"""
    try:
        from .config import LORA_FINETUNE_CONFIG
        from .lora_finetuner import ETFAdvisorLoRATuner
        
        df = agent.fetcher.get_history(request.symbol, request.period)
        if df.empty:
            raise HTTPException(404, f"无法获取 {request.symbol} 的数据")
        
        train_data = []
        for i in range(len(df) - 60, len(df) - 10):
            row = df.iloc[i]
            train_data.append({
                "instruction": f"分析ETF {request.symbol} 的技术指标并预测未来走势",
                "output": f"当前价格: {row['close']:.3f}, 建议: {'买入' if row['close'] > df.iloc[i-1]['close'] else '卖出'}"
            })
        
        train_df = pd.DataFrame(train_data[:200])
        
        tuner = ETFAdvisorLoRATuner(LLM_API_CONFIG.get("model_path"))
        tuner.load_model_and_tokenizer()
        tuner.train_lora(train_df, str(LORA_FINETUNE_CONFIG["output_dir"]))
        
        return to_python(ensure_date_fields({
            "status": "success",
            "message": "LoRA 微调完成",
            "samples": len(train_data),
            "output_dir": str(LORA_FINETUNE_CONFIG["output_dir"])
        }))
    except Exception as e:
        return to_python(ensure_date_fields({"status": "error", "message": str(e)}))

# ============================================================
# 系统状态 API
# ============================================================

@app.get("/api/status", dependencies=[Depends(verify_token)])
async def get_system_status():
    """获取系统状态"""
    try:
        status = agent.get_status()
        return to_python(ensure_date_fields({
            "status": "success",
            "details": status
        }))
    except Exception as e:
        raise HTTPException(500, str(e))

# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    port = API_PORT
    print("="*60)
    print("🚀 ETF-Smart Advisor")
    print("="*60)
    print(f"📡 API 端口: {port}")
    print(f"📚 API 文档: http://localhost:{port}/docs")
    print(f"❤️  健康检查: http://localhost:{port}/health")
    print("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )