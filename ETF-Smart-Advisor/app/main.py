from fastapi import FastAPI, HTTPException,Security, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os,Request
import json
import hashlib
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from .agent import ETFAdvisorAgent
from .config import BASE_DIR,DATA_DIR  

app = FastAPI(
    title="ETF-Smart Advisor",
    description="基于AMD ROCm的ETF智能投顾Agent",
    version="1.0.0"
)
API_KEY = os.environ.get("API_KEY", "abc-123")

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

# 初始化Agent
agent = ETFAdvisorAgent()

# 请求模型
class ChatRequest(BaseModel):
    message: str
    symbol: Optional[str] = None

class PredictRequest(BaseModel):
    symbol: str

class FineTuneRequest(BaseModel):
    data_path: str
    output_dir: str = "./lora_etf_advisor"
    
# 静态文件
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    """首页（公开）"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("""
    <html>
        <head><title>ETF-Smart Advisor</title></head>
        <body>
            <h1>ETF-Smart Advisor</h1>
            <p>基于AMD ROCm的ETF智能投顾Agent</p>
            <p>API文档: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """)

@app.post("/api/chat", dependencies=[Depends(verify_token)])
async def chat(request: ChatRequest):
    """聊天接口"""
    result = await agent.chat(request.message, request.symbol)
    return result

@app.post("/api/recommend", dependencies=[Depends(verify_token)])
async def get_recommendation(request: PredictRequest):
    """获取投资建议"""
    df = agent.fetcher.get_history(request.symbol)
    if df.empty:
        raise HTTPException(404, f"无法获取 {request.symbol} 的数据")
    
    advice = agent.advisor.get_recommendation(request.symbol, df)
    return advice

@app.post("/api/predict", dependencies=[Depends(verify_token)])
async def get_prediction(request: PredictRequest):
    """获取价格预测"""
    df = agent.fetcher.get_history(request.symbol)
    if df.empty:
        raise HTTPException(404, f"无法获取 {request.symbol} 的数据")
    
    pred = agent.predictor.predict(df)
    if not pred.get('success', False):
        raise HTTPException(400, pred.get('error', '预测失败'))
    
    return pred

@app.get("/api/quote/{symbol}", dependencies=[Depends(verify_token)])
async def get_quote(symbol: str):
    """获取实时行情"""
    quote = agent.fetcher.get_etf_quote(symbol)
    if not quote:
        raise HTTPException(404, f"未找到 {symbol}")
    return quote

@app.get("/api/etfs", dependencies=[Depends(verify_token)])
async def list_etfs():
    """列出默认ETF"""
    return {"etfs": agent.fetcher.get_etf_list()}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "service": "ETF-Smart Advisor"}

class AnalyzeRequest(BaseModel):
    symbol: str
    depth: str = "full"  # quick, full

class CompareRequest(BaseModel):
    symbols: List[str]

@app.post("/api/analyze", dependencies=[Depends(verify_token)])
async def analyze_complete(request: AnalyzeRequest):
    """完整分析接口（多步骤任务规划）"""
    result = await agent._analyze_complete(request.symbol)
    return {"response": result, "symbol": request.symbol}

@app.post("/api/chat/session", dependencies=[Depends(verify_token)])
async def chat_with_session(request: ChatRequest):
    """带会话记忆的聊天接口"""
    session_id = request.symbol or f"session_{datetime.now().timestamp()}"
    result = await agent.chat(request.message, request.symbol, session_id)
    return result

@app.get("/api/memory/{session_id}", dependencies=[Depends(verify_token)])
async def get_memory(session_id: str):
    """获取会话记忆"""
    if agent.memory:
        context = agent.memory.get_context(session_id)
        return {"session_id": session_id, "context": context}
    return {"error": "记忆功能未启用"}

@app.post("/api/knowledge/search", dependencies=[Depends(verify_token)])
async def search_knowledge(query: str):
    """知识库检索"""
    result = await agent._search_knowledge(query)
    return {"query": query, "result": result}



@app.post("/api/finetune", dependencies=[Depends(verify_token)])
async def start_finetune(request: FineTuneRequest):
    """启动 LoRA 微调"""
    try:
        # 加载数据
        df = pd.read_csv(request.data_path)
        
        # 执行微调
        result = agent.predictor.fine_tune(df, request.output_dir)
        
        # 重新加载 LoRA 适配器
        agent.predictor.load_lora_adapter(request.output_dir)
        
        return {
            "status": "success",
            "message": "微调完成并已加载",
            "output_dir": request.output_dir
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/finetune/upload", dependencies=[Depends(verify_token)])
async def upload_finetune_data(file: UploadFile = File(...)):
    """上传微调数据文件"""
    data_path = DATA_DIR / "finetune_data.csv"
    content = await file.read()
    with open(data_path, 'wb') as f:
        f.write(content)
    return {"status": "success", "data_path": str(data_path)}

# main.py - 在文件末尾添加

# ✅ 新增：获取 Top 推荐 API
@app.get("/api/top-recommendations", dependencies=[Depends(verify_token)])
async def get_top_recommendations():
    """获取每日 Top 3 买入/卖出/持有 ETF 推荐"""
    top_results = agent.advisor.get_top_recommendations()
    return {
        "status": "success",
        "data": top_results,
        "timestamp": datetime.now().isoformat()
    }

# ✅ 新增：双模型集成预测 API
@app.post("/api/predict/ensemble", dependencies=[Depends(verify_token)])
async def get_ensemble_prediction(request: PredictRequest):
    """获取双模型集成预测"""
    df = agent.fetcher.get_history(request.symbol)
    if df.empty:
        raise HTTPException(404, f"无法获取 {request.symbol} 的数据")
    
    pred = agent.predictor.predict(df, use_ensemble=True)
    if not pred.get('success', False):
        raise HTTPException(400, pred.get('error', '预测失败'))
    
    return pred

# ✅ 新增：批量分析 API
@app.post("/api/batch-analysis", dependencies=[Depends(verify_token)])
async def batch_analysis(symbols: List[str]):
    """批量分析多个 ETF"""
    results = []
    for symbol in symbols[:10]:  # 限制最多 10 个
        df = agent.fetcher.get_history(symbol, "6mo")
        if not df.empty:
            advice = agent.advisor.get_recommendation(symbol, df)
            pred = agent.predictor.predict(df, use_ensemble=True)
            results.append({
                "symbol": symbol,
                "recommendation": advice,
                "prediction": pred if pred.get('success') else None
            })
    return {"status": "success", "results": results}

@app.post("/api/dify/predict", dependencies=[Depends(verify_token)])
async def dify_predict(request: PredictRequest):
    """通过 Dify 统一管理所有预测模型"""
    df = agent.fetcher.get_history(request.symbol)
    if df.empty:
        raise HTTPException(404, f"无法获取 {request.symbol} 的数据")
    
    results = await agent._call_all_dify_agents(request.symbol, df)
    return {
        "symbol": request.symbol,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

# ✅ LoRA微调端点
@app.post("/api/lora/finetune", dependencies=[Depends(verify_token)])
async def lora_finetune(request: Request):
    """使用历史基金数据微调Qwen"""
    try:
        from .config import LORA_FINETUNE_CONFIG
        data = await request.json()
        symbol = data.get('symbol', '510300')
        
        df = agent.fetcher.get_history(symbol, "3y")
        if df.empty:
            raise HTTPException(404, f"无法获取 {symbol} 的数据")
        
        train_data = []
        for i in range(len(df) - 60, len(df) - 10):
            row = df.iloc[i]
            train_data.append({
                "instruction": f"分析ETF {symbol} 的技术指标并预测未来走势",
                "output": f"当前价格: {row['close']:.3f}, 建议: {'买入' if row['close'] > df.iloc[i-1]['close'] else '卖出'}"
            })
        
        import pandas as pd
        train_df = pd.DataFrame(train_data[:200])
        
        from .lora_finetuner import ETFAdvisorLoRATuner
        tuner = ETFAdvisorLoRATuner(LLM_CONFIG["model_name"])
        tuner.load_model_and_tokenizer()
        tuner.train_lora(train_df, str(LORA_FINETUNE_CONFIG["output_dir"]))
        
        return {
            "status": "success",
            "message": "LoRA微调完成",
            "samples": len(train_data),
            "output_dir": str(LORA_FINETUNE_CONFIG["output_dir"])
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
    