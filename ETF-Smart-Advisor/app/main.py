from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from .agent import ETFAdvisorAgent
from .config import BASE_DIR

app = FastAPI(
    title="ETF-Smart Advisor",
    description="基于AMD ROCm的ETF智能投顾Agent",
    version="2.0.0"
)

# 初始化Agent
agent = ETFAdvisorAgent()

# 请求模型
class ChatRequest(BaseModel):
    message: str
    symbol: Optional[str] = None

class PredictRequest(BaseModel):
    symbol: str

# 静态文件
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    """首页"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("""
    <html>
        <head><title>ETF-Smart Advisor</title></head>
        <body>
            <h1>📊 ETF-Smart Advisor</h1>
            <p>基于AMD ROCm的ETF智能投顾Agent</p>
            <p>API文档: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """聊天接口"""
    result = await agent.chat(request.message, request.symbol)
    return result

@app.post("/api/recommend")
async def get_recommendation(request: PredictRequest):
    """获取投资建议"""
    df = agent.fetcher.get_history(request.symbol)
    if df.empty:
        raise HTTPException(404, f"无法获取 {request.symbol} 的数据")
    
    advice = agent.advisor.get_recommendation(request.symbol, df)
    return advice

@app.post("/api/predict")
async def get_prediction(request: PredictRequest):
    """获取价格预测"""
    df = agent.fetcher.get_history(request.symbol)
    if df.empty:
        raise HTTPException(404, f"无法获取 {request.symbol} 的数据")
    
    pred = agent.predictor.predict(df)
    if not pred.get('success', False):
        raise HTTPException(400, pred.get('error', '预测失败'))
    
    return pred

@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """获取实时行情"""
    quote = agent.fetcher.get_etf_quote(symbol)
    if not quote:
        raise HTTPException(404, f"未找到 {symbol}")
    return quote

@app.get("/api/etfs")
async def list_etfs():
    """列出默认ETF"""
    return {"etfs": agent.fetcher.get_etf_list()}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "service": "ETF-Smart Advisor"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)