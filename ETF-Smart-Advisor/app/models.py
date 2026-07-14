from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    message: str
    symbol: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    recommendation: Optional[str] = None
    timestamp: datetime = datetime.now()

class PredictionRequest(BaseModel):
    symbol: str
    period: int = 20

class PredictionResponse(BaseModel):
    symbol: str
    dates: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    recommendation: Dict[str, Any]
    confidence: float

class AnalysisRequest(BaseModel):
    symbol: str
    period: str = "6mo"

class AnalysisResponse(BaseModel):
    symbol: str
    name: str
    current_price: float
    signal: str
    recommendation: str
    reasons: List[str]
    technical_indicators: Dict[str, Any]
    risk_level: str
    predicted_trend: Optional[Dict[str, Any]]