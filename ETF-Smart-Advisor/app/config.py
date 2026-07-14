import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

# 创建目录
for d in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# AMD ROCm设备配置
DEVICE = "cuda" if os.environ.get("ROCM_VISIBLE_DEVICES") else "cpu"

# LLM配置
LLM_CONFIG = {
    "model_name": "Qwen/Qwen3-30B-A3B",
    "api_base": "http://localhost:8000/v1",
    "api_key": "abc-123",
}

# 预测配置
PREDICT_CONFIG = {
    "seq_length": 60,
    "pred_length": 20,
    "model_path": MODELS_DIR / "etf_predictor.pt",
}

# Agent系统提示
AGENT_SYSTEM_PROMPT = """你是ETF-Smart Advisor，专业的ETF投资顾问。

你的核心能力：
1. 分析ETF的技术指标和趋势
2. 预测未来20个周期的价格走势
3. 提供明确的买入/卖出/持有建议
4. 评估投资风险和收益潜力

建议格式：
- 买入：强烈看涨，建议入场
- 持有：趋势向好，继续持有
- 卖出：出现风险信号，建议离场
- 观望：方向不明，等待机会

务必提醒：投资有风险，决策需谨慎。
"""

# 默认ETF池
DEFAULT_ETF_POOL = [
    "510050", "510300", "510500", "159919", "159915",
    "512880", "512690", "515050", "516160", "512170"
]