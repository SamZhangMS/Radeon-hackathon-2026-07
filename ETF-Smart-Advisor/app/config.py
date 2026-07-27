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

# ============================================================
# Qwen 模型配置 - 使用本地模型
# ============================================================

QWEN_MODEL_PATH = "./models/Qwen/mapfinben-qwen35-9b"
QWEN_MODEL_NAME = "mapfinben-qwen35-9b"

# LLM配置 - 直接使用 transformers
LLM_CONFIG = {
    "model_path": QWEN_MODEL_PATH,
    "model_name": QWEN_MODEL_NAME,
    "trust_remote_code": True,
    "torch_dtype": "auto",
    "device_map": "auto",
    "enable_thinking": False,  # 重要：关闭思考模式
}

# VLLM配置（用于API模式）
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "abc-123")
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", QWEN_MODEL_PATH)

# RAG 配置
RAG_CONFIG = {
    "collection_name": "etf_knowledge",
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,
}

# 记忆配置
MEMORY_CONFIG = {
    "enabled": True,
    "max_history": 10,
    "memory_path": str(DATA_DIR / "memory.json"),
}

# 任务规划配置
PLANNER_CONFIG = {
    "enabled": True,
    "max_steps": 5,
}

LORA_CONFIG = {
    "enabled": True,
    "model_path": MODELS_DIR / "lora_etf_advisor",
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# 预测配置
PREDICT_CONFIG = {
    "seq_length": 60,
    "pred_length": 20,
    "model_path": MODELS_DIR / "etf_predictor.pt",
    "lora_path": LORA_CONFIG["model_path"],
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

# 扩展 Agent 系统提示词
AGENT_SYSTEM_PROMPT_EXTENDED = """
你具备以下工具能力：
1. get_quote - 获取ETF实时行情
2. get_history - 获取ETF历史数据
3. analyze_technical - 技术指标分析
4. predict_price - 未来20周期价格预测
5. get_recommendation - 投资建议
6. search_knowledge - 知识库检索
7. generate_report - 生成完整分析报告
8. compare_etfs - 对比多个ETF

对于复杂任务，请自动分解为多个步骤并依次执行。
"""

# 默认ETF池
DEFAULT_ETF_POOL = [
    "510050", "510300", "510500", "159919", "159915",
    "512880", "512690", "515050", "516160", "512170"
]

# 推荐配置
RECOMMEND_CONFIG = {
    "top_k": 3,
    "min_data_days": 60,
    "score_threshold": 0.5,
}

# 集成预测配置
ENSEMBLE_CONFIG = {
    "transformer_weight": 0.6,
    "lstm_weight": 0.4,
    "min_confidence": 0.3,
}

GPU_LOCAL_PREDICTORS = {
    "lstm_light": {
        "name": "LSTM-Light (GPU)",
        "enabled": True,
        "weight": 0.35,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1
    },
    "transformer_light": {
        "name": "Transformer-Light (GPU)",
        "enabled": True,
        "weight": 0.45,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1
    }
}

# ============================================================
# 大模型API配置（可扩展）
# ============================================================

LLM_API_CONFIG = {
    "deepseek": {
        "name": "DeepSeek-V4-Flash",
        "api_base": os.environ.get("DEEPSEEK_API_BASE", "https://radeon.anruicloud.com/api/v1/chat/completions"),
        "api_key": os.environ.get("DEEPSEEK_API_KEY", "rc-7011453aecbbe0901a4b6d73980aea50852ad2c3002d8bf6"),
        "model": os.environ.get("DEEPSEEK_MODEL", "DeepSeek-V4-Flash"),
        "enabled": True,
        "weight": 0.3,
        "type": "remote"
    },
    "qwen_local": {
        "name": "Qwen-Local",
        "api_base": os.environ.get("QWEN_API_BASE", VLLM_API_BASE),
        "api_key": os.environ.get("QWEN_API_KEY", VLLM_API_KEY),
        "model": os.environ.get("QWEN_MODEL_NAME", QWEN_MODEL_NAME),
        "enabled": True,
        "weight": 0.4,
        "type": "local"
    }
}

# ============================================================
# LoRA微调配置
# ============================================================

LORA_FINETUNE_CONFIG = {
    "enabled": True,
    "output_dir": MODELS_DIR / "lora_etf_advisor",
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.1,
    "batch_size": 2,
    "epochs": 3,
    "learning_rate": 2e-4,
    "max_seq_length": 2048
}