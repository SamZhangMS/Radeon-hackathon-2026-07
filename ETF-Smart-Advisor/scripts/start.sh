#!/bin/bash
# scripts/start.sh - Service startup script

set -e

echo "=============================================="
echo "🚀 Starting ETF-Smart Advisor"
echo "=============================================="

# 1. Get project root directory
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 2. Activate virtual environment
echo ""
echo "📦 Activating virtual environment..."
if [ -d "etfadvisorvenv" ]; then
    source etfadvisorvenv/bin/activate
else
    echo "❌ Virtual environment not found, please run setup_env.sh first"
    exit 1
fi

# 3. 从 config.py 读取配置
echo ""
echo "📖 Reading configuration from app/config.py..."

# 使用 Python 读取配置
read_config() {
    python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
try:
    from app.config import LLM_MODEL_PATH, LLM_MODEL_NAME, LLM_MODEL_ID, VLLM_CONFIG
    print(f'MODEL_PATH={LLM_MODEL_PATH}')
    print(f'MODEL_NAME={LLM_MODEL_NAME}')
    print(f'MODEL_ID={LLM_MODEL_ID}')
    print(f'VLLM_PORT={VLLM_CONFIG.get(\"port\", 8000)}')
    print(f'VLLM_GPU_MEM={VLLM_CONFIG.get(\"gpu_memory_utilization\", 0.85)}')
    print(f'VLLM_MAX_MODEL_LEN={VLLM_CONFIG.get(\"max_model_len\", 8192)}')
    print(f'VLLM_ENABLED={VLLM_CONFIG.get(\"enabled\", True)}')
except ImportError as e:
    print(f'ERROR={e}', file=sys.stderr)
    sys.exit(1)
"
}

# 读取配置
CONFIG_OUTPUT=$(read_config)
if [ $? -ne 0 ]; then
    echo "❌ Failed to read config.py"
    echo "   Please ensure app/config.py exists and contains LLM_MODEL_PATH, LLM_MODEL_NAME, LLM_MODEL_ID, VLLM_CONFIG"
    exit 1
fi

# 解析配置
eval "$CONFIG_OUTPUT"

# 检查必要配置是否存在
if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ] || [ -z "$MODEL_ID" ]; then
    echo "❌ LLM_MODEL_PATH, LLM_MODEL_NAME or LLM_MODEL_ID not set in config.py"
    exit 1
fi

echo "  ✅ Model path: $MODEL_PATH"
echo "  ✅ Model name: $MODEL_NAME"
echo "  ✅ Model ID: $MODEL_ID"
echo "  ✅ vLLM enabled: $VLLM_ENABLED"
echo "  ✅ vLLM port: $VLLM_PORT"
echo "  ✅ GPU memory: $VLLM_GPU_MEM"
echo "  ✅ Max model len: $VLLM_MAX_MODEL_LEN"

# 4. Set GPU optimization environment variables
echo ""
echo "🔧 Setting GPU optimization environment variables..."
export PYTORCH_ROCM_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_ROCM_GRAPH=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# 5. Check ROCm environment
echo ""
echo "🔍 Checking ROCm environment..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm detected successfully!"
    rocm-smi --showproductname
else
    echo "⚠️ ROCm not detected, using CPU mode"
fi

# 6. Check AMD GPU
echo ""
echo "🔍 Checking AMD GPU..."
if [ -d /dev/dri ]; then
    echo "✅ AMD GPU detected"
    export ROCM_VISIBLE_DEVICES=0
else
    echo "⚠️ AMD GPU not detected"
fi

# 7. Check VRAM
echo ""
echo "💾 Checking VRAM..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showmeminfo vram
fi

# 8. Check Python dependencies
echo ""
echo "🔍 Checking Python dependencies..."
MISSING_PKGS=""
for pkg in torch transformers fastapi uvicorn pymilvus sentence-transformers; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done
if [ -n "$MISSING_PKGS" ]; then
    echo "  ⚠️ Installing missing packages:$MISSING_PKGS"
    pip install $MISSING_PKGS
else
    echo "  ✅ All dependencies installed"
fi

# 9. Check Qwen model
echo ""
echo "🔍 Checking Qwen model..."

if [ ! -d "$MODEL_PATH" ]; then
    echo "  ❌ Model not found at: $MODEL_PATH"
    echo "  💡 Please check LLM_MODEL_PATH in app/config.py"
    echo "  💡 Or run setup_env.sh to download the model"
    exit 1
fi

# 检查模型文件是否完整
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "  ❌ Model config.json not found. Model may be incomplete."
    exit 1
fi

echo "  ✅ Model exists at: $MODEL_PATH"
echo "  ✅ Model files verified"

# 10. Check Milvus Lite
echo ""
echo "🔍 Checking Milvus Lite..."
python -c "from app.milvus_client import get_milvus_client; client = get_milvus_client(); print(f'✅ Milvus Lite: {client.get_stats()}')" 2>/dev/null || echo "  ⚠️ Milvus Lite will start on demand"

# 11. Check ports
echo ""
echo "🔍 Checking ports..."
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        echo "  ⚠️ Port $port is already in use"
        return 1
    else
        echo "  ✅ Port $port is available"
        return 0
    fi
}
check_port $VLLM_PORT || echo "  💡 vLLM may fail if port $VLLM_PORT is occupied"
check_port 7860 || echo "  💡 Web service may fail if port 7860 is occupied"

# 12. Check vLLM availability
echo ""
echo "🔍 Checking vLLM..."
USE_VLLM=true # false
# if [ "$VLLM_ENABLED" = true ] && python -c "import vllm" 2>/dev/null; then
#     echo "  ✅ vLLM installed and enabled"
#     USE_VLLM=true
# elif [ "$VLLM_ENABLED" = true ]; then
#     echo "  ⚠️ vLLM enabled in config but not installed"
#     echo "  💡 Install with: pip install vllm"
#     USE_VLLM=false
# else
#     echo "  ℹ️ vLLM disabled in config"
# fi

# 13. Start vLLM (if available and enabled)
VLLM_PID=""
if [ "$USE_VLLM" = true ]; then
    echo ""
    echo "🚀 Starting vLLM inference service..."
    echo "  Model: $MODEL_PATH"
    echo "  GPU memory utilization: $VLLM_GPU_MEM"
    echo "  Port: $VLLM_PORT"
    echo "  Max model len: $VLLM_MAX_MODEL_LEN"
    
    # 启动 vLLM（在后台运行）
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --port $VLLM_PORT \
        --trust-remote-code \
        --gpu-memory-utilization=$VLLM_GPU_MEM \
        --max-num-seqs=8 \
        --dtype=auto \
        --max-model-len=$VLLM_MAX_MODEL_LEN \
        > vllm.log 2>&1 &
    
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"
    
    # 等待 vLLM 启动
    echo ""
    echo "⏳ Waiting for vLLM to be ready ..."
    MAX_WAIT=1200 # 120
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            echo "  ✅ vLLM is ready (took ${WAIT_COUNT}s)"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        echo -n "."
    done
    
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo ""
        echo "  ⚠️ vLLM startup timeout, checking logs..."
        tail -20 vllm.log 2>/dev/null || echo "  No logs available"
        echo "  💡 Falling back to Transformers mode"
        USE_VLLM=false
        if [ -n "$VLLM_PID" ]; then
            kill $VLLM_PID 2>/dev/null || true
        fi
    else
        echo ""
        echo "  ✅ vLLM service running successfully"
    fi
else
    echo ""
    echo "ℹ️ Using Transformers mode (direct PyTorch ROCm)"
fi

# 14. Check LLM Client
echo ""
echo "🔍 Checking LLM Client..."
python -c "
from app.llm_client import get_llm_client
try:
    llm = get_llm_client()
    status = llm.get_model_status()
    print(f'  ✅ LLM Client ready')
    print(f'  📊 vLLM 可用: {status.get(\"vllm_available\", False)}')
    print(f'  📊 Transformers 已加载: {status.get(\"transformers_loaded\", False)}')
    print(f'  📊 推理模式: {\"vLLM\" if status.get(\"vllm_available\", False) else \"Transformers\"}')
    print(f'  📊 模型路径: {status.get(\"model_path\", \"N/A\")}')
except Exception as e:
    print(f'  ⚠️ LLM Client: {e}')
"

# 15. Start application
echo ""
echo "🚀 Starting ETF-Smart Advisor Web service..."
echo "  📊 Web UI: http://localhost:7860"
echo "  📚 API Docs: http://localhost:7860/docs"
echo "  🔧 推理模式: $( [ "$USE_VLLM" = true ] && echo "vLLM (高性能)" || echo "Transformers (兼容)" )"
echo "  🗄️  知识库: Milvus Lite"
echo "  📁 模型: $MODEL_NAME"
echo "  🆔 模型ID: $MODEL_ID"
echo "  ⏹️  Press Ctrl+C to stop"
echo ""

# 设置环境变量
export VLLM_ENABLED=$USE_VLLM
export VLLM_PORT=$VLLM_PORT
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export MODEL_ID=$MODEL_ID

# 启动应用
PYTHONPATH=. python -m app.main

# 16. Cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down..."

    # 1. 停止 Milvus Lite
    echo "  Stopping Milvus Lite..."
    python3 -c "
from app.milvus_client import get_milvus_client
get_milvus_client().stop()
print('  ✅ Milvus Lite stopped')
" 2>/dev/null || true

    # 2. 停止 vLLM
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "  Stopping vLLM (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        sleep 2
        echo "  ✅ vLLM stopped"
    fi
    echo "  ✅ Service stopped"
    exit 0
}

trap cleanup EXIT