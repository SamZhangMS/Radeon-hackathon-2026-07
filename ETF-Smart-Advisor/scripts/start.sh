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

# 3. Set GPU optimization environment variables
echo ""
echo "🔧 Setting GPU optimization environment variables..."
export PYTORCH_ROCM_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_ROCM_GRAPH=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 4. Check ROCm environment
echo ""
echo "🔍 Checking ROCm environment..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm detected successfully!"
    rocm-smi --showproductname
else
    echo "⚠️ ROCm not detected, using CPU mode"
fi

# 5. Check AMD GPU
echo ""
echo "🔍 Checking AMD GPU..."
if [ -d /dev/dri ]; then
    echo "✅ AMD GPU detected"
    export ROCM_VISIBLE_DEVICES=0
else
    echo "⚠️ AMD GPU not detected"
fi

# 6. Check VRAM
echo ""
echo "💾 Checking VRAM..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showmeminfo vram
fi

# ============================================================
# 7. Check Qwen model and LoRA adapter
# ============================================================
echo ""
echo "🔍 Checking Qwen model..."

# 模型本地路径
MODEL_PATH="./models/Qwen/mapfinben-qwen35-9b"
# LoRA 适配器路径
LORA_PATH="./data/models/lora_etf_advisor"
# 模型服务名称
MODEL_SERVED_NAME="mapfinben-qwen35-9b"
# ModelScope 上的模型ID
MODEL_SCOPE_ID="Qwen/mapfinben-qwen35-9b"
# 量化类型
QUANTIZATION="gptq"
# 最大模型长度
MAX_MODEL_LEN=8192

if [ ! -d "$MODEL_PATH" ]; then
    echo "  ❌ Model not found at: $MODEL_PATH"
    echo "  Please run: modelscope download --model $MODEL_SCOPE_ID --local_dir $MODEL_PATH"
    exit 1
else
    echo "  ✅ Model exists at: $MODEL_PATH"
fi

# 检查 LoRA 适配器
if [ -f "$LORA_PATH/adapter_model.safetensors" ]; then
    echo "  ✅ LoRA adapter found at: $LORA_PATH"
    export LORA_ADAPTER_PATH="$LORA_PATH"
else
    echo "  ℹ️ LoRA adapter not found, using base model"
    export LORA_ADAPTER_PATH=""
fi

# 8. Start vLLM with optimized settings
echo ""
echo "🚀 Starting vLLM inference service..."
echo "  Using model: $MODEL_PATH"
echo "  GPU memory utilization: 0.80"

# 构建量化参数
QUANTIZATION_ARG=""
if [ -n "$QUANTIZATION" ]; then
    QUANTIZATION_ARG="--quantization $QUANTIZATION"
fi

# 构建 LoRA 参数
LORA_ARG=""
if [ -n "$LORA_ADAPTER_PATH" ]; then
    LORA_ARG="--enable-lora --lora-modules qwen_lora=$LORA_ADAPTER_PATH"
fi

VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_SERVED_NAME" \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --gpu-memory-utilization=0.80 \
    --max-num-seqs=16 \
    --dtype=float16 \
    $QUANTIZATION_ARG \
    --max-model-len="$MAX_MODEL_LEN" \
    $LORA_ARG &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# 9. Wait for vLLM to be ready
echo ""
echo "⏳ Waiting for vLLM service to be ready (approx 60 seconds)..."
sleep 60

# Check if vLLM is running properly
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️ vLLM may not have started properly, check logs"
    echo "   tail -f vllm.log"
fi

# 10. Run benchmark test (optional)
echo ""
echo "📊 Running performance benchmark test..."
PYTHONPATH=. python -m scripts.benchmark --gpu 2>/dev/null || echo "  ⚠️ Benchmark skipped"

# 11. Start application
echo ""
echo "🚀 Starting ETF-Smart Advisor Web service..."
echo "  📊 Web UI: http://localhost:7860"
echo "  📚 API Docs: http://localhost:7860/docs"
echo "  ⏹️  Press Ctrl+C to stop"
echo ""

PYTHONPATH=. python -m app.main

# 12. Cleanup
trap "echo '🛑 Shutting down...'; kill $VLLM_PID 2>/dev/null" EXIT