#!/bin/bash
# scripts/start.sh - Service startup script (assumes environment is ready)

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
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

# 7. Check Qwen model
echo ""
echo "🔍 Checking Qwen model..."
MODEL_PATH="./models/Qwen3-30B-A3B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "  ❌ Qwen3-30B-A3B model not found!"
    echo "  Please run: modelscope download --model Qwen/Qwen3-30B-A3B --local_dir ./models/Qwen3-30B-A3B"
    exit 1
else
    echo "  ✅ Model exists"
fi

# 8. Start vLLM with optimized settings
echo ""
echo "🚀 Starting vLLM inference service..."
echo "  Using GPU memory utilization: 0.80 (reduced to avoid OOM)"

VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve "$MODEL_PATH" \
    --served-model-name Qwen3-30B-A3B \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --gpu-memory-utilization=0.80 \
    --max-num-seqs=16 \
    --dtype=bfloat16 \
    --quantization=awq \
    --max-model-len=8192 &

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