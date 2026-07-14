#!/bin/bash
# scripts/start.sh - 增强版启动脚本

set -e

echo "启动 ETF-Smart Advisor (优化版)"
echo "="*60

# 1. 检查 ROCm 环境
echo "检查 ROCm 环境..."
if command -v rocm-smi &> /dev/null; then
    echo "ROCm 检测成功"
    rocm-smi --showproductname
else
    echo "ROCm 未检测到，使用 CPU 模式"
fi

# 2. 检查 AMD GPU
echo "检查 AMD GPU..."
if [ -d /dev/dri ]; then
    echo "AMD GPU 检测成功"
    export ROCM_VISIBLE_DEVICES=0
else
    echo "AMD GPU 未检测到"
fi

# 3. 设置环境变量（GPU优化）
echo "🔧 设置环境变量..."
export PYTORCH_ROCM_ALLOC_CONF="max_split_size_mb:128"
export TORCH_ROCM_GRAPH=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 4. 检查显存
echo "检查显存..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showmeminfo vram
fi

# 5. 启动 vLLM (GPU优化)
echo "启动 vLLM 推理服务..."
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-30B-A3B \
    --served-model-name Qwen3-30B-A3B \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --gpu-memory-utilization=0.95 \
    --max-num-seqs=32 \
    --dtype=bfloat16 &

VLLM_PID=$!

# 6. 等待 vLLM 就绪
echo "等待 vLLM 服务就绪..."
sleep 30

# 7. 启动应用
echo "启动 ETF-Smart Advisor..."
cd /workspace

# 运行性能基准测试
echo "运行性能基准测试..."
python -m scripts.benchmark --gpu

# 启动主应用
python -m app.main

# 8. 清理
trap "kill $VLLM_PID" EXIT