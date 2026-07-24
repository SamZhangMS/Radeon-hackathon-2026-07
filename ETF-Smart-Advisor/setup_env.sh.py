#!/bin/bash
# setup_env.sh - 一键创建 AMD ROCm 7.2.1 运行环境

set -e

echo "=============================================="
echo "🚀 ETF-Smart Advisor 环境搭建 (ROCm 7.2.1)"
echo "=============================================="

# 1. 获取项目根目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "📁 项目目录: $PROJECT_DIR"

# 2. 检查 ROCm 环境
echo ""
echo "🔍 检查 ROCm 环境..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm 检测成功!"
    rocm-smi --showproductname
else
    echo "❌ ROCm 未检测到，请先安装 ROCm 7.2.1"
    exit 1
fi

# 3. 创建虚拟环境
echo ""
echo "📦 创建 Python 虚拟环境..."
if [ -d "venv" ]; then
    echo "⚠️ 虚拟环境已存在，删除旧环境..."
    rm -rf venv
fi
python3 -m venv venv
source venv/bin/activate

# 4. 升级 pip
echo ""
echo "⬆️ 升级 pip..."
pip install --upgrade pip

# 5. 下载并安装 ROCm 版 PyTorch
echo ""
echo "📥 安装 ROCm 版 PyTorch (2.9.1+rocm7.2.1)..."
cd /tmp

# 检查 wheel 文件是否存在，不存在则下载
PYTORCH_WHEEL="torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
TORCHVISION_WHEEL="torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
TORCHAUDIO_WHEEL="torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
TRITON_WHEEL="triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

download_if_missing() {
    local url="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/$1"
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "  下载 $file..."
        wget -q "$url" -O "$file" || echo "  ⚠️ 下载失败，尝试继续..."
    else
        echo "  ✅ $file 已存在"
    fi
}

echo "  检查并下载 wheel 文件..."
download_if_missing "$PYTORCH_WHEEL"
download_if_missing "$TORCHVISION_WHEEL"
download_if_missing "$TORCHAUDIO_WHEEL"
download_if_missing "$TRITON_WHEEL"

echo "  安装 PyTorch 组件..."
pip install --no-deps "$PYTORCH_WHEEL" "$TORCHVISION_WHEEL" "$TORCHAUDIO_WHEEL" "$TRITON_WHEEL" 2>/dev/null || \
pip install "$PYTORCH_WHEEL" "$TORCHVISION_WHEEL" "$TORCHAUDIO_WHEEL" "$TRITON_WHEEL"

# 6. 验证 PyTorch
echo ""
echo "🔬 验证 PyTorch 安装..."
python -c "
import torch
print(f'✅ PyTorch 版本: {torch.__version__}')
print(f'✅ GPU 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU 名称: {torch.cuda.get_device_name(0)}')
    print(f'✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 7. 安装 uv 和 vLLM
echo ""
echo "📦 安装 uv 和 vLLM (ROCm 版)..."
pip install uv
uv pip install vllm==0.18.0+rocm700 \
    --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700

# 8. 安装其他依赖
echo ""
echo "📦 安装项目其他依赖..."
cd "$PROJECT_DIR"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 9. 创建数据目录
echo ""
echo "📁 创建数据目录..."
mkdir -p data/models data/cache

# 10. 验证安装
echo ""
echo "=============================================="
echo "✅ 环境搭建完成!"
echo "=============================================="
echo ""
echo "📊 启动服务:"
echo "   source venv/bin/activate"
echo "   PYTHONPATH=. python -m app.main"
echo ""
echo "🧪 验证 GPU:"
echo "   python -c \"import torch; print(torch.cuda.is_available())\""