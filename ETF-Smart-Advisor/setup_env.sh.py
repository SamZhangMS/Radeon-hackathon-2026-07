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
echo "📦 创建 Python 虚拟环境 (etfadvisorvenv)..."
if [ -d "etfadvisorvenv" ]; then
    echo "⚠️ 虚拟环境已存在，删除旧环境..."
    rm -rf etfadvisorvenv
fi
python3 -m venv etfadvisorvenv
source etfadvisorvenv/bin/activate

# 4. 升级 pip
echo ""
echo "⬆️ 升级 pip..."
pip install --upgrade pip

# 5. 安装 PyTorch 依赖
echo ""
echo "📦 安装 PyTorch 依赖..."
pip install typing_extensions numpy

# 6. 下载并安装 ROCm 版 PyTorch
echo ""
echo "📥 安装 ROCm 版 PyTorch (2.9.1+rocm7.2.1)..."
cd /tmp

# 定义 wheel 文件名
PYTORCH_WHEEL="torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
TORCHVISION_WHEEL="torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
TORCHAUDIO_WHEEL="torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
TRITON_WHEEL="triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

# 下载函数
download_if_missing() {
    local url="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/$1"
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "  下载 $file..."
        wget -q --show-progress "$url" -O "$file" || echo "  ⚠️ 下载失败，尝试继续..."
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
pip install "$PYTORCH_WHEEL" "$TORCHVISION_WHEEL" "$TORCHAUDIO_WHEEL" "$TRITON_WHEEL"

# 7. 验证 PyTorch
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

# 8. 安装 uv 和 vLLM
echo ""
echo "📦 安装 uv 和 vLLM (ROCm 版)..."
pip install uv
uv pip install vllm==0.18.0+rocm700 \
    --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700

# 9. 安装其他依赖
echo ""
echo "📦 安装项目其他依赖..."
cd "$PROJECT_DIR"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 10. 创建数据目录
echo ""
echo "📁 创建数据目录..."
mkdir -p data/models data/cache

# ============================================================
# 11. 清理临时文件
# ============================================================
cleanup_temp_files() {
    echo ""
    echo "🧹 清理临时文件以节省空间..."
    
    # 清理 /tmp 目录下的 wheel 文件
    local tmp_files=(
        "/tmp/$PYTORCH_WHEEL"
        "/tmp/$TORCHVISION_WHEEL"
        "/tmp/$TORCHAUDIO_WHEEL"
        "/tmp/$TRITON_WHEEL"
    )
    
    local freed_space=0
    for file in "${tmp_files[@]}"; do
        if [ -f "$file" ]; then
            local size=$(du -b "$file" | cut -f1)
            freed_space=$((freed_space + size))
            rm -f "$file"
            echo "  已删除: $(basename "$file") ($(numfmt --to=iec $size))"
        fi
    done
    
    # 清理 pip 缓存
    echo "  清理 pip 缓存..."
    pip cache purge 2>/dev/null || true
    
    # 清理 uv 缓存
    echo "  清理 uv 缓存..."
    uv cache clean 2>/dev/null || true
    
    if [ $freed_space -gt 0 ]; then
        echo "  ✅ 共释放空间: $(numfmt --to=iec $freed_space)"
    else
        echo "  ✅ 没有需要清理的临时文件"
    fi
}

echo ""
echo "💡 是否清理下载的临时 wheel 文件？(约 2GB)"
read -p "  输入 y 确认清理，输入 n 跳过 (默认 y): " -r CLEANUP_CHOICE
CLEANUP_CHOICE=${CLEANUP_CHOICE:-y}

if [[ "$CLEANUP_CHOICE" =~ ^[Yy]$ ]]; then
    cleanup_temp_files
else
    echo "  跳过清理，wheel 文件保留在 /tmp 目录"
    echo "  可手动运行: rm -f /tmp/torch-*.whl /tmp/triton-*.whl"
fi

# ============================================================
# 12. 验证安装
# ============================================================
echo ""
echo "=============================================="
echo "✅ 环境搭建完成!"
echo "=============================================="
echo ""
echo "📊 启动服务:"
echo "   source etfadvisorvenv/bin/activate"
echo "   PYTHONPATH=. python -m app.main"
echo ""
echo "🧪 验证 GPU:"
echo "   python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "📁 临时文件已清理，当前磁盘空间:"
df -h /tmp /workspace 2>/dev/null || df -h .