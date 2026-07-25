#!/bin/bash
# setup_env.sh - One-click setup script for AMD ROCm 7.2.1 environment

set -e

echo "=============================================="
echo "🚀 ETF-Smart Advisor Environment Setup"
echo "=============================================="

# 1. Get project root directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "📁 Project directory: $PROJECT_DIR"

# 2. Check ROCm environment
echo ""
echo "🔍 Checking ROCm environment..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm detected successfully!"
    rocm-smi --showproductname
else
    echo "❌ ROCm not detected, please install ROCm first"
    exit 1
fi

# 3. Create virtual environment
echo ""
echo "📦 Creating Python virtual environment (etfadvisorvenv)..."
if [ -d "etfadvisorvenv" ]; then
    echo "⚠️ Virtual environment already exists, removing old one..."
    rm -rf etfadvisorvenv
fi
python3 -m venv etfadvisorvenv
source etfadvisorvenv/bin/activate

# 4. Upgrade pip
echo ""
echo "⬆️ Upgrading pip..."
pip install --upgrade pip
pip install typing_extensions numpy
pip install pandas pyarrow --upgrade
pip install fastparquet
pip install cython
pip install akshare
pip install ta-lib
pip install baostock
pip install xgboost
pip install catboost
pip install optuna
pip install requests
pip install robust_json_parser
pip install pyqlib
pip install -r ../work/lib/Kronos/requirements.txt

# 5. Install PyTorch dependencies
echo ""
echo "📦 Installing PyTorch dependencies..."
pip install typing_extensions numpy

# 6. Download and install ROCm PyTorch
echo ""
echo "📥 Installing ROCm PyTorch (2.9.1+rocm7.2.1)..."
cd /tmp

# Define wheel filenames
PYTORCH_WHEEL="torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
TORCHVISION_WHEEL="torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
TORCHAUDIO_WHEEL="torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
TRITON_WHEEL="triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

# Download function
download_if_missing() {
    local url="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/$1"
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "  Downloading $file..."
        wget -q --show-progress "$url" -O "$file" || echo "  ⚠️ Download failed, continuing..."
    else
        echo "  ✅ $file already exists"
    fi
}

echo "  Checking and downloading wheel files..."
download_if_missing "$PYTORCH_WHEEL"
download_if_missing "$TORCHVISION_WHEEL"
download_if_missing "$TORCHAUDIO_WHEEL"
download_if_missing "$TRITON_WHEEL"

echo "  Installing PyTorch components..."
pip install "$PYTORCH_WHEEL" "$TORCHVISION_WHEEL" "$TORCHAUDIO_WHEEL" "$TRITON_WHEEL"

# 7. Verify PyTorch
echo ""
echo "🔬 Verifying PyTorch installation..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU name: {torch.cuda.get_device_name(0)}')
    print(f'✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 8. Install uv and vLLM
echo ""
echo "📦 Installing uv and vLLM (ROCm version)..."
pip install uv
uv pip install vllm --upgrade \
    --extra-index-url https://wheels.vllm.ai/rocm/

# 9. Install other project dependencies
echo ""
echo "📦 Installing other project dependencies..."
cd "$PROJECT_DIR"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 10. Create data directories
echo ""
echo "📁 Creating data directories..."
mkdir -p data/models data/cache

# 11. Check and download Qwen model
echo ""
echo "📥 Checking Qwen3-30B-A3B-GPTQ-Int4..."
MODEL_PATH="./models/Qwen/Qwen3-30B-A3B-GPTQ-Int4"
if [ ! -d "$MODEL_PATH" ]; then
    echo "  Model not found, downloading ..."
    mkdir -p models/Qwen
    pip install modelscope -q
    modelscope download --model Qwen/Qwen3-30B-A3B-GPTQ-Int4 --local_dir "$MODEL_PATH"
else
    echo "  ✅ Model already exists"
fi

# ============================================================
# 12. LoRA 调优（使用 ETF 历史数据）
# ============================================================
echo ""
echo "🔧 Running LoRA fine-tuning on Qwen with ETF data..."

# 检查数据目录是否存在
if [ -d "./data/1D" ] && [ "$(ls -A ./data/1D 2>/dev/null)" ]; then
    echo "  ✅ ETF data found, starting fine-tuning..."
    
    # 设置环境变量
    export FINETUNE_MODEL_PATH="$MODEL_PATH"
    export FINETUNE_OUTPUT_DIR="./data/models/lora_etf_advisor"
    
    # 安装额外依赖
    pip install transformers==4.48.0
    pip install optimum==1.21.0
    pip install accelerate==0.34.0
    pip install bitsandbytes
    pip install peft trl datasets scikit-learn
    # 运行调优脚本
    PYTHONPATH=. python scripts/finetune_qwen.py
    
    # 检查调优是否成功
    if [ -f "./data/models/lora_etf_advisor/adapter_model.safetensors" ]; then
        echo "  ✅ LoRA fine-tuning completed successfully!"
        echo "  📁 LoRA weights saved to: ./data/models/lora_etf_advisor"
    else
        echo "  ⚠️ LoRA fine-tuning may have failed, check logs above"
    fi
else
    echo "  ⚠️ ETF data not found at ./data/1D, skipping fine-tuning"
    echo "  💡 Please add ETF historical data files (*.txt) to ./data/1D/"
    echo "  💡 Example format: date,open,high,low,close,volume,money"
fi


# 13. Clean up temporary files
cleanup_temp_files() {
    echo ""
    echo "🧹 Cleaning up temporary files to save disk space..."
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
            echo "  Deleted: $(basename "$file") ($(numfmt --to=iec $size))"
        fi
    done
    pip cache purge 2>/dev/null || true
    uv cache clean 2>/dev/null || true
    if [ $freed_space -gt 0 ]; then
        echo "  ✅ Freed space: $(numfmt --to=iec $freed_space)"
    else
        echo "  ✅ No files to clean"
    fi
}

echo ""
echo "💡 Clean up temporary wheel files? (approx 2GB)"
read -p "  Enter y to confirm, n to skip (default y): " -r CLEANUP_CHOICE
CLEANUP_CHOICE=${CLEANUP_CHOICE:-y}
if [[ "$CLEANUP_CHOICE" =~ ^[Yy]$ ]]; then
    cleanup_temp_files
else
    echo "  Skipping cleanup"
fi

# 14. Start the service
echo ""
echo "=============================================="
echo "✅ Environment setup complete, starting service..."
echo "=============================================="
echo ""

# Call start.sh to launch the service
bash scripts/start.sh