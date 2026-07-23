# 使用官方 ROCm PyTorch 镜像作为基础
FROM pytorch/pytorch:2.5.0-rocm-5.7-ubuntu22.04

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    ROCM_VISIBLE_DEVICES=0 \
    PYTHONPATH=/workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目源码
COPY . .

# 创建必要的数据目录
RUN mkdir -p data/models data/cache

# 暴露端口
EXPOSE 8000 7860

# 设置启动命令
CMD ["bash", "scripts/start.sh"]