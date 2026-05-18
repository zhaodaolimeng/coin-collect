# 印尼智能催收系统 Docker 镜像
# 基于 Python 3.10 slim，面向 Linux/CPU 测试环境
FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENV=production
ENV TZ=Asia/Jakarta

# HuggingFace 模型缓存目录（可通过 volume 持久化）
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt
COPY requirements.txt .

# 1. CPU-only PyTorch 先装（防止 transformers 等包拉取 CUDA 版 torch）
# 2. 其余依赖（grep -v 排除 torch 行）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir $(grep -vE '^(torch|#|$)' requirements.txt)

COPY . .

RUN mkdir -p /app/data /app/logs /app/tmp /app/.cache/huggingface

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
