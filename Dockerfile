# 可选增强项（非硬性验收要求）：CPU 版部署，避免携带 CUDA 依赖
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==22.0.0

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY models/ ./models/

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')"

# 用 gunicorn 启动；模型在 app.py 导入时加载，加载失败进程直接退出
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--chdir", "src", "app:app"]
