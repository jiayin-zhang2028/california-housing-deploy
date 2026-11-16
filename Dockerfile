# 用 slim 而不是完整版：镜像 120MB 和 1.2GB 的差别，就是冷启动 3 秒和 30 秒的差别。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/model_bundle.joblib \
    PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

# 先只拷 requirements 再装依赖：这样改代码不会让 pip 层缓存失效，
# 每次构建从十几分钟降到几秒。这是 Docker 分层缓存最基本的利用方式。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY monitor/ ./monitor/
COPY scripts/ ./scripts/

# 模型文件单独一层：模型变了才重传这一层
COPY artifacts/ ./artifacts/

# 不要用 root 跑服务：容器逃逸时这是最后一道墙。
# 同时把日志目录建好并交给非 root 用户，否则运行时写日志会 Permission denied。
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查：让 docker / K8s 知道这个容器是不是真的能干活
# start-period 给 60s：sklearn 版几秒就起来，但 Keras 版要加载 TensorFlow，
# 20s 会误判成 unhealthy 然后被反复重启（真会踩的坑）。
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# --workers 2：模型推理是 CPU 密集，多进程才能吃到多核（Python GIL 决定线程没用）。
# 注意 workers 数 ≈ CPU 核数，而不是越大越好：每个 worker 都会各自加载一份模型进内存。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
