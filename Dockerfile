# 蔗循智策 —— Docker 部署镜像
# 基于 Python 3.11 官方轻量镜像

FROM python:3.11-slim

LABEL maintainer="蔗循智策项目团队"
LABEL description="面向中国-东盟的甘蔗副产物循环经济决策系统"
LABEL version="1.0.0"

# 设置工作目录
WORKDIR /app

# 安装系统依赖（编译某些 Python 包所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单并安装（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目全部代码与数据
COPY . .

# 暴露 Streamlit 和 FastAPI 端口
EXPOSE 8501
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 同时启动 Streamlit 可视化界面和 FastAPI 服务
CMD ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0 & uvicorn api:app --host 0.0.0.0 --port 8000"]
