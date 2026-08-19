# 蔗循智策 —— 部署指南

本文档提供项目的多种部署方式，确保评委和用户体验**一键启动、开箱即用**。

---

## 1. 环境要求

| 项目 | 最低配置 | 推荐配置 |
|:-----|:---------|:---------|
| Docker | 20.10+ | 24.0+ |
| Docker Compose | 2.0+ | 2.20+ |
| 内存 | 2 GB | 4 GB |
| 磁盘 | 1 GB | 2 GB |
| 网络 | 本地离线可运行 | — |

---

## 2. 方式一：Docker Compose 一键部署（推荐）

### 2.1 快速启动

```bash
# 1. 进入项目目录
cd /path/to/project

# 2. 一键构建并启动全部服务
docker-compose up --build -d

# 3. 查看服务状态
docker-compose ps
```

### 2.2 访问服务

| 服务 | 本地地址 | 说明 |
|:-----|:---------|:-----|
| Streamlit 可视化 | http://localhost:8501 | 决策系统主界面 |
| FastAPI 接口文档 | http://localhost:8000/docs | Swagger UI 在线调试 |
| FastAPI 健康检查 | http://localhost:8000/health | 系统状态 |
| 数据安全体检 | http://localhost:8000/api/security/status | 安全评分 |

### 2.3 停止服务

```bash
docker-compose down
```

---

## 3. 方式二：Docker 单镜像运行

### 3.1 构建镜像

```bash
docker build -t sczc:latest .
```

### 3.2 运行容器（双服务模式）

```bash
docker run -d \
  --name sczc \
  -p 8501:8501 \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/logs:/app/logs \
  sczc:latest
```

### 3.3 单独运行 Web 或 API

```bash
# 仅启动 Streamlit
docker run -d -p 8501:8501 sczc:latest \
  streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# 仅启动 FastAPI
docker run -d -p 8000:8000 sczc:latest \
  uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 4. 方式三：本地 Python 环境运行

### 4.1 安装依赖

```bash
pip install -r requirements.txt
```

### 4.2 运行测试（验证环境）

```bash
python test.py
```

预期输出：`17 tests passed`。

### 4.3 启动服务

```bash
# 终端 1：启动可视化界面
streamlit run app.py

# 终端 2：启动 API 服务
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4.4 Windows 快捷启动

双击项目根目录下的 `启动.bat` 或 `启动.ps1`。

---

## 5. 服务验证清单

部署完成后，按以下清单逐项验证：

- [ ] `docker-compose ps` 显示 `sczc-web` 和 `sczc-api` 状态为 `Up`
- [ ] 访问 http://localhost:8501 出现 Streamlit 决策系统界面
- [ ] 访问 http://localhost:8000/docs 出现 FastAPI Swagger 文档
- [ ] 访问 http://localhost:8000/health 返回 `{"status": "ok"}`
- [ ] 在 Swagger 中调用 `/api/decision`，传入参数后返回决策结果
- [ ] 在 Streamlit 中点击"来宾市一键加载"，页面正常刷新并加载参数
- [ ] 运行 `python test.py`，全部测试通过

---

## 6. 常见问题

### Q1: 端口被占用

```bash
# 修改 docker-compose.yml 中的端口映射，例如将 8501 改为 8502
ports:
  - "8502:8501"
```

### Q2: 内存不足导致模型加载失败

确保 Docker 分配内存 >= 2GB。若使用 Docker Desktop，请在设置中调整资源限制。

### Q3: 数据文件权限问题（Linux/macOS）

```bash
chmod -R 755 data/
```

---

## 7. 生产环境建议

| 项目 | 建议 |
|:-----|:-----|
| API 密钥 | 修改 `docker-compose.yml` 中的 `SUGARCANE_API_KEY` |
| 日志管理 | 挂载外部卷到 `/app/logs`，配合日志轮转工具 |
| 反向代理 | 使用 Nginx/Caddy 代理 8501/8000 端口，启用 HTTPS |
| 监控 | 接入 Prometheus + Grafana 监控容器资源 |

---

## 8. 联系与支持

- 项目文档：`README.md`、`DATA_SECURITY.md`、`CARBON_METHODOLOGY.md`
- 测试报告：`测试报告.pdf`
- 数据安全说明：`DATA_SECURITY.md`

---

> **部署声明**：本项目全部依赖和数据集均已封装在镜像内，离线环境亦可正常运行。数据量仅约 0.05 MB，符合大赛 1GB 限制。
