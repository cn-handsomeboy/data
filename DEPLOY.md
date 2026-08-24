# 蔗循智策 —— 部署指南

本文档提供项目的部署方式，确保评委和用户体验**一键启动、开箱即用**。

---

## 1. 环境要求

| 项目 | 最低配置 | 推荐配置 |
|:-----|:---------|:---------|
| 操作系统 | Windows 10/11 / macOS / Linux | Windows 11 / Linux |
| Python | 3.8+ | 3.11 |
| 内存 | 2 GB | 4 GB |
| 网络 | 离线可运行（本地模式） | 联网（LLM 增强） |

依赖详见 `requirements.txt` 与 `运行环境说明.md`（含准确版本号）。

---

## 2. 方式一：本地 Python 运行（推荐）

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 运行测试（验证环境）

```bash
python test.py
```

预期输出：`17 tests passed`。

### 2.3 启动服务

```bash
# 终端 1：启动可视化界面（Streamlit）
streamlit run app.py

# 终端 2：启动 API 服务（FastAPI，可选）
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 2.4 Windows 快捷启动

双击项目根目录下的 `启动.bat` 或 `启动.ps1`。

---

## 3. 方式二：Streamlit Cloud 在线部署（云端）

项目已通过 Streamlit Cloud 从 GitHub 仓库自动部署，评审无需安装环境即可在线访问：

1. 将源码推送到 GitHub 仓库（`git push`）
2. 在 Streamlit Cloud 绑定该仓库
3. Streamlit Cloud 自动检测依赖并启动 `streamlit run app.py`

**云端密钥配置（不写入仓库）**：在 Streamlit Cloud 平台的 Secrets 中配置 `SCZC_LLM_API_KEY` 等环境变量，未配置时系统自动回退规则模板，功能不中断。

---

## 4. 服务验证清单

部署完成后，按以下清单逐项验证：

- [ ] 本地运行 `python test.py`，全部测试通过
- [ ] 访问 http://localhost:8501 出现 Streamlit 决策系统界面
- [ ] 访问 http://localhost:8000/docs 出现 FastAPI Swagger 文档（API 模式）
- [ ] 访问 http://localhost:8000/health 返回 `{"status": "ok"}`
- [ ] 在 Swagger 中调用 `/api/decision`，传入参数后返回决策结果
- [ ] 在 Streamlit 中点击"来宾市一键加载"，页面正常刷新并加载参数

---

## 5. 常见问题

### Q1: 端口被占用

Streamlit 默认 8501、FastAPI 默认 8000。若被占用，可用 `--server.port` / `--port` 指定其他端口。

### Q2: 未配置 DeepSeek Key，影响大吗？

不影响。未配置或调用失败时，系统自动回退规则模板生成报告，核心功能（产量预测、碳核算、多目标优化）完全正常。

### Q3: 数据文件权限问题（Linux/macOS）

```bash
chmod -R 755 data/
```

---

## 6. 生产环境建议

| 项目 | 建议 |
|:-----|:-----|
| API 密钥 | 通过环境变量注入，勿写入仓库或镜像 |
| 日志管理 | 挂载外部目录到 `logs/`，配合日志轮转工具 |
| 反向代理 | 使用 Nginx/Caddy 代理 8501/8000 端口，启用 HTTPS |
| 监控 | 接入 Prometheus + Grafana（如需） |

---

## 7. 联系与支持

- 项目文档：`README.md`、`DATA_SECURITY.md`、`CARBON_METHODOLOGY.md`
- 测试报告：`测试报告.pdf`
- 数据安全说明：`DATA_SECURITY.md`

---

> **部署声明**：本项目全部依赖均为公开主流开源包（见 `requirements.txt`），本地 Python 环境即可离线运行，无需额外容器依赖。数据集规模约 0.05 MB，符合大赛 1GB 限制。
>
> **提交说明**：本项目以「源码 + requirements.txt + 运行环境说明.md（含准确版本号）」提交运行环境，未引入 Docker 镜像，从源头规避密钥打包进镜像的风险。