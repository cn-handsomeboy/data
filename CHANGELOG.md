# Changelog

## [1.3.0] - 2026-08-20

### 数据层（东盟真实气象 + 真实产量数据点）
- 新增东盟四国真实气象补全：`weather_data_asean.csv`（Open-Meteo ERA5，泰/越/缅/老各 2 个主产蔗区，
  2010-2024 逐日，43832 行，与广西同源可溯源）。
- 新增 `asean_climate_normals.json`：四国生长季（5-10月）逐年聚合后的气候基准
  （泰国 27.9°C/1118mm、越南 28.0°C/1627mm、缅甸 27.1°C/2478mm、老挝 26.9°C/1803mm）。
- 新增 `asean_yield_weather.csv`：真实 FAO 产量 × 东盟真实气象，4国×10年 = **40 个真实
  国家-年级产量+气象数据点**，补齐跨境真实产量数据。
- `data/.file_hashes.json` 登记以上新增文件及 `weather_data_expanded.csv` 的 SHA-256 指纹。

### Agent（agent.py）
- `_cross_country_compare` 改为使用各国真实生长季气候基准，不再复用中国的入门气象输入；
  跨境对比表新增"生长季均温/降水"列，由于气候差异使各国最优方案自然分化
  （如泰/越为最优循环、缅/老为进阶循环）。文件缺失时自动回退复现中国天气。

### 运行台账（实际成效证据）
- 新增 `run_stats.py` + `GET /api/run/stats` 端点 + 前端"真实运行台账"面板：
  聚合真实审计日志（API调用/端点/安全拦截/跨境授权）与真实反馈闭环（条数/MAPE/校准状态），
  所有数字直接读取 `logs/security_audit.log` 与 `data/user_feedback.json`，可溯源、无模拟注入，
  为"实际成效"评审项提供可复核的真实运行凭证。

## [1.2.0] - 2026-08-19

### 模型层（models.py）
- 修复 `predict()` 扩展变量缺口：模型已用 9 维气象特征训练，但预测仅输入温度/降水/日照
  三变量，导致 DataFrame 选列 KeyError。现预测时自动按城市从 `CITY_CLIMATE_NORMALS`
  （Open-Meteo ERA5 2010-2024 生长季基准）补齐 humidity/pressure/wind/ET0/soil_temp/soil_moisture。
- 确认使用扩展气象数据训练：`weather_data_expanded.csv`（38,353 条，2010-2024，7 市，13 变量），
  训练合并样本 70（城市-年细粒度），GBRT LOOCV R²≈0.84。

### 数据层（data/）
- 删除合成假数据 `bootstrap_training_data.csv`（7000 条 bootstrap 合成样本，非真实采集），
  维护数据真实性；`data_expansion_summary.json` 重建为合法 JSON 并移除 `bootstrap_samples` 字段。

### LLM 增强（新增 llm_agent.py）
- 新增可选大语言模型增强模块：决策报告润色 + 自然语言问数；仅作表述增强，不参与任何计算。
- OpenAI 兼容 chat/completions，标准库 urllib 实现，零新增运行依赖。
- 内置滑动窗口限流（默认 10 次/分钟，可收紧）+ 超时控制 + 全程可回退（无 key/失败/限流→规则模板）。
- 通过环境变量配置：`SCZC_LLM_API_KEY` / `SCZC_LLM_BASE_URL` / `SCZC_LLM_MODEL` 等（见 `.env.example`）。

### Agent（agent.py）
- `chat()` 接入 LLM 报告润色：有 key 时输出"AI 智能决策报告"（正文）+ 结构化核算附表（数字可复核），
  无 key 回退"规则引擎版"；新增 `answer_question()` 自然语言问数并带规则兜底。
- 文档与表述明确定位：规则引擎 + LLM 口语化增强，LLM 不参与计算。

### API（api.py）
- 新增 `POST /api/report`：决策报告 + 推理链 + 可选 LLM 润色（`llm_used`/`llm_available` 标记）。
- 新增 `POST /api/ask`：自然语言问数，LLM 缺省时返回规则兜底。
- 新增 `GET /api/llm/status`：LLM 增强是否配置，供前端展示开关。

## [1.1.0] - 2026-07-31

### API 与数据产品（api.py）
- 修正 OpenAPI 描述：产量预测模型明确为 GBRT LOOCV R²=0.893，五方案多目标优化，五国跨境决策
- 扩展 `DecisionRequest.country` 枚举示例至 China/Thailand/Vietnam/Myanmar/Laos
- 决策接口新增 `request_id` / `request_time` 并在审计日志中透传，提升可审计性
- 强化异常处理：输入校验失败、业务校验失败均记录安全事件，模型失败返回结构化 HTTP 错误而非堆栈
- 补全五方案中文名称映射（传统/改良传统/基础循环/进阶循环/最优循环）

### 模型与系统健壮性（models.py）
- `_check_numeric` 与 `_validate_decision_inputs` 防御 `None` / `NaN` / 非数值输入
- `YieldPredictor.load_model` 显式捕获 `pickle.UnpicklingError` / `EOFError`，损坏模型自动回退重训练
- 新增 `warm_start_models()` 热加载/预加载辅助，优先加载已保存模型，减少 API 冷启动时间
- 确认 `get_default_carbon_price` 已覆盖中国-泰国-越南-缅甸-老挝五国碳价默认值

### 数据安全（data_security.py）
- 顶部新增速率限制实现说明（无外部依赖，依赖网关层）
- `DataMasker.mask_api_response` 支持嵌套列表/字典递归扫描，字符串中 PII 自动脱敏
- `SecurityManager.full_security_check` 新增 API Key 强度检查，识别开发随机密钥/弱密钥

### 数据产品与报告（data_product.py）
- 数据血缘节点 `src_intl` 补充五国说明
- 测试报告 PDF 更新为 17 项测试通过，结论与摘要同步五国、五方案、R²=0.893

### 部署与工程化
- 新增 `.dockerignore`，排除缓存、模型、日志、敏感文档等冗余文件
- `.env.example` 增加安全提示，`.gitignore` 增加 `.env`
- 新增 `run_tests.py` 自动化测试运行器，支持 `--pdf` 参数一键生成测试报告

### 文档一致性
- `README.md`：测试数量更新为 17 项；保持五国/五方案/R²=0.893 一致
- `DEPLOY.md` 服务验证清单已核对
- `test.py`：测试 1/7 的打印与 docstring 描述更新为 GBRT LOOCV、五国跨境对比

## [1.0.0] - 2026-07-22

### 数据层
- 广西7市×10年甘蔗产量数据（70样本，广西统计年鉴）
- 7市×10年×12月气象数据（840条，tianqi24.com + Open-Meteo ERA5）
- 中国-泰国-越南-缅甸-老挝FAO数据（2015-2024）
- IPCC AR6 GWP-100排放因子（23条，含44/28转换）
- 全国碳市场CEA价格（2021-07至2026-07，62个月）
- 副产物参数（文献综述，12种副产物路径）
- 中-泰-越-缅-老市场价格（90+条，含来宾27亿环保餐具产业数据）

### 模型层
- Ridge/RF/GBRT/ElasticNet 四模型竞赛 + GridSearchCV
- LOOCV交叉验证（R²=0.893）
- RepeatedKFold(5×10)稳健估计（R²=0.842±0.072）
- Stacking Ensemble（R²=0.806）
- 特征重要性（permutation importance）
- IPCC AR6 Tier 1全链条碳排放核算（种植+机械+电力+焚烧+替代+填埋+土壤碳汇）
- 多目标优化（Min-max标准化，收益70%+碳30%）
- 中-泰-越-缅-老五国跨境参数化决策

### 应用层
- Streamlit可视化系统（模型质量仪表盘 + 三方案对比 + 跨境对比）
- AI Agent自然语言决策（正则解析 + 推理链生成）
- FastAPI RESTful数据产品接口（9条路由 + API Key鉴权）
- CSV决策报告 + JSON数据产品

### 文档层
- 申报书（对齐GB/T 47950-2026、GB/T 46353-2025）
- 碳排放方法学溯源（CARBON_METHODOLOGY.md，AR6全链路）
- 数据安全文档（DATA_SECURITY.md）
- 学术对标报告（石杰锋2023等）
- 获奖项目特征自查报告（北大荒对标）
- 答辩Q&A预备文档（12题）
- 演示视频脚本（5分钟，8场景）
- 数据资产登记卡（GB/T标准对齐）
- 数据血缘可视化（HTML）
- 数据产品规格说明书