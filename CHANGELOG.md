# Changelog

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