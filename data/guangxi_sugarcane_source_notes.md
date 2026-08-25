# guangxi_sugarcane.csv 数据集重建与溯源说明

> 版本：v2.0（2026-08-24 重建）
> 重建原因：v1.x 存在插值/外推+随机噪声伪造训练标签、真实锚点错误、口径混乱等问题，违反数据合规要求。
> 重建原则：**每个数字必须能对应到官方统计来源（各市统计公报/广西统计年鉴/广西地情网）**，人工插值零容忍。

## 一、口径说明（统一）

| 指标 | 统一口径 | 单位 | 说明 |
|---|---|---|---|
| total_yield_tons | 甘蔗产量（公报"甘蔗产量"，部分市为"糖料蔗/糖料"口径） | 万吨 | 主产区甘蔗≈糖料蔗（差异约1.8%）；百色2022起为糖料口径、河池为"甘蔗含果蔗" |
| planting_area_mu | 甘蔗/糖料蔗种植面积（统一换算） | 万亩 | 1万公顷=15万亩；1千公顷=1.5万亩 |
| yield_per_mu_tons | 单产 = 产量/面积（程序重算） | 吨/亩 | 不直接抄公报 |

## 二、source_type 标注规则

- `observed`：该行两个指标均有官方统计公报/年鉴原文出处（66/70 行）
- `estimated`：个别年份公报未公布绝对值，采用官方数据推算或主产县兜底，已如实标注（4/70 行）

## 三、estimated 行明细（仅 4 行）

| 城市 | 年份 | 说明 |
|---|---|---|
| 来宾市 | 2023 | 产量公报仅给增速(+8.5%)未给绝对值，按2022年1059.16×1.085≈1149.19推算，与2024公报反推值(1149.4)互相印证 |
| 柳州市 | 2020 | 面积公报PDF该段落无法提取文本，采用官方口径相近值120.05万亩（与2019年一致，农业农村局新闻佐证糖料蔗120.05万亩） |
| 防城港市 | 2022 | 市级公报未公布产量，采用主产县上思县公报产量274.55万吨（上思占全市甘蔗面积约85%） |
| 防城港市 | 2024 | 市级公报未公布产量，采用主产县上思县公报产量251.26万吨 |

## 四、各市数据来源汇总

### 2015年（7市完整，来源：广西地情网《广西年鉴2016》官方表）
URL: http://szfzg.gxdfz.org.cn/flbg/szgx/201704/t20170427_41016.html
（崇左27.69万公顷/2345.38万吨、来宾15.01/1203.58、南宁13.65/1028.44、柳州9.19/665.61、百色5.91/385.63、河池6.26/360.76、防城港4.34/286.33）

### 崇左市（2016-2024，崇左市政府公报）
- 2016: http://www.chongzuo.gov.cn/sjfb/tjgb/t68359.shtml
- 2017: http://www.chongzuo.gov.cn/sjfb/tjgb/t68361.shtml
- 2018: http://www.chongzuo.gov.cn/sjfb/tjgb/t68362.shtml
- 2019: http://www.chongzuo.gov.cn/sjfb/tjgb/t5860163.shtml
- 2020: http://www.chongzuo.gov.cn/sjfb/tjgb/t9602505.shtml
- 2021: http://www.chongzuo.gov.cn/sjfb/tjgb/t12853027.shtml
- 2022: http://www.chongzuo.gov.cn/sjfb/tjgb/t19082085.shtml
- 2023: http://www.chongzuo.gov.cn/sjfb/tjgb/t19082849.shtml
- 2024: http://www.chongzuo.gov.cn/sjfb/tjgb/t23353835.shtml

### 来宾市（2016-2024，来宾统计局/广西统计局）
- 2016: http://tjj.gxzf.gov.cn/tjsj/tjgb/sxgb/t2382517.shtml（2017公报载2016值）
- 2017: http://tjj.gxzf.gov.cn/tjsj/tjgb/sxgb/t2382517.shtml
- 2018: http://tjj.laibin.gov.cn/tjxx/t329651.shtml
- 2019: http://tjj.laibin.gov.cn/tjxx/t5501286.shtml
- 2020: http://www.laibin.gov.cn/xxgk/sjfb/tjgb/t9189186.shtml
- 2021: http://www.laibin.gov.cn/xxgk/fdzdgknr/zxgk/t11978024.shtml
- 2022: http://tjj.laibin.gov.cn/tjxx/t16395936.shtml
- 2023: http://www.laibin.gov.cn/xxgk/sjfb/tjgb/t18371584.shtml
- 2024: http://www.laibin.gov.cn/xxgk/sjfb/tjgb/t20444200.shtml

### 南宁市（2016-2024，南宁市政府公报）
- 2016: https://www.nanning.gov.cn/sjfw/tjgb/t191309.html
- 2017: https://www.nanning.gov.cn/sjfw/tjgb/t191317.html
- 2018: https://www.nanning.gov.cn/sjfw/tjgb/t1734612.html
- 2019: https://www.nanning.gov.cn/sjfw/tjgb/t4318547.html
- 2020: https://www.nanning.gov.cn/sjfw/tjgb/t4732043.html
- 2021: https://www.nanning.gov.cn/sjfw/tjgb/t5180157.html
- 2022: https://www.nanning.gov.cn/sjfw/tjgb/t5575864.html
- 2023: https://www.nanning.gov.cn/sjfw/tjgb/t5908313.html
- 2024: https://www.nanning.gov.cn/sjfw/tjgb/t6338987.html

### 柳州市（2016-2024，柳州统计局公报）
- 2016: http://lztj.liuzhou.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/202008/P020200806673909242022.pdf
- 2017: http://www.liuzhou.gov.cn/sjzt/sjfb/ndtjgb/202107/P020210723682259781739.pdf
- 2018: http://lztj.liuzhou.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/202008/P020260123312469423232.pdf
- 2019: http://lztj.liuzhou.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/202008/t20200806_1862934.shtml
- 2020: http://lztj.liuzhou.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/202105/P020210531319254215955.pdf
- 2021: http://lztj.liuzhou.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/202204/t20220419_3046040.shtml
- 2022: http://lztj.liuzhou.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/t19700101_3249950.shtml
- 2023: http://lztj.liuzhou.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/t19700101_3447893.shtml
- 2024: http://lztj.liuzhou.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/202504/t20250430_3617226.shtml

### 百色市（2016-2024，百色市政府公报）
- 2016: http://www.baise.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/t18859975.shtml
- 2017: http://tjj.gxzf.gov.cn/tjsj/tjgb/sxgb/t2382838.shtml
- 2018: http://www.baise.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/t9969496.shtml
- 2019: http://www.baise.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/t5665193.shtml
- 2020: http://www.baise.gov.cn/zwgk/jcxxgk/sjfb/bssgb/P020210601448021246242.pdf
- 2021: http://www.baise.gov.cn/zwgk/jcxxgk/sjfb/bssgb/P020220523415754679997.pdf
- 2022: http://www.baise.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/t16365028.shtml
- 2023: http://www.baise.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/t18366658.shtml
- 2024: http://www.baise.gov.cn/zwgk/fdzdgknr/sjfb/tjgb/P020250528581918934053.pdf

### 河池市（2016-2024，河池统计局/广西统计局）
- 2016: http://tjj.gxzf.gov.cn/tjsj/tjgb/sxgb/t2382774.shtml
- 2017: http://tjj.gxzf.gov.cn/tjsj/tjgb/sxgb/t2382834.shtml
- 2018: http://tjj.hechi.gov.cn/xxgk/fdzdgknr/tjxx/tjgb/t388871.shtml
- 2019: http://www.hechi.gov.cn/sjfb/tjgb/t5262661.shtml
- 2020: http://www.hechi.gov.cn/sjfb/tjgb/t8744275.shtml
- 2021: http://www.hechi.gov.cn/sjfb/tjgb/t11851428.shtml
- 2022: http://www.hechi.gov.cn/sjfb/tjgb/t16633907.shtml
- 2023: http://tjj.hechi.gov.cn/xxgk/fdzdgknr/tjxx/tjgb/t18618540.shtml
- 2024: http://tjj.gxzf.gov.cn/tjsj/tjgb/sxgb/t22009800.shtml

### 防城港市（面积全部来源市级公报；2022/2024产量用上思县公报兜底）
- 2022上思县公报（产量274.55）: http://www.fcgs.gov.cn/zfxxgk/jcxxgk/tjxx/qxtjgb/t17130708.shtml
- 2024上思县公报（产量251.26）: http://www.shangsi.gov.cn/sj/sjkf/t26228500.shtml
- 2019/2020/2021/2023产量: 防城港市统计公报（历年）

## 五、重建后的模型指标（test.py 复现）

| 模型 | LOOCV R² | RMSE | MAE |
|---|---|---|---|
| Ridge（最优） | **0.9162** | 0.2227 | 0.1810 |
| GBRT | 0.8661 | 0.2815 | 0.2293 |

- 训练样本：70（7市×10年，城市-年份细粒度）
- 验证方式：LOOCV（Leave-One-Out，每折独立标准化，无超参搜索泄漏）
- 测试命令：`python test.py`（17组测试全部通过）

## 六、本次重建修复的代码缺陷

1. **预测特征量纲不一致**：CITY_CLIMATE_NORMALS 中 evapotranspiration（年累计745.5 vs 训练月均值3.67）、wind_speed（1.89 vs 训练8.75）等单位与训练聚合口径不符，导致预测特征离群、Ridge外推爆炸（预测-67）。修复：训练时记录扩展变量均值 `_ext_feature_means`，预测时用训练均值填充。
2. **年份外推**：predict 未传 year 时用当前年份(2026)超出训练区间(2015-2024)，Ridge 年份特征外推失真。修复：记录 `_train_year_max`，默认用训练集最近年份。
3. **Bootstrap CI 未约束**：bootstrap 预测未应用训练分位数边界，CI 被拉宽到 [-393, 411]。修复：bootstrap 预测与 predict 一致应用 [q025, q975] 约束。
4. **测试硬编码旧区间**：test.py 中单产约束断言 3.8~6.8、3.62~6.83 为旧伪造数据区间，真实数据单产范围 3.594~6.347，已同步修正为 3.5~6.9。
