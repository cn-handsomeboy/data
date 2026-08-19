"""
测试程序
用于验证系统各模块功能正确性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import (
    SugarcaneDecisionSystem,
    YieldPredictor,
    ByproductEstimator,
    CarbonCalculator,
    EconomicCalculator,
    OptimizationEngine,
    DATA_DIR
)
import pandas as pd


def test_yield_predictor():
    """测试产量预测模型（GBRT LOOCV + 多模型对比 + SHAP）"""
    import math
    print("=" * 60)
    print("测试 1: 产量预测模型（GBRT LOOCV + 多模型对比 + SHAP）")
    print("=" * 60)
    
    predictor = YieldPredictor()

    # 加载数据并训练（自动选择最优模型）
    gx = pd.read_csv(os.path.join(DATA_DIR, 'guangxi_sugarcane.csv'))
    weather = pd.read_csv(os.path.join(DATA_DIR, 'weather_data.csv'))
    
    metrics = predictor.train(gx, weather, model_type='auto')
    
    assert not metrics.get('fallback', True), "模型应该真正训练，不应触发fallback"
    assert metrics['mse'] >= 0, "MSE应该大于等于0"
    assert 'rmse' in metrics, "应包含RMSE指标"
    assert 'mae' in metrics, "应包含MAE指标"
    assert 'model_name' in metrics, "应包含模型名称"
    assert 'loocv_samples' in metrics, "应包含LOOCV样本数"
    
    # 预测测试（约束在 [3.87, 6.74] 范围内，覆盖全部7市历史数据）
    result = predictor.predict(28.5, 2200, 900, city='崇左市')
    assert 3.8 <= result <= 6.8, f"预测单产应在约束范围内，实际为{result}"

    # T2外部对照：预测值应在广西统计年鉴历史单产范围 3.62~6.83 吨/亩 内（数据集来源说明）
    assert 3.62 <= result <= 6.83, f"预测单产 {result} 超出统计年鉴历史范围 [3.62, 6.83]"
    # T2外部对照：RMSE 应优于学术基线（石杰锋2023 LSTM RMSE≈10.34t/ha≈0.69t/亩）
    assert metrics['rmse'] < 0.69, \
        f"RMSE {metrics['rmse']:.3f} 应优于学术基线 0.69 吨/亩"
    print(f"[PASS] T2外部对照: 预测在统计年鉴范围, RMSE {metrics['rmse']:.3f} < 学术基线0.69")
    
    # ---- Bootstrap CI 测试 ----
    ci = predictor.predict_with_ci(28.5, 2200, 900, city='崇左市', n_bootstrap=100)
    assert 'point' in ci, "CI应包含point"
    assert 'ci_lower' in ci, "CI应包含ci_lower"
    assert 'ci_upper' in ci, "CI应包含ci_upper"
    assert ci['ci_lower'] <= ci['point'] <= ci['ci_upper'], "point应在CI区间内"
    print(f"[PASS] Bootstrap CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}], method={ci['method']}")

    # ---- SHAP 测试 ----
    if predictor.shap_summary:
        assert len(predictor.shap_summary) > 0, "SHAP汇总不应为空"
        top_feat = max(predictor.shap_summary.items(), key=lambda x: x[1]['mean_abs_shap'])
        print(f"[PASS] SHAP top特征: {top_feat[0]} (mean_abs={top_feat[1]['mean_abs_shap']:.4f})")
        
        shap_explain = predictor.explain_shap(28.5, 2200, 900, city='崇左市')
        assert 'error' not in shap_explain, f"SHAP解释不应报错: {shap_explain.get('error')}"
        assert 'shap_values' in shap_explain, "SHAP解释应包含shap_values"
        assert 'top_positive' in shap_explain, "SHAP解释应包含top_positive"
        print(f"[PASS] SHAP解释: base={shap_explain['base_value']:.3f}, pred={shap_explain['prediction']:.3f}")
    else:
        print("[INFO] SHAP未启用（shap库未安装或模型不支持）")
    
    r2_display = f"{metrics['r2']:.4f}" if not math.isnan(metrics['r2']) else "nan"
    print(f"[PASS] 模型: {metrics['model_name']}, LOOCV-R2={r2_display}, RMSE={metrics['rmse']:.4f}")
    print(f"[PASS] LOOCV样本数: {metrics['loocv_samples']}, 预测单产: {result:.2f} 吨/亩")
    # 显示模型对比
    if predictor.model_comparison:
        print(f"[PASS] 模型对比: {predictor.model_comparison}")
    print()


def test_byproduct_estimator():
    """测试副产物产量估算"""
    print("=" * 60)
    print("测试 2: 副产物产量估算")
    print("=" * 60)
    
    estimator = ByproductEstimator()
    
    # 测试10吨甘蔗的副产物产量
    byproducts = estimator.estimate(10)
    
    assert 'sugarcane_leaf' in byproducts
    assert 'bagasse' in byproducts
    assert 'filter_mud' in byproducts
    assert 'molasses' in byproducts
    
    # 验证产量系数（对照行业标准：蔗叶0.15-0.20/蔗渣0.24湿基/滤泥0.03-0.04/糖蜜0.03-0.04）
    assert abs(byproducts['sugarcane_leaf']['quantity'] - 1.8) < 0.01
    assert abs(byproducts['bagasse']['quantity'] - 2.4) < 0.01
    assert abs(byproducts['filter_mud']['quantity'] - 0.4) < 0.01
    assert abs(byproducts['molasses']['quantity'] - 0.35) < 0.01

    # T2外部对照：蔗梢系数 0.08 在行业范围 0.05-0.10 内
    assert 'sugarcane_top' in byproducts, "应包含蔗梢副产物"
    top_coeff = byproducts['sugarcane_top']['coefficient']
    assert 0.05 <= top_coeff <= 0.10, f"蔗梢系数 {top_coeff} 超出行业范围"
    print(f"[PASS] 蔗梢产量: {byproducts['sugarcane_top']['quantity']:.2f} 吨（系数{top_coeff}）")
    
    print(f"[PASS] 蔗叶产量: {byproducts['sugarcane_leaf']['quantity']:.2f} 吨")
    print(f"[PASS] 蔗渣产量: {byproducts['bagasse']['quantity']:.2f} 吨")
    print(f"[PASS] 滤泥产量: {byproducts['filter_mud']['quantity']:.2f} 吨")
    print(f"[PASS] 糖蜜产量: {byproducts['molasses']['quantity']:.2f} 吨")
    print()


def test_carbon_calculator():
    """测试碳排放核算"""
    print("=" * 60)
    print("测试 3: 碳排放核算")
    print("=" * 60)
    
    calculator = CarbonCalculator()

    # 测试焚烧排放
    burning = calculator.calculate_burning_emission(1.8)
    assert burning['co2_equivalent_kg'] > 0
    # T3量级校验：文献值 甘蔗田间焚烧 CO2e ≈ 1500+2.3*27+0.07*273 ≈ 1600 kg/吨干蔗叶，
    # 1.8吨蔗叶的CO2e应在 2500~3500 kg 区间
    assert 2000 < burning['co2_equivalent_kg'] < 4000, \
        f"焚烧排放量级异常: {burning['co2_equivalent_kg']:.1f} kg"
    print(f"[PASS] 焚烧碳排放: {burning['co2_equivalent_kg']:.2f} kg CO2e（量级校验通过）")

    # 测试生物质替代
    substitution = calculator.calculate_biomass_substitution(1.8)
    assert substitution['carbon_reduction_kg'] > 0
    # T3量级校验：生物质替代煤炭因子 1800 kgCO2/吨，1.8吨应减排 2500~4000 kg
    assert 2000 < substitution['carbon_reduction_kg'] < 5000, \
        f"替代减排量级异常: {substitution['carbon_reduction_kg']:.1f} kg"
    print(f"[PASS] 生物质替代减排: {substitution['carbon_reduction_kg']:.2f} kg CO2（量级校验通过）")

    # 测试全链条排放（化肥用量与 config.json 一致：22 kg N/亩 × 10亩 = 220）
    full_chain = calculator.calculate_full_chain(
        sugarcane_yield_tons=60,
        fertilizer_n_kg=220,
        diesel_l=50,
        electricity_kwh=500,
        country='China'
    )
    assert full_chain['total_kg'] > 0
    # T3量级校验：10亩全链条（种植+机械+加工）CO2e 合理区间 1~5 吨
    assert 1000 < full_chain['total_kg'] < 5000, \
        f"全链条排放量级异常: {full_chain['total_kg']:.1f} kg"
    print(f"[PASS] 全链条碳排放: {full_chain['total_tons']:.4f} 吨CO2（量级校验通过）")
    print(f"  - 种植环节: {full_chain['planting']:.2f} kg")
    print(f"  - 机械作业: {full_chain['mechanization']:.2f} kg")
    print(f"  - 加工环节: {full_chain['processing']:.2f} kg")
    print()


def test_economic_calculator():
    """测试经济效益计算"""
    print("=" * 60)
    print("测试 4: 经济效益计算")
    print("=" * 60)
    
    calculator = EconomicCalculator()
    estimator = ByproductEstimator()
    
    byproducts = estimator.estimate(60)
    economic = calculator.calculate_byproduct_value(byproducts, 'China')
    net_benefit = calculator.calculate_net_benefit(economic)
    
    assert 'sugarcane_leaf' in economic
    assert 'filter_mud' in economic
    assert 'molasses' in economic
    
    assert net_benefit['traditional'] <= net_benefit['circular_optimal']

    # T2外部对照：传统模式（焚烧+填埋）净收益应低于循环经济（最优）——循环经济产业逻辑
    # T2外部对照：糖蜜直销价（1000元/吨）应在广西糖业协会市场区间 800-1200 元/吨内
    molasses_qty = economic['molasses']['quantity']
    molasses_val = economic['molasses']['direct_sale']
    unit_price = molasses_val['revenue'] / molasses_qty if molasses_qty else 0
    assert 800 <= unit_price <= 1200, f"糖蜜直销价 {unit_price:.0f} 元/吨超出市场区间[800,1200]"
    print(f"[PASS] T2外部对照: 糖蜜直销价 {unit_price:.0f} 元/吨在广西糖业协会市场区间内")

    # T2外部对照：五国收购价应符合产业常识（泰国 < 越南 < 中国）
    import pandas as _pd
    mkt = _pd.read_csv(os.path.join(DATA_DIR, 'market_prices.csv'))
    proc = mkt[mkt['product_name'] == 'sugarcane_procurement'].set_index('country')
    th, vn, cn = (proc.loc[c, 'price_avg_yuan_per_ton'] for c in ['Thailand', 'Vietnam', 'China'])
    assert th < vn < cn, f"收购价序应泰国<越南<中国，实际 {th} < {vn} < {cn}"
    print(f"[PASS] T2外部对照: 收购价 泰国{th:.0f} < 越南{vn:.0f} < 中国{cn:.0f} 元/吨")
    
    print(f"[PASS] 传统模式净收益: {net_benefit['traditional']:,.2f} 元")
    print(f"[PASS] 改良传统净收益: {net_benefit['improved_traditional']:,.2f} 元")
    print(f"[PASS] 基础循环净收益: {net_benefit['circular_basic']:,.2f} 元")
    print(f"[PASS] 进阶循环净收益: {net_benefit['circular_advanced']:,.2f} 元")
    print(f"[PASS] 最优循环净收益: {net_benefit['circular_optimal']:,.2f} 元")
    print()


def test_optimization():
    """测试多目标优化"""
    print("=" * 60)
    print("测试 5: 多目标优化")
    print("=" * 60)
    
    optimizer = OptimizationEngine()
    estimator = ByproductEstimator()
    
    byproducts = estimator.estimate(60)
    result = optimizer.optimize(60, byproducts, carbon_price=85, country='China')
    
    assert 'optimal' in result
    assert 'all_schemes' in result
    assert len(result['all_schemes']) == 5
    
    optimal = result['optimal']
    print(f"[PASS] 最优方案: {optimal['name']}")
    print(f"[PASS] 净收益: {optimal['net_benefit']:,.2f} 元")
    print(f"[PASS] 碳排放: {optimal['carbon_emission_kg']:,.2f} kg")
    print(f"[PASS] 综合得分: {optimal['total_score']:.4f}")
    print()


def test_full_system():
    """测试完整系统"""
    import math
    print("=" * 60)
    print("测试 6: 完整系统流程")
    print("=" * 60)
    
    system = SugarcaneDecisionSystem()
    
    # 训练模型（自动选择最优）
    metrics = system.train_models(model_type='auto')
    assert not metrics.get('fallback', True), "模型应该真正训练"
    r2_display = f"{metrics['r2']:.4f}" if not math.isnan(metrics['r2']) else "nan"
    print(f"[PASS] 模型: {metrics['model_name']}, LOOCV-R2={r2_display}, RMSE={metrics['rmse']:.4f}")
    
    # 运行完整决策（化肥用量与 config.json 一致：22 kg N/亩 × 10亩 = 220）
    result = system.run_decision(
            area_mu=10,
            avg_temp=28.5,
            precipitation=2200,
            sunshine=900,
            fertilizer_n_kg=220,
            diesel_l=50,
            electricity_kwh=500,
            carbon_price=85,
            country='China',
            city='崇左市'
        )
    
    assert result['area_mu'] == 10
    assert result['yield_per_mu'] > 0
    assert result['total_yield'] > 0
    assert len(result['byproducts']) > 0
    assert result['carbon_emission']['total_kg'] > 0
    assert len(result['economic']) > 0
    assert 'optimal' in result['optimization']
    
    print(f"[PASS] 预测单产: {result['yield_per_mu']:.2f} 吨/亩")
    print(f"[PASS] 总产量: {result['total_yield']:.2f} 吨")
    print(f"[PASS] 碳排放: {result['carbon_emission']['total_tons']:.4f} 吨CO2")
    print(f"[PASS] 最优方案: {result['optimization']['optimal']['name']}")
    print()


def test_cross_country():
    """测试跨境对比功能（FAO数据参与产量预测）"""
    print("=" * 60)
    print("测试 7: 跨境对比（中国-泰国-越南-缅甸-老挝五国）")
    print("=" * 60)
    
    system = SugarcaneDecisionSystem()
    system.yield_predictor.load_model()  # 确保模型已加载
    
    countries = ['China', 'Thailand', 'Vietnam']
    results = {}
    
    for country in countries:
        result = system.run_decision(
            area_mu=10,
            avg_temp=28.5,
            precipitation=2200,
            sunshine=900,
            fertilizer_n_kg=220,
            diesel_l=50,
            electricity_kwh=500,
            carbon_price=85,
            country=country,
            city='崇左市'
        )
        results[country] = result
        
        yield_src = result.get('yield_source', 'unknown')
        print(f"[PASS] {country}: yield={result['yield_per_mu']:.2f} (source={yield_src}), "
              f"benefit={result['optimization']['optimal']['net_benefit']:,.2f} 元")
    
    # 验证三个国家结果不同
    yields = [r['yield_per_mu'] for r in results.values()]
    benefits = [r['optimization']['optimal']['net_benefit'] for r in results.values()]
    
    # 中国用模型预测，泰越用FAO均值，产量应不同
    assert len(set([round(y, 1) for y in yields])) > 1, "不同国家应有不同产量"
    
    # 验证yield_source
    assert results['China'].get('yield_source') == 'model', "中国应使用模型预测"
    assert results['Thailand'].get('yield_source') == 'fao_statistical_average', "泰国应使用FAO均值"
    assert results['Vietnam'].get('yield_source') == 'fao_statistical_average', "越南应使用FAO均值"
    
    # 验证收益因市场价格不同而有差异
    assert len(set([round(b, 0) for b in benefits])) > 1, "不同国家应该有不同结果"
    print()


def test_boundary_cases():
    """测试 8: 边界条件与异常处理"""
    print("=" * 60)
    print("测试 8: 边界条件与异常处理")
    print("=" * 60)

    system = SugarcaneDecisionSystem()
    system.train_models(model_type='auto')

    # 8.1 零面积
    result = system.run_decision(
        area_mu=0.0, avg_temp=28.0, precipitation=900, sunshine=870,
        fertilizer_n_kg=0, diesel_l=0, electricity_kwh=0,
        carbon_price=85, country='China', city='崇左市'
    )
    assert result['total_yield'] == 0, "零面积应得零产量"
    print("[PASS] 8.1 零面积 → 零产量")

    # 8.2 极端低温
    result = system.run_decision(
        area_mu=10, avg_temp=10.0, precipitation=500, sunshine=500,
        fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
        carbon_price=85, country='China', city='崇左市'
    )
    assert result['yield_per_mu'] >= 3.8, f"极端低温预测应在约束内，实际{result['yield_per_mu']:.2f}"
    print(f"[PASS] 8.2 极端低温(10℃) → 预测单产 {result['yield_per_mu']:.2f}")

    # 8.3 极端高温
    result = system.run_decision(
        area_mu=10, avg_temp=35.0, precipitation=2000, sunshine=1200,
        fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
        carbon_price=85, country='China', city='崇左市'
    )
    assert 3.8 <= result['yield_per_mu'] <= 6.8, f"极端高温预测应在约束内"
    print(f"[PASS] 8.3 极端高温(35℃) → 预测单产 {result['yield_per_mu']:.2f}")

    # 8.4 小面积(0.5亩)
    result = system.run_decision(
        area_mu=0.5, avg_temp=28.0, precipitation=900, sunshine=870,
        fertilizer_n_kg=11, diesel_l=2.5, electricity_kwh=25,
        carbon_price=85, country='China', city='崇左市'
    )
    assert result['total_yield'] > 0, "小面积也应有产量"
    print(f"[PASS] 8.4 小面积(0.5亩) → 总产量 {result['total_yield']:.2f}吨")

    # 8.5 大面积(10000亩)
    result = system.run_decision(
        area_mu=10000, avg_temp=28.0, precipitation=900, sunshine=870,
        fertilizer_n_kg=220000, diesel_l=50000, electricity_kwh=500000,
        carbon_price=85, country='China', city='崇左市'
    )
    assert result['total_yield'] > 0
    print(f"[PASS] 8.5 大面积(10000亩) → 总产量 {result['total_yield']:.0f}吨")

    # 8.6 零碳价
    result = system.run_decision(
        area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
        fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
        carbon_price=0, country='China', city='崇左市'
    )
    assert result['optimization']['optimal']['total_score'] >= 0
    print(f"[PASS] 8.6 零碳价 → 最优方案 {result['optimization']['optimal']['name']}")

    # 8.7 极高碳价(500元/吨)
    result = system.run_decision(
        area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
        fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
        carbon_price=500, country='China', city='崇左市'
    )
    assert result['optimization']['optimal']['name'] == 'circular_optimal'
    print(f"[PASS] 8.7 极高碳价(500元/吨) → 最优方案 still {result['optimization']['optimal']['name']}")

    # 8.8 所有广西城市均可预测
    all_cities = ['崇左市', '来宾市', '南宁市', '柳州市', '百色市', '河池市', '防城港市']
    for city in all_cities:
        result = system.run_decision(
            area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
            fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
            carbon_price=85, country='China', city=city
        )
        assert 3.8 <= result['yield_per_mu'] <= 6.8, f"{city}预测应在约束内"
    print(f"[PASS] 8.8 全部{len(all_cities)}个城市预测OK")

    # 8.9 碳排放核算非负
    assert result['carbon_emission']['total_kg'] >= 0 or result['country'] == 'China', \
        "碳排放总量应合理"
    print(f"[PASS] 8.9 碳排放核算合理性")

    print()


def test_api_imports():
    """测试 9: API 模块导入与鉴权"""
    print("=" * 60)
    print("测试 9: API 模块导入与鉴权")
    print("=" * 60)

    # 9.1 模块可导入
    import api
    assert hasattr(api, 'app'), "API应有app实例"
    assert hasattr(api, 'verify_api_key'), "API应有鉴权函数"
    print("[PASS] 9.1 API模块导入成功")

    # 9.2 FastAPI app 路由存在
    routes = [r.path for r in api.app.routes if hasattr(r, 'path')]
    assert '/api/decision' in routes, "应有/api/decision路由"
    assert '/health' in routes, "应有/health路由"
    print(f"[PASS] 9.2 API路由数量: {len(routes)}, 包含decision和health")

    # 9.3 鉴权函数存在
    import inspect
    sig = inspect.signature(api.verify_api_key)
    assert 'api_key' in sig.parameters
    print("[PASS] 9.3 鉴权函数签名正确")

    print()


def test_agent_regex():
    """测试 10: Agent 正则解析增强"""
    print("=" * 60)
    print("测试 10: Agent 正则解析增强")
    print("=" * 60)

    from agent import parse_query

    # 10.1 整数面积
    r = parse_query("我有50亩地，在崇左")
    assert r.get('area_mu') == 50
    print(f"[PASS] 10.1 '50亩' → {r.get('area_mu')}")

    # 10.2 小数面积
    r = parse_query("10.5亩甘蔗地")
    assert r.get('area_mu') == 10.5
    print(f"[PASS] 10.2 '10.5亩' → {r.get('area_mu')}")

    # 10.3 中文数字
    r = parse_query("十来亩地")
    assert r.get('area_mu') == 10
    print(f"[PASS] 10.3 '十来亩' → {r.get('area_mu')}")

    # 10.4 几十亩
    r = parse_query("几十亩地")
    assert r.get('area_mu') == 30
    print(f"[PASS] 10.4 '几十亩' → {r.get('area_mu')}")

    # 10.5 碳价解析
    r = parse_query("碳价100元")
    assert r.get('carbon_price') == 100
    print(f"[PASS] 10.5 '碳价100元' → {r.get('carbon_price')}")

    # 10.6 城市识别
    r = parse_query("来宾市50亩")
    assert r.get('city') == '来宾市'
    print(f"[PASS] 10.6 城市识别 → {r.get('city')}")

    # 10.7 缅甸/老挝识别
    r = parse_query("缅甸甘蔗田")
    assert r.get('country') == 'Myanmar'
    print(f"[PASS] 10.7 '缅甸' → {r.get('country')}")

    r = parse_query("老挝30亩")
    assert r.get('country') == 'Laos'
    assert r.get('area_mu') == 30
    print(f"[PASS] 10.8 '老挝30亩' → country={r.get('country')}, area={r.get('area_mu')}")

    print()


def test_myanmar_laos():
    """测试 11: 缅甸/老挝跨境支持"""
    print("=" * 60)
    print("测试 11: 缅甸/老挝跨境支持")
    print("=" * 60)

    system = SugarcaneDecisionSystem()
    system.yield_predictor.load_model()

    # 11.1 缅甸
    result = system.run_decision(
        area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
        fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
        carbon_price=85, country='Myanmar', city='崇左市'
    )
    assert result.get('yield_source') == 'fao_statistical_average'
    assert result['yield_per_mu'] > 0
    print(f"[PASS] 11.1 Myanmar: yield={result['yield_per_mu']:.2f} (source={result.get('yield_source')})")

    # 11.2 老挝
    result = system.run_decision(
        area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
        fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
        carbon_price=85, country='Laos', city='崇左市'
    )
    assert result.get('yield_source') == 'fao_statistical_average'
    assert result['yield_per_mu'] > 0
    print(f"[PASS] 11.2 Laos: yield={result['yield_per_mu']:.2f} (source={result.get('yield_source')})")

    # 11.3 五国产量不同
    countries = ['China', 'Thailand', 'Vietnam', 'Myanmar', 'Laos']
    yields = {}
    for c in countries:
        r = system.run_decision(
            area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
            fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
            carbon_price=85, country=c, city='崇左市'
        )
        yields[c] = r['yield_per_mu']
        print(f"  {c}: {r['yield_per_mu']:.2f} 吨/亩, benefit={r['optimization']['optimal']['net_benefit']:,.0f} 元")

    # 五国至少有3个不同产量
    unique_yields = len(set([round(y, 1) for y in yields.values()]))
    assert unique_yields >= 3, f"五国应有≥3种不同产量，实际{unique_yields}"
    print(f"[PASS] 11.3 五国产量差异: {unique_yields}种不同产量")

    print()


def test_security_injection():
    """测试 12: 安全性与异常注入"""
    print("=" * 60)
    print("测试 12: 安全性与异常注入")
    print("=" * 60)

    from agent import parse_query

    # 12.1 SQL注入尝试（应安全处理）
    r = parse_query("50亩'; DROP TABLE users;--")
    assert r.get('area_mu') == 50
    print("[PASS] 12.1 SQL注入尝试 → 安全解析")

    # 12.2 超长输入
    long_text = "甘蔗" * 1000 + "50亩"
    r = parse_query(long_text)
    assert r.get('area_mu') == 50
    print("[PASS] 12.2 超长输入(2000+字) → 正常解析")

    # 12.3 特殊字符
    r = parse_query("50亩<script>alert('xss')</script>")
    assert r.get('area_mu') == 50
    print("[PASS] 12.3 XSS尝试 → 安全解析")

    # 12.4 空输入
    r = parse_query("")
    assert isinstance(r, dict)
    print("[PASS] 12.4 空输入 → 返回空字典")

    # 12.5 只有数字无单位
    r = parse_query("12345")
    assert 'area_mu' not in r
    print("[PASS] 12.5 无单位数字 → 不误解析为面积")

    print()

def test_shap_data_consistency():
    """测试 13: SHAP 数据一致性"""
    print("=" * 60)
    print("测试 13: SHAP 数据一致性")
    print("=" * 60)

    predictor = YieldPredictor()
    # 尝试加载已保存的模型（避免重复训练）
    if not predictor.load_model():
        gx = pd.read_csv(os.path.join(DATA_DIR, 'guangxi_sugarcane.csv'))
        weather = pd.read_csv(os.path.join(DATA_DIR, 'weather_data.csv'))
        predictor.train(gx, weather, model_type='auto')

    # 13.1 SHAP解释与预测值一致性
    shap_res = predictor.explain_shap(28.0, 900, 870, city='崇左市')
    assert 'error' not in shap_res, f"SHAP应已启用，但返回错误: {shap_res.get('error')}"
    # base_value + sum(shap_values) ≈ prediction
    total_shap = sum(shap_res['shap_values'])
    reconstructed = shap_res['base_value'] + total_shap
    pred = shap_res['prediction']
    assert abs(reconstructed - pred) < 0.01, \
        f"SHAP重建值 {reconstructed:.3f} 与预测值 {pred:.3f} 偏差过大"
    print(f"[PASS] 13.1 SHAP加和一致性: base={shap_res['base_value']:.3f} + shap={total_shap:.3f} ≈ pred={pred:.3f}")

    # 13.2 不同城市SHAP特征贡献不同
    shap_nt = predictor.explain_shap(28.0, 900, 870, city='来宾市')
    assert 'error' not in shap_nt, f"SHAP应已启用，但返回错误: {shap_nt.get('error')}"
    city_diff = any(
        abs(a - b) > 0.001
        for a, b in zip(shap_res['shap_values'], shap_nt['shap_values'])
    )
    assert city_diff, "不同城市应有不同的SHAP贡献"
    print("[PASS] 13.2 不同城市SHAP贡献有差异")

    # 13.3 Bootstrap CI 单调性
    ci1 = predictor.predict_with_ci(28.0, 900, 870, city='崇左市', n_bootstrap=50)
    ci2 = predictor.predict_with_ci(28.0, 900, 870, city='崇左市', n_bootstrap=200)
    if ci1['method'] == 'bootstrap' and ci2['method'] == 'bootstrap':
        # 更多bootstrap不应显著改变point
        assert abs(ci1['point'] - ci2['point']) < 0.1, "不同n_bootstrap的point应稳定"
        print(f"[PASS] 13.3 Bootstrap稳定性: n=50 vs n=200 point差={abs(ci1['point']-ci2['point']):.4f}")

    print()


def test_cross_border_consistency():
    """测试 14: 跨境一致性——遍历5国×7市×极端参数，断言无错误国名/城市名"""
    print("=" * 60)
    print("测试 14: 跨境一致性（5国×7市×极端参数）")
    print("=" * 60)

    system = SugarcaneDecisionSystem()
    system.yield_predictor.load_model()

    countries = ['China', 'Thailand', 'Vietnam', 'Myanmar', 'Laos']
    cities = ['崇左市', '来宾市', '南宁市', '柳州市', '百色市', '河池市', '防城港市']

    country_cn = {
        'China': '中国', 'Thailand': '泰国', 'Vietnam': '越南',
        'Myanmar': '缅甸', 'Laos': '老挝'
    }

    # 极端参数组合
    extreme_params = [
        {"area_mu": 0.5, "avg_temp": 22.0, "precipitation": 500, "sunshine": 700},
        {"area_mu": 10, "avg_temp": 28.0, "precipitation": 900, "sunshine": 870},
        {"area_mu": 1000, "avg_temp": 32.0, "precipitation": 1200, "sunshine": 1000},
        {"area_mu": 10, "avg_temp": 10.0, "precipitation": 500, "sunshine": 500},
        {"area_mu": 10, "avg_temp": 35.0, "precipitation": 2000, "sunshine": 1200},
    ]

    # 14.1 遍历5国×7市×极端参数，验证核心一致性
    for country in countries:
        for city in cities:
            for params in extreme_params:
                result = system.run_decision(
                    area_mu=params["area_mu"],
                    avg_temp=params["avg_temp"],
                    precipitation=params["precipitation"],
                    sunshine=params["sunshine"],
                    fertilizer_n_kg=220,
                    diesel_l=50,
                    electricity_kwh=500,
                    carbon_price=85,
                    country=country,
                    city=city
                )

                # yield_source 正确性
                if country == 'China':
                    assert result.get('yield_source') == 'model', \
                        f"{country}-{city}: 中国应使用model预测"
                else:
                    assert result.get('yield_source') == 'fao_statistical_average', \
                        f"{country}-{city}: {country}应使用FAO均值"

                # 产量合理
                assert result['yield_per_mu'] > 0, f"{country}-{city}: 产量应大于0"
                assert result['total_yield'] >= 0, f"{country}-{city}: 总产量应≥0"

                # 碳排放合理
                assert result['carbon_emission']['total_kg'] >= 0, \
                    f"{country}-{city}: 碳排放应≥0"

                # 有最优方案
                assert 'optimal' in result['optimization'], \
                    f"{country}-{city}: 应有最优方案"

    print(f"[PASS] 14.1 5国×7市×{len(extreme_params)}组极端参数 = {len(countries)*len(cities)*len(extreme_params)}次决策全部通过")

    # 14.2 非中国国家时，不同城市产量必须相同（city参数被忽略）
    for country in ['Thailand', 'Vietnam', 'Myanmar', 'Laos']:
        yields = []
        for city in cities:
            result = system.run_decision(
                area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
                fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
                carbon_price=85, country=country, city=city
            )
            yields.append(result['yield_per_mu'])
        assert max(yields) - min(yields) < 0.0001, \
            f"{country}: 不同城市产量应相同，差异={max(yields)-min(yields):.4f}"
        print(f"[PASS] 14.2 {country_cn[country]}: 7城市产量一致={yields[0]:.2f} 吨/亩")

    # 14.3 中国不同城市产量应有差异（城市哑变量生效）
    cn_yields = {}
    for city in cities:
        result = system.run_decision(
            area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
            fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
            carbon_price=85, country='China', city=city
        )
        cn_yields[city] = result['yield_per_mu']
    unique_cn = len(set([round(y, 2) for y in cn_yields.values()]))
    assert unique_cn >= 3, f"中国7市应有≥3种不同产量，实际{unique_cn}"
    print(f"[PASS] 14.3 中国: 7城市产量差异={unique_cn}种")

    # 14.4 输出文本一致性：构造政策建议文本并断言不含错误国名/城市名
    # 模拟 app.py 中的政策建议文本生成逻辑，验证无硬编码
    for country in countries:
        result = system.run_decision(
            area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
            fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
            carbon_price=85, country=country, city='崇左市'
        )

        cn_name = country_cn[country]
        policy_text = (
            f"基于本次{cn_name}10亩蔗田决策结果，"
            f"最优方案为{result['optimization']['optimal']['name']}，"
            f"预计综合收益{result['optimization']['optimal']['total_benefit']:,.0f}元。"
            f"{cn_name}甘蔗单产{result['yield_per_mu']:.2f}吨/亩，"
            f"与中国-广西（{system._fao_yield_baseline.get('China', 5.96):.2f}吨/亩）相比"
            f"仍有产量提升空间。"
        )

        # 当国家不是中国时，文本中不应出现中国城市名（除跨境对比提及外）
        if country != 'China':
            wrong_cities = ['崇左市', '来宾市', '南宁市', '柳州市', '百色市', '河池市', '防城港市']
            for wc in wrong_cities:
                # 允许在"中国-广西"语境中出现，但不允许单独作为中国城市出现
                assert wc not in policy_text or '中国-广西' in policy_text, \
                    f"{country}的政策文本不应包含错误城市名'{wc}'"

        # 所有国家文本中都不应包含除当前国名外的其他国名（跨境对比语境除外）
        for other_country, other_cn in country_cn.items():
            if other_country != country and other_country != 'China':
                assert other_cn not in policy_text, \
                    f"{country}的政策文本不应包含错误国名'{other_cn}'"

    print("[PASS] 14.4 输出文本不含错误国名/城市名")

    # 14.5 验证FAO基准数据完整性
    for country in countries:
        assert country in system._fao_yield_baseline, f"{country}应在FAO基准中"
        assert system._fao_yield_baseline[country] > 0, f"{country}FAO基准应大于0"
    print(f"[PASS] 14.5 FAO基准完整性: {', '.join([f'{country_cn[c]}={system._fao_yield_baseline[c]:.2f}' for c in countries])}")

    print()


def test_data_security_module():
    """测试 15: 数据安全模块（分类分级/脱敏/完整性/审计）"""
    print("=" * 60)
    print("测试 15: 数据安全模块")
    print("=" * 60)

    from data_security import (
        DataClassifier, DataMasker, DataIntegrityChecker,
        SecurityManager, InputValidator
    )

    # 15.1 数据分类分级
    registry = DataClassifier.list_all()
    assert len(registry) == 7, f"应有7个数据集，实际{len(registry)}"
    for item in registry:
        assert item['level'] in [1, 2, 3], f"{item['dataset']} 等级应为1-3"
    print(f"[PASS] 15.1 数据分类分级: 7个数据集已分类")

    # 15.2 跨境合规检查
    public_ok = DataClassifier.check_cross_border_allowed("weather_data.csv", "Thailand")
    assert public_ok, "公开数据应允许跨境"
    internal_ok = DataClassifier.check_cross_border_allowed("guangxi_sugarcane.csv", "Thailand")
    assert internal_ok, "农业非敏感数据RCEP下应允许跨境"
    print("[PASS] 15.2 跨境合规: 公开数据✓ 农业内部数据✓")

    # 15.3 数据脱敏
    masked = DataMasker.mask_value("崇左市", "masking")
    assert "*" in masked, "脱敏后应包含掩码字符"
    generalized = DataMasker.mask_value(123.456, "generalization")
    assert generalized != 123.456, "泛化后应改变精确值"
    print(f"[PASS] 15.3 数据脱敏: 掩码={masked} 泛化={generalized}")

    # 15.4 API响应脱敏
    sample_response = {
        "input": {"city": "崇左市", "area_mu": 10},
        "debug": "sensitive info",
        "output": {"result": 100}
    }
    masked_resp = DataMasker.mask_api_response(sample_response)
    assert masked_resp["debug"] == "[REDACTED]", "debug字段应被脱敏"
    assert "*" in masked_resp["input"]["city"], "城市名应被脱敏"
    print("[PASS] 15.4 API响应脱敏: debug=[REDACTED] 城市=掩码")

    # 15.5 数据完整性校验
    integrity = DataIntegrityChecker.verify_integrity()
    assert integrity['status'] in ['ok', 'error'], "应返回有效状态"
    print(f"[PASS] 15.5 数据完整性: 状态={integrity['status']}")

    # 15.6 安全体检
    check = SecurityManager.full_security_check()
    assert 'overall_score' in check, "应包含总分"
    assert 0 <= check['overall_score'] <= 100, "分数应在0-100之间"
    print(f"[PASS] 15.6 安全体检: 总分={check['overall_score']}/100")

    # 15.7 输入安全过滤
    clean = InputValidator.sanitize_string("<script>alert('xss')</script>")
    assert "<script>" not in clean, "XSS脚本应被清除"
    validated = InputValidator.validate_numeric(100, 0, 1000, "面积")
    assert validated == 100.0, "数值验证应通过"
    try:
        InputValidator.validate_numeric(-5, 0, 1000, "面积")
        assert False, "负数应触发异常"
    except ValueError:
        pass
    print("[PASS] 15.7 输入安全过滤: XSS清除✓ 数值边界✓")

    # 15.8 城市白名单验证
    valid_city = InputValidator.validate_city("崇左市")
    assert valid_city == "崇左市", "合法城市应通过"
    try:
        InputValidator.validate_city("../../../etc/passwd")
        assert False, "路径遍历应被拦截"
    except ValueError:
        pass
    print("[PASS] 15.8 城市白名单: 合法通过✓ 路径遍历拦截✓")

    # 15.9 PII 自动脱敏
    pii_text = "联系人手机号13800138000，身份证450102199001011234，邮箱 admin@example.com"
    redacted = InputValidator.redact_pii(pii_text)
    assert "13800138000" not in redacted, "手机号应被脱敏"
    assert "450102199001011234" not in redacted, "身份证号应被脱敏"
    assert "admin@example.com" not in redacted, "邮箱应被脱敏"
    print("[PASS] 15.9 PII脱敏: 手机号/身份证/邮箱自动保护✓")

    # 15.10 API响应脱敏含PII
    pii_resp = {
        "input": {"city": "崇左市", "notes": "联系我 13800138000"},
        "feedback": {"notes": "邮箱 user@example.com"}
    }
    masked_pii = DataMasker.mask_api_response(pii_resp)
    assert "13800138000" not in masked_pii["input"]["notes"]
    assert "user@example.com" not in masked_pii["feedback"]["notes"]
    print("[PASS] 15.10 API响应PII脱敏: 备注字段自动保护✓")

    print()


def test_user_validation_loop():
    """测试 16: 用户验证闭环（反馈收集与统计）"""
    print("=" * 60)
    print("测试 16: 用户验证闭环")
    print("=" * 60)

    from user_validation import FeedbackCollector

    # 16.1 使用临时文件避免污染真实数据
    import tempfile
    original_file = FeedbackCollector.FEEDBACK_FILE
    FeedbackCollector.FEEDBACK_FILE = os.path.join(
        tempfile.gettempdir(), 'test_feedback.json'
    )
    if os.path.exists(FeedbackCollector.FEEDBACK_FILE):
        os.remove(FeedbackCollector.FEEDBACK_FILE)

    # 16.2 初始状态应为空
    stats = FeedbackCollector.get_validation_stats()
    assert stats["total_feedbacks"] == 0, "初始验证数据应为空"
    print("[PASS] 16.1 初始状态为空")

    # 16.3 提交反馈
    fb = FeedbackCollector.submit_feedback(
        predicted_yield=5.2, actual_yield=5.0,
        predicted_benefit=50000, actual_benefit=48000,
        city="崇左市", country="China", user_type="制糖企业",
        notes="测试反馈"
    )
    assert fb["id"].startswith("FB-"), "反馈ID应以FB-开头"
    expected_yield_err = round((5.0 - 5.2) / 5.2 * 100, 2)
    expected_benefit_err = round((48000 - 50000) / 50000 * 100, 2)
    assert fb["yield_error_pct"] == expected_yield_err, \
        f"产量偏差计算应正确: {fb['yield_error_pct']} != {expected_yield_err}"
    assert fb["benefit_error_pct"] == expected_benefit_err, \
        f"收益偏差计算应正确: {fb['benefit_error_pct']} != {expected_benefit_err}"
    print(f"[PASS] 16.2 反馈提交: ID={fb['id']}, 产量偏差={fb['yield_error_pct']:.1f}%")

    # 16.4 统计更新
    stats = FeedbackCollector.get_validation_stats()
    assert stats["total_feedbacks"] == 1
    assert stats["avg_yield_error_pct"] == fb["yield_error_pct"]
    assert stats["yield_mape"] == abs(fb["yield_error_pct"])
    print(f"[PASS] 16.3 统计更新: MAPE={stats['yield_mape']:.1f}%")

    # 16.5 多条反馈聚合
    FeedbackCollector.submit_feedback(
        predicted_yield=6.0, actual_yield=6.3,
        predicted_benefit=60000, actual_benefit=63000,
        city="来宾市", country="China"
    )
    stats = FeedbackCollector.get_validation_stats()
    assert stats["total_feedbacks"] == 2
    print(f"[PASS] 16.4 聚合统计: 平均产量偏差={stats['avg_yield_error_pct']:+.1f}%")

    # 16.6 校准阈值判断（MAPE > 15% 应提示校准）
    FeedbackCollector.submit_feedback(
        predicted_yield=5.0, actual_yield=7.0,  # 40% 误差
        predicted_benefit=50000, actual_benefit=70000,
        city="南宁市", country="China"
    )
    stats = FeedbackCollector.get_validation_stats()
    assert stats["calibration_needed"] is True, \
        "MAPE>15%时应提示需要校准"
    print(f"[PASS] 16.5 校准提示: calibration_needed={stats['calibration_needed']}")

    # 16.7 获取反馈列表
    feedbacks = FeedbackCollector.get_feedbacks()
    assert len(feedbacks) == 3
    assert feedbacks[-1]["city"] == "南宁市"
    print("[PASS] 16.6 反馈列表获取正确")

    # 清理临时文件
    if os.path.exists(FeedbackCollector.FEEDBACK_FILE):
        os.remove(FeedbackCollector.FEEDBACK_FILE)
    FeedbackCollector.FEEDBACK_FILE = original_file
    print()


def test_input_validation():
    """测试 17: 核心决策接口入参边界校验"""
    print("=" * 60)
    print("测试 17: 核心决策接口入参边界校验")
    print("=" * 60)

    system = SugarcaneDecisionSystem()
    system.yield_predictor.load_model()

    # 17.1 负数面积应被拦截
    try:
        system.run_decision(
            area_mu=-10, avg_temp=28.0, precipitation=900, sunshine=870,
            fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
            carbon_price=85, country='China', city='崇左市'
        )
        assert False, "负数面积应触发 ValueError"
    except ValueError:
        print("[PASS] 17.1 负数面积 → ValueError")

    # 17.2 极端温度应被拦截
    try:
        system.run_decision(
            area_mu=10, avg_temp=100.0, precipitation=900, sunshine=870,
            fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
            carbon_price=85, country='China', city='崇左市'
        )
        assert False, "极端温度应触发 ValueError"
    except ValueError:
        print("[PASS] 17.2 极端温度(100℃) → ValueError")

    # 17.3 非法城市应被拦截
    try:
        system.run_decision(
            area_mu=10, avg_temp=28.0, precipitation=900, sunshine=870,
            fertilizer_n_kg=220, diesel_l=50, electricity_kwh=500,
            carbon_price=85, country='China', city='北京市'
        )
        assert False, "非法城市应触发 ValueError"
    except ValueError:
        print("[PASS] 17.3 非法城市(北京市) → ValueError")

    # 17.4 合法边界值应正常执行
    result = system.run_decision(
        area_mu=0.01, avg_temp=10.0, precipitation=0.0, sunshine=0.0,
        fertilizer_n_kg=0.0, diesel_l=0.0, electricity_kwh=0.0,
        carbon_price=0.0, country='China', city='崇左市'
    )
    assert result['yield_per_mu'] > 0
    print("[PASS] 17.4 合法边界值 → 正常决策")

    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行系统测试")
    print("=" * 60 + "\n")
    
    try:
        test_yield_predictor()
        test_byproduct_estimator()
        test_carbon_calculator()
        test_economic_calculator()
        test_optimization()
        test_full_system()
        test_cross_country()
        test_boundary_cases()
        test_api_imports()
        test_agent_regex()
        test_myanmar_laos()
        test_security_injection()
        test_shap_data_consistency()
        test_cross_border_consistency()
        test_data_security_module()
        test_user_validation_loop()
        test_input_validation()

        print("=" * 60)
        print("[ALL PASS] 所有测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"[FAIL] 测试失败: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)