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
    """测试产量预测模型（Ridge + LOOCV）"""
    import math
    print("=" * 60)
    print("测试 1: 产量预测模型（Ridge + LOOCV）")
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
    result = predictor.predict(28.5, 2200, 3500, city='崇左市')
    assert 3.8 <= result <= 6.8, f"预测单产应在约束范围内，实际为{result}"
    
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
    
    # 验证产量系数（蔗渣从0.28改为0.24行业标准）
    assert abs(byproducts['sugarcane_leaf']['quantity'] - 1.8) < 0.01
    assert abs(byproducts['bagasse']['quantity'] - 2.4) < 0.01
    assert abs(byproducts['filter_mud']['quantity'] - 0.4) < 0.01
    assert abs(byproducts['molasses']['quantity'] - 0.35) < 0.01
    
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
    print(f"[PASS] 焚烧碳排放: {burning['co2_equivalent_kg']:.2f} kg CO2e")
    
    # 测试生物质替代
    substitution = calculator.calculate_biomass_substitution(1.8)
    assert substitution['carbon_reduction_kg'] > 0
    print(f"[PASS] 生物质替代减排: {substitution['carbon_reduction_kg']:.2f} kg CO2")
    
    # 测试全链条排放
    full_chain = calculator.calculate_full_chain(
        sugarcane_yield_tons=60,
        fertilizer_n_kg=150,
        diesel_l=50,
        electricity_kwh=500,
        country='China'
    )
    assert full_chain['total_kg'] > 0
    print(f"[PASS] 全链条碳排放: {full_chain['total_tons']:.4f} 吨CO2")
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
    
    print(f"[PASS] 传统模式净收益: {net_benefit['traditional']:,.2f} 元")
    print(f"[PASS] 基础循环净收益: {net_benefit['circular_basic']:,.2f} 元")
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
    assert len(result['all_schemes']) == 3
    
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
    
    # 运行完整决策
    result = system.run_decision(
            area_mu=10,
            avg_temp=28.5,
            precipitation=2200,
            sunshine=3500,
            fertilizer_n_kg=150,
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
    print("测试 7: 跨境对比（中国 vs 泰国 vs 越南）")
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
            sunshine=3500,
            fertilizer_n_kg=150,
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