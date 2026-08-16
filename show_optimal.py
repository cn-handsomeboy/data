"""
输出完整的最优方案详情报告
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from models import SugarcaneDecisionSystem


def main():
    system = SugarcaneDecisionSystem()
    system.yield_predictor.load_model()

    # 运行决策
    result = system.run_decision(
        area_mu=10,
        avg_temp=28.5,
        precipitation=2200,
        sunshine=900,
        fertilizer_n_kg=150,
        diesel_l=50,
        electricity_kwh=500,
        carbon_price=85,
        country='China'
    )

    scheme_name_cn = {
        'traditional': '传统模式（焚烧+填埋+直接出售）',
        'circular_basic': '基础循环（还田+有机肥+直接出售）',
        'circular_optimal': '最优循环（生物质颗粒+有机肥+深加工）'
    }
    byproduct_name_cn = {
        'sugarcane_leaf': '蔗叶',
        'bagasse': '蔗渣',
        'filter_mud': '滤泥',
        'molasses': '糖蜜',
        'sugarcane_top': '蔗梢'
    }

    print("=" * 70)
    print("         面向中国-东盟的甘蔗副产物循环经济跨境数据协同决策系统")
    print("                      【 最 优 方 案 完 整 报 告 】")
    print("=" * 70)

    # 一、基础参数
    print("\n【一、基础参数】")
    print(f"  国家/地区：中国-广西")
    print(f"  种植面积：{result['area_mu']:.0f} 亩")
    print(f"  年平均气温：28.5 ℃")
    print(f"  年降水量：2,200 mm")
    print(f"  年日照时数：3,500 h")
    print(f"  氮肥用量：150 kg N")
    print(f"  柴油用量：50 L")
    print(f"  用电量：500 kWh")
    print(f"  碳价：85 元/吨CO2")

    # 二、产量预测
    print("\n【二、产量预测】")
    print(f"  预测单产：{result['yield_per_mu']:.2f} 吨/亩")
    print(f"  总产量：{result['total_yield']:.2f} 吨")

    # 三、副产物产量估算
    print("\n【三、副产物产量估算】")
    for bp_name, bp_data in result['byproducts'].items():
        name = byproduct_name_cn.get(bp_name, bp_name)
        print(f"  {name}：{bp_data['quantity']:.2f} 吨（系数 {bp_data['coefficient']:.2f}）")

    # 四、全链条碳排放
    print("\n【四、全链条碳排放核算】")
    ce = result['carbon_emission']
    print(f"  种植环节（化肥N2O）：{ce['planting']:.2f} kg CO2e")
    print(f"  机械作业（柴油）：{ce['mechanization']:.2f} kg CO2e")
    print(f"  加工环节（电力）：{ce['processing']:.2f} kg CO2e")
    print(f"  全链条总计：{ce['total_tons']:.4f} 吨CO2e")
    print(f"  折合每亩：{ce['total_tons'] / result['area_mu']:.4f} 吨CO2e/亩")

    # 五、方案对比
    print("\n【五、三种方案详细对比】")
    schemes = result['optimization']['all_schemes']
    for i, scheme in enumerate(schemes, 1):
        name = scheme_name_cn.get(scheme['name'], scheme['name'])
        marker = " ★★★ 推荐方案 ★★★" if scheme['name'] == result['optimization']['optimal']['name'] else ""
        print(f"\n  方案{i}：{name}{marker}")
        print(f"    净收益：{scheme['net_benefit']:,.2f} 元")
        print(f"    碳排放：{scheme['carbon_emission_kg']:,.2f} kg CO2e")
        print(f"    综合得分：{scheme['total_score']:.4f}")

    # 六、经济效益明细
    print("\n【六、经济效益明细】")
    for bp_name, bp_econ in result['economic'].items():
        name = byproduct_name_cn.get(bp_name, bp_name)
        print(f"\n  {name}（产量：{bp_econ.get('quantity', 'N/A')} 吨）")
        for method, values in bp_econ.items():
            if method == 'quantity':
                continue
            net = values['revenue'] - values['cost']
            method_cn = {
                'burn': '田间焚烧', 'biomass_pellet': '生物质颗粒',
                'landfill': '填埋处理', 'organic_fertilizer': '有机肥',
                'direct_sale': '直接出售', 'deep_processed': '深加工',
                'boiler_fuel': '锅炉燃料'
            }
            print(f"    {method_cn.get(method, method)}：收入 {values['revenue']:,.2f} 元，"
                  f"成本 {values['cost']:,.2f} 元，净收益 {net:,.2f} 元")

    # 七、总结建议
    print("\n【七、总结与建议】")
    opt = result['optimization']['optimal']
    trad = schemes[0] if schemes[0]['name'] == 'traditional' else schemes[-1]
    improve = opt['net_benefit'] - trad['net_benefit']
    print(f"  推荐方案：{scheme_name_cn.get(opt['name'], opt['name'])}")
    print(f"  相比传统模式增收：{improve:,.2f} 元")
    print(f"  收益率提升：{improve / max(abs(trad['net_benefit']), 1) * 100:.1f}%")
    print(f"  碳排放减少：{trad['carbon_emission_kg'] - opt['carbon_emission_kg']:,.2f} kg CO2e")

    print("\n" + "=" * 70)
    print("                          —— 报告结束 ——")
    print("=" * 70)


if __name__ == '__main__':
    main()
