"""LLM 端到端测试脚本"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question

print('=' * 60)
print('测试 1：LLM 基础对话连接')
print('=' * 60)

cfg = LLMConfig()
client = get_client()
print('API Key 配置状态:', '已配置' if cfg.enabled else '未配置')
print('Client available:', client.available)

if client.available:
    try:
        resp = client.complete([
            {'role': 'system', 'content': '你是农业专家，请回复：测试通过'},
            {'role': 'user', 'content': '验证 LLM 连接'}
        ])
        print('LLM 响应:', resp[:100])
        print('结果: 通过')
    except Exception as e:
        print('LLM 调用失败:', e)
        print('结果: 失败')
else:
    print('结果: 未启用')

print()
print('=' * 60)
print('测试 2：决策报告增强')
print('=' * 60)

test_params = {
    'city': '南宁', 'area_mu': 100, 'avg_temp': 24,
    'precipitation': 1200, 'sunshine': 1600, 'carbon_price': 50
}
test_result = {
    'yield_per_mu': 5.2, 'total_yield': 520, 'yield_source': 'model',
    'byproducts': {
        'sugarcane_leaf': {'quantity': 156},
        'bagasse': {'quantity': 260},
        'filter_mud': {'quantity': 52},
        'molasses': {'quantity': 26},
        'sugarcane_top': {'quantity': 78}
    },
    'carbon_emission': {
        'total_tons': 45.3, 'planting': 12000,
        'mechanization': 8000, 'processing': 25000
    },
    'optimization': {
        'optimal': {
            'name': 'circular_optimal',
            'net_benefit': 150000,
            'total_benefit': 180000,
            'carbon_revenue': 5000
        },
        'all_schemes': [
            {'name': 'traditional', 'net_benefit': 80000, 'carbon_emission_kg': 60000, 'carbon_revenue': 0, 'total_score': 0.6},
            {'name': 'improved_traditional', 'net_benefit': 100000, 'carbon_emission_kg': 50000, 'carbon_revenue': 1000, 'total_score': 0.72},
            {'name': 'circular_basic', 'net_benefit': 120000, 'carbon_emission_kg': 35000, 'carbon_revenue': 2500, 'total_score': 0.82},
            {'name': 'circular_advanced', 'net_benefit': 135000, 'carbon_emission_kg': 25000, 'carbon_revenue': 4000, 'total_score': 0.89},
            {'name': 'circular_optimal', 'net_benefit': 150000, 'carbon_emission_kg': 20000, 'carbon_revenue': 5000, 'total_score': 0.95}
        ]
    }
}

try:
    report = enhance_decision_report(test_params, test_result)
    if report:
        print('报告生成成功，长度:', len(report), '字符')
        print('报告前300字:')
        print(report[:300])
        print('结果: 通过')
    else:
        print('报告返回 None')
        print('结果: 失败')
except Exception as e:
    print('决策报告增强失败:', e)
    print('结果: 失败')

print()
print('=' * 60)
print('测试 3：自然语言问数')
print('=' * 60)

try:
    answer = answer_question('碳减排原理是什么', test_params, test_result)
    if answer:
        print('回答生成成功，长度:', len(answer), '字符')
        print('回答内容:')
        print(answer[:400])
        print('结果: 通过')
    else:
        print('回答返回 None')
        print('结果: 失败')
except Exception as e:
    print('自然语言问数失败:', e)
    print('结果: 失败')

print()
print('=' * 60)
print('测试完成')
print('=' * 60)
