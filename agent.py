"""
蔗循智策 对话式决策助手（规则引擎 + 可选 LLM 增强）

功能：自然语言输入 → 规则式参数解析（regex + keyword）→ 模型调用 →
推理链生成 → 自然语言决策报告

技术定位说明（重要）：
- 意图识别与参数提取基于正则与关键词匹配，核心计算由确定性
  规则引擎 / 统计模型完成，保证结果可复现、可解释；
- 可选 LLM 增强（见 llm_agent.py）：在计算结果之上提供
  ① 决策报告润色（结构化结果→给蔗农/糖企/政府的自然语言报告）
  ② 自然语言问数（基于已核算事实回答开放式提问）。
  配置 SCZC_LLM_API_KEY 后自动启用；未配置 / 调用失败 / 限流时
  自动回退到规则模板，系统功能不受影响；
- 对外展示请如实表述为"对话式决策助手（规则引擎 + LLM 口语化增强）"，
  LLM 不参与任何计算，仅负责报告的表述润色与答疑。
"""

import json
import logging
import os
import re

from models import SugarcaneDecisionSystem, get_default_carbon_price

logger = logging.getLogger("agent")

# 东盟气候基准缓存（按需加载，见 _load_asean_climate）
_ASEAN_CLIMATE = None
_ASEAN_CLIMATE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'asean_climate_normals.json')


def _load_asean_climate() -> dict:
    """按需加载东盟四国生长季气候基准（Open-Meteo ERA5）。

    用于跨境对比：每个国家使用其真实区域气候（温度/降水/日照），
    而非复用中国的入门气象输入。未找到文件/解析失败时返回空 dict（回退现状）。
    """
    global _ASEAN_CLIMATE
    if _ASEAN_CLIMATE is not None:
        return _ASEAN_CLIMATE
    try:
        with open(_ASEAN_CLIMATE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _ASEAN_CLIMATE = data.get('normals', {})
    except Exception as e:
        logger.warning("读取东盟气候基准失败，跨境对比回退复用中国天气: %s", e)
        _ASEAN_CLIMATE = {}
    return _ASEAN_CLIMATE

# ============================================================
# 栽培参数（从 config.json 读取）
# ============================================================
def _load_cultivation_config():
    """从 config.json 加载栽培默认参数"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg.get('default_cultivation', {})
    except Exception:
        return {}

CULTIVATION = _load_cultivation_config()
FERTILIZER_N_PER_MU = CULTIVATION.get('fertilizer_n_kg_per_mu', 22)
DIESEL_PER_MU = CULTIVATION.get('diesel_l_per_mu', 5)
ELECTRICITY_PER_MU = CULTIVATION.get('electricity_kwh_per_mu', 50)
FERTILIZER_P2O5_PER_MU = CULTIVATION.get('fertilizer_p2o5_kg_per_mu', 10)
FERTILIZER_K2O_PER_MU = CULTIVATION.get('fertilizer_k2o_kg_per_mu', 18)

# ============================================================
# 意图识别 + 参数提取
# ============================================================

CITY_ALIASES = {
    '崇左': '崇左市', '来宾': '来宾市', '南宁': '南宁市',
    '柳州': '柳州市', '百色': '百色市', '河池': '河池市',
    '防城港': '防城港市', '防城': '防城港市',
}

SCHEME_CN = {
    'traditional': '传统模式（焚烧+填埋+直销+锅炉）',
    'circular_basic': '基础循环（饲料+有机肥+直销+锅炉）',
    'circular_optimal': '最优循环（生物质颗粒+有机肥+深加工+环保浆料）',
}

BYPRODUCT_CN = {
    'sugarcane_leaf': '蔗叶', 'bagasse': '蔗渣',
    'filter_mud': '滤泥', 'molasses': '糖蜜', 'sugarcane_top': '蔗梢'
}

COUNTRY_ALIASES = {
    '中国': 'China', '广西': 'China', '泰国': 'Thailand',
    '越南': 'Vietnam', '缅甸': 'Myanmar', '老挝': 'Laos',
    'china': 'China', 'thailand': 'Thailand',
    'vietnam': 'Vietnam', 'myanmar': 'Myanmar', 'laos': 'Laos',
}


def parse_query(text: str) -> dict:
    """从自然语言中提取参数（增强容错版）"""
    params = {}

    # 面积：支持 "50亩" "50 亩" "10.5亩" "十来亩" "几十亩"
    # 先处理中文数量词，按长度降序避免"十亩"匹配到"几十亩"
    cn_num_map = {'十来': 10, '几十': 30, '二十': 20, '三十': 30,
                  '五十': 50, '一百': 100, '十': 10, '几': 5}
    for cn_word, val in cn_num_map.items():
        if cn_word + '亩' in text or cn_word + ' 亩' in text:
            params['area_mu'] = float(val)
            break
    if 'area_mu' not in params:
        area_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:亩|mu)', text)
        if area_match:
            params['area_mu'] = float(area_match.group(1))

    # 城市
    for alias, city in CITY_ALIASES.items():
        if alias in text:
            params['city'] = city
            break

    # 国家
    for alias, country in COUNTRY_ALIASES.items():
        if alias in text:
            params['country'] = country
            break

    # 气象
    # 温度：支持 "28度" "28℃" "温度28" "均温28"
    temp_match = re.search(r'(?:温度|气温|均温).*?(\d+[\.\d]*)|(\d+[\.\d]*)\s*[度℃°]', text)
    if temp_match:
        val = next((v for v in temp_match.groups() if v), None)
        if val:
            params['avg_temp'] = float(val)

    # 降水：支持 "降水900" "900毫米" "900mm" "雨量900"
    rain_match = re.search(r'(?:降水|降雨|雨量).*?(\d+)|(\d+)\s*(?:毫米|mm)', text)
    if not rain_match:
        rain_match = re.search(r'(\d+)\s*毫米', text)
    if rain_match:
        val = next((v for v in rain_match.groups() if v), None)
        if val:
            params['precipitation'] = float(val)

    # 日照：支持 "日照870" "870小时" "870h" "光照870"
    sun_match = re.search(r'(?:日照|光照).*?(\d+)|(\d+)\s*(?:小时|h)', text)
    if sun_match:
        val = next((v for v in sun_match.groups() if v), None)
        if val:
            params['sunshine'] = float(val)

    # 碳价：支持 "碳价100" "碳价 100元"
    cp_match = re.search(r'碳价\s*(\d+(?:\.\d+)?)', text)
    if cp_match:
        params['carbon_price'] = float(cp_match.group(1))

    # 意图
    intents = []
    if any(w in text for w in ['优化', '方案', '推荐', '建议', '决策', '怎么', '如何']):
        intents.append('recommend')
    if any(w in text for w in ['碳', '排放', '减排', 'CO2', '温室']):
        intents.append('carbon')
    if any(w in text for w in ['收益', '赚钱', '增收', '利润', '收入', '经济']):
        intents.append('economic')
    if any(w in text for w in ['蔗渣', '蔗叶', '滤泥', '糖蜜', '副产物']):
        intents.append('byproduct')
    if any(w in text for w in ['泰国', '越南', '东盟', '跨境', '对比']):
        intents.append('cross_country')
    if any(w in text for w in ['天气', '气象', '温度', '降水', '干旱', '雨水']):
        intents.append('weather')
    if not intents:
        intents.append('recommend')

    params['intents'] = intents
    return params


# ============================================================
# 推理链生成 (Chain-of-Thought)
# ============================================================

def generate_reasoning_chain(result: dict, params: dict, system) -> str:
    """生成决策推理链（Markdown格式，适配Streamlit渲染）"""
    opt = result['optimization']['optimal']
    trad = result['optimization']['all_schemes'][-1]
    ce = result['carbon_emission']
    model_name = system.yield_predictor.metrics.get('model_name', 'ridge')
    r2 = system.yield_predictor.metrics.get('r2', 0)
    loocv_n = system.yield_predictor.metrics.get('loocv_samples', '?')

    steps = []

    # Step 1: 产量预测
    yield_src = result.get('yield_source', 'model')
    if yield_src == 'model':
        steps.append(
            f"**1. 产量预测**\n\n"
            f"模型 `{model_name.upper()}` (LOOCV R²={r2:.3f}, {loocv_n}样本) "
            f"预测 **{params.get('city', '未知')}** 单产 **{result['yield_per_mu']:.2f} 吨/亩**，"
            f"总产量 **{result['total_yield']:.2f} 吨**"
        )
    else:
        steps.append(
            f"**1. 产量基准**\n\n"
            f"FAO 十年统计数据基准："
            f"**{result['yield_per_mu']:.2f} 吨/亩**"
        )

    # Step 2: 副产物
    bp_items = []
    for k, v in result['byproducts'].items():
        bp_items.append(f"{BYPRODUCT_CN.get(k, k)} {v['quantity']:.2f} 吨")
    steps.append(
        f"**2. 副产物估算**\n\n"
        + " · ".join(bp_items)
    )

    # Step 3: 碳排放
    steps.append(
        f"**3. 碳排放核算** (IPCC Tier 1)\n\n"
        f"全链条排放 **{ce['total_tons']:.2f} 吨CO₂e** "
        f"(化肥 {ce['planting']:.0f}kg + 柴油 {ce['mechanization']:.0f}kg + 电力 {ce['processing']:.0f}kg)  "
        f"最优方案额外碳减排 **{abs(opt['carbon_emission_kg']):.0f} kg**"
    )

    # Step 4: 方案对比（用Markdown表格）
    table_rows = []
    for s in result['optimization']['all_schemes']:
        mark = '⭐' if s['name'] == opt['name'] else ''
        cn = SCHEME_CN.get(s['name'], s['name'])
        table_rows.append(
            f"| {mark} {cn} | {s['net_benefit']:,.0f} | "
            f"{s['carbon_emission_kg']:+,.0f} | "
            f"{s['total_score']:.3f} |"
        )
    steps.append(
        f"**4. 多目标优化** (收益70% + 碳30%)\n\n"
        f"| 方案 | 净收益(元) | 碳排放(kg) | 综合得分 |\n"
        f"|------|-----------|------------|--------|\n"
        + '\n'.join(table_rows)
    )

    # Step 5: 推荐理由
    improve_pct = (opt['net_benefit'] / max(abs(trad['net_benefit']), 1) - 1) * 100
    carbon_line = ""
    if opt['carbon_revenue'] > 0:
        carbon_line = f"\n→ 碳交易收益 **+{opt['carbon_revenue']:,.0f} 元**"
    else:
        carbon_line = f"\n→ 传统模式碳交易支出 **{opt['carbon_revenue']:,.0f} 元**"

    steps.append(
        f"**5. 推荐结论**\n\n"
        f"选择 **{SCHEME_CN.get(opt['name'], opt['name'])}**，"
        f"比传统模式增收 **{improve_pct:.0f}%**，"
        f"综合收益 **{opt['total_benefit']:,.0f} 元**"
        f"{carbon_line}"
    )

    return '\n\n---\n\n'.join(steps)


# ============================================================
# Agent 主类
# ============================================================

# 必填参数及追问话术
REQUIRED_PARAMS = {
    'area_mu': {'label': '种植面积', 'unit': '亩', 'prompt': '请问种植面积是多少亩？'},
    'city': {'label': '所在城市', 'unit': '', 'prompt': '请问是广西哪个城市？（崇左/来宾/南宁/柳州等）'},
}

# 可选参数（有默认值，缺失时不追问但会提示使用默认值）
OPTIONAL_PARAMS = {
    'avg_temp': {'label': '生长季均温', 'unit': '℃', 'default': 28.0},
    'precipitation': {'label': '生长季降水', 'unit': 'mm', 'default': 900.0},
    'sunshine': {'label': '生长季日照', 'unit': 'h', 'default': 870.0},
    'carbon_price': {'label': '碳价', 'unit': '元/吨', 'default': None},  # 动态默认值
}


class SugarcaneAgent:
    """蔗循智策 对话式决策助手（规则引擎实现，非大模型 Agent）

    通过正则+关键词解析自然语言参数，支持多轮追问收集参数，
    参数齐全后调用决策模型生成报告。预留 LLM 接口可升级为 AI Agent。

    用法：
        agent = SugarcaneAgent(system=已有的SugarcaneDecisionSystem实例)
        # 单轮：response, done = agent.chat("崇左10亩，怎么处理蔗叶最赚钱？", state={})
        # 多轮：state维护对话状态，多轮调用间保留已收集的参数
    """

    def __init__(self, system=None):
        """初始化Agent，支持传入已有的system实例避免重复训练"""
        self.system = system or SugarcaneDecisionSystem()
        if system is None:
            self.system.train_models(model_type='auto')
        self.default_carbon_price = get_default_carbon_price()

    def chat(self, query: str, state: dict = None) -> tuple:
        """自然语言对话接口（支持多轮追问）

        Args:
            query: 用户输入
            state: 对话状态字典，用于多轮对话间保留上下文

        Returns:
            (response_text, is_done, state)
            - response_text: Agent回复
            - is_done: True=已生成决策，False=仍在追问参数
            - state: 更新后的对话状态
        """
        if state is None:
            state = {'collected': {}, 'intents': [], 'history': []}

        # 解析当前轮次参数
        current_params = parse_query(query)
        current_intents = current_params.pop('intents', [])

        # 合并到已收集参数（新参数覆盖旧参数）
        for k, v in current_params.items():
            state['collected'][k] = v

        # 合并意图
        for intent in current_intents:
            if intent not in state['intents']:
                state['intents'].append(intent)

        state['history'].append({'role': 'user', 'content': query})

        # 检查必填参数是否齐全
        missing = []
        for key, info in REQUIRED_PARAMS.items():
            if key not in state['collected']:
                missing.append(key)

        # 多轮追问兜底：按追问上下文定向解析用户输入
        # 如果 parse_query 没提取到参数，但用户输入是纯数字/纯城市名，按追问意图识别
        if not current_params:
            asking = state.get('asking')  # 上一轮追问的参数
            query_clean = query.strip()
            # 纯数字兜底
            num_match = re.match(r'^\d+(\.\d+)?$', query_clean)
            if num_match:
                val = float(num_match.group())
                if asking == 'area_mu' or (missing and missing[0] == 'area_mu'):
                    state['collected']['area_mu'] = val
                elif asking == 'avg_temp':
                    state['collected']['avg_temp'] = val
                elif asking == 'precipitation':
                    state['collected']['precipitation'] = val
                elif asking == 'sunshine':
                    state['collected']['sunshine'] = val
                elif asking == 'carbon_price':
                    state['collected']['carbon_price'] = val
            # 城市名兜底
            elif asking == 'city' or (missing and missing[0] == 'city'):
                for alias, city in CITY_ALIASES.items():
                    if alias in query:
                        state['collected']['city'] = city
                        break

        # 重新计算缺失
        missing = [k for k in REQUIRED_PARAMS if k not in state['collected']]

        # 还有必填参数缺失 → 追问
        if missing:
            first_missing = missing[0]
            state['asking'] = first_missing  # 记录当前追问的参数
            info = REQUIRED_PARAMS[first_missing]
            collected_str = self._format_collected(state['collected'])
            response = (
                f"我还需要了解一些信息才能帮你决策：\n\n"
                f"**{info['prompt']}**\n\n"
                f"---\n"
                f"*已收集：{collected_str}*"
            )
            state['history'].append({'role': 'assistant', 'content': response})
            return response, False, state

        # 参数齐全 → 填充默认值并生成决策
        params = state['collected'].copy()
        params['intents'] = state['intents'] if state['intents'] else ['recommend']
        params.setdefault('country', 'China')
        params.setdefault('avg_temp', 28.0)
        params.setdefault('precipitation', 900.0)
        params.setdefault('sunshine', 870.0)
        params.setdefault('carbon_price', self.default_carbon_price)

        # 边界保护：防止用户输入超出模型可信范围
        area_mu = max(0.01, min(float(params['area_mu']), 100000.0))
        avg_temp = max(10.0, min(float(params['avg_temp']), 45.0))
        precipitation = max(0.0, min(float(params['precipitation']), 5000.0))
        sunshine = max(0.0, min(float(params['sunshine']), 3000.0))
        carbon_price = max(0.0, min(float(params['carbon_price']), 10000.0))

        # 调用决策模型
        result = self.system.run_decision(
            area_mu=area_mu,
            avg_temp=avg_temp,
            precipitation=precipitation,
            sunshine=sunshine,
            fertilizer_n_kg=FERTILIZER_N_PER_MU * area_mu,
            diesel_l=DIESEL_PER_MU * area_mu,
            electricity_kwh=ELECTRICITY_PER_MU * area_mu,
            carbon_price=carbon_price,
            country=params['country'],
            city=params['city'],
        )

        # 推理链
        reasoning = generate_reasoning_chain(result, params, self.system)

        # LLM 增强：决策报告润色（未配置 key / 失败自动返回 None → 规则模板）
        llm_report = self._enhance_report(params, result, reasoning)

        # 跨境对比
        cross = ""
        if 'cross_country' in params['intents'] and params['country'] == 'China':
            cross = self._cross_country_compare(params, result)

        # 生成最终回复
        response = self._format_response(params, result, reasoning, cross, llm_report)
        state['history'].append({'role': 'assistant', 'content': response})
        state['last_result'] = result
        state['last_params'] = params
        state['llm_report'] = bool(llm_report)
        return response, True, state

    def _enhance_report(self, params: dict, result: dict, reasoning: str):
        """调用 LLM 润色决策报告；未启用/失败返回 None（保持规则模板）。"""
        try:
            from llm_agent import enhance_decision_report
            return enhance_decision_report(params, result, reasoning)
        except Exception as e:
            logger.warning("LLM 报告增强异常，使用规则模板: %s", e)
            return None

    def answer_question(self, question: str, state: dict = None,
                        params: dict = None, result: dict = None) -> tuple:
        """自然语言问数：基于最近一次（或传入的）决策结果回答开放式问题。

        返回 (answer, use_llm)：
            - answer: 回答文本；LLM 未启用/失败时返回基于规则模板的兜底回复
            - use_llm: True=LLM 生成，False=规则兜底
        """
        # 收集本次问数所需的决策上下文
        q_params = params
        q_result = result
        if state and state.get('last_result') and state.get('last_params'):
            q_params = q_params or state['last_params']
            q_result = q_result or state['last_result']
        # 尚无决策结果：用默认参数跑一次决策作为上下文
        if q_params is None or q_result is None:
            q_params = {
                'area_mu': 10.0, 'city': '崇左市', 'country': 'China',
                'avg_temp': 28.0, 'precipitation': 900.0,
                'sunshine': 870.0, 'carbon_price': self.default_carbon_price,
                'intents': ['recommend'],
            }
            q_result = self.system.run_decision(
                area_mu=q_params['area_mu'],
                avg_temp=q_params['avg_temp'],
                precipitation=q_params['precipitation'],
                sunshine=q_params['sunshine'],
                fertilizer_n_kg=FERTILIZER_N_PER_MU * q_params['area_mu'],
                diesel_l=DIESEL_PER_MU * q_params['area_mu'],
                electricity_kwh=ELECTRICITY_PER_MU * q_params['area_mu'],
                carbon_price=q_params['carbon_price'],
                country=q_params['country'],
                city=q_params['city'],
            )
        # LLM 解答
        try:
            from llm_agent import answer_question as _llm_answer
            answer = _llm_answer(question, q_params, q_result)
            if answer:
                return answer, True
        except Exception:
            pass
        # 规则兜底：给出可用事实与可答建议
        opt = q_result['optimization']['optimal']
        fallback = (
            "我暂时无法调用语言模型来开放作答（未配置 LLM 服务或调用失败），"
            "以下是本次决策的确定性结论，供你参考：\n\n"
            f"- 最优方案：{opt['name']}，综合收益 {opt.get('total_benefit', opt['net_benefit']):,.0f} 元\n"
            f"- 净收益：{opt['net_benefit']:,.0f} 元，碳交易收益 {opt.get('carbon_revenue', 0):+,.0f} 元\n\n"
            "配置 SCZC_LLM_API_KEY 后可针对任意子方案、碳减排原理或副产物利用路径"
            "进行开放问答。想问具体技术细节，请先配置语言模型接口。"
        )
        return fallback, False

    def _format_collected(self, collected: dict) -> str:
        """格式化已收集参数的可读字符串"""
        if not collected:
            return '暂无'
        parts = []
        label_map = {
            'area_mu': '面积', 'city': '城市', 'country': '国家',
            'avg_temp': '温度', 'precipitation': '降水', 'sunshine': '日照',
            'carbon_price': '碳价',
        }
        unit_map = {
            'area_mu': '亩', 'avg_temp': '℃', 'precipitation': 'mm',
            'sunshine': 'h', 'carbon_price': '元/吨',
        }
        for k, v in collected.items():
            label = label_map.get(k, k)
            unit = unit_map.get(k, '')
            if isinstance(v, float):
                parts.append(f"{label}={v:.0f}{unit}")
            else:
                parts.append(f"{label}={v}")
        return '、'.join(parts)

    def _cross_country_compare(self, params: dict, china_result: dict) -> str:
        country_names = {'China': '中国', 'Thailand': '泰国', 'Vietnam': '越南', 'Myanmar': '缅甸', 'Laos': '老挝'}
        rows = []
        area_mu = max(0.01, min(float(params.get('area_mu', 10.0)), 100000.0))
        avg_temp = max(10.0, min(float(params.get('avg_temp', 28.0)), 45.0))
        precipitation = max(0.0, min(float(params.get('precipitation', 900.0)), 5000.0))
        sunshine = max(0.0, min(float(params.get('sunshine', 870.0)), 3000.0))
        carbon_price = max(0.0, min(float(params.get('carbon_price', self.default_carbon_price)), 10000.0))
        city = params.get('city', '崇左市')
        # 东盟四国真实生长季气候（Open-Meteo ERA5）；缺失时回退复用中国输入
        asean_climate = _load_asean_climate()
        for c in ['Thailand', 'Vietnam', 'Myanmar', 'Laos']:
            cli = asean_climate.get(c, {})
            c_temp = cli.get('avg_temp_c', avg_temp)
            c_precip = cli.get('precipitation_mm', precipitation)
            c_sun = cli.get('sunshine_hours', sunshine)
            r = self.system.run_decision(
                area_mu=area_mu,
                avg_temp=c_temp,
                precipitation=c_precip,
                sunshine=c_sun,
                fertilizer_n_kg=22 * area_mu,
                diesel_l=5 * area_mu,
                electricity_kwh=50 * area_mu,
                carbon_price=carbon_price,
                country=c, city=city,
            )
            o = r['optimization']['optimal']
            src = r.get('yield_source', 'N/A')
            rows.append(
                f"| {country_names[c]} | {r['yield_per_mu']:.2f} | {src} | "
                f"{o['total_benefit']:,.0f} | {o['carbon_emission_kg']:+,.0f} | "
                f"{c_temp:.1f}°C/{c_precip:.0f}mm |"
            )
        return (
            f"**跨境对比**（气象=各国真实生长季基准，Open-Meteo ERA5）\n\n"
            f"| 国家 | 单产(吨/亩) | 产量来源 | 综合收益(元) | 碳排放(kg) | 生长季均温/降水 |\n"
            f"|------|------------|---------|------------|------------|----------------|\n"
            + '\n'.join(rows)
        )

    def _format_response(self, params, result, reasoning, cross, llm_report=None):
        opt = result['optimization']['optimal']
        model_name = self.system.yield_predictor.metrics.get('model_name', 'ridge')
        r2 = self.system.yield_predictor.metrics.get('r2', 0)
        loocv_n = self.system.yield_predictor.metrics.get('loocv_samples', '?')
        yield_src = result.get('yield_source', 'model')
        src_label = 'LOOCV回归模型' if yield_src == 'model' else 'FAO统计均值'

        # LLM 增强的报告为正文，结构化推理链作为附表保留（保证数字可复核）
        if llm_report:
            header = (
                f"## 🌱 蔗循智策 智能决策报告（AI）\n\n"
                f"{llm_report}\n\n"
                f"---\n\n"
                f"### 📊 结构化核算（数字可复核）\n"
            )
            lines = [
                header,
                f"**{params.get('city', '未知')}** {params['area_mu']:.0f}亩 | "
                f"碳价 {params['carbon_price']:.0f} 元/吨 | 产量来源: {src_label}",
                "",
                reasoning,
            ]
        else:
            lines = [
                f"## 🌱 蔗循智策 决策报告（规则引擎版）",
                f"",
                f"> **{params.get('city', '未知')}** {params['area_mu']:.0f}亩 | "
                f"碳价 {params['carbon_price']:.0f} 元/吨 | 产量来源: {src_label}",
                f"",
                reasoning,
            ]
        if cross:
            lines.append(cross)
        lines.extend([
            "",
            "---",
            f"",
            f"*推理完成 · 模型 `{model_name.upper()}` LOOCV R²={r2:.3f} | {loocv_n}样本 | LLM增强={'开' if llm_report else '关'}*",
        ])
        return '\n'.join(lines)

    def ask(self, query: str) -> dict:
        """编程接口：返回结构化结果"""
        params = parse_query(query)
        params.setdefault('area_mu', 10.0)
        params.setdefault('city', '崇左市')
        params.setdefault('country', 'China')
        params.setdefault('avg_temp', 28.0)
        params.setdefault('precipitation', 900.0)
        params.setdefault('sunshine', 870.0)
        params.setdefault('carbon_price', self.default_carbon_price)

        area_mu = max(0.01, min(float(params['area_mu']), 100000.0))
        avg_temp = max(10.0, min(float(params['avg_temp']), 45.0))
        precipitation = max(0.0, min(float(params['precipitation']), 5000.0))
        sunshine = max(0.0, min(float(params['sunshine']), 3000.0))
        carbon_price = max(0.0, min(float(params['carbon_price']), 10000.0))

        result = self.system.run_decision(
            area_mu=area_mu,
            avg_temp=avg_temp,
            precipitation=precipitation,
            sunshine=sunshine,
            fertilizer_n_kg=22 * area_mu,
            diesel_l=5 * area_mu,
            electricity_kwh=50 * area_mu,
            carbon_price=carbon_price,
            country=params['country'],
            city=params['city'],
            scenario='optimal'
        )
        return {
            'params': params,
            'result': result,
            'reasoning': generate_reasoning_chain(result, params, self.system),
            'optimal_scheme': result['optimization']['optimal']['name'],
            'optimal_benefit': result['optimization']['optimal']['total_benefit'],
        }


# ============================================================
# CLI 入口
# ============================================================

if __name__ == '__main__':
    import sys

    agent = SugarcaneAgent()

    test_queries = [
        "我在崇左有10亩蔗田，温度28度降水900日照870，怎么处理蔗叶最赚钱？",
        "来宾50亩甘蔗，碳价100元，推荐最优方案",
        "对比中国泰国越南的蔗渣利用收益差异",
    ]

    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        print(agent.chat(query))
    else:
        for q in test_queries:
            print(agent.chat(q))
            print()
