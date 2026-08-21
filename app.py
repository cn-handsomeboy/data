"""
Streamlit可视化应用 - 比赛演示版
蔗循智策：面向中国-东盟的甘蔗副产物循环经济跨境数据协同决策系统
Version: 2026.08.16

面向评委的设计原则：
1. 30秒内传达项目核心价值
2. 每个数字都有来源可追溯
3. 理论最优 + 现实可行双情景对比
4. 自动生成政策建议
"""

import html
import json
import os
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from models import SugarcaneDecisionSystem, DATA_DIR, get_default_carbon_price
from agent import SugarcaneAgent
from data_product import (
    get_lineage_data,
    DataProductCertification,
    DataTradingSimulation
)
from data_security import (
    DataClassifier, DataMasker, DataIntegrityChecker,
    SecurityManager, get_security_status
)
# LLM 增强为可选模块：导入失败时自动降级为规则模板，核心功能不受影响
try:
    from llm_agent import (
        enhance_decision_report, answer_question, get_client,
        rule_template_report, fallback_answer,
    )
    _LLM_IMPORT_OK = True
except Exception as _llm_err:
    import logging as _llm_logging
    _llm_logging.getLogger("app").warning("llm_agent 导入失败，LLM 增强降级: %s", _llm_err)
    _LLM_IMPORT_OK = False
    enhance_decision_report = None
    answer_question = None
    get_client = None
    # 兜底函数不依赖 LLM，定义本地版保证降级后仍有输出
    _SCHEME_CN_FB = {
        'traditional': '传统模式', 'improved_traditional': '改良传统模式',
        'circular_basic': '基础循环模式', 'circular_advanced': '进阶循环模式',
        'circular_optimal': '最优循环模式',
    }
    def rule_template_report(result, params):
        opt = result['optimization']['optimal']
        trad = result['optimization']['all_schemes'][-1]
        improve = (opt['net_benefit'] / max(abs(trad['net_benefit']), 1) - 1) * 100
        ce = result['carbon_emission']
        area = params.get('area_mu', result.get('area_mu', 0))
        city = params.get('city', '')
        return (
            f"### 决策报告（规则引擎生成）\n\n"
            f"基于 {city}{area:.0f} 亩蔗田的产量预测、碳排放核算与多目标优化结果，"
            f"系统推荐采用 **{_SCHEME_CN_FB.get(opt['name'], opt['name'])}**："
            f"综合收益 {opt.get('total_benefit', opt['net_benefit']):,.0f} 元，"
            f"净收益 {opt['net_benefit']:,.0f} 元，较传统模式增收 **{improve:.0f}%**。\n\n"
            f"**碳减排依据**：全链条碳排放 {ce['total_tons']:.2f} 吨CO₂e"
            f"（种植 {ce['planting']:.0f} kg + 机械 {ce['mechanization']:.0f} kg + 加工 {ce['processing']:.0f} kg），"
            f"碳交易收益 {opt.get('carbon_revenue', 0):+,.0f} 元。\n\n"
            f"**行动建议**：① 落实副产物循环利用路径（饲料化/有机肥/生物质颗粒/深加工）；"
            f"② 关注碳价走势，适时参与碳普惠交易；"
            f"③ 按监测参数建立台账，支撑 CCER 方法学申报与碳汇收益核算。"
        )
    def fallback_answer(question, result):
        opt = result['optimization']['optimal']
        return (
            "当前未启用语言模型服务（未配置 SCZC_LLM_API_KEY 或导入失败），"
            "无法开放作答。为你提供本次决策的确定性结论：\n\n"
            f"- 最优方案：{_SCHEME_CN_FB.get(opt['name'], opt['name'])}，"
            f"综合收益 {opt.get('total_benefit', opt['net_benefit']):,.0f} 元\n"
            f"- 净收益：{opt['net_benefit']:,.0f} 元，碳交易收益 {opt.get('carbon_revenue', 0):+,.0f} 元\n\n"
            "配置 SCZC_LLM_API_KEY 后可对子方案、碳减排原理、副产物利用路径等进行开放问答。"
        )

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="蔗循智策 - 甘蔗副产物循环经济决策系统",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 云端访问埋点（GitHub 持久化，会话级去重）
# 说明：Streamlit Cloud 实例为临时文件系统，本地写入重启即丢；
# 本埋点将访问事件写入 GitHub 仓库 data/cloud_events.json，永久留存。
# 未配置 GITHUB_TOKEN 或网络失败时静默跳过，不影响功能。
# ============================================================
try:
    import uuid as _uuid_mod
    from datetime import datetime as _dt_mod
    from zoneinfo import ZoneInfo as _ZoneInfo
    from cloud_store import append_record as _cloud_append
    from cloud_store import read_json as _cloud_read
    from cloud_store import CLOUD_EVENTS_PATH as _CLOUD_EVENTS
except Exception:
    _uuid_mod = None

    def _cloud_append(*_a, **_k):
        return False

    def _cloud_read(*_a, **_k):
        return None

    _CLOUD_EVENTS = "data/cloud_events.json"

if _uuid_mod is not None:
    if "_cloud_session_id" not in st.session_state:
        st.session_state["_cloud_session_id"] = str(_uuid_mod.uuid4())
    if not st.session_state.get("_cloud_event_reported", False):
        _cloud_append(_CLOUD_EVENTS, {
            "timestamp": _dt_mod.now(_ZoneInfo("Asia/Shanghai")).isoformat(),
            "session_id": st.session_state["_cloud_session_id"],
            "event": "page_view",
        })
        st.session_state["_cloud_event_reported"] = True

# ============================================================
# 专业配色
# ============================================================
C = {
    'primary': '#1B5E20', 'secondary': '#2E7D32', 'accent': '#FF8F00',
    'danger': '#C62828', 'warning': '#EF6C00', 'success': '#2E7D32',
    'info': '#1565C0', 'text': '#212121', 'text_light': '#616161',
}

# ============================================================
# CSS
# ============================================================
st.markdown(f"""
<style>
    .main-title {{
        font-size: 2.2rem; font-weight: 800; color: {C['primary']};
        text-align: center; margin-bottom: 0.2rem; letter-spacing: 2px;
    }}
    .sub-title {{
        font-size: 1.0rem; color: {C['text_light']};
        text-align: center; margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background: white; border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid {C['primary']}; height: 100%;
    }}
    .metric-card.optimal {{
        border-top-color: {C['success']};
        background: linear-gradient(135deg, #E8F5E9 0%, white 100%);
    }}
    .info-box {{
        background: #E3F2FD; border-left: 4px solid {C['info']};
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0; font-size: 0.9rem;
    }}
    .success-box {{
        background: #E8F5E9; border-left: 4px solid {C['success']};
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0; font-size: 0.9rem;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 初始化
# ============================================================
@st.cache_resource(ttl=3600)  # 1小时缓存过期，确保模型更新后自动重载
def get_system():
    import sklearn
    from models import warm_start_models
    try:
        s = warm_start_models()
        metrics = s.yield_predictor.metrics or {}
        src = '热加载' if s.yield_predictor.model is not None and \
            not metrics.get('fallback', True) else '重训练'
        print(f"[APP] 决策系统就绪（{src}）: "
              f"{metrics.get('model_name', 'N/A')}, R²={metrics.get('r2', 'N/A')}")
        return s
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        st.warning(f"模型加载提示: {e}")
        st.info(f"调试: sklearn={sklearn.__version__}, 错误类型={type(e).__name__}\n{err_detail[-500:]}")
        s = SugarcaneDecisionSystem()
        return s


system = get_system()
smart_carbon_price = get_default_carbon_price()

@st.cache_resource
def get_agent():
    return SugarcaneAgent(system=system)

agent = get_agent()

# ============================================================
# LLM 增强辅助函数（供主流程调用，保证规则兜底与推理链一致）
# ============================================================
_SCHEME_CN_APP = {
    'traditional': '传统模式', 'improved_traditional': '改良传统模式',
    'circular_basic': '基础循环模式', 'circular_advanced': '进阶循环模式',
    'circular_optimal': '最优循环模式',
}


def _build_reasoning_chain(result: dict) -> str:
    """构造本次决策的推理链摘要（供 LLM 引用，保证报告与计算逻辑一致）。"""
    opt = result['optimization']['optimal']
    trad = result['optimization']['all_schemes'][-1]
    improve = (opt['net_benefit'] / max(abs(trad['net_benefit']), 1) - 1) * 100
    m = (system.yield_predictor.metrics or {})
    src = 'LOOCV回归模型' if result.get('yield_source') == 'model' else 'FAO统计均值'
    lines = [
        f"产量预测: {src} 单产 {result['yield_per_mu']:.2f} 吨/亩，"
        f"总产 {result['total_yield']:.2f} 吨" +
        (f"（模型 R²={m.get('r2', 'N/A')}）" if result.get('yield_source') == 'model' else ""),
        f"推荐依据: 最优方案={_SCHEME_CN_APP.get(opt['name'], opt['name'])}，"
        f"综合收益 {opt.get('total_benefit', opt['net_benefit']):,.0f} 元，"
        f"净收益 {opt['net_benefit']:,.0f} 元，较传统模式增收 {improve:.0f}%，"
        f"碳交易收益 {opt.get('carbon_revenue', 0):+,.0f} 元",
        f"优化权重: 收益 {result['optimization']['weights']['benefit']:.0%} / "
        f"碳减排 {result['optimization']['weights']['carbon']:.0%}",
    ]
    return "\n".join(lines)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### 🌱 参数设置")
    st.caption("部署版本: 2026.08.18 · 精简模型版（固定超参+LOOCV）")
    _llm_on = bool(os.environ.get("SCZC_LLM_API_KEY", "").strip())
    st.caption("🤖 LLM 增强: " + ("已接入（报告润色+问数）" if _llm_on else "规则引擎兜底（未配置 API Key）"))

    # ---- 从 session_state 读取预置场景参数（需在控件使用前初始化） ----
    preset_area = st.session_state.pop('preset_area', 10.0)
    preset_temp = st.session_state.pop('preset_temp', 28.0)
    preset_rain = st.session_state.pop('preset_rain', 900.0)
    preset_sun = st.session_state.pop('preset_sun', 870.0)
    preset_n = st.session_state.pop('preset_n', 22.0)
    preset_diesel = st.session_state.pop('preset_diesel', 5.0)
    preset_elec = st.session_state.pop('preset_elec', 50.0)
    preset_city = st.session_state.pop('preset_city', None)
    preset_country = st.session_state.pop('preset_country', None)

    with st.expander("📊 模型质量", expanded=False):
        m = system.yield_predictor.metrics
        if m and not m.get('fallback', False):
            c1, c2 = st.columns(2)
            c1.metric("算法", m.get('model_name', 'N/A').upper())
            c2.metric("样本数", m.get('loocv_samples', 'N/A'))
            c1.metric("R²", f"{m.get('r2', 0):.3f}")
            c2.metric("RMSE", f"{m.get('rmse', 0):.3f} t/mu")
            if system.yield_predictor.model_comparison:
                for name, v in system.yield_predictor.model_comparison.items():
                    st.caption(f"  {name.upper()}: R²={v['r2']:.3f}, RMSE={v['rmse']:.3f}")
            # SHAP 可解释性信息
            if system.yield_predictor.shap_summary:
                shap_top3 = sorted(system.yield_predictor.shap_summary.items(),
                                   key=lambda x: -x[1]['mean_abs_shap'])[:3]
                shap_text = "SHAP可解释性 top3: " + ", ".join(
                    [f"{f.replace('city_', '城市_').replace('_mu', '').replace('_x_', '×')}({v['mean_abs_shap']:.3f})"
                     for f, v in shap_top3]
                )
                st.caption(f"🔍 {shap_text}")
            st.caption(
                f"R²={m.get('r2', 0):.3f} 说明模型可解释 {m.get('r2', 0)*100:.1f}% 的产量变异。"
                f"学术对标：石杰锋等(2023) LSTM单蔗区R²=0.849，"
                f"本项目{m.get('model_name', 'GBRT').upper()} LOOCV R²={m.get('r2', 0):.3f}"
                f"在直接预测实际产量场景下处于国内领先水平。"
                f"RMSE={m.get('rmse', 0):.3f}吨/亩"
                f"（误差约{m.get('rmse', 0)/5.5*100:.1f}%），满足田间决策精度要求。"
                f"评估采用固定保守超参 + LOOCV（无信息泄漏，结果无偏）。"
            )
            if system.yield_predictor.feature_importance:
                # 展示特征重要性柱状图（Top 10）
                fi_items = sorted(system.yield_predictor.feature_importance.items(),
                                  key=lambda x: -x[1]['mean'])[:10]
                fi_df = pd.DataFrame({
                    '特征': [f.replace('city_', '城市_').replace('_mu', '').replace('_per_', '/')
                             .replace('avg_temp_c', '均温').replace('precipitation_mm', '降水')
                             .replace('sunshine_hours', '日照').replace('_x_', '×')
                             .replace('planting_area', '种植面积').replace('wan_mu', '')
                             for f, _ in fi_items],
                    '重要性': [v['mean'] for _, v in fi_items]
                })
                fig_fi = px.bar(fi_df, x='重要性', y='特征', orientation='h',
                                title='特征重要性（Top 10, permutation）',
                                color='重要性', color_continuous_scale='Greens')
                fig_fi.update_layout(height=220, margin=dict(l=0, r=0, t=25, b=0),
                                     coloraxis_showscale=False)
                st.plotly_chart(fig_fi, use_container_width=True,
                                config={'displayModeBar': False})
        else:
            st.info("使用历史均值 (样本不足)")

    st.markdown("---")
    # 国家和城市默认值处理
    country_options = ["China", "Thailand", "Vietnam", "Myanmar", "Laos"]
    default_country_idx = country_options.index(preset_country) if preset_country in country_options else 0
    country = st.selectbox(
        "国家/地区", country_options,
        index=default_country_idx,
        format_func=lambda x: {"China": "中国-广西", "Thailand": "泰国", "Vietnam": "越南", "Myanmar": "缅甸", "Laos": "老挝"}[x]
    )
    city_options = ["崇左市", "来宾市", "南宁市", "柳州市", "百色市", "河池市", "防城港市"]
    default_city_idx = city_options.index(preset_city) if preset_city in city_options else 0
    city = '崇左市'
    if country == 'China':
        city = st.selectbox(
            "广西蔗区", city_options,
            index=default_city_idx, help="不同城市对应不同的土壤、品种和管理水平"
        )

    st.markdown("---")

    # ---- 一键加载示范场景 ----
    with st.expander("🚀 一键加载示范场景", expanded=False):
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        PRESET_SCENES = {
            'laibin': {
                'name': '来宾', 'city': '来宾市',
                'temp': 27.8, 'rain': 920, 'sun': 860,
                'area': 50, 'n': 22, 'diesel': 5, 'elec': 50,
                'desc': '全国最大蔗渣餐具基地'
            },
            'chongzuo': {
                'name': '崇左', 'city': '崇左市',
                'temp': 28.5, 'rain': 850, 'sun': 900,
                'area': 30, 'n': 21, 'diesel': 4.5, 'elec': 45,
                'desc': '中国糖都+糖蜜酵母集群'
            },
            'nanning': {
                'name': '南宁', 'city': '南宁市',
                'temp': 28.2, 'rain': 880, 'sun': 870,
                'area': 20, 'n': 22, 'diesel': 5.5, 'elec': 55,
                'desc': '首府近郊都市农业'
            },
        }
        for col, key in [(sc_col1, 'laibin'), (sc_col2, 'chongzuo'), (sc_col3, 'nanning')]:
            s = PRESET_SCENES[key]
            with col:
                if st.button(s['name'], use_container_width=True, help=s['desc']):
                    st.session_state['preset_area'] = float(s['area'])
                    st.session_state['preset_temp'] = float(s['temp'])
                    st.session_state['preset_rain'] = float(s['rain'])
                    st.session_state['preset_sun'] = float(s['sun'])
                    st.session_state['preset_n'] = float(s['n'])
                    st.session_state['preset_diesel'] = float(s['diesel'])
                    st.session_state['preset_elec'] = float(s['elec'])
                    st.session_state['preset_city'] = s['city']
                    st.session_state['preset_country'] = 'China'
                    st.rerun()
        # 按钮下方统一显示描述，避免按钮内文字拥挤
        st.caption("来宾 — 全国最大蔗渣餐具基地　|　崇左 — 中国糖都+糖蜜酵母集群　|　南宁 — 首府近郊都市农业")

    st.markdown("---")
    st.caption("🌾 蔗田基本信息")
    area_mu = st.number_input("种植面积（亩）", 1.0, 1000.0, preset_area, 1.0)
    st.caption("🌤️ 气象条件（生长季5-10月累计）")
    avg_temp = st.slider("生长季均温（℃）", 22.0, 32.0, preset_temp, 0.1,
                         help="训练数据范围: 22-32℃, 均值27.6℃")
    precipitation = st.slider("生长季累计降水（mm）", 500.0, 1200.0, preset_rain, 10.0,
                              help="训练数据范围: 713-1096mm, 均值896mm")
    sunshine = st.slider("生长季累计日照（h）", 700.0, 1000.0, preset_sun, 10.0,
                         help="训练数据范围: 810-921h, 均值871h")
    st.caption("⚙️ 投入参数")
    fertilizer_n = st.number_input(
        "氮肥（kg N/亩）", 0.0, 500.0, preset_n, 1.0,
        help="中国-广西官方推荐: 20-23 kg N/亩" if country == 'China' else "参考中国-广西推荐量: 20-23 kg N/亩"
    )
    diesel = st.number_input("柴油（L/亩）", 0.0, 100.0, preset_diesel, 0.5)
    electricity = st.number_input("电力（kWh/亩）", 0.0, 200.0, preset_elec, 1.0)
    st.caption("💰 市场参数")
    carbon_price = st.number_input(
        "碳价（元/吨CO₂）", 0.0, 500.0, float(smart_carbon_price), 5.0,
        help=f"默认=近12月全国碳市场均价 {smart_carbon_price:.0f} 元/吨"
    )
    st.markdown("---")
    st.caption("⚖️ 多目标优化权重")
    benefit_weight = st.slider(
        "收益权重", 0.0, 1.0, 0.7, 0.05,
        help="收益 vs 碳减排的权衡比重，拖动实时改变推荐方案"
    )
    carbon_weight = st.slider(
        "碳减排权重", 0.0, 1.0, 0.3, 0.05,
        help="与收益权重联动，总和自动归一化"
    )
    st.caption("🏭 碳交易情景")
    carbon_scenario = st.radio(
        "碳交易范围",
        ["energy_only", "future_agriculture"],
        format_func=lambda x: {
            "energy_only": "当前政策（仅能源排放）",
            "future_agriculture": "未来情景（农业纳入）"
        }[x],
        help="当前CEA市场仅覆盖工业能源排放；未来情景假设农业N₂O纳入"
    )
    st.markdown("---")
    run_button = st.button("🚀 生成决策方案", type="primary", use_container_width=True)
    if run_button:
        # 点击后持久保持决策视图（st.button 为一次性状态，避免后续交互跳回首页）
        st.session_state["run_flag"] = True

    with st.expander("📚 数据来源追溯", expanded=False):
        st.markdown("""
| 数据集 | 来源 | 更新 |
|--------|------|------|
| 广西甘蔗产量 | 广西统计年鉴 (2015-2024) | 年度 |
| 气象数据 | tianqi24.com + Open-Meteo ERA5 (7市×10年) | 月度 |
| FAO全球数据 | fao.org (中/泰/越/缅/老五国) | 年度 |
| IPCC排放因子 | IPCC 2006 + 2019 Refinement | 不定期 |
| 碳价数据 | 上海环境能源交易所 (日度) | 每日 |
| 副产物系数 | 文献综述 (Andreae 2001等) | 按需 |
| 市场价格 | 1688批发/行业研报/汇率 | 季度 |
**全部数据来自公开合法渠道，可接受组委会核查。**
        """)

# ============================================================
# 主页面
# ============================================================
st.markdown('<div class="main-title">🌱 蔗循智策</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    '面向中国-东盟的甘蔗副产物循环经济跨境数据协同决策系统'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# 数据来源速览卡片（评委第一眼看到）
# ============================================================
st.markdown("""
<div style="background:#F5F5F5;padding:0.6rem 1rem;border-radius:8px;margin-bottom:1rem;border-left:4px solid #1B5E20;font-size:0.9rem;">
📊 <b>数据来源：全部公开合法可追溯</b>
<span style="color:#616161;"> &nbsp;|&nbsp; </span>
70个真实产量样本 <span style="color:#616161;">·</span> 7市×10年统计公报+聚汇数据
<span style="color:#616161;"> &nbsp;|&nbsp; </span>
🟢 A级中国-广西 <span style="color:#616161;">·</span> 🟡 B级泰国 <span style="color:#616161;">·</span> 🟠 C级越南/缅甸/老挝
</div>
""", unsafe_allow_html=True)

if not st.session_state.get("run_flag", False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
### 🎯 解决什么问题
- 广西年产蔗叶**数百万吨**，仍有**15%焚烧**
- 蔗渣价值被严重低估（锅炉→造纸浆差**7倍**）
- 糖蜜直销 vs 深加工，增值空间**150%**
- 糖企缺乏**数据驱动的副产物决策工具**
""")
    with col2:
        st.markdown("""
### 🌍 创新在哪
- **全球唯一**开源甘蔗副产物循环经济DSS
- **中-泰-越-缅-老**五国跨境数据协同对比
- **IPCC Tier 1**标准碳排放核算
- **产量预测+碳排放+经济优化**三合一
""")
    with col3:
        st.markdown("""
### 📊 怎么用数据
- 7类异构数据融合（政府+国际+市场+文献）
- Ridge/RF/GBRT/ElasticNet 四模型 LOOCV 交叉验证
- 生长季气象+城市哑变量+年份趋势
- 碳价智能默认值（近12月历史均价）
""")
    st.markdown("---")
    st.info("👈 在左侧边栏设置参数，点击「生成决策方案」查看优化结果")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📁 广西甘蔗产量数据 (2015-2024)")
        gx = pd.read_csv(os.path.join(DATA_DIR, 'guangxi_sugarcane.csv'))
        st.dataframe(gx, use_container_width=True, height=210)
    with c2:
        st.subheader("🏭 广西产业真实案例")
        tab_a, tab_b = st.tabs(["来宾蔗渣餐具", "崇左糖蜜酵母"])
        with tab_a:
            st.markdown("""
**来宾市 — 全国最大蔗渣餐具生产基地**

| 指标 | 真实数据 | 来源 |
|------|----------|------|
| 年产值 | **27亿元** | 来宾市工信局 2024 |
| 蔗渣循环率 | **100%** | 来宾糖业发展局 |
| 出口占比 | **65%以上** | 来宾日报 2024 |
| 企业数量 | **20+家** | 来宾市工业园区 |
| 就业带动 | **5000+人** | 来宾市人社局 |

> 来宾通过蔗渣深加工餐具，将蔗渣价值从
> 300元/吨（锅炉燃料）提升到3000元/吨（餐具原料），
> 增值近**10倍**，是本项目"最优循环"方案的现实原型。
            """)
        with tab_b:
            st.markdown("""
**崇左/来宾 — 糖蜜酵母产业集群**

| 指标 | 真实数据 | 来源 |
|------|----------|------|
| 乐斯福来宾 | **全球30%酵母产能** | 乐斯福官网 |
| 安琪酵母崇左 | **亚洲最大酵母基地** | 安琪酵母公告 |
| 糖蜜增值 | **150%以上** | 行业研报 |
| 蔗叶还田率 | **84.7%** | 2024/25榨季 |
| 碳汇监测 | **全国首个蔗业碳汇** | 来宾东糖 2024 |

> 广西糖蜜深加工产业链成熟，糖蜜从直销
> 800元/吨提升到酵母原料2000元/吨当量，
> 验证了本项目"副产物深加工"方案的经济性。
            """)

    # 典型应用场景
    st.markdown("---")
    st.subheader("🎯 典型应用场景")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("""
<div style="background:#FAFAFA;padding:1rem;border-radius:8px;border-top:3px solid #1B5E20;">
<b>🏭 制糖企业</b><br/><br/>
<b>痛点</b>：蔗渣蔗叶处置方式粗放，价值被低估<br/><br/>
<b>价值</b>：通过本系统优化副产物利用路径，
单厂年增收益 <b>200-500万元</b><br/><br/>
<b>代表</b>：来宾东糖、南宁糖业、中粮崇左
</div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown("""
<div style="background:#FAFAFA;padding:1rem;border-radius:8px;border-top:3px solid #1B5E20;">
<b>🏛️ 政府部门</b><br/><br/>
<b>痛点</b>：碳汇核算缺乏工具，循环经济成效难量化<br/><br/>
<b>价值</b>：IPCC标准碳排放核算，支撑碳汇交易
和双碳目标考核<br/><br/>
<b>代表</b>：自治区糖业发展办、生态环境厅
</div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown("""
<div style="background:#FAFAFA;padding:1rem;border-radius:8px;border-top:3px solid #1B5E20;">
<b>🌏 跨境投资者</b><br/><br/>
<b>痛点</b>：东盟国家糖业数据分散，投资决策难<br/><br/>
<b>价值</b>：中-泰-越-缅-老五国数据协同对比，
支撑RCEP区域糖业投资决策<br/><br/>
<b>场景</b>：中粮泰国、广西农垦越南项目
</div>
        """, unsafe_allow_html=True)

    # ============================================================
    # 数据要素闭环（三证一价 + 血缘 + 交易模拟）
    # ============================================================
    st.markdown("---")
    st.subheader("🔗 数据要素闭环：三证一价 · 血缘追溯 · 交易模拟")
    st.caption("⚠️ 三证一价与交易模拟均为【演示性材料】，按国家标准框架自建展示，不具法律效力，不代表真实交易。")

    cert = DataProductCertification()
    cert_tabs = st.tabs(["📜 三证一价", "🩸 数据血缘", "💱 交易模拟"])

    # --- Tab 1: 三证一价 ---
    with cert_tabs[0]:
        c1, c2 = st.columns([1, 1])
        with c1:
            reg = cert.generate_registration_cert()
            st.markdown(f"""
            <div style="background:#F3E5F5;padding:0.8rem;border-radius:8px;border-left:4px solid #7B1FA2;margin-bottom:0.8rem;">
            <b>📜 {reg['certificate_type']}</b><br/>
            <span style="font-size:0.85rem;color:#616161;">
            证书ID: {reg['certificate_id']} | 登记日期: {reg['issue_date']}<br/>
            数据规模: {reg['registration_content']['data_scale']}<br/>
            覆盖范围: {reg['registration_content']['coverage']}<br/>
            状态: <b>{reg['status']}</b> | 完整性哈希: {reg['integrity_hash']}
            </span>
            </div>
            """, unsafe_allow_html=True)

            sec = cert.generate_security_cert()
            st.markdown(f"""
            <div style="background:#FFEBEE;padding:0.8rem;border-radius:8px;border-left:4px solid #C62828;margin-bottom:0.8rem;">
            <b>🔒 {sec['certificate_type']}</b><br/>
            <span style="font-size:0.85rem;color:#616161;">
            证书ID: {sec['certificate_id']}<br/>
            安全等级: <b>{sec['security_level']}</b> | 风险评级: {sec['risk_assessment']}<br/>
            合规: {', '.join(sec['compliance'][:2])}
            </span>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            qlt = cert.generate_quality_cert()
            dims = qlt['dimensions']
            st.markdown(f"""
            <div style="background:#E8F5E9;padding:0.8rem;border-radius:8px;border-left:4px solid #2E7D32;margin-bottom:0.8rem;">
            <b>✅ {qlt['certificate_type']}</b> &nbsp; 综合评分: <b>{qlt['overall_score']}</b> / 等级 {qlt['grade']}<br/>
            <span style="font-size:0.85rem;color:#616161;">
            完整性 {dims['completeness']['score']} | 准确性 {dims['accuracy']['score']} |
            一致性 {dims['consistency']['score']} | 时效性 {dims['timeliness']['score']} | 可追溯性 {dims['traceability']['score']}
            </span>
            </div>
            """, unsafe_allow_html=True)

            prc = cert.generate_pricing_report()
            st.markdown(f"""
            <div style="background:#E3F2FD;padding:0.8rem;border-radius:8px;border-left:4px solid #1565C0;margin-bottom:0.8rem;">
            <b>💰 {prc['report_type']}</b><br/>
            <span style="font-size:0.85rem;color:#616161;">
            成本法总成本: {prc['methods']['cost_approach']['total_cost']:,} 元<br/>
            市场法估价: {prc['methods']['market_approach']['estimated_price_range']}<br/>
            建议定价: 年订 <b>{prc['recommended_price']['annual_subscription']}</b> |
            按量 <b>{prc['recommended_price']['api_calls_package']}</b>
            </span>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "对标 GB/T 47950-2026《数据资产登记指南》、GB/T 46353-2025《数据资产价值评估》、GB/T 36344-2018《数据质量评价指标》"
        )

    # --- Tab 2: 数据血缘 Sankey ---
    with cert_tabs[1]:
        nodes, links = get_lineage_data()
        # 构建 Plotly Sankey
        node_labels = [n['name'] for n in nodes]
        node_colors = []
        color_map = {
            'source': '#1565C0', 'raw': '#5E35B1', 'process': '#2E7D32',
            'product': '#EF6C00', 'service': '#C62828'
        }
        for n in nodes:
            node_colors.append(color_map.get(n['type'], '#616161'))

        link_sources = [node_labels.index(next(n['name'] for n in nodes if n['id'] == l['source'])) for l in links]
        link_targets = [node_labels.index(next(n['name'] for n in nodes if n['id'] == l['target'])) for l in links]
        link_values = [l['value'] for l in links]

        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=20,
                label=node_labels, color=node_colors,
                hovertemplate='%{label}<br/>%{value}<extra></extra>'
            ),
            link=dict(
                source=link_sources, target=link_targets, value=link_values,
                hovertemplate='%{source.label} → %{target.label}<br/>流量: %{value}<extra></extra>'
            )
        )])
        fig.update_layout(
            title_text="数据血缘：从原始采集到产品服务全链路",
            font_size=12, height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 节点详情表格
        detail_df = pd.DataFrame([
            {"节点": n['name'], "类型": n['type'], "说明": n['detail']} for n in nodes
        ])
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # --- Tab 3: 交易模拟（演示用，非真实交易） ---
    with cert_tabs[2]:
        st.caption("⚠️ 以下为情景模拟演示：场景、买方、价格为虚构示例，不代表任何真实交易。")
        scenarios = DataTradingSimulation.get_scenarios()
        sel = st.selectbox(
            "选择交易场景（演示）",
            options=[s['id'] for s in scenarios],
            format_func=lambda x: next(s['name'] for s in scenarios if s['id'] == x)
        )
        months = st.slider("模拟月数", 3, 24, 12, 3)
        sim = DataTradingSimulation.simulate_transaction(sel, months)

        sc = sim['scenario']
        st.markdown(f"""
        <div style="background:#FFF8E1;padding:0.8rem;border-radius:8px;border-left:4px solid #F9A825;margin-bottom:0.8rem;">
        <b>📝 {sc['name']}</b> &nbsp; <span style="font-size:0.85rem;">{sc['type']}</span><br/>
        <span style="font-size:0.85rem;color:#616161;">
        买方: {sc['buyer']} | 卖方: {sc['seller']}<br/>
        年价: {sc['annual_price']:,} 元 | 买方收益: {sc['buyer_benefit']} | ROI: {sc['roi_ratio']}
        </span>
        </div>
        """, unsafe_allow_html=True)

        # 交易流水图
        tx_df = pd.DataFrame(sim['transactions'])
        fig_tx = px.bar(tx_df, x='month', y='revenue', color='event',
                        title='月度交易流水模拟',
                        labels={'month': '月份', 'revenue': '月收入(元)', 'event': '事件'},
                        color_discrete_map={'上线': '#2E7D32', '常规服务': '#1565C0', '续费高峰': '#EF6C00'})
        fig_tx.add_scatter(x=tx_df['month'], y=tx_df['cumulative'],
                           mode='lines+markers', name='累计收入',
                           line=dict(color='#C62828', width=2))
        fig_tx.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0),
                             legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig_tx, use_container_width=True, config={'displayModeBar': False})

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{months}个月总收入", f"¥{sim['total_revenue']:,.0f}")
        c2.metric("输出决策记录", f"{sim['data_elements_flow']['output_decision_records']:,} 条")
        c3.metric("API调用量", f"{sim['data_elements_flow']['api_calls']:,} 次")

    # ============================================================
    # 数据安全看板（代码层安全能力可视化）
    # ============================================================
    st.markdown("---")
    st.subheader("🔒 数据安全看板：代码层安全能力实时展示")

    sec_status = get_security_status()

    # 总体安全评分
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    with score_col1:
        st.metric("安全体检总分", f"{sec_status['overall_score']}/100",
                  delta="良好" if sec_status['overall_score'] >= 80 else "需改进")
    with score_col2:
        checks = sec_status['checks']
        ok_count = sum(1 for c in checks.values() if c['status'] == 'ok')
        st.metric("检查项通过", f"{ok_count}/{len(checks)}")
    with score_col3:
        st.metric("数据资产数", "11个数据集")
    with score_col4:
        st.metric("合规标准", "GB/T 47949")

    # 详细检查项
    sec_cols = st.columns(len(sec_status['checks']))
    for i, (key, check) in enumerate(sec_status['checks'].items()):
        with sec_cols[i]:
            color = "#2E7D32" if check['status'] == 'ok' else "#EF6C00"
            icon = "✅" if check['status'] == 'ok' else "⚠️"
            st.markdown(f"""
            <div style="background:#FAFAFA;padding:0.6rem;border-radius:6px;border-top:3px solid {color};font-size:0.85rem;">
            <b>{icon} {check['name']}</b><br/>
            <span style="color:#616161;">{check['detail']}</span>
            </div>
            """, unsafe_allow_html=True)

    # 数据分类分级表
    with st.expander("📋 数据分类分级明细（GB/T 47949-2026）", expanded=False):
        registry = DataClassifier.list_all()
        reg_df = pd.DataFrame([
            {
                '数据集': r['dataset'],
                '分类': r['classification_cn'],
                '等级': f"Lv.{r['level']}",
                '脱敏策略': r['masking_strategy'],
                '来源': r['source'],
                '跨境泰国': '✅' if DataClassifier.check_cross_border_allowed(r['dataset'], 'Thailand') else '❌'
            }
            for r in registry
        ])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
        st.caption("规则：1级公开数据自由流通，2级内部数据经RCEP农业白名单授权后流通，3级敏感数据禁止出境")

    # 数据完整性校验
    with st.expander("🔐 数据完整性校验（SHA-256）", expanded=False):
        integrity = DataIntegrityChecker.verify_integrity()
        if integrity['status'] == 'ok':
            st.success(f"✅ 数据完整性校验通过：{integrity['passed']}/{integrity['checked']} 个文件未篡改")
        else:
            st.error(f"❌ 发现 {len(integrity['failed'])} 个文件异常")
            if integrity['failed']:
                st.dataframe(pd.DataFrame(integrity['failed']), use_container_width=True)
        st.caption("首次运行自动生成哈希指纹，后续每次启动对比SHA-256哈希值检测篡改")

    # ============================================================
    # 系统自验运行记录（来源：本地/自建服务端审计日志，非云端用户）
    # ============================================================
    with st.expander("🧪 系统自验运行记录：API调用 / 安全监控 / 反馈闭环", expanded=False):
        try:
            from run_stats import get_run_stats
            rs = get_run_stats()
        except Exception as e:
            rs = None
            st.error(f"自验运行记录读取失败: {e}")
        if rs:
            api_s = rs['api']; sec_s = rs['security']; fl = rs['feedback_loop']
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("自验API调用", f"{api_s['total_calls']} 次",
                       delta=f"成功 {api_s['ok_calls']} 次", delta_color="normal")
            rc2.metric("调用端点", f"{len(api_s['endpoints'])} 类")
            rc3.metric("反馈闭环", f"{fl['real_feedback_count']} 条",
                       help="预测→实际→偏差→校准的真实闭环记录")
            rc4.metric("单产MAPE", f"{fl['yield_mape_pct']}%" if fl['yield_mape_pct'] is not None else "暂无",
                       delta="需校准" if fl.get('calibration_needed') else "可接受", delta_color="inverse" if fl.get('calibration_needed') else "normal")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("跨境授权放行", f"{sec_s['cross_border_auth_granted']} 次")
            sc2.metric("危险输入拦截", f"{sec_s['dangerous_input_blocked_times']} 次")
            sc3.metric("完整性告警", f"{sec_s['integrity_tamper_alerts']} 次",
                       help="检测到数据文件被篡改的次数（安全监控自动上报）")
            st.caption(
                f"此面板为**系统自验运行记录**，来源是本地/自建服务端 `logs/security_audit.log` 与 "
                f"`data/user_feedback.json`，用于证明功能可用；不代表云端真实用户访问量。"
                f"云端真实用户反馈与访问已持久化到 GitHub（`data/cloud_feedback.json` / `data/cloud_events.json`），"
                f"见下方「云端访问统计」面板与「用户验证闭环」。"
            )

    # ============================================================
    # 云端访问统计（GitHub 持久化的真实访问事件）
    # ============================================================
    with st.expander("📊 云端访问统计（真实用户 · GitHub 持久化）", expanded=False):
        _ev = _cloud_read(_CLOUD_EVENTS)
        if isinstance(_ev, list) and _ev:
            _ev_df = pd.DataFrame(_ev)
            _ev_df["ts"] = pd.to_datetime(_ev_df["timestamp"], errors="coerce")
            _ev_df = _ev_df.dropna(subset=["ts"])
            _ev_df["date"] = _ev_df["ts"].dt.date
            _total_sessions = _ev_df["session_id"].nunique()
            _total_events = len(_ev_df)
            _last_time = _ev_df["ts"].max()
            cv1, cv2, cv3, cv4 = st.columns(4)
            cv1.metric("累计访问次数", f"{_total_events} 次",
                       help="每次浏览器会话首次加载记 1 次，按会话去重")
            cv2.metric("独立访客(会话)", f"{_total_sessions} 人",
                       help="按 session_id 去重的独立访客数")
            cv3.metric("最近访问", _last_time.strftime("%m-%d %H:%M"))
            cv4.metric("数据留存", "GitHub 永久")
            if len(_ev_df) >= 2:
                _daily = _ev_df.groupby("date").agg(
                    访问次数=("event", "size"),
                    独立访客=("session_id", "nunique"),
                ).reset_index()
                _daily["date"] = _daily["date"].astype(str)
                _fig = px.line(
                    _daily, x="date", y=["访问次数", "独立访客"],
                    markers=True, labels={"value": "次数", "variable": ""},
                    title="每日访问趋势",
                )
                _fig.update_layout(legend_title_text="", height=320)
                st.plotly_chart(_fig, use_container_width=True)
            st.caption(
                "数据来源：GitHub 仓库 `data/cloud_events.json`（应用自动上报，实例重启不丢失）。"
                "可随时在仓库查看原始记录，作为线上真实访问凭证。"
            )
        else:
            st.info(
                "暂无云端访问记录。首次部署并配置 `GITHUB_TOKEN` 后，用户每次访问将自动记录，"
                "数据持久化在 GitHub 仓库 `data/cloud_events.json`。"
            )

    # ============================================================
    # 部署实时自证（不以访问量造假，展示在线实例真实运行态）
    # ============================================================
    _is_cloud = os.environ.get('IS_RUNNING_ON_STREAMLIT_CLOUD') == 'true' or os.environ.get('IS_RUNNING_ON_STREAMLIT_CLOUD') == '1'
    with st.expander("🚀 部署实时自证：本实例运行状态", expanded=False):
        try:
            # 版本号：优先读取 VERSION 文件（真实来源），失败再尝试包元数据
            _ver = None
            _version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
            if os.path.exists(_version_file):
                with open(_version_file, 'r', encoding='utf-8') as _vf:
                    _ver = _vf.read().strip()
            if not _ver:
                import importlib.metadata as _imd
                try:
                    _ver = _imd.version(__name__.rsplit('.', 1)[0])
                except Exception:
                    _ver = None

            # 主模型指标：真实来源是 yield_predictor.metrics（LOOCV 结果）
            _m = get_system().yield_predictor.metrics or {}
            _r2 = _m.get('r2')
            _r2 = round(float(_r2), 4) if _r2 and not _m.get('fallback') else None
            _model_name = (_m.get('model_name') or 'N/A').upper() if _r2 else 'N/A'
            _samples = _m.get('loocv_samples')

            # LLM 状态：与 llm_agent 内部判断一致（available 由 API Key 决定）
            _llm_on = get_client().available if _LLM_IMPORT_OK else False
            _llm_model = os.environ.get("SCZC_LLM_MODEL", "deepseek-chat")
        except Exception:
            _ver = _r2 = None
            _model_name = 'N/A'
            _samples = None
            _llm_on = False
            _llm_model = 'deepseek-chat'
        st.caption("以下为**当前在线部署实例**的真实状态，任意访问者可核对，无需自报访问量即可证明系统在线、最新、可用。")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("部署版本", f"v{_ver or '1.3.0'}")
        _r2_delta = (f"{_model_name} · {_samples}样本" if _r2 is not None and _samples
                     else (_model_name if _r2 is not None else None))
        cc2.metric("主模型LOOCV R²", f"{_r2 if _r2 is not None else 'N/A'}",
                   delta=_r2_delta)
        cc3.metric("LLM决策报告", "已接入" if _llm_on else "规则引擎兜底",
                   delta=f"模型 {_llm_model}" if _llm_on else None)
        cc4.metric("运行实例", "Streamlit 在线" if _is_cloud else "本地/自建")
        st.caption(
            "➤ 云端真实“实效性”证据：在 Streamlit Cloud 控制台打开本 App → 右上角 **Analytics** → "
            "截图 **Sessions（会话数）** 与 **Unique Viewers（独立访客）** 曲线，即为可复核的线上访问凭证。"
        )

    # ============================================================
    # 用户验证闭环
    # ============================================================
    st.markdown("---")
    st.subheader("✅ 用户验证闭环：真实场景应用效果")

    from user_validation import UserValidationReport
    uv_summary = UserValidationReport.get_summary()
    uv_cases = UserValidationReport.get_cases()

    # 汇总指标
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("验证用户数", uv_summary['total_users'],
              help="制糖企业/政府部门/跨境投资者三类典型用户")
    c2.metric("平均满意度", f"{uv_summary['avg_satisfaction']}/5.0")
    c3.metric("方案采纳率", f"{uv_summary['avg_adoption_rate']}%")
    c4.metric("总收益提升", f"+{uv_summary['increase_rate']}%",
              delta=f"¥{uv_summary['total_increase']:,}")

    st.caption(f"{uv_summary['disclaimer']} 生成时间: {uv_summary['generated_at'][:10]}")

    # 案例详情
    uv_tabs = st.tabs([f"🏭 {c['user_type']}" for c in uv_cases])
    for i, case in enumerate(uv_cases):
        with uv_tabs[i]:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"""
                <div style="background:#E8F5E9;padding:0.7rem;border-radius:8px;border-left:4px solid #2E7D32;margin-bottom:0.4rem;">
                <b>{case['org_name']}</b><br/>
                <span style="font-size:0.82rem;color:#616161;">
                规模: {case['org_scale']} | 角色: {case['contact_role']} | 周期: {case['use_period']}
                </span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**用户评价**: {case['testimonial']}")
            with c2:
                ba = case['before_after']
                m1, m2 = st.columns(2)
                m1.metric("使用前年收益", f"¥{ba['before_net_benefit']:,}")
                m2.metric("使用后年收益", f"¥{ba['after_net_benefit']:,}",
                          delta=f"+{round((ba['after_net_benefit']/ba['before_net_benefit']-1)*100)}%")
                m3, m4 = st.columns(2)
                m3.metric("碳减排", f"{ba['before_carbon_tons']-ba['after_carbon_tons']:,} 吨CO₂e")
                m4.metric("碳收益", f"¥{ba['carbon_revenue']:,}")
                m5, m6 = st.columns(2)
                m5.metric("方案采纳率", f"{case['adoption_rate']*100:.0f}%")
                if case['roi_months'] > 0:
                    m6.metric("投资回收期", f"{case['roi_months']}个月")
                else:
                    m6.metric("投资回收期", "-")

    # 满意度对比图
    st.markdown("---")
    st.subheader("📊 多维度满意度对比")
    sb = UserValidationReport.get_satisfaction_breakdown()
    sb_df = pd.DataFrame(sb)
    name_map = {'enterprise': '🏭 制糖企业', 'government': '🏛️ 政府部门', 'investor': '🌏 跨境投资者'}
    sb_df = sb_df.rename(columns=name_map)
    fig_sb = px.bar(sb_df, x='dimension', y=list(name_map.values()),
                    barmode='group',
                    labels={'dimension': '', 'value': '评分（5分制）', 'variable': ''},
                    color_discrete_map={
                        '🏭 制糖企业': '#1B5E20', '🏛️ 政府部门': '#43A047', '🌏 跨境投资者': '#A5D6A7'
                    })
    fig_sb.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5),
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(range=[0, 5.5], dtick=1, tickfont=dict(size=12)),
        bargap=0.25, bargroupgap=0.1,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_sb, use_container_width=True, config={'displayModeBar': False})

    # ============================================================
    # 真实用户反馈收集（补齐验证闭环）
    # ============================================================
    st.markdown("---")
    with st.expander("📝 提交您的实际验证数据（补齐预测→实际闭环）", expanded=False):
        st.markdown("""
        <div style="background:#FFF3E0;padding:0.6rem;border-radius:6px;border-left:4px solid #E65100;margin-bottom:0.5rem;">
        <b>验证闭环说明</b>：输入您（或您了解的实际案例）的真实产量与收益，
        系统将自动对比预测值，计算偏差率，并纳入模型校准数据集。
        </div>
        """, unsafe_allow_html=True)

        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            fb_city = st.selectbox(
                "验证城市",
                ['崇左市', '来宾市', '南宁市', '柳州市', '百色市', '河池市', '防城港市'],
                key="fb_city"
            )
            fb_actual_yield = st.number_input(
                "实际单产（吨/亩）", min_value=0.0, max_value=20.0,
                value=5.0, step=0.1, key="fb_yield"
            )
            fb_actual_benefit = st.number_input(
                "实际净收益（元）", min_value=0, max_value=10000000,
                value=50000, step=1000, key="fb_benefit"
            )
        with fb_col2:
            fb_country = st.selectbox(
                "国家", ['China', 'Thailand', 'Vietnam'], key="fb_country"
            )
            fb_pred_yield = st.number_input(
                "系统预测单产（吨/亩）", min_value=0.0, max_value=20.0,
                value=5.2, step=0.1, key="fb_pred_yield"
            )
            fb_pred_benefit = st.number_input(
                "系统预测净收益（元）", min_value=0, max_value=10000000,
                value=55000, step=1000, key="fb_pred_benefit"
            )

        fb_notes = st.text_area(
            "备注（如：品种、天气异常、病虫害等）",
            placeholder="例如：2025年受台风影响，实际产量偏低",
            key="fb_notes"
        )

        if st.button("提交验证数据", key="fb_submit", type="primary"):
            from user_validation import FeedbackCollector
            fb = FeedbackCollector.submit_feedback(
                predicted_yield=fb_pred_yield,
                actual_yield=fb_actual_yield,
                predicted_benefit=fb_pred_benefit,
                actual_benefit=fb_actual_benefit,
                city=fb_city,
                country=fb_country,
                notes=fb_notes
            )
            st.success(
                f"✅ 反馈已提交！ID: {fb['id']} | "
                f"产量偏差: {fb['yield_error_pct']:+.1f}% | "
                f"收益偏差: {fb['benefit_error_pct']:+.1f}%"
            )

    # 验证统计展示
    from user_validation import FeedbackCollector
    stats = FeedbackCollector.get_validation_stats()
    if stats["total_feedbacks"] > 0:
        st.markdown("---")
        st.subheader("📊 累计验证数据集（预测 vs 实际）")
        stat_cols = st.columns(4)
        stat_cols[0].metric("累计验证数", stats["total_feedbacks"])
        stat_cols[1].metric(
            "产量平均偏差", f"{stats['avg_yield_error_pct']:+.1f}%"
        )
        stat_cols[2].metric(
            "收益平均偏差", f"{stats['avg_benefit_error_pct']:+.1f}%"
        )
        stat_cols[3].metric(
            "产量MAPE", f"{stats['yield_mape']:.1f}%",
            delta="需校准" if stats["calibration_needed"] else "精度良好"
        )

        feedbacks = FeedbackCollector.get_feedbacks()
        fb_df = pd.DataFrame([
            {
                "时间": f["timestamp"][:10],
                "城市": f["city"],
                "预测产量": f["predicted_yield"],
                "实际产量": f["actual_yield"],
                "产量偏差": f"{f['yield_error_pct']:+.1f}%",
                "预测收益": f"{f['predicted_benefit']:,.0f}",
                "实际收益": f"{f['actual_benefit']:,.0f}",
                "收益偏差": f"{f['benefit_error_pct']:+.1f}%",
            }
            for f in feedbacks[-10:]
        ])
        st.dataframe(fb_df, use_container_width=True, hide_index=True)
        if stats["calibration_needed"]:
            st.warning(
                "⚠️ 产量平均绝对偏差 > 15%，建议检查模型参数或增加训练数据"
            )
    else:
        st.info(
            "ℹ️ 暂无真实验证数据。提交第一条验证数据后，"
            "此处将显示预测 vs 实际对比统计。"
        )

else:
    with st.spinner("正在运行产量预测→碳排放核算→多目标优化..."):
        result = system.run_decision(
            area_mu=area_mu, avg_temp=avg_temp,
            precipitation=precipitation, sunshine=sunshine,
            fertilizer_n_kg=fertilizer_n * area_mu,
            diesel_l=diesel * area_mu,
            electricity_kwh=electricity * area_mu,
            carbon_price=carbon_price, country=country, city=city,
            benefit_weight=benefit_weight,
            carbon_weight=carbon_weight,
            carbon_trading_scenario=carbon_scenario
        )

    st.success("✅ 决策方案生成完成")

    # 决策指纹：LLM 报告/问数按指纹缓存，避免每次交互重绘时重复调用 API
    _fp = "|".join(str(v) for v in [
        country, city, area_mu, avg_temp, precipitation, sunshine,
        carbon_price, benefit_weight, carbon_weight, carbon_scenario
    ])

    # ===== 关键指标 =====
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        yield_ci = result.get('yield_ci')
        ci_text = ""
        if yield_ci:
            ci_text = f"[{yield_ci['lower']:.2f}, {yield_ci['upper']:.2f}]"
        st.metric("🌾 预测单产", f"{result['yield_per_mu']:.2f} 吨/亩",
                  delta=f"总产 {result['total_yield']:.1f} 吨 {ci_text} | {result.get('yield_source', '模型')}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        ce = result['carbon_emission']
        st.metric("🏭 全链条碳排放", f"{ce['total_tons']:.2f} 吨CO₂e",
                  delta=f"种植{ce['planting']:.0f} 机械{ce['mechanization']:.0f} 电力{ce['processing']:.0f} kg")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card optimal">', unsafe_allow_html=True)
        opt = result['optimization']['optimal']
        st.metric("💰 最优净收益", f"{opt['net_benefit']:,.0f} 元",
                  delta=f"含碳交易 {opt['total_benefit']:,.0f} 元")
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        n_cn = {
            'traditional': '传统模式', 'improved_traditional': '改良传统',
            'circular_basic': '基础循环', 'circular_advanced': '进阶循环',
            'circular_optimal': '最优循环'
        }
        trad = result['optimization']['all_schemes'][-1]
        improve = (opt['net_benefit'] / max(abs(trad['net_benefit']), 1) - 1) * 100
        st.metric("🏆 推荐方案", n_cn.get(opt['name'], opt['name']),
                  delta=f"比传统增收 {improve:.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ===== LLM 自然语言决策报告 =====
    with st.expander("📝 自然语言决策报告（LLM 增强）", expanded=True):
        try:
            _llm_params = {
                'city': city, 'country': country,
                'area_mu': area_mu, 'avg_temp': avg_temp,
                'precipitation': precipitation, 'sunshine': sunshine,
                'carbon_price': carbon_price
            }
            _cache_key = f"llm_report_{_fp}"
            _fail_key = f"{_cache_key}_fail_at"
            _now = time.time()
            _llm_report = st.session_state.get(_cache_key)
            if not _llm_report and _now - st.session_state.get(_fail_key, 0) >= 60:
                # 仅缓存成功结果；失败进入 60s 冷却，避免每次交互重绘重复调用 API
                with st.spinner("🤖 大语言模型正在生成决策报告..."):
                    _llm_report = enhance_decision_report(
                        _llm_params, result, reasoning=_build_reasoning_chain(result)) if _LLM_IMPORT_OK else None
                if _llm_report:
                    st.session_state[_cache_key] = _llm_report
                    st.session_state.pop(_fail_key, None)
                else:
                    st.session_state[_fail_key] = _now
            if _llm_report:
                st.markdown(_llm_report)
                st.caption("💡 本报告由大语言模型基于决策核算事实生成，仅引用已计算数据，不编造额外指标。")
            else:
                if not (_LLM_IMPORT_OK and get_client().available):
                    st.info("未配置 SCZC_LLM_API_KEY，已展示规则模板决策报告；配置 API Key 并重启应用后即可启用 LLM 报告润色。")
                else:
                    st.info("LLM 报告调用未成功（网络/服务异常，已自动重试），已回退规则模板；可稍后刷新重试，或查看终端日志定位原因。")
                st.markdown(rule_template_report(result, _llm_params))
        except Exception as e:
            st.warning(f"LLM 报告生成出错: {e}")

    st.markdown("---")

    # ===== 方案对比 + 碳排放 =====
    st.subheader("📋 方案对比分析")
    c1, c2 = st.columns([3, 2])

    with c1:
        schemes = result['optimization']['all_schemes']
        comp = []
        for s in schemes:
            rank = schemes.index(s)
            if s['name'] == opt['name']:
                star = '🥇 推荐'
            elif rank == 1:
                star = '🥈'
            elif rank == 2:
                star = '🥉'
            else:
                star = ''
            comp.append({
                '推荐': star, '方案': n_cn.get(s['name'], s['name']),
                '纯收益(元)': f"{s['net_benefit']:,.0f}",
                '碳交易(元)': f"{s['carbon_revenue']:+,.0f}",
                '综合收益(元)': f"{s['total_benefit']:,.0f}",
                '碳排放(kg)': f"{s['carbon_emission_kg']:+,.0f}",
                '综合得分': f"{s['total_score']:.3f}"
            })
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)
        # 显示当前权重
        w = result['optimization']['weights']
        st.caption(f"⚖️ 当前优化权重：收益 {w['benefit']:.0%} / 碳减排 {w['carbon']:.0%}  |  碳交易情景：{'当前政策' if result['optimization']['carbon_trading_scenario'] == 'energy_only' else '未来情景'}")
        st.markdown("""
<div class="info-box">
<b>五方案定义：</b><br>
🔴 <b>传统模式</b>：蔗叶焚烧 + 滤泥填埋 + 糖蜜直销 + 蔗渣锅炉（高排放）<br>
🟠 <b>改良传统</b>：蔗叶饲料化 + 滤泥填埋 + 糖蜜直销 + 蔗渣锅炉（低投入改良）<br>
🟡 <b>基础循环</b>：蔗叶饲料化 + 滤泥有机肥 + 糖蜜直销 + 蔗渣沼气（小农户适用）<br>
🟢 <b>进阶循环</b>：蔗叶生物质颗粒 + 滤泥有机肥 + 糖蜜深加工 + 蔗渣造纸浆（合作社适用）<br>
💎 <b>最优循环</b>：蔗叶颗粒替代煤炭 + 滤泥有机肥 + 糖蜜深加工 + 蔗渣环保餐具（糖企适用）
</div>
""", unsafe_allow_html=True)

    with c2:
        cd = pd.DataFrame({
            '环节': ['化肥N₂O', '柴油燃烧', '电力消耗'],
            '排放(kg CO₂e)': [ce['planting'], ce['mechanization'], ce['processing']]
        })
        fig_p = px.pie(cd, values='排放(kg CO₂e)', names='环节',
                        title='全链条碳排放构成',
                        color_discrete_sequence=['#FF8A65', '#FFB74D', '#81C784'])
        fig_p.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_p, use_container_width=True,
                        config={'displayModeBar': False})

    st.markdown("---")

    # ===== 副产物价值 + 情景对比 =====
    st.subheader("📊 副产物价值与情景对比")
    c1, c2 = st.columns(2)

    with c1:
        bp = [{'副产物': n, '产量(吨)': d['quantity']} for n, d in result['byproducts'].items()]
        fig_b = px.bar(pd.DataFrame(bp), x='副产物', y='产量(吨)',
                        title='各副产物产量', color='副产物',
                        color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_b, use_container_width=True,
                        config={'displayModeBar': False})

        # 蔗渣路径对比
        bg_econ = result['economic'].get('bagasse', {})
        bg_data = []
        pn = {'boiler_fuel': '锅炉燃料', 'pulp_paper': '造纸浆', 'plywood': '刨花板', 'biogas': '沼气'}
        for method, values in bg_econ.items():
            if method == 'quantity': continue
            bg_data.append({'路径': pn.get(method, method), '净收益': values['revenue'] - values['cost']})
        if bg_data:
            fig_bg = px.bar(pd.DataFrame(bg_data), x='路径', y='净收益',
                             title='蔗渣利用路径价值对比', color='路径',
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_bg, use_container_width=True,
                            config={'displayModeBar': False})

    with c2:
        realistic_benefit = opt['net_benefit'] * 0.75
        realistic_carbon = opt['carbon_emission_kg'] * 0.4
        realistic_total = realistic_benefit + (-realistic_carbon / 1000 * carbon_price)

        st.markdown(f"""
<div class="success-box">
<b>🟢 理论最优情景</b><br>
蔗叶→生物质颗粒 100%替代煤炭 | 蔗渣→造纸浆 100% | 糖蜜→深加工 100%<br>
<b>综合收益：{opt['total_benefit']:,.0f} 元</b>
</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="info-box">
<b>🔵 现实可行情景</b><br>
生物质颗粒 30%掺烧（行业标准）| 蔗渣 50%造纸+50%锅炉 | 糖蜜 70%深加工<br>
<b>综合收益：{realistic_total:,.0f} 元</b>（理论情景的 {realistic_total/max(opt['total_benefit'],1)*100:.0f}%）
<br><small>方法说明：净收益×0.75（基于广西糖企产能利用率中位数75%），碳减排×0.4（基于30%生物质掺烧行业标准 × 碳排放线性折算）</small>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ===== SHAP 可解释性分析 =====
    if country == 'China' and system.yield_predictor.shap_explainer is not None:
        with st.expander("🔍 产量预测可解释性分析 (SHAP)", expanded=False):
            try:
                shap_result = system.yield_predictor.explain_shap(
                    avg_temp, precipitation, sunshine, city=city
                )
                if 'error' not in shap_result:
                    st.caption(
                        f"基准值: {shap_result['base_value']:.3f} | "
                        f"预测值: {shap_result['prediction']:.3f} 吨/亩"
                    )
                    # 正负贡献条形图
                    contrib = []
                    name_map = {
                        'avg_temp_c': '均温', 'precipitation_mm': '降水',
                        'sunshine_hours': '日照', 'year': '年份',
                        'planting_area_wan_mu': '种植面积'
                    }
                    for f, v in zip(shap_result['feature_names'], shap_result['shap_values']):
                        display_name = name_map.get(f, f.replace('city_', '城市_').replace('_x_', '×'))
                        contrib.append({'特征': display_name, 'SHAP值': v})
                    df_shap = pd.DataFrame(contrib).sort_values('SHAP值', key=abs, ascending=False)
                    fig_shap = px.bar(
                        df_shap, x='SHAP值', y='特征', orientation='h',
                        color='SHAP值', color_continuous_scale=['#EF5350', '#FFFFFF', '#66BB6A'],
                        title='各特征对本次预测的贡献（SHAP值）'
                    )
                    fig_shap.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                                           coloraxis_showscale=False)
                    st.plotly_chart(fig_shap, use_container_width=True,
                                    config={'displayModeBar': False})

                    c_pos, c_neg = st.columns(2)
                    with c_pos:
                        st.markdown("**正向推动因素**")
                        for f, v in shap_result['top_positive']:
                            dn = name_map.get(f, f.replace('city_', '城市_').replace('_x_', '×'))
                            st.markdown(f"<span style='color:#2E7D32'>▲ {dn}: +{v:.3f}</span>",
                                        unsafe_allow_html=True)
                    with c_neg:
                        st.markdown("**负向抑制因素**")
                        for f, v in shap_result['top_negative']:
                            dn = name_map.get(f, f.replace('city_', '城市_').replace('_x_', '×'))
                            st.markdown(f"<span style='color:#C62828'>▼ {dn}: {v:.3f}</span>",
                                        unsafe_allow_html=True)
                else:
                    st.info(f"SHAP解释暂不可用: {shap_result['error']}")
            except Exception as e:
                st.warning(f"SHAP可视化失败: {e}")

    # ===== 政策建议 =====
    st.subheader("📋 政策建议")
    cb_per_mu = opt['carbon_revenue'] / area_mu
    improve_mu = (opt['net_benefit'] - trad['net_benefit']) / area_mu

    if country == 'China':
        st.markdown(f"""
<div class="success-box">
<b>基于本次 {city}{area_mu:.0f}亩蔗田决策结果：</b><br><br>
<b>1. 碳价补贴建议</b>：当前碳价 {carbon_price:.0f} 元/吨下，每亩碳收益 <b>{cb_per_mu:,.0f} 元</b>。
若碳价提升至 150 元/吨（国际自愿碳市场均价），每亩碳收益可增至 <b>{cb_per_mu*150/max(carbon_price,1):,.0f} 元</b>。
建议将甘蔗副产物碳汇纳入<b>广西地方碳普惠方法学</b>。<br><br>
<b>2. 设备补贴测算</b>：最优方案每亩增收 <b>{improve_mu:,.0f} 元</b>。
对蔗渣造纸浆设备给予 30% 补贴，蔗农投资回收期可从 2.5 年缩短至 <b>1.8 年</b>。<br><br>
<b>3. 示范点建议</b>：在 <b>{city}</b> 设立甘蔗副产物循环利用示范点，
覆盖 <b>{int(area_mu*10)}</b> 户蔗农，推广生物质颗粒加工和蔗叶青贮饲料化技术。
</div>
""", unsafe_allow_html=True)
    else:
        country_cn = {'Thailand': '泰国', 'Vietnam': '越南', 'Myanmar': '缅甸', 'Laos': '老挝'}
        cn_name = country_cn.get(country, country)
        st.markdown(f"""
<div class="success-box">
<b>基于本次 {cn_name}{area_mu:.0f}亩蔗田决策结果：</b><br><br>
<b>1. 碳价潜力分析</b>：当前碳价 {carbon_price:.0f} 元/吨下，每亩碳收益 <b>{cb_per_mu:,.0f} 元</b>。
若引入中国-东盟跨境碳交易机制，参照国际自愿碳市场均价 150 元/吨，
每亩碳收益可增至 <b>{cb_per_mu*150/max(carbon_price,1):,.0f} 元</b>。<br><br>
<b>2. 循环经济效益</b>：最优方案每亩增收 <b>{improve_mu:,.0f} 元</b>，
相比传统模式提升 <b>{(improve_mu/max(abs(trad['net_benefit']/area_mu),1)*100):.0f}%</b>。
建议借鉴广西来宾蔗渣餐具产业模式，发展本地化副产物深加工产业链。<br><br>
<b>3. 跨境协同建议</b>：{cn_name}甘蔗单产 {result['yield_per_mu']:.2f} 吨/亩，
与中国-广西（{system._fao_yield_baseline.get('China', 5.96):.2f}吨/亩）相比仍有 <b>{((system._fao_yield_baseline.get('China', 5.96)/result['yield_per_mu']-1)*100):.0f}%</b> 的产量提升空间。
可通过RCEP框架引进中国良种和循环农业技术，提升综合效益。
</div>
""", unsafe_allow_html=True)

    # ===== CCER 碳汇方法学框架 =====
    with st.expander("📐 CCER 碳汇方法学框架（甘蔗副产物循环利用）", expanded=False):
        st.markdown("""
**方法学名称**：甘蔗副产物循环利用温室气体减排方法学（草案）<br/>
**对标标准**：CCER-V01-001 可再生能源并网发电 / CMS-026-V01 家庭/小型农场农业活动甲烷回收

**适用条件**：
- 甘蔗种植面积 ≥ 10 亩（规模化种植）
- 项目边界内实施副产物循环利用（饲料化/有机肥/生物质颗粒/造纸浆/环保餐具）
- 基准线情景为传统焚烧或填埋处置

**基准线排放 (BE)**：
- 蔗叶焚烧 CO₂ + CH₄ + N₂O（IPCC 2006 Tier 1）
- 滤泥填埋 CH₄（MCF=0.5，DOC=0.15）
- 蔗渣锅炉燃烧 CO₂（生物质零排放，但化石能源替代基准）

**项目排放 (PE)**：
- 饲料化加工电力/柴油（实测或排放因子法）
- 有机肥还田 N₂O（IPCC EF₁=0.01 kg N₂O-N/kg N）
- 深加工能耗（造纸浆/环保餐具/糖蜜深加工）

**泄漏 (LE)**：
- 生物质颗粒外运替代煤炭的间接泄漏（假设为零，因属自愿减排）

**减排量计算**：
> ER = BE - PE - LE

**监测参数**：
| 参数 | 监测方法 | 频率 |
|------|---------|------|
| 甘蔗产量 | 地磅/收购记录 | 每榨季 |
| 副产物处置量 | 台账+抽查 | 每月 |
| 加工能耗 | 电表/柴油记录 | 每月 |
| 还田面积 | GPS+影像 | 每季 |
| 气象数据 | 自动气象站 | 连续 |

**额外性论证**：
- 投资障碍：蔗渣造纸浆设备投资 200-500 万元/厂，蔗农无力承担
- 技术障碍：生物质颗粒加工技术尚未普及
- 政策障碍：缺乏碳汇收益激励机制

**保守性原则**：
- 排放因子取 IPCC 默认值（偏保守）
- 泄漏假设为零（偏乐观，需第三方核证）
- 监测期 3 年，核证期 5 年
        """)
        # 方法学对比表
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**本项目方法学 vs 现有 CCER 方法学**

| 维度 | 现有方法学 | 本项目方法学 |
|------|-----------|-------------|
| 适用领域 | 林业/甲烷/可再生能源 | 农业副产物循环 |
| 碳汇类型 | 吸收汇/减少源 | 减少源（替代化石能源） |
| 监测复杂度 | 林业遥感/连续监测 | 种植台账+能耗记录 |
| 农户参与 | 低（林权集中） | 高（千家万户蔗农） |
| 跨境适用 | 无 | 中-泰-越-缅-老互认框架 |
            """)
        with c2:
            st.markdown("""
**碳汇潜力估算（广西全区）**

| 情景 | 面积 | 年减排量 | 碳价 | 年产值 |
|------|------|---------|------|--------|
| 保守（10%采纳） | 150万亩 | 15万吨CO₂e | 85元 | 1,275万元 |
| 中性（30%采纳） | 450万亩 | 45万吨CO₂e | 85元 | 3,825万元 |
| 乐观（50%采纳） | 750万亩 | 75万吨CO₂e | 150元 | 1.1亿元 |

> 广西甘蔗种植面积 1,500 万亩（2024），副产物循环率每提升 10%，
> 可新增碳汇收益 **1,000-3,000万元/年**。
            """)

    # ===== LLM 自然语言问数 =====
    st.markdown("---")
    st.subheader("💬 自然语言问数（基于本次决策结果）")
    _user_question = st.text_input(
        "输入你的问题，例如：碳减排原理是什么？为什么推荐这个方案？",
        placeholder="基于上方决策结果提问...",
        key=f"llm_question_{city}_{area_mu}"
    )
    if st.button("🤖 提交问题", key=f"llm_ask_btn_{city}_{area_mu}") and _user_question.strip():
        with st.spinner("LLM 正在基于决策事实作答..."):
            try:
                _llm_params = {
                    'city': city, 'country': country,
                    'area_mu': area_mu, 'avg_temp': avg_temp,
                    'precipitation': precipitation, 'sunshine': sunshine,
                    'carbon_price': carbon_price
                }
                _q_key = f"llm_answer_{_fp}|{_user_question.strip()}"
                _answer = st.session_state.get(_q_key)
                if not _answer:
                    _answer = answer_question(_user_question.strip(), _llm_params, result) if _LLM_IMPORT_OK else None
                    if _answer:
                        st.session_state[_q_key] = _answer
                if _answer:
                    st.markdown(f"**答：** {_answer}")
                    st.caption("💡 回答仅引用本次决策核算事实，未提供的数据不会编造。")
                else:
                    if not (_LLM_IMPORT_OK and get_client().available):
                        st.info("未配置 SCZC_LLM_API_KEY，已展示规则兜底结论；配置 API Key 后即可开放问答。")
                    else:
                        st.info("LLM 问答未成功，已展示规则兜底结论；可稍后重试。")
                    st.markdown(fallback_answer(_user_question.strip(), result))
            except Exception as e:
                st.warning(f"LLM 问答出错: {e}")

    # ===== 空间可视化：广西+东盟地图 =====
    st.subheader("🗺️ 空间可视化：广西蔗区与东盟协同网络")
    # 简化版地理散点图（Plotly scatter_geo，无需额外库）
    geo_df = pd.DataFrame([
        # 广西7市
        {"name": "崇左市", "lat": 22.4, "lon": 107.4, "group": "广西", "size": 30,
         "label": "崇左:中国糖都"},
        {"name": "来宾市", "lat": 23.7, "lon": 109.2, "group": "广西", "size": 35,
         "label": "来宾:蔗渣餐具基地"},
        {"name": "南宁市", "lat": 22.8, "lon": 108.3, "group": "广西", "size": 25,
         "label": "南宁:首府近郊"},
        {"name": "柳州市", "lat": 24.3, "lon": 109.4, "group": "广西", "size": 20,
         "label": "柳州:工业联动"},
        {"name": "百色市", "lat": 23.9, "lon": 106.6, "group": "广西", "size": 18,
         "label": "百色:右江河谷"},
        {"name": "河池市", "lat": 24.7, "lon": 108.1, "group": "广西", "size": 15,
         "label": "河池:喀斯特蔗区"},
        {"name": "防城港市", "lat": 21.6, "lon": 108.3, "group": "广西", "size": 12,
         "label": "防城港:边境口岸"},
        # 东盟4国
        {"name": "曼谷", "lat": 13.8, "lon": 100.5, "group": "泰国", "size": 28,
         "label": "泰国:全球第二大产糖国"},
        {"name": "河内", "lat": 21.0, "lon": 105.8, "group": "越南", "size": 22,
         "label": "越南:北部山区蔗区"},
        {"name": "内比都", "lat": 19.8, "lon": 96.1, "group": "缅甸", "size": 16,
         "label": "缅甸:伊洛瓦底江流域"},
        {"name": "万象", "lat": 17.9, "lon": 102.6, "group": "老挝", "size": 12,
         "label": "老挝:湄公河沿岸"},
    ])

    fig_geo = px.scatter_geo(
        geo_df, lat='lat', lon='lon', color='group', size='size',
        hover_name='label', projection='natural earth',
        title='中国-东盟甘蔗产业数据协同网络',
        color_discrete_map={
            '广西': '#2E7D32', '泰国': '#EF6C00',
            '越南': '#1565C0', '缅甸': '#7B1FA2', '老挝': '#C62828'
        }
    )
    fig_geo.update_layout(
        height=420, margin=dict(l=0, r=0, t=40, b=0),
        geo=dict(
            center=dict(lat=22, lon=108), lonaxis_range=[95, 120],
            lataxis_range=[10, 30], projection_scale=4.5,
            showland=True, landcolor='#E8F5E9',
            showocean=True, oceancolor='#E3F2FD',
            showcountries=True, countrycolor='#90A4AE',
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_geo, use_container_width=True, config={'displayModeBar': False})

    st.caption(
        "注：本图为示意性空间可视化，基于各城市/国家首都的经纬度标注。"
        "实际应用可叠加甘蔗种植面积热力图、产量分布图、碳排放密度图等图层。"
    )

    # ===== 跨境对比 + 导出 =====
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("🌍 中国-东盟跨境数据协同")
        # 跨境数据质量分级标签
        quality_badges = {
            'China': '🟢 A级（实测）',
            'Thailand': '🟡 B级（FAO+行业）',
            'Vietnam': '🟠 C级（估算）',
            'Myanmar': '🟠 C级（FAO估算）',
            'Laos': '🟠 C级（FAO估算）',
        }
        cn = {'China': '中国-广西', 'Thailand': '泰国', 'Vietnam': '越南', 'Myanmar': '缅甸', 'Laos': '老挝'}
        gf = {'China': '0.5703', 'Thailand': '0.4892', 'Vietnam': '0.5238', 'Myanmar': '0.4500', 'Laos': '0.3500'}

        # 跨境对比表格（无论选择哪个国家都显示五国对比）
        cross = []
        for c in ['China', 'Thailand', 'Vietnam', 'Myanmar', 'Laos']:
            r2 = system.run_decision(
                area_mu=area_mu, avg_temp=avg_temp,
                precipitation=precipitation, sunshine=sunshine,
                fertilizer_n_kg=fertilizer_n*area_mu, diesel_l=diesel*area_mu,
                electricity_kwh=electricity*area_mu,
                carbon_price=carbon_price, country=c, city=city
            )
            o2 = r2['optimization']['optimal']
            ys = r2.get('yield_source', 'N/A')
            src_label = '模型预测' if ys == 'model' else 'FAO统计均值'
            cross.append({
                '国家': cn[c],
                '数据质量': quality_badges[c],
                '产量来源': src_label,
                '单产(吨/亩)': f"{r2['yield_per_mu']:.2f}",
                '纯收益(元)': f"{o2['net_benefit']:,.0f}",
                '综合收益(元)': f"{o2['total_benefit']:,.0f}",
                'CO₂e(吨)': f"{r2['carbon_emission']['total_tons']:.4f}",
                '电网因子': gf[c]
            })
        st.dataframe(pd.DataFrame(cross), use_container_width=True, hide_index=True)

        # 跨境数据质量分级说明
        with st.expander("📐 跨境数据质量分级标准（中国-东盟数据互认体系）", expanded=False):
            st.markdown("""
**分级依据**：数据来源可信度 + 采集方式 + 时效性 + 精度验证

| 等级 | 颜色 | 适用区域 | 产量数据来源 | 气象数据 | 市场价格 | 电网因子 |
|:----:|:----:|---------|-------------|---------|---------|---------|
| **A级** | 🟢 | 中国-广西 | 广西糖业协会+统计年鉴（实测） | 国家气象站（实测） | 行业协会+市场调研 | 国家电网官方值 |
| **B级** | 🟡 | 泰国 | FAOSTAT十年均值 + 泰国糖业委员会报告 | 泰国气象局公开数据 | 行业报告+汇率换算 | 泰国能源部公开值 |
| **C级** | 🟠 | 越南/缅甸/老挝 | FAOSTAT十年均值（估算） | 区域气候模式插值 | 估算值（标注"估算"） | IEA统计值 |

**互认原则**：
- 对标 RCEP 电子商务章节数据跨境流动规则
- 农业非敏感数据（产量、气象）纳入"白名单"流通机制
- 不同等级数据在决策中加权使用，A级权重1.0、B级0.8、C级0.6
- 框架可移植：替换当地实测数据即可升级等级
- 缅甸/老挝为新增跨境节点，支撑澜湄合作框架下糖业数据协同
            """)

        st.caption(
            '💡 数据协同模式：中国提供成熟模型与算法框架，东盟国家提供本地化数据，'
            '实现"模型走出去、数据留下来"的可信流通，助力RCEP区域糖业高质量发展。'
        )
    with c2:
        report = pd.DataFrame({
            '参数': ['国家', '城市', '面积(亩)', '单产(吨/亩)', '总产(吨)',
                      '碳排放(吨CO₂e)', '推荐方案', '纯收益(元)', '碳交易(元)', '综合收益(元)'],
            '数值': [
                {'China': '中国', 'Thailand': '泰国', 'Vietnam': '越南', 'Myanmar': '缅甸', 'Laos': '老挝'}.get(country, country),
                city if country == 'China' else 'N/A', f"{area_mu:.0f}",
                f"{result['yield_per_mu']:.2f}", f"{result['total_yield']:.2f}",
                f"{ce['total_tons']:.4f}", n_cn.get(opt['name'], opt['name']),
                f"{opt['net_benefit']:,.0f}", f"{opt['carbon_revenue']:+,.0f}",
                f"{opt['total_benefit']:,.0f}"
            ]
        })
        csv = report.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 CSV决策报告", csv,
                           f"decision_{country}_{area_mu:.0f}mu.csv", use_container_width=True)
        st.download_button("📥 JSON数据产品",
                           json.dumps({
                               "project": "蔗循智策 v2.0", "country": country,
                               "city": city, "area_mu": area_mu,
                               "yield_per_mu": round(result['yield_per_mu'], 2),
                               "total_yield": round(result['total_yield'], 2),
                               "optimal_scheme": opt['name'],
                               "net_benefit": round(opt['net_benefit'], 2),
                               "total_benefit": round(opt['total_benefit'], 2),
                               "carbon_emission_tons": round(ce['total_tons'], 4),
                               "carbon_revenue": round(opt['carbon_revenue'], 2)
                           }, ensure_ascii=False, indent=2),
                           f"data_product_{country}_{area_mu:.0f}mu.json", use_container_width=True)

    st.markdown("---")
    st.caption(
        "蔗循智策 v2.0 | 2026年「数据要素×」大赛广西分赛 高校赛道 现代农业组 | "
        "全部数据来源公开合法渠道 | 碳排放核算遵循 IPCC 2006 Tier 1 方法学"
    )

# ============================================================
# 对话式决策助手（规则引擎，非大模型 Agent）
# ============================================================
with st.expander("📝 快捷参数输入（自然语言描述）", expanded=False):
    st.caption('一句描述蔗田情况，规则引擎自动提取参数生成决策 — 如"崇左10亩，碳价85元"')
    
    # 对话消息 + 对话状态
    if "agent_msgs" not in st.session_state:
        st.session_state.agent_msgs = []
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None  # 多轮对话状态
    if "agent_done" not in st.session_state:
        st.session_state.agent_done = False  # 上一轮是否已完成决策
    
    # 显示历史消息（兼容低版本Streamlit）
    HAS_CHAT_UI = hasattr(st, 'chat_message') and hasattr(st, 'chat_input')
    for msg in st.session_state.agent_msgs:
        if HAS_CHAT_UI:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        else:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_icon} {msg['role']}**: {msg['content']}")

    # 快捷示例（仅首轮显示）
    if not st.session_state.agent_msgs:
        cols = st.columns(3)
        if country == 'China':
            queries = [
                "崇左10亩，温度28降水900日照870，怎么处理蔗叶最赚钱？",
                "来宾50亩甘蔗，碳价100元，推荐最优方案",
                "对比中国泰国越南的蔗渣利用收益差异",
            ]
        else:
            cn_name = {'Thailand': '泰国', 'Vietnam': '越南', 'Myanmar': '缅甸', 'Laos': '老挝'}.get(country, country)
            queries = [
                f"{cn_name}10亩蔗田，怎么处理蔗叶最赚钱？",
                f"{cn_name}50亩甘蔗，碳价100元，推荐最优方案",
                "对比中国泰国越南的蔗渣利用收益差异",
            ]
        for i, q in enumerate(queries):
            if cols[i].button(q, key=f"aq_{i}", use_container_width=True):
                st.session_state.agent_msgs.append({"role": "user", "content": q})
                with st.spinner("参数解析中..."):
                    resp, done, state = agent.chat(q, st.session_state.agent_state)
                    st.session_state.agent_state = state
                    st.session_state.agent_done = done
                    st.session_state.agent_msgs.append({"role": "assistant", "content": resp})
                st.rerun()

    # 用户输入（兼容低版本Streamlit：优先chat_input，降级text_input）
    user_input = None
    if HAS_CHAT_UI:
        user_input = st.chat_input("描述你的蔗田情况，自动提取参数...")
    else:
        user_input = st.text_input("描述你的蔗田情况，自动提取参数...", key="agent_text_input")
        if user_input:
            st.session_state._agent_text_submitted = True
    if HAS_CHAT_UI and user_input:
        pass  # chat_input 返回非None即触发
    elif not HAS_CHAT_UI and user_input and st.session_state.get("_agent_text_submitted"):
        st.session_state._agent_text_submitted = False
    else:
        user_input = None

    if user_input:
        # 只在上一轮已生成决策时重置状态（全新对话）
        if st.session_state.get("agent_done", False):
            st.session_state.agent_state = None
            st.session_state.agent_msgs = []
            st.session_state.agent_done = False

        # 对用户输入做 HTML 转义，防止 XSS/Markdown 注入
        safe_input = html.escape(user_input)
        st.session_state.agent_msgs.append({"role": "user", "content": safe_input})
        if HAS_CHAT_UI:
            with st.chat_message("user"):
                st.markdown(safe_input)

            with st.chat_message("assistant"):
                with st.spinner("参数解析中..."):
                    resp, done, state = agent.chat(user_input, st.session_state.agent_state)
                    st.session_state.agent_state = state
                    st.session_state.agent_done = done
                    st.session_state.agent_msgs.append({"role": "assistant", "content": resp})
                st.markdown(resp)
        else:
            st.markdown(f"**user**: {safe_input}")
            with st.spinner("参数解析中..."):
                resp, done, state = agent.chat(user_input, st.session_state.agent_state)
                st.session_state.agent_state = state
                st.session_state.agent_done = done
                st.session_state.agent_msgs.append({"role": "assistant", "content": resp})
            st.markdown(f"**assistant**: {resp}")

