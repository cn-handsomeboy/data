"""
llm_agent.py — 蔗循智策 大语言模型增强模块（可选扩展，非必须）

定位（重要）：
- 核心决策计算由确定性规则引擎 / 统计模型完成（见 models.py / agent.py），
  本模块【不参与任何计算】，仅在结构化结果之上提供两类可选增强：
    1) 决策报告润色：把计算结果转成给蔗农/糖企/政府的自然语言报告；
    2) 自然语言问数：基于已生成的计算结果，回答开放式提问。
- 全程可回退：未配置 API Key / 调用失败 / 超时 / 触发限流 / 网络中断时，
  所有增强接口一律返回 None，调用方回退到规则模板，系统功能不受影响。

配置（通过环境变量，见 .env.example）：
    SCZC_LLM_API_KEY           必填；未设置则整体禁用 LLM 增强
    SCZC_LLM_BASE_URL          兼容 OpenAI 的接口地址，默认 https://api.deepseek.com/v1
    SCZC_LLM_MODEL             模型名，默认 deepseek-chat
    SCZC_LLM_MAX_TOKENS        单次最大生成 token，默认 800
    SCZC_LLM_TIMEOUT_SEC       请求超时秒数，默认 25
    SCZC_LLM_RATE_LIMIT        每分钟最大调用次数，默认 10（防滥用；可收紧为 1）
    SCZC_LLM_TEMPERATURE       采样温度，默认 0.3（倾向确定性，减少编造）

实现说明：
- 使用 OpenAI 兼容 chat/completions 协议，通过 Python 标准库 urllib 实现，
  不引入外部 SDK 依赖，便于离线 / 受限环境部署。
- 所有模型输出都要求【仅基于传入的已核算事实】，任何未提供的数据一律不得编造。
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request

# 自动加载 .env 文件（本地开发时从文件读取，云端则由 Secrets 注入环境变量）
try:
    from dotenv import load_dotenv
    _dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
except Exception:
    pass

logger = logging.getLogger("llm_agent")

_ENABLE_LLM_ENV = "SCZC_LLM_API_KEY"

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class LLMConfig:
    """LLM 增强配置（从环境变量读取，可被调用方覆盖以便测试）"""

    def __init__(self, api_key=None, base_url=None, model=None,
                 max_tokens=None, timeout=None, rate_limit=None,
                 temperature=None):
        self.api_key = api_key if api_key is not None else os.environ.get(_ENABLE_LLM_ENV)
        self.base_url = base_url or os.environ.get(
            "SCZC_LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.model = model or os.environ.get("SCZC_LLM_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens or _env_int("SCZC_LLM_MAX_TOKENS", 800)
        self.timeout = timeout if timeout is not None else _env_float("SCZC_LLM_TIMEOUT_SEC", 25)
        self.rate_limit = rate_limit or _env_int("SCZC_LLM_RATE_LIMIT", 10)
        self.temperature = temperature if temperature is not None else _env_float("SCZC_LLM_TEMPERATURE", 0.3)

    @property
    def enabled(self) -> bool:
        """是否已配置 API Key（未配置则整体禁用）"""
        return bool(self.api_key and str(self.api_key).strip())


# ---------------------------------------------------------------------------
# 进程内滑动窗口限流器（防滥用，单进程粒度）
# ---------------------------------------------------------------------------

class _SlidingWindowRateLimiter:
    """固定时间窗口限流：限制每分钟最多 N 次调用。"""

    def __init__(self, limit: int):
        self.limit = max(1, int(limit))
        self._timestamps = []
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            if len(self._timestamps) >= self.limit:
                return False
            self._timestamps.append(now)
            return True


# ---------------------------------------------------------------------------
# LLM 客户端（OpenAI 兼容）
# ---------------------------------------------------------------------------

class LLMClient:
    """轻量 OpenAI 兼容客户端，含限流与统一错误处理。

    complete() 不在内部捕获业务异常（超时/网络/HTTP 错误会抛出），
    由上层使用方捕获并回退，避免假成功。
    """

    def __init__(self, config: LLMConfig = None):
        self.cfg = config or LLMConfig()
        self._limiter = _SlidingWindowRateLimiter(self.cfg.rate_limit)

    @property
    def available(self) -> bool:
        return self.cfg.enabled

    def complete(self, messages, temperature=None, max_tokens=None) -> str:
        """调用 chat/completions，返回文本内容。失败抛异常。"""
        if not self.cfg.enabled:
            raise RuntimeError("LLM 未启用：未配置 SCZC_LLM_API_KEY")

        if not self._limiter.allow():
            raise RuntimeError("LLM 调用频率超限，请稍后再试")

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "stream": False,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"LLM 响应格式异常: {e}")


# 全局单例（复用配置与限流窗口）
_client: LLMClient = None
_client_lock = threading.Lock()


def get_client(config: LLMConfig = None) -> LLMClient:
    """获取全局 LLM 客户端单例；显式传入 config 时新建（便于测试）。"""
    global _client
    if config is not None:
        return LLMClient(config)
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = LLMClient(LLMConfig())
    return _client


# ---------------------------------------------------------------------------
# 决策上下文序列化（把结构化结果转成 LLM 可读的事实文本）
# ---------------------------------------------------------------------------

_SCHEME_CN = {
    'traditional': '传统模式（焚烧+填埋+直销+锅炉）',
    'improved_traditional': '改良传统模式',
    'circular_basic': '基础循环（饲料+有机肥+直销+锅炉）',
    'circular_advanced': '进阶循环模式',
    'circular_optimal': '最优循环（生物质颗粒+有机肥+深加工+环保浆料）',
}

_BYPRODUCT_CN = {
    'sugarcane_leaf': '蔗叶', 'bagasse': '蔗渣',
    'filter_mud': '滤泥', 'molasses': '糖蜜', 'sugarcane_top': '蔗梢',
}


def _build_decision_context(params: dict, result: dict) -> str:
    """将决策计算结果压缩为一段事实性上下文（供 LLM 引用，禁止其编造其他数据）。"""
    buf = []
    p = params
    opt = result['optimization']['optimal']

    # 输入
    buf.append("【决策输入】")
    buf.append(
        f"城市={p.get('city', '未知')}，国家={p.get('country', 'China')}，"
        f"面积={p.get('area_mu', '?')}亩，生长季均温={p.get('avg_temp', '?')}℃，"
        f"降水={p.get('precipitation', '?')}mm，日照={p.get('sunshine', '?')}h，"
        f"碳价={p.get('carbon_price', '?')}元/吨"
    )

    # 产量
    buf.append("")
    buf.append("【产量预测】")
    src = result.get('yield_source', 'model')
    src_label = 'LOOCV回归模型' if src == 'model' else 'FAO统计均值'
    buf.append(
        f"{src_label}预测单产 {result['yield_per_mu']:.2f} 吨/亩，"
        f"总产量 {result['total_yield']:.2f} 吨"
    )

    # 副产物
    buf.append("")
    buf.append("【副产物估算】")
    bp = "，".join(
        f"{_BYPRODUCT_CN.get(k, k)} {v['quantity']:.2f} 吨"
        for k, v in result['byproducts'].items()
    )
    buf.append(bp)

    # 碳排放
    ce = result['carbon_emission']
    buf.append("")
    buf.append("【碳排放核算 (IPCC Tier 1)】")
    buf.append(
        f"全链条 {ce['total_tons']:.2f} 吨CO₂e"
        f"（种植 {ce['planting']:.0f}kg + 机械 {ce['mechanization']:.0f}kg + 加工 {ce['processing']:.0f}kg）"
    )

    # 方案对比表
    buf.append("")
    buf.append("【多目标优化方案对比（收益权重70%，碳减排30%）】")
    buf.append("| 方案 | 净收益(元) | 碳排放(kg) | 碳交易收益(元) | 综合得分 |")
    buf.append("|---|---|---|---|---|")
    for s in result['optimization']['all_schemes']:
        buf.append(
            f"| {_SCHEME_CN.get(s['name'], s['name'])} | {s['net_benefit']:,.0f} | "
            f"{s['carbon_emission_kg']:+,.0f} | {s.get('carbon_revenue', 0):+,.0f} | "
            f"{s['total_score']:.3f} |"
        )

    # 最优方案
    buf.append("")
    buf.append("【推荐方案】")
    buf.append(
        f"{_SCHEME_CN.get(opt['name'], opt['name'])}，"
        f"综合收益 {opt.get('total_benefit', opt['net_benefit']):,.0f} 元，"
        f"净收益 {opt['net_benefit']:,.0f} 元，"
        f"碳交易收益 {opt.get('carbon_revenue', 0):+,.0f} 元"
    )
    return "\n".join(buf)


# ---------------------------------------------------------------------------
# 增强 1：决策报告润色
# ---------------------------------------------------------------------------

_DECISION_REPORT_SYSTEM = (
    "你是'蔗循智策'甘蔗副产物循环经济决策系统的农业专家。"
    "你的任务是把一份结构化的决策核算结果，改写成面向种植户/糖厂/政府三类读者"
    "都能读懂的、专业且克制的自然语言决策报告。"
    "\n\n硬性要求："
    "\n1. 只允许引用下面'决策事实'中明确给出的数字与结论，一律不得编造或外推任何指标；"
    "\n2. 报告需包含：一段结论性导语、最优方案及其增收/减排依据、1-2条可落地的行动建议；"
    "\n3. 使用简体中文，避免空泛套话，措辞专业、可信；"
    "\n4. 如某方面数据未提供，不得虚构，说明'当前数据不足以支持该判断'即可；"
    "\n5. 输出为纯文本，不要使用 Markdown 表格。"
)


def enhance_decision_report(params: dict, result: dict, reasoning: str = "",
                            client: LLMClient = None) -> str:
    """生成自然语言决策报告。任何失败返回 None（调用方回退规则模板）。"""
    client = client or get_client()
    if not client.available:
        return None
    try:
        context = _build_decision_context(params, result)
        if reasoning:
            context += "\n\n【推理链参考】\n" + reasoning[:1500]
        messages = [
            {"role": "system", "content": _DECISION_REPORT_SYSTEM},
            {"role": "user", "content": f"【决策事实】\n{context}\n\n请据此给出决策报告。"},
        ]
        return client.complete(messages)
    except Exception as e:
        logger.warning("决策报告 LLM 增强失败，回退规则模板: %s", e)
        return None


# ---------------------------------------------------------------------------
# 增强 2：自然语言问数
# ---------------------------------------------------------------------------

_ASK_SYSTEM = (
    "你是'蔗循智策'甘蔗副产物循环经济决策系统的智能问数助手。"
    "用户会基于下面给定的一次决策事实，向你提出开放式问题"
    "（例如某方案为什么赚钱、碳减排原理、某项副产物的利用建议等）。"
    "\n\n硬性要求："
    "\n1. 只依据'决策事实'中给出的数据回答，不得编造数字或外推结论；"
    "\n2. 问题超出给定数据范围时，明确说明'现有决策数据不足以支持该回答'，并礼貌建议可补充的信息；"
    "\n3. 用简体中文，回答简洁、专业、口语化易懂；"
    "\n4. 不输出 Markdown 表格。"
)


def answer_question(question: str, params: dict, result: dict,
                    client: LLMClient = None) -> str:
    """基于一次决策结果，回答用户的开放式问数。任何失败返回 None。"""
    client = client or get_client()
    if not client.available:
        return None
    try:
        context = _build_decision_context(params, result)
        messages = [
            {"role": "system", "content": _ASK_SYSTEM},
            {"role": "user", "content":
                f"【决策事实】\n{context}\n\n【用户问题】{question}\n\n请作答。"},
        ]
        return client.complete(messages)
    except Exception as e:
        logger.warning("自然语言问数 LLM 失败，回退规则模板: %s", e)
        return None