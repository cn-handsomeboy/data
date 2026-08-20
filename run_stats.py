"""
蔗循智策 — 真实运行台账统计
==========================
聚合系统真实运行痕迹，形成可量化、可审计的"实际成效"证据：

- API 调用审计：解析 logs/security_audit.log 中的 API_ACCESS 记录
  （真实时间戳 / 客户端IP / 端点 / 状态码 / 耗时 / 响应字节）。
- 安全监控：跨境授权放行、危险输入拦截、数据完整性告警等 SECURITY 事件。
- 反馈闭环：读取 data/user_feedback.json，展示真实 预测→实际→偏差→校准。

所有数字均直接来自真实运行日志与真实反馈文件，不注入任何模拟数据。
"""
import json
import os
import re
from collections import Counter
from datetime import datetime

_PROJECT_DIR = os.path.dirname(__file__)
_LOG_FILE = os.path.join(_PROJECT_DIR, 'logs', 'security_audit.log')
_FEEDBACK_FILE = os.path.join(_PROJECT_DIR, 'data', 'user_feedback.json')


def _load_log_lines():
    if not os.path.exists(_LOG_FILE):
        return []
    with open(_LOG_FILE, 'r', encoding='utf-8') as f:
        return [ln.rstrip('\n') for ln in f if ln.strip()]


def get_run_stats() -> dict:
    """聚合真实运行台账统计"""
    lines = _load_log_lines()

    api_access = []
    security_events = Counter()
    cross_border_count = 0
    tamper_alerts = 0
    dangerous_input_count = 0

    for ln in lines:
        if 'API_ACCESS | ' in ln:
            try:
                payload = ln.split('API_ACCESS | ', 1)[1]
                api_access.append(json.loads(payload))
            except Exception:
                continue
        elif 'SECURITY_ALERT | ' in ln or 'SECURITY_EVENT | ' in ln:
            kind = 'SECURITY_ALERT' if 'SECURITY_ALERT' in ln else 'SECURITY_EVENT'
            security_events[kind] += 1
        elif '跨境授权' in ln:
            cross_border_count += 1
        elif '数据文件被篡改' in ln:
            tamper_alerts += 1
        elif '检测到危险输入模式' in ln:
            dangerous_input_count += 1

    # 端点统计
    endpoint_counter = Counter(a.get('endpoint', '?') for a in api_access)
    client_ips = set(a.get('client_ip') for a in api_access if a.get('client_ip'))
    ok_calls = sum(1 for a in api_access if a.get('status_code', 400) < 400)

    # 反馈闭环真实统计
    feedback = []
    if os.path.exists(_FEEDBACK_FILE):
        with open(_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback = json.load(f)
    yield_errs = [abs(f.get('yield_error_pct', 0) or 0) for f in feedback]
    yield_mape = round(sum(yield_errs) / len(yield_errs), 2) if yield_errs else None
    calibration_needed = (yield_mape or 0) > 15.0

    return {
        "source": "系统真实审计日志(security_audit.log) + 真实反馈闭环(user_feedback.json)，可溯源、未注入模拟数据",
        "api": {
            "total_calls": len(api_access),
            "ok_calls": ok_calls,
            "fail_calls": len(api_access) - ok_calls,
            "first_access": api_access[0]['timestamp'] if api_access else None,
            "last_access": api_access[-1]['timestamp'] if api_access else None,
            "unique_client_ips": sorted(client_ips),
            "endpoints": dict(endpoint_counter),
        },
        "security": {
            "cross_border_auth_granted": cross_border_count,
            "dangerous_input_blocked_times": dangerous_input_count,
            "integrity_tamper_alerts": tamper_alerts,
        },
        "feedback_loop": {
            "real_feedback_count": len(feedback),
            "yield_mape_pct": yield_mape,
            "calibration_needed": calibration_needed,
            "latest_feedback_time": feedback[-1].get('timestamp') if feedback else None,
        },
        "generated_at": datetime.now().isoformat(),
    }