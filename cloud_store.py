"""
cloud_store.py — 蔗循智策 云端持久化模块（GitHub 即存储）

定位：
- Streamlit Cloud 实例为临时文件系统，重启后本地写入全部丢失。
- 本模块通过 GitHub Contents API 将反馈记录与访问事件持久化到仓库，
  数据随仓库永久留存，可作为测试报告"真实用户使用记录"的可溯源证据。
- 全程可回退：未配置 GITHUB_TOKEN / 网络失败 / 限流时，一律返回 None/False，
  调用方回退到本地文件，系统功能不受影响。

配置（环境变量 / Streamlit Secrets）：
    GITHUB_TOKEN  必填；GitHub Personal Access Token（需要 repo 写权限）
    GITHUB_REPO   可选；格式 owner/repo，默认 cn-handsomeboy/data

实现说明：
- 使用 Python 标准库 urllib 调用 GitHub REST API，不引入外部依赖。
- 数据文件：data/user_feedback.json（反馈）、data/cloud_events.json（访问事件）
- 更新文件需要携带 sha（ETag），append 采用"读-改-写 + 重试"应对并发。
"""

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger("cloud_store")

_API_BASE = "https://api.github.com"
_DEFAULT_REPO = "cn-handsomeboy/data"
# 云持久化文件（位于仓库 data/ 下，本地 .gitignore 忽略，由 API 独占写入）
CLOUD_FEEDBACK_PATH = "data/user_feedback.json"
CLOUD_EVENTS_PATH = "data/cloud_events.json"
# 写入时对并发冲突的最大重试次数
_MAX_RETRY = 3


def is_configured() -> bool:
    """是否已配置 GitHub Token"""
    return bool(_token())


def _read_st_secrets(key: str) -> str:
    """从 Streamlit Secrets 读取（若在 Streamlit 环境且已配置）。

    Streamlit Cloud 的 Secrets 默认存放在 secrets.toml，除非部署设置里
    勾选 "Export as environment variable"，否则不会注入 os.environ。
    此函数做兼容兜底，保证"只在 Secrets 配置了 GITHUB_TOKEN"也能生效。
    """
    try:
        import streamlit as st
        secrets = st.secrets if hasattr(st, "secrets") else None
        if secrets is not None:
            # st.secrets 支持 dict 风格取值；取不到时抛异常或返回 None
            try:
                val = secrets.get(key, "")
            except Exception:
                val = ""
            if isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        pass
    return ""


def _load_dotenv_fallback(env_path: str) -> None:
    """本地开发兜底：从 .env 读取（仅当环境变量/Secrets 均未配置时）。

    与 api.py / llm_agent.py 的 _load_env_file 保持一致，不覆盖已存在的变量。
    """
    if not env_path or not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except Exception:
        pass
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


# 模块加载时尝试从 .env 补充（仅本地开发；云端由环境变量/Secrets 提供）
_load_dotenv_fallback(os.path.join(os.path.dirname(__file__), ".env"))


def _token() -> str:
    """依次从 环境变量 -> Streamlit Secrets -> .env 读取 GITHUB_TOKEN"""
    val = os.environ.get("GITHUB_TOKEN", "").strip()
    if val:
        return val
    return _read_st_secrets("GITHUB_TOKEN")


def _repo() -> str:
    """依次从 环境变量 -> Streamlit Secrets -> .env 读取 GITHUB_REPO"""
    val = os.environ.get("GITHUB_REPO", "").strip()
    if val:
        return val
    secrets_repo = _read_st_secrets("GITHUB_REPO")
    if secrets_repo:
        return secrets_repo
    return _DEFAULT_REPO


def _api(method: str, url: str, body=None, timeout: int = 20):
    """调用 GitHub API，返回 (status, parsed_json_or_raw_str)。

    读取（GET）公开仓库时允许匿名访问（无需 token）；写入（PUT/POST）
    仍需 token，未配置时返回 401 语义，由调用方自行回退。
    """
    req = urllib.request.Request(url, method=method)
    tok = _token()
    if tok:
        req.add_header("Authorization", f"token {tok}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "sugarcane-cloud-store")
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except Exception as e:
        logger.warning("GitHub API 请求失败: %s", e)
        return None, None


def read_json(rel_path: str):
    """
    读取仓库文件并解析 JSON。
    返回: 解析后的数据；文件不存在返回 None；失败返回 None。

    说明：读取允许匿名访问（公开仓库可读）；私有仓库无 token 时
    GitHub 返回 401，本函数返回 None，调用方回退本地文件。
    """
    url = f"{_API_BASE}/repos/{_repo()}/contents/{rel_path}"
    status, data = _api("GET", url)
    if status != 200 or not isinstance(data, dict):
        return None
    try:
        content = base64.b64decode(data.get("content", "")).decode("utf-8")
        return json.loads(content)
    except Exception as e:
        logger.warning("解析云端文件 %s 失败: %s", rel_path, e)
        return None


def write_json(rel_path: str, payload, message: str = "update") -> bool:
    """整体覆写仓库文件（自动带 sha）。成功返回 True。"""
    if not is_configured():
        return False
    url = f"{_API_BASE}/repos/{_repo()}/contents/{rel_path}"
    body = {
        "message": message,
        "content": base64.b64encode(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("utf-8"),
    }
    # 更新已有文件需要 sha
    _, existing = _api("GET", url)
    if isinstance(existing, dict) and existing.get("sha"):
        body["sha"] = existing["sha"]
    status, resp = _api("PUT", url, body)
    if status in (200, 201):
        logger.info("已写入云端 %s (%s)", rel_path, message)
        return True
    logger.warning("写入云端 %s 失败: %s %s", rel_path, status, resp if isinstance(resp, str) else "")
    return False


def append_record(rel_path: str, record: dict) -> bool:
    """
    追加一条记录到云端 JSON 数组文件（读-改-写，带并发重试）。
    成功返回 True；未配置/失败返回 False。
    """
    if not is_configured():
        return False
    for attempt in range(_MAX_RETRY):
        records = read_json(rel_path) or []
        if not isinstance(records, list):
            records = []
        records.append(record)
        if write_json(rel_path, records, message="append record"):
            return True
        time.sleep(1 + attempt)  # 简单退避，等待并发冲突窗口
    logger.warning("追加云端记录 %s 重试 %d 次仍失败", rel_path, _MAX_RETRY)
    return False
