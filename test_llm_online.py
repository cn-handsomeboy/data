"""
本地模拟线上"""
本地模拟线上环境，完整测试 LLM 对话 +"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os
"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" *"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLM"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态:"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system","""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role":"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f""""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print(""""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

    test_params = {
        "city": """"
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

    test_params = {
        "city": "南宁", "area_mu": 100, """"
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

    test_params = {
        "city": "南宁", "area_mu": 100, "avg_temp": 24,
        "prec"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

    test_params = {
        "city": "南宁", "area_mu": 100, "avg_temp": 24,
        "precipitation": 1200, "sunshine"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

    test_params = {
        "city": "南宁", "area_mu": 100, "avg_temp": 24,
        "precipitation": 1200, "sunshine": 1600, "carbon_price": 50
    }
    test_result = {
        "yield_per_mu": 5"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

    test_params = {
        "city": "南宁", "area_mu": 100, "avg_temp": 24,
        "precipitation": 1200, "sunshine": 1600, "carbon_price": 50
    }
    test_result = {
        "yield_per_mu": 5.2, "total_yield": 520,"""
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

    test_params = {
        "city": "南宁", "area_mu": 100, "avg_temp": 24,
        "precipitation": 1200, "sunshine": 1600, "carbon_price": 50
    }
    test_result = {
        "yield_per_mu": 5.2, "total_yield": 520, "yield_source": "model",
        """"
本地模拟线上环境，完整测试 LLM 对话 + 决策报告增强 + 自然语言问数
"""
import sys
import os

# 模拟线上 Streamlit Cloud 环境
os.environ['IS_RUNNING_ON_STREAMLIT_CLOUD'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from llm_agent import LLMConfig, get_client, enhance_decision_report, answer_question


def test_basic_chat():
    print("=" * 60)
    print("测试 1：LLM 基础对话连接")
    print("=" * 60)

    cfg = LLMConfig()
    client = get_client()
    print(f"API Key 配置状态: {'已配置' if cfg.enabled else '未配置'}")
    print(f"Base URL: {cfg.base_url}")
    print(f"Model: {cfg.model}")
    print(f"Client available: {client.available}")

    if not client.available:
        print("结果: 未启用（未配置 API Key）")
        return False

    try:
        resp = client.complete([
            {"role": "system", "content": "你是农业专家，请回复：测试通过"},
            {"role": "user", "content": "验证 LLM 连接"}
        ])
        print(f"LLM 响应: {resp[:120]}...")
        print("结果: 通过")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        print("结果: 失败")
        return False


def test_decision_report():
    print()
    print("=" * 60)
    print("测试 2：决策报告增强")
    print("=" * 60)

    test_params = {
        "city": "南宁", "area_mu": 100, "avg_temp": 24,
        "precipitation": 1200, "sunshine": 1600, "carbon_price": 50
    }
    test_result = {
        "yield_per_mu": 5.2, "total_yield": 520, "yield_source": "model",
        "byproducts": {
            "sugarc