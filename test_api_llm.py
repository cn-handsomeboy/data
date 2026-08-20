"""
test_api_llm.py — LLM 相关 API 接口层测试（P0 完善项）

覆盖内容：
- GET  /api/llm/status    LLM 增强状态接口
- POST /api/report        决策报告接口（use_llm 回退 / 推理链 / 权重参数）
- POST /api/decision      权重参数生效（output.weights 透传）
- POST /api/ask           自然语言问数接口（规则兜底）
- 鉴权：无 Key 401、错误 Key 403

运行方式：
    python test_api_llm.py

实现说明：
- 使用 httpx.ASGITransport 直连 FastAPI ASGI 应用（不依赖 TestClient，
  规避 httpx>=0.28 与旧版 starlette TestClient 的兼容性问题）；
- 测试启动时将环境变量 SCZC_LLM_API_KEY 置空，强制走"未配置 LLM"路径，
  不消耗真实 API 配额、不影响演示数据；
- 首次运行会加载/训练模型（warm_start_models），约需 1-3 分钟，请耐心等待。
"""
import asyncio
import os
import sys

# 强制禁用 LLM（需在 import 任何项目模块之前设置，覆盖项目 .env 自动注入的 Key）
os.environ["SCZC_LLM_API_KEY"] = ""
# 固定 API Key 供鉴权测试使用
os.environ["SUGARCANE_API_KEY"] = "test-api-key-llm-2026"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx  # noqa: E402

import api  # noqa: E402
from llm_agent import get_client  # noqa: E402

BASE_URL = "http://testserver"
HEADERS = {"X-API-Key": "test-api-key-llm-2026"}

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


async def main():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL,
                                 timeout=600.0) as client:
        # 0. LLM 客户端状态（测试期间必须为禁用，确保不调用真实 API）
        check("测试环境 LLM 已禁用（available=False）",
              get_client().available is False,
              f"available={get_client().available}")

        print("\n== 1. GET /api/llm/status 状态接口 ==")
        r = await client.get("/api/llm/status", headers=HEADERS)
        check("返回 200", r.status_code == 200, f"status={r.status_code}")
        body = r.json()
        check("llm_configured 字段存在且为 False",
              body.get("llm_configured") is False, str(body))
        check("note 字段存在（说明 LLM 定位）", bool(body.get("note")), str(body))

        print("\n== 2. 鉴权 ==")
        r = await client.get("/api/llm/status")
        check("无 Key 返回 401", r.status_code == 401, f"status={r.status_code}")
        r = await client.get("/api/llm/status", headers={"X-API-Key": "wrong-key"})
        check("错误 Key 返回 403", r.status_code == 403, f"status={r.status_code}")

        print("\n== 3. POST /api/report 报告接口（LLM 回退路径） ==")
        payload = {
            "country": "China", "city": "崇左市", "area_mu": 10.0,
            "avg_temp": 28.0, "precipitation": 900.0, "sunshine": 870.0,
            "use_llm": True,
        }
        r = await client.post("/api/report", json=payload, headers=HEADERS)
        check("返回 200", r.status_code == 200, f"status={r.status_code}")
        body = r.json()
        check("llm_used=false（无 Key 自动回退）", body.get("llm_used") is False,
              f"llm_used={body.get('llm_used')}")
        check("llm_available=false", body.get("llm_available") is False)
        check("llm_report 为 null（由调用方决定兜底）", body.get("llm_report") is None)
        check("reasoning 推理链非空", bool(body.get("reasoning")), "")
        out = body.get("output", {})
        check("output.optimal_scheme.name_cn 非空",
              bool(out.get("optimal_scheme", {}).get("name_cn")),
              str(out.get("optimal_scheme")))

        print("\n== 4. POST /api/report 权重参数透传 ==")
        payload_w = dict(payload)
        payload_w.update({"benefit_weight": 0.4, "carbon_weight": 0.6,
                          "carbon_trading_scenario": "future_agriculture"})
        r = await client.post("/api/report", json=payload_w, headers=HEADERS)
        check("带权重参数返回 200", r.status_code == 200, f"status={r.status_code}")

        print("\n== 5. POST /api/decision 权重参数生效 ==")
        base = {"country": "China", "city": "崇左市", "area_mu": 10.0,
                "avg_temp": 28.0, "precipitation": 900.0, "sunshine": 870.0}
        r = await client.post("/api/decision", json=base, headers=HEADERS)
        check("默认权重请求返回 200", r.status_code == 200,
              f"status={r.status_code}")
        d_default = r.json()["output"]

        r = await client.post("/api/decision", json=dict(base, **{
            "benefit_weight": 0.0, "carbon_weight": 1.0,
            "carbon_trading_scenario": "future_agriculture",
        }), headers=HEADERS)
        check("自定义权重请求返回 200", r.status_code == 200,
              f"status={r.status_code}")
        d_weighted = r.json()["output"]

        check("output 含 weights 字段（新增透明输出）",
              "weights" in d_default and "weights" in d_weighted)
        w1 = d_default.get("weights", {})
        w2 = d_weighted.get("weights", {})
        check("默认权重 benefit≈0.7", abs(w1.get("benefit", 0) - 0.7) < 1e-6, str(w1))
        check("自定义权重生效 benefit=0.0", abs(w2.get("benefit", 1) - 0.0) < 1e-6, str(w2))
        check("碳交易情景透传 future_agriculture",
              w2.get("carbon_trading_scenario") == "future_agriculture", str(w2))

        print("\n== 6. POST /api/ask 问数接口（规则兜底） ==")
        r = await client.post("/api/ask", json={
            "question": "为什么推荐这个方案？",
            "country": "China", "city": "崇左市", "area_mu": 10.0,
            "avg_temp": 28.0, "precipitation": 900.0, "sunshine": 870.0,
            "use_llm": True,
        }, headers=HEADERS)
        check("返回 200", r.status_code == 200, f"status={r.status_code}")
        body = r.json()
        check("use_llm=false（规则兜底）", body.get("use_llm") is False,
              f"use_llm={body.get('use_llm')}")
        check("llm_available=false", body.get("llm_available") is False)
        check("兜底回答包含'最优方案'结论", "最优方案" in body.get("answer", ""),
              body.get("answer", "")[:100])
        check("兜底回答方案名为中文", any(k in body.get("answer", "")
              for k in ["传统模式", "循环", "改良"]), body.get("answer", "")[:100])

        print("\n== 7. POST /api/ask 空问题校验 ==")
        r = await client.post("/api/ask", json={"question": ""}, headers=HEADERS)
        check("空问题返回 400", r.status_code == 400, f"status={r.status_code}")

    print()
    print(f"结果: {_PASS} 通过, {_FAIL} 失败")
    if _FAIL:
        print("存在失败用例，请检查！")
        sys.exit(1)
    print("全部通过 ✓")


if __name__ == "__main__":
    asyncio.run(main())
