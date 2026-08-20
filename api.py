"""
REST API 数据产品接口

基于 FastAPI 构建，将决策系统封装为可调用的数据产品 API。
体现"数据要素市场化"理念——数据产品可通过 API 被其他系统调用和集成。

运行方式：
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

或：
    python api.py
"""

import hmac
import json
import os
import sys
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))
from models import SugarcaneDecisionSystem
from data_security import (
    AuditLogger, DataMasker, DataIntegrityChecker,
    SecurityManager, get_security_status, InputValidator
)

# ---------------------------------------------------------------------------
# API Key 鉴权
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("SUGARCANE_API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

if not API_KEY:
    # 本地开发允许启动，但给出强警告；生产环境必须设置
    import logging
    logging.getLogger("api").warning(
        "未设置 SUGARCANE_API_KEY 环境变量，使用随机临时密钥。"
        "生产环境请务必设置强密钥！"
    )
    API_KEY = "dev-" + os.urandom(16).hex()


async def verify_api_key(api_key: str = Security(api_key_header)):
    """验证 API Key（使用恒定时间比较防御时序攻击）"""
    if api_key is None:
        raise HTTPException(status_code=401, detail="缺少 API Key，请在请求头中添加 X-API-Key")
    # 防御时序攻击：hmac.compare_digest 不受输入长度差异影响
    if not hmac.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=403, detail="API Key 无效")
    return api_key

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="蔗循智策 - 甘蔗副产物循环经济决策数据产品 API",
    description="""面向中国-东盟的甘蔗副产物循环经济跨境数据协同决策系统 · 数据产品接口

## 数据产品特性

- **产量预测**：GBRT LOOCV，R²=0.893，7市×10年训练数据
- **碳排放核算**：IPCC AR6 Tier 1，含N₂O 44/28转换，23条排放因子可溯源
- **多目标优化**：经济收益70%+碳减排30%，五方案自动对比（传统/改良传统/基础循环/进阶循环/最优循环）
- **跨境对比**：中国-泰国-越南-缅甸-老挝五国参数化决策
- **数据产品化**：对齐 GB/T 47950-2026《数据资产登记指南》、GB/T 46353-2025《数据资产价值评估》

## 鉴权方式

在请求头中添加 `X-API-Key: <你的API密钥>`（通过环境变量 `SUGARCANE_API_KEY` 设置）

## 学术对标

石杰锋等 (2023) 《智慧农业(中英文)》DOI: 10.12133/j.smartag.SA202304004
    """,
    version="1.3.0",
    docs_url="/docs" if os.environ.get("SCZC_ENABLE_DOCS", "true").lower() == "true" else None,
    redoc_url="/redoc" if os.environ.get("SCZC_ENABLE_DOCS", "true").lower() == "true" else None,
    openapi_tags=[
        {"name": "数据产品", "description": "核心决策接口"},
        {"name": "系统", "description": "健康检查与系统信息"},
    ],
)

# CORS 跨域支持
# 生产环境应通过 SCZC_CORS_ORIGINS 环境变量配置白名单，多个来源用逗号分隔
# 例如：SCZC_CORS_ORIGINS=https://your-domain.com,https://admin.your-domain.com
_default_origins = os.environ.get("SCZC_CORS_ORIGINS", "*")
_allow_origins = [o.strip() for o in _default_origins.split(",") if o.strip()]
# 安全加固：当 allow_origins 包含 * 时，禁止 allow_credentials=True
_allow_credentials = False if "*" in _allow_origins else (
    os.environ.get("SCZC_CORS_CREDENTIALS", "false").lower() == "true"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 全局系统实例（单例，应用启动时加载）
import threading
_system: Optional[SugarcaneDecisionSystem] = None
_lock = threading.Lock()


def _get_client_ip(request) -> str:
    """获取真实客户端 IP，优先读取 X-Forwarded-For，但仅取第一个可信值"""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        # 取第一个 IP，防止伪造链中追加的虚假 IP 被利用做日志欺骗
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "0.0.0.0"


def get_system() -> SugarcaneDecisionSystem:
    """获取或初始化决策系统实例（线程安全）

    优先热加载已保存的模型文件（warm_start_models 内部逻辑）：
    - 本地/生产已有 models/yield_predictor.pkl → 直接加载，秒级启动
    - 无模型文件或加载失败 → 快速重训练（精简后约2秒，70样本×2模型LOOCV）
    避免在请求线程内做完整训练导致首请求超时。
    """
    global _system
    if _system is None:
        with _lock:
            if _system is None:  # 双重检查锁定
                from models import warm_start_models
                try:
                    _system = warm_start_models()
                    metrics = _system.yield_predictor.metrics or {}
                    src = '热加载' if _system.yield_predictor.model is not None and \
                        not metrics.get('fallback', True) else '重训练'
                    print(f"[API] 决策系统就绪（{src}）: "
                          f"{metrics.get('model_name', 'N/A')}, "
                          f"R²={metrics.get('r2', 'N/A')}")
                except Exception as e:
                    # 训练失败不阻断启动，使用 fallback 模型，但记录错误
                    AuditLogger.log_security_event(
                        event_type="model_load_failure",
                        severity="HIGH",
                        description="API 启动时模型加载/训练失败，已回退到 fallback 模型",
                        details={"error": str(e)},
                    )
                    print(f"[API] 模型加载提示: {e}")
                    _system = SugarcaneDecisionSystem()
    return _system


# ---------------------------------------------------------------------------
# 请求/响应模型（数据产品契约）
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    """决策请求参数"""
    country: str = Field(
        default="China",
        description="国家（China/Thailand/Vietnam/Myanmar/Laos）",
        examples=["China", "Thailand", "Vietnam", "Myanmar", "Laos"],
    )
    city: str = Field(
        default="崇左市",
        description="广西城市（崇左市/来宾市/南宁市/柳州市/百色市/河池市/防城港市）",
        examples=["崇左市", "来宾市", "南宁市", "柳州市"],
    )
    area_mu: float = Field(
        default=10.0, ge=0.0, le=100000.0,
        description="种植面积（亩）",
        examples=[10.0],
    )
    avg_temp: float = Field(
        default=28.0, ge=10.0, le=45.0,
        description="生长季均温（℃）",
        examples=[28.0],
    )
    precipitation: float = Field(
        default=900.0, ge=0.0, le=5000.0,
        description="生长季累计降水（mm）",
        examples=[900.0],
    )
    sunshine: float = Field(
        default=870.0, ge=0.0, le=3000.0,
        description="生长季累计日照（h）",
        examples=[870.0],
    )
    fertilizer_n_kg: float = Field(
        default=220.0, ge=0.0, le=22000000.0,
        description="氮肥用量（kg N），模型内部按亩均 ≤220 kg N/亩 二次校验",
        examples=[220.0],
    )
    diesel_l: float = Field(
        default=50.0, ge=0.0, le=5000000.0,
        description="柴油用量（L），模型内部按亩均 ≤50 L/亩 二次校验",
        examples=[50.0],
    )
    electricity_kwh: float = Field(
        default=500.0, ge=0.0, le=50000000.0,
        description="用电量（kWh），模型内部按亩均 ≤500 kWh/亩 二次校验",
        examples=[500.0],
    )
    carbon_price: float = Field(
        default=85.0, ge=0.0, le=10000.0,
        description="碳价（元/吨CO2），不传则使用近12月碳市场均价",
        examples=[85.0],
    )


class DecisionResponse(BaseModel):
    """决策响应（数据产品格式）"""
    project: str = "蔗循智策"
    version: str = "1.0"
    timestamp: str
    # 数据产品元数据
    metadata: dict
    # 输入参数
    input: dict
    # 输出结果
    output: dict


class SchemeInfo(BaseModel):
    """方案对比信息"""
    name: str
    name_cn: str
    net_benefit: float
    carbon_emission_kg: float
    carbon_revenue: float
    total_benefit: float
    total_score: float


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "蔗循智策 API",
        "version": "1.3.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/security/status", tags=["系统"])
async def security_status():
    """
    获取系统数据安全状态

    返回数据完整性、分类分级、审计日志、跨境合规等安全体检结果。
    """
    return get_security_status()


@app.post("/api/security/verify", tags=["系统"])
async def verify_data_integrity():
    """
    手动触发数据完整性校验

    重新计算所有数据文件SHA-256哈希，与记录值比对，检测篡改。
    """
    result = DataIntegrityChecker.verify_integrity()
    return {
        "message": "数据完整性校验完成",
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/countries", tags=["数据产品"])
async def list_countries():
    """获取支持的国家/地区列表"""
    return {
        "countries": [
            {
                "code": "China",
                "name": "中国-广西",
                "description": "广西甘蔗主产区（崇左/来宾/南宁/柳州）",
            },
            {
                "code": "Thailand",
                "name": "泰国",
                "description": "泰国甘蔗产区",
            },
            {
                "code": "Vietnam",
                "name": "越南",
                "description": "越南甘蔗产区",
            },
            {
                "code": "Myanmar",
                "name": "缅甸",
                "description": "缅甸甘蔗产区（FAO统计均值，数据质量B级）",
            },
            {
                "code": "Laos",
                "name": "老挝",
                "description": "老挝甘蔗产区（FAO统计均值，数据质量B级）",
            },
        ]
    }


@app.get("/api/datasets", tags=["数据产品"])
async def list_datasets():
    """获取系统使用的数据集信息"""
    return {
        "datasets": [
            {
                "name": "guangxi_sugarcane",
                "description": "广西7市甘蔗种植与产量数据",
                "source": "广西统计年鉴",
                "years": "2015-2024",
                "cities": ["崇左市", "来宾市", "南宁市", "柳州市", "百色市", "河池市", "防城港市"],
                "type": "政府开放数据",
            },
            {
                "name": "fao_global",
                "description": "中国-东盟五国甘蔗生产对比数据",
                "source": "FAOSTAT",
                "years": "2015-2024",
                "countries": ["China", "Thailand", "Vietnam", "Myanmar", "Laos"],
                "type": "国际组织开放数据",
            },
            {
                "name": "carbon_price",
                "description": "全国碳排放权交易市场（CEA）价格数据",
                "source": "上海环境能源交易所",
                "years": "2021-2026",
                "type": "政府开放数据",
            },
            {
                "name": "weather_data",
                "description": "广西7市逐月气象观测数据",
                "source": "中国气象数据网",
                "years": "2015-2024",
                "cities": ["崇左市", "来宾市", "南宁市", "柳州市", "百色市", "河池市", "防城港市"],
                "type": "政府开放数据",
            },
            {
                "name": "ipcc_factors",
                "description": "IPCC碳排放因子数据库",
                "source": "IPCC 2006国家温室气体清单指南",
                "type": "国际标准",
            },
            {
                "name": "byproduct_params",
                "description": "甘蔗副产物生产参数",
                "source": "学术文献/行业标准",
                "type": "学术研究",
            },
            {
                "name": "market_prices",
                "description": "中-泰-越-缅-老五国副产物市场价格（90条，含环保浆料真实产业数据；东盟条目为估算并已标注）",
                "source": "1688批发/行业研报/来宾工信局/FAO/东盟行业估算",
                "type": "市场数据",
            },
        ]
    }


@app.post("/api/decision", tags=["数据产品"])
async def run_decision(
    request: DecisionRequest,
    api_key: str = Depends(verify_api_key),
    req: Request = None
):
    """
    运行决策，返回优化方案和数据产品

    这是核心数据产品接口，接收种植参数，返回最优方案、
    经济效益、碳排放等完整决策结果。
    """
    import time
    import uuid
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request_time = datetime.now().isoformat()
    system = get_system()
    client_ip = _get_client_ip(req) if req is not None else "0.0.0.0"

    # 输入安全校验（防御直接调用 Python API 时的非法入参）
    try:
        validated_city = InputValidator.validate_city(request.city)
        validated_country = InputValidator.sanitize_string(request.country, max_length=50)
    except ValueError as e:
        AuditLogger.log_security_event(
            event_type="input_validation_failure",
            severity="MEDIUM",
            description="决策接口输入校验失败",
            details={"request_id": request_id, "error": str(e), "country": request.country},
        )
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = system.run_decision(
            area_mu=request.area_mu,
            avg_temp=request.avg_temp,
            precipitation=request.precipitation,
            sunshine=request.sunshine,
            fertilizer_n_kg=request.fertilizer_n_kg,
            diesel_l=request.diesel_l,
            electricity_kwh=request.electricity_kwh,
            carbon_price=request.carbon_price,
            country=validated_country,
            city=validated_city,
        )
    except ValueError as e:
        # 业务校验失败（如参数越界）返回 400
        AuditLogger.log_security_event(
            event_type="business_validation_failure",
            severity="MEDIUM",
            description="决策业务校验失败",
            details={"request_id": request_id, "error": str(e), "country": request.country},
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 审计日志：记录失败（不返回内部错误详情）
        duration = (time.time() - start_time) * 1000
        failure_params = request.model_dump()
        failure_params["request_id"] = request_id
        AuditLogger.log_api_access(
            endpoint="/api/decision",
            client_ip=client_ip,
            api_key=api_key,
            params=failure_params,
            status_code=500,
            response_size=0,
            duration_ms=duration,
            country=request.country,
        )
        # 安全加固：不将内部异常详情返回给客户端
        AuditLogger.log_security_event(
            event_type="decision_error",
            severity="HIGH",
            description="决策计算异常",
            details={"request_id": request_id, "error": str(e), "country": request.country},
        )
        raise HTTPException(status_code=500, detail="决策计算失败，请检查输入参数或联系管理员")

    # 五方案名称映射
    scheme_name_cn = {
        "traditional": "传统模式",
        "improved_traditional": "改良传统",
        "circular_basic": "基础循环",
        "circular_advanced": "进阶循环",
        "circular_optimal": "最优循环",
    }

    # 最优方案
    opt = result["optimization"]["optimal"]

    # 构建方案对比列表
    all_schemes = []
    for s in result["optimization"]["all_schemes"]:
        all_schemes.append({
            "name": s["name"],
            "name_cn": scheme_name_cn.get(s["name"], s["name"]),
            "net_benefit": round(s["net_benefit"], 2),
            "carbon_emission_kg": round(s["carbon_emission_kg"], 2),
            "carbon_revenue": round(s.get("carbon_revenue", 0), 2),
            "total_benefit": round(s.get("total_benefit", s["net_benefit"]), 2),
            "total_score": round(s["total_score"], 4),
        })

    # 输出（数据产品核心）
    output = {
        "yield": {
            "yield_per_mu": round(result["yield_per_mu"], 2),
            "total_yield": round(result["total_yield"], 2),
            "unit": "吨/亩",
            "source": result.get("yield_source", "model"),
        },
        "carbon_emission": {
            "total_tons": round(result["carbon_emission"]["total_tons"], 4),
            "planting_kg": round(result["carbon_emission"]["planting"], 2),
            "mechanization_kg": round(result["carbon_emission"]["mechanization"], 2),
            "processing_kg": round(result["carbon_emission"]["processing"], 2),
        },
        "byproducts": {
            bp: round(info["quantity"], 2)
            for bp, info in result["byproducts"].items()
        },
        "optimal_scheme": {
            "name": opt["name"],
            "name_cn": scheme_name_cn.get(opt["name"], opt["name"]),
            "net_benefit": round(opt["net_benefit"], 2),
            "total_benefit": round(opt.get("total_benefit", opt["net_benefit"]), 2),
            "carbon_revenue": round(opt.get("carbon_revenue", 0), 2),
            "total_score": round(opt["total_score"], 4),
        },
        "all_schemes": all_schemes,
    }

    # 获取模型指标（复用已有实例）
    model_metrics = system.yield_predictor.metrics or {}

    # 数据产品元数据
    metadata = {
        "data_product_id": f"SCZC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "request_id": request_id,
        "request_time": request_time,
        "product_type": "决策支持数据产品",
        "update_frequency": "实时计算",
        "data_freshness": "气象数据至2024年，产量数据至2024年(7市×10年=70样本)，碳价数据近12月滚动更新",
        "model_info": {
            "name": model_metrics.get("model_name", "N/A"),
            "method": "LOOCV交叉验证" if request.country == "China" else "FAO统计均值",
            "r2": round(model_metrics.get("r2", 0), 4) if not model_metrics.get("fallback") else None,
            "rmse": round(model_metrics.get("rmse", 0), 4) if not model_metrics.get("fallback") else None,
            "training_samples": model_metrics.get("loocv_samples"),
            "cities": 7,
        },
        "data_sources": [
            {"name": "广西甘蔗种植数据", "source": "广西统计年鉴", "type": "政府开放数据"},
            {"name": "气象数据", "source": "中国气象数据网", "type": "政府开放数据"},
            {"name": "FAO全球数据", "source": "FAOSTAT", "type": "国际组织数据"},
            {"name": "碳排放系数", "source": "IPCC 2006", "type": "国际标准"},
            {"name": "碳交易价格", "source": "上海环境能源交易所", "type": "政府开放数据"},
            {"name": "市场价格", "source": "1688批发/行业研报/来宾工信局/FAO/东盟行业估算", "type": "市场数据"},
        ],
        "coverage": {
            "region": "中国-广西（崇左/来宾/南宁/柳州/百色/河池/防城港）、泰国、越南、缅甸、老挝",
            "time_range": "2015-2024",
            "cross_border_framework": "RCEP农业数据合作 + 中国-东盟跨境产业对比",
        },
        "disclaimer": "本数据产品仅供决策参考，不构成投资建议。跨境数据采用FAO统计均值与东盟市场估算，实际应用需结合当地实测数据校准。",
    }

    # 构建响应
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "input": request.model_dump(),
        "output": output,
    }

    # 审计日志：记录成功访问
    duration = (time.time() - start_time) * 1000
    response_size = len(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))
    audit_params = request.model_dump()
    audit_params["request_id"] = request_id
    AuditLogger.log_api_access(
        endpoint="/api/decision",
        client_ip=client_ip,
        api_key=api_key,
        params=audit_params,
        status_code=200,
        response_size=response_size,
        duration_ms=duration,
        country=request.country,
    )

    # 响应脱敏（隐藏内部细节）
    masked_response = DataMasker.mask_api_response(response_data)

    return DecisionResponse(
        timestamp=masked_response["timestamp"],
        metadata=masked_response["metadata"],
        input=masked_response["input"],
        output=masked_response["output"],
    )


@app.get("/api/decision", tags=["数据产品"])
async def run_decision_get(
    api_key: str = Depends(verify_api_key),
    req: Request = None,
    country: str = Query("China", description="国家"),
    city: str = Query("崇左市", description="广西城市"),
    area_mu: float = Query(10.0, ge=0.0, le=100000.0, description="种植面积（亩）"),
    avg_temp: float = Query(28.0, ge=10.0, le=45.0, description="生长季均温（℃）"),
    precipitation: float = Query(900.0, ge=0.0, le=5000.0, description="生长季累计降水（mm）"),
    sunshine: float = Query(870.0, ge=0.0, le=3000.0, description="生长季累计日照（h）"),
    fertilizer_n_kg: float = Query(220.0, ge=0.0, le=22000000.0, description="氮肥用量（kg N），按亩均 ≤220 kg N/亩 二次校验"),
    diesel_l: float = Query(50.0, ge=0.0, le=5000000.0, description="柴油用量（L），按亩均 ≤50 L/亩 二次校验"),
    electricity_kwh: float = Query(500.0, ge=0.0, le=50000000.0, description="用电量（kWh），按亩均 ≤500 kWh/亩 二次校验"),
    carbon_price: float = Query(85.0, ge=0.0, le=10000.0, description="碳价（元/吨CO2）"),
):
    """
    通过 GET 请求运行决策（方便浏览器直接访问和 curl 测试）

    示例：
        curl -H "X-API-Key: $SUGARCANE_API_KEY" "http://localhost:8000/api/decision?country=China&area_mu=10.0"
    """
    request = DecisionRequest(
        country=country,
        city=city,
        area_mu=area_mu,
        avg_temp=avg_temp,
        precipitation=precipitation,
        sunshine=sunshine,
        fertilizer_n_kg=fertilizer_n_kg,
        diesel_l=diesel_l,
        electricity_kwh=electricity_kwh,
        carbon_price=carbon_price,
    )
    return await run_decision(request=request, api_key=api_key, req=req)


# ---------------------------------------------------------------------------
# LLM 增强接口（决策报告 + 自然语言问数）
# 核心计算仍由规则引擎/统计模型完成；LLM 仅负责报告润色与开放式答疑。
# 未配置 SCZC_LLM_API_KEY 时，这些接口自动回退：llm_used=False，功能不中断。
# ---------------------------------------------------------------------------

class ReportRequest(DecisionRequest):
    """决策报告请求（含可选 LLM 润色）"""
    use_llm: bool = Field(
        default=True, description="是否尝试用 LLM 生成自然语言报告（无 key 时自动回退规则模板）"
    )


class AskRequest(BaseModel):
    """自然语言问数请求"""
    question: str = Field(..., description="开放式问题", examples=["蔗渣生物质颗粒为什么能赚钱？"])
    use_llm: bool = Field(default=True, description="是否尝试用 LLM 作答")
    country: str = Field(default="China", description="国家")
    city: str = Field(default="崇左市", description="广西城市")
    area_mu: float = Field(default=10.0, ge=0.0, le=100000.0, description="种植面积（亩）")
    avg_temp: float = Field(default=28.0, ge=10.0, le=45.0, description="生长季均温（℃）")
    precipitation: float = Field(default=900.0, ge=0.0, le=5000.0, description="生长季累计降水（mm）")
    sunshine: float = Field(default=870.0, ge=0.0, le=3000.0, description="生长季累计日照（h）")
    carbon_price: float | None = Field(default=None, description="碳价（元/吨，缺省取市场均价）")


_DECISION_SCHEME_CN = {
    "traditional": "传统模式", "improved_traditional": "改良传统",
    "circular_basic": "基础循环", "circular_advanced": "进阶循环",
    "circular_optimal": "最优循环",
}


@app.post("/api/report", tags=["数据产品"])
async def run_report(
    request: ReportRequest,
    api_key: str = Depends(verify_api_key),
    req: Request = None
):
    """
    生成决策报告（可选 LLM 润色）

    在 /api/decision 的结构化结果之上，额外返回推理链 reasoning；
    当 use_llm=true 且系统已配置 LLM 服务时，返回 llm_report（自然语言报告）
    与 llm_used=true；否则 llm_used=false，llm_report 为 null，调用方可回退规则模板。
    """
    from agent import generate_reasoning_chain
    from llm_agent import enhance_decision_report, get_client

    system = get_system()
    client_ip = _get_client_ip(req) if req is not None else "0.0.0.0"
    llm_available = get_client().available

    try:
        validated_city = InputValidator.validate_city(request.city)
        validated_country = InputValidator.sanitize_string(request.country, max_length=50)
        result = system.run_decision(
            area_mu=request.area_mu,
            avg_temp=request.avg_temp,
            precipitation=request.precipitation,
            sunshine=request.sunshine,
            fertilizer_n_kg=request.fertilizer_n_kg,
            diesel_l=request.diesel_l,
            electricity_kwh=request.electricity_kwh,
            carbon_price=request.carbon_price,
            country=validated_country,
            city=validated_city,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        AuditLogger.log_security_event(
            event_type="decision_error", severity="HIGH",
            description="决策报告接口计算异常",
            details={"error": str(e), "country": request.country},
        )
        raise HTTPException(status_code=500, detail="决策计算失败，请检查输入或联系管理员")

    opt = result["optimization"]["optimal"]
    params = {
        "country": validated_country, "city": validated_city,
        "area_mu": request.area_mu, "avg_temp": request.avg_temp,
        "precipitation": request.precipitation, "sunshine": request.sunshine,
        "carbon_price": request.carbon_price,
    }
    reasoning = generate_reasoning_chain(result, params, system)

    # LLM 报告（失败自动返回 None）
    llm_report = None
    llm_used = False
    if request.use_llm and llm_available:
        llm_report = enhance_decision_report(params, result, reasoning)
        if llm_report:
            llm_used = True

    output = {
        "optimal_scheme": {
            "name": opt["name"],
            "name_cn": _DECISION_SCHEME_CN.get(opt["name"], opt["name"]),
            "net_benefit": round(opt["net_benefit"], 2),
            "total_benefit": round(opt.get("total_benefit", opt["net_benefit"]), 2),
            "carbon_revenue": round(opt.get("carbon_revenue", 0), 2),
        },
        "yield_per_mu": round(result["yield_per_mu"], 2),
        "total_yield": round(result["total_yield"], 2),
        "carbon_total_tons": round(result["carbon_emission"]["total_tons"], 4),
    }

    AuditLogger.log_api_access(
        endpoint="/api/report", client_ip=client_ip, api_key=api_key,
        params={"city": validated_city, "country": validated_country},
        status_code=200, response_size=0, duration_ms=0, country=validated_country,
    )

    return {
        "project": "蔗循智策",
        "timestamp": datetime.now().isoformat(),
        "input": request.model_dump(),
        "output": output,
        "reasoning": reasoning,
        "llm_report": llm_report,
        "llm_used": llm_used,
        "llm_available": llm_available,
    }


@app.post("/api/ask", tags=["数据产品"])
async def ask_question(
    request: AskRequest,
    api_key: str = Depends(verify_api_key),
    req: Request = None
):
    """
    自然语言问数（可选用 LLM 作答）

    基于给定（或默认）场景工况运行一次确定性决策作为事实上下文，
    再让 LLM 回答开放式问题。未配置 LLM 时返回规则兜底，use_llm=false。
    """
    from agent import (SugarcaneAgent, FERTILIZER_N_PER_MU, DIESEL_PER_MU,
                       ELECTRICITY_PER_MU)
    from llm_agent import get_client

    client_ip = _get_client_ip(req) if req is not None else "0.0.0.0"
    llm_available = get_client().available
    question = InputValidator.sanitize_string(request.question, max_length=1000)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        validated_city = InputValidator.validate_city(request.city)
        validated_country = InputValidator.sanitize_string(request.country, max_length=50)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    agent = SugarcaneAgent(system=get_system())
    params = {
        "area_mu": request.area_mu, "city": validated_city, "country": validated_country,
        "avg_temp": request.avg_temp, "precipitation": request.precipitation,
        "sunshine": request.sunshine,
        "carbon_price": (request.carbon_price if request.carbon_price is not None
                         else agent.default_carbon_price),
    }
    result = agent.system.run_decision(
        area_mu=params["area_mu"], avg_temp=params["avg_temp"],
        precipitation=params["precipitation"], sunshine=params["sunshine"],
        fertilizer_n_kg=FERTILIZER_N_PER_MU * params["area_mu"],
        diesel_l=DIESEL_PER_MU * params["area_mu"],
        electricity_kwh=ELECTRICITY_PER_MU * params["area_mu"],
        carbon_price=params["carbon_price"],
        country=params["country"], city=params["city"],
    )

    if request.use_llm and llm_available:
        try:
            from llm_agent import answer_question as _llm_answer
            answer = _llm_answer(question, params, result)
            if answer:
                AuditLogger.log_api_access(
                    endpoint="/api/ask", client_ip=client_ip, api_key=api_key,
                    params={"city": validated_city, "country": validated_country,
                            "llm": True}, status_code=200, response_size=0,
                    duration_ms=0, country=validated_country,
                )
                return {"question": question, "answer": answer, "use_llm": True,
                        "llm_available": True, "timestamp": datetime.now().isoformat()}
        except Exception:
            pass

    # 规则兜底
    opt = result["optimization"]["optimal"]
    fallback = (
        "当前未启用语言模型服务（未配置 SCZC_LLM_API_KEY 或调用失败），"
        "无法开放作答。为你提供本次决策的确定性结论：\n\n"
        f"- 最优方案：{_DECISION_SCHEME_CN.get(opt['name'], opt['name'])}，"
        f"综合收益 {opt.get('total_benefit', opt['net_benefit']):,.0f} 元\n"
        f"- 净收益：{opt['net_benefit']:,.0f} 元，碳交易收益 {opt.get('carbon_revenue', 0):+,.0f} 元\n\n"
        "配置 SCZC_LLM_API_KEY 后可对子方案、碳减排原理、副产物利用路径等进行开放问答。"
    )
    AuditLogger.log_api_access(
        endpoint="/api/ask", client_ip=client_ip, api_key=api_key,
        params={"city": validated_city, "country": validated_country, "llm": False},
        status_code=200, response_size=0, duration_ms=0, country=validated_country,
    )
    return {"question": question, "answer": fallback, "use_llm": False,
            "llm_available": llm_available, "timestamp": datetime.now().isoformat()}


@app.get("/api/llm/status", tags=["系统"])
async def llm_status(api_key: str = Depends(verify_api_key)):
    """获取 LLM 增强服务状态（是否已配置，供前端展示增强开关）"""
    from llm_agent import get_client
    return {
        "llm_configured": get_client().available,
        "note": "LLM 仅负责报告润色与开放式答疑，不参与核心计算",
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# 三证一价与数据要素接口
# ---------------------------------------------------------------------------

@app.get("/api/certifications", tags=["数据要素"])
async def get_certifications(api_key: str = Depends(verify_api_key)):
    """
    获取数据产品"三证一价"

    返回数据资产登记证书、数据质量证书、数据安全证书、数据定价报告。
    对标 GB/T 47950-2026《数据资产登记指南》、GB/T 46353-2025《数据资产价值评估》。
    """
    from data_product import DataProductCertification
    cert = DataProductCertification()
    return cert.get_all_certifications()


@app.get("/api/certifications/{cert_type}", tags=["数据要素"])
async def get_single_certificate(cert_type: str, api_key: str = Depends(verify_api_key)):
    """
    获取单张证书/报告

    - registration: 数据资产登记证书
    - quality: 数据质量证书
    - security: 数据安全证书
    - pricing: 数据定价报告
    """
    from data_product import DataProductCertification
    cert = DataProductCertification()

    if cert_type == "registration":
        return cert.generate_registration_cert()
    elif cert_type == "quality":
        return cert.generate_quality_cert()
    elif cert_type == "security":
        return cert.generate_security_cert()
    elif cert_type == "pricing":
        return cert.generate_pricing_report()
    else:
        raise HTTPException(status_code=400, detail=f"不支持的证书类型: {cert_type}")


@app.get("/api/lineage", tags=["数据要素"])
async def get_data_lineage():
    """
    获取数据血缘信息（无需鉴权，公开透明）

    返回数据从原始采集到最终产品服务的完整流转链路。
    """
    from data_product import get_lineage_data
    nodes, links = get_lineage_data()
    return {"nodes": nodes, "links": links}


@app.get("/api/trading/scenarios", tags=["数据要素"])
async def list_trading_scenarios():
    """获取数据交易场景列表"""
    from data_product import DataTradingSimulation
    return {"scenarios": DataTradingSimulation.get_scenarios()}


@app.get("/api/trading/simulate", tags=["数据要素"])
async def simulate_trading(
    scenario_id: str,
    months: int = Query(12, ge=1, le=120, description="模拟月数（1-120）")
):
    """
    模拟数据交易流水

    - scenario_id: 场景ID（从 /api/trading/scenarios 获取）
    - months: 模拟月数（默认12个月，上限120防止资源耗尽）
    """
    from data_product import DataTradingSimulation
    result = DataTradingSimulation.simulate_transaction(scenario_id, months)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# 用户验证闭环接口
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """用户反馈请求"""
    predicted_yield: float = Field(..., description="系统预测单产（吨/亩）")
    actual_yield: float = Field(..., description="实际单产（吨/亩）")
    predicted_benefit: float = Field(..., description="系统预测净收益（元）")
    actual_benefit: float = Field(..., description="实际净收益（元）")
    city: str = Field(default="崇左市", description="城市")
    country: str = Field(default="China", description="国家")
    user_type: str = Field(default="其他", description="用户类型")
    notes: str = Field(default="", description="备注")


@app.post("/api/feedback", tags=["数据产品"])
async def submit_feedback(
    request: FeedbackRequest,
    api_key: str = Depends(verify_api_key),
    req: Request = None
):
    """
    提交用户实际验证数据

    实现预测→实际→偏差分析的验证闭环。系统自动计算产量和收益的偏差百分比。
    """
    from user_validation import FeedbackCollector
    from data_security import InputValidator

    client_ip = _get_client_ip(req) if req is not None else "0.0.0.0"

    # 输入安全校验
    try:
        city = InputValidator.validate_city(request.city)
        notes = InputValidator.sanitize_string(request.notes, max_length=500)
        user_type = InputValidator.sanitize_string(request.user_type, max_length=50)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    fb = FeedbackCollector.submit_feedback(
        predicted_yield=request.predicted_yield,
        actual_yield=request.actual_yield,
        predicted_benefit=request.predicted_benefit,
        actual_benefit=request.actual_benefit,
        city=city,
        country=request.country,
        user_type=user_type,
        notes=notes,
    )

    AuditLogger.log_api_access(
        endpoint="/api/feedback",
        client_ip=client_ip,
        api_key=api_key,
        params={"city": city, "country": request.country},
        status_code=200,
        response_size=0,
        duration_ms=0,
        country=request.country,
    )

    return {
        "message": "反馈提交成功",
        "feedback": fb,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/validation/stats", tags=["数据产品"])
async def get_validation_stats(api_key: str = Depends(verify_api_key)):
    """
    获取用户验证统计

    返回累计验证数据集的统计指标，包括平均偏差、MAPE等。
    """
    from user_validation import FeedbackCollector
    stats = FeedbackCollector.get_validation_stats()
    return {
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/run/stats", tags=["数据产品"])
async def get_run_stats(api_key: str = Depends(verify_api_key)):
    """
    获取系统真实运行台账统计

    聚合真实审计日志（API调用次数/端点/客户端IP/安全拦截/跨境授权）
    与真实反馈闭环（反馈条数/MAPE/是否需校准）。所有数字直接来自
    logs/security_audit.log 与 data/user_feedback.json，可溯源、无模拟注入。
    """
    from run_stats import get_run_stats
    return get_run_stats()


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       蔗循智策 - 数据产品 API v1.3.0                    ║
    ║       面向中国-东盟的甘蔗副产物循环经济决策系统          ║
    ║       支持中国-泰国-越南-缅甸-老挝五国跨境决策          ║
    ╠══════════════════════════════════════════════════════════╣
    ║  API 文档: http://localhost:8000/docs                   ║
    ║  健康检查: http://localhost:8000/health                 ║
    ║  决策接口: POST http://localhost:8000/api/decision      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")