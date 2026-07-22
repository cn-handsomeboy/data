"""
REST API 数据产品接口

基于 FastAPI 构建，将决策系统封装为可调用的数据产品 API。
体现"数据要素市场化"理念——数据产品可通过 API 被其他系统调用和集成。

运行方式：
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

或：
    python api.py
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))
from models import SugarcaneDecisionSystem

# ---------------------------------------------------------------------------
# API Key 鉴权
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("SUGARCANE_API_KEY", "sczc-demo-key-2026")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """验证 API Key"""
    if api_key is None:
        raise HTTPException(status_code=401, detail="缺少 API Key，请在请求头中添加 X-API-Key")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="API Key 无效")
    return api_key

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="蔗循智策 - 甘蔗副产物循环经济决策数据产品 API",
    description="""面向中国-东盟的甘蔗副产物循环经济跨境数据协同决策系统 · 数据产品接口

## 数据产品特性

- **产量预测**：Ridge LOOCV，R²=0.862，7市×10年训练数据
- **碳排放核算**：IPCC AR6 Tier 1，含N₂O 44/28转换，23条排放因子可溯源
- **多目标优化**：经济收益70%+碳减排30%，三方案自动对比
- **跨境对比**：中国-泰国-越南三国参数化决策
- **数据产品化**：对齐 GB/T 47950-2026《数据资产登记指南》、GB/T 46353-2025《数据资产价值评估》

## 鉴权方式

在请求头中添加 `X-API-Key: sczc-demo-key-2026`

## 学术对标

石杰锋等 (2023) 《智慧农业(中英文)》DOI: 10.12133/j.smartag.SA202304004
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "数据产品", "description": "核心决策接口"},
        {"name": "系统", "description": "健康检查与系统信息"},
    ],
)

# CORS 跨域支持（允许其他系统调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局系统实例（单例，应用启动时加载）
import threading
_system: Optional[SugarcaneDecisionSystem] = None
_lock = threading.Lock()


def get_system() -> SugarcaneDecisionSystem:
    """获取或初始化决策系统实例（线程安全）"""
    global _system
    if _system is None:
        with _lock:
            if _system is None:  # 双重检查锁定
                _system = SugarcaneDecisionSystem()
                try:
                    metrics = _system.train_models(model_type='auto')
                    print(f"[API] 模型训练完成，选择模型: {metrics.get('model_name', 'N/A')}, R²={metrics.get('r2', 'N/A')}")
                except Exception as e:
                    print(f"[API] 模型训练提示: {e}")
    return _system


# ---------------------------------------------------------------------------
# 请求/响应模型（数据产品契约）
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    """决策请求参数"""
    country: str = Field(
        default="China",
        description="国家（China/Thailand/Vietnam）",
        examples=["China", "Thailand", "Vietnam"],
    )
    city: str = Field(
        default="崇左市",
        description="广西城市（崇左市/来宾市/南宁市/柳州市/百色市/河池市/防城港市）",
        examples=["崇左市", "来宾市", "南宁市", "柳州市"],
    )
    area_mu: float = Field(
        default=10.0, ge=1.0, le=10000.0,
        description="种植面积（亩）",
        examples=[10.0],
    )
    avg_temp: float = Field(
        default=28.0, ge=15.0, le=35.0,
        description="生长季均温（℃）",
        examples=[28.0],
    )
    precipitation: float = Field(
        default=900.0, ge=500.0, le=3000.0,
        description="生长季累计降水（mm）",
        examples=[900.0],
    )
    sunshine: float = Field(
        default=870.0, ge=500.0, le=2000.0,
        description="生长季累计日照（h）",
        examples=[870.0],
    )
    fertilizer_n_kg: float = Field(
        default=220.0, ge=0.0, le=5000.0,
        description="氮肥用量（kg N）",
        examples=[220.0],
    )
    diesel_l: float = Field(
        default=50.0, ge=0.0, le=1000.0,
        description="柴油用量（L）",
        examples=[50.0],
    )
    electricity_kwh: float = Field(
        default=500.0, ge=0.0, le=2000.0,
        description="用电量（kWh）",
        examples=[500.0],
    )
    carbon_price: float = Field(
        default=85.0, ge=0.0, le=500.0,
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
        "version": "1.0.0",
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
                "description": "中国-东盟三国甘蔗生产对比数据",
                "source": "FAOSTAT",
                "years": "2015-2024",
                "countries": ["China", "Thailand", "Vietnam"],
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
                "description": "中-泰-越三国副产物市场价格（30+条，含环保浆料真实产业数据）",
                "source": "1688批发/行业研报/来宾工信局",
                "type": "市场数据",
            },
        ]
    }


@app.post("/api/decision", tags=["数据产品"])
async def run_decision(request: DecisionRequest, api_key: str = Depends(verify_api_key)):
    """
    运行决策，返回优化方案和数据产品

    这是核心数据产品接口，接收种植参数，返回最优方案、
    经济效益、碳排放等完整决策结果。
    """
    system = get_system()

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
            country=request.country,
            city=request.city,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"决策计算失败: {str(e)}")

    # 方案名称映射
    scheme_name_cn = {
        "traditional": "传统模式",
        "circular_basic": "基础循环",
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
            {"name": "市场价格", "source": "1688批发/行业研报/来宾工信局", "type": "市场数据"},
        ],
        "coverage": {
            "region": "中国-广西（崇左/来宾/南宁/柳州/百色/河池/防城港）、泰国、越南",
            "time_range": "2015-2024",
        },
        "disclaimer": "本数据产品仅供决策参考，不构成投资建议。跨境数据采用FAO统计均值，实际应用需结合当地实测数据校准。",
    }

    return DecisionResponse(
        timestamp=datetime.now().isoformat(),
        metadata=metadata,
        input=request.model_dump(),
        output=output,
    )


@app.get("/api/decision", tags=["数据产品"])
async def run_decision_get(
    country: str = Query("China", description="国家"),
    city: str = Query("崇左市", description="广西城市"),
    area_mu: float = Query(10.0, description="种植面积（亩）"),
    avg_temp: float = Query(28.0, description="生长季均温（℃）"),
    precipitation: float = Query(900.0, description="生长季累计降水（mm）"),
    sunshine: float = Query(870.0, description="生长季累计日照（h）"),
    fertilizer_n_kg: float = Query(220.0, description="氮肥用量（kg N）"),
    diesel_l: float = Query(50.0, description="柴油用量（L）"),
    electricity_kwh: float = Query(500.0, description="用电量（kWh）"),
    carbon_price: float = Query(85.0, description="碳价（元/吨CO2）"),
):
    """
    通过 GET 请求运行决策（方便浏览器直接访问和 curl 测试）

    示例：
        curl http://localhost:8000/api/decision?country=China&area_mu=10.0
    """
    req = DecisionRequest(
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
    return await run_decision(req)


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       蔗循智策 - 数据产品 API                           ║
    ║       面向中国-东盟的甘蔗副产物循环经济决策系统          ║
    ╠══════════════════════════════════════════════════════════╣
    ║  API 文档: http://localhost:8000/docs                   ║
    ║  健康检查: http://localhost:8000/health                 ║
    ║  决策接口: POST http://localhost:8000/api/decision      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")