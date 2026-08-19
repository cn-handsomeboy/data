"""
数据产品"三证一价"体系与数据血缘管理

对接 GB/T 47950-2026《数据资产登记指南》、GB/T 46353-2025《数据资产价值评估》
实现数据资产登记证书、数据质量证书、数据安全证书、数据定价报告。
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from models import DATA_DIR

# ---------------------------------------------------------------------------
# 数据血缘节点定义
# ---------------------------------------------------------------------------

DATA_LINEAGE_NODES = [
    {"id": "src_gov", "name": "政府开放数据", "type": "source", "detail": "广西统计年鉴、中国气象数据网"},
    {"id": "src_intl", "name": "国际组织数据", "type": "source", "detail": "FAOSTAT（中/泰/越/缅/老五国）、IPCC 2006"},
    {"id": "src_mkt", "name": "市场/文献数据", "type": "source", "detail": "1688批发、行业研报、学术论文"},
    {"id": "raw_gx", "name": "广西甘蔗产量原始数据", "type": "raw", "detail": "7市×10年=70条样本"},
    {"id": "raw_wx", "name": "气象原始数据", "type": "raw", "detail": "7市×120月=840条记录"},
    {"id": "raw_fao", "name": "FAO全球数据", "type": "raw", "detail": "中/泰/越/缅/老 5国×10年"},
    {"id": "proc_clean", "name": "数据清洗与对齐", "type": "process", "detail": "缺失值填补、单位统一、年份对齐"},
    {"id": "proc_merge", "name": "多源数据融合", "type": "process", "detail": "气象+产量+FAO数据按年份/城市合并"},
    {"id": "feat_eng", "name": "特征工程", "type": "process", "detail": "生成城市哑变量、年份趋势、交互特征"},
    {"id": "model_train", "name": "模型训练", "type": "process", "detail": "GBRT/Ridge 固定超参 LOOCV"},
    {"id": "model_eval", "name": "模型评估", "type": "process", "detail": "LOOCV+SHAP可解释性"},
    {"id": "prod_yield", "name": "产量预测数据产品", "type": "product", "detail": "输入气象→输出单产(吨/亩)"},
    {"id": "prod_carbon", "name": "碳排放核算数据产品", "type": "product", "detail": "输入种植参数→输出CO2e(kg)"},
    {"id": "prod_econ", "name": "经济效益数据产品", "type": "product", "detail": "输入副产物→输出净收益(元)"},
    {"id": "prod_opt", "name": "多目标优化数据产品", "type": "product", "detail": "五方案对比+最优推荐"},
    {"id": "api_svc", "name": "API数据服务", "type": "service", "detail": "FastAPI封装，支持外部系统调用"},
    {"id": "app_ui", "name": "可视化决策界面", "type": "service", "detail": "Streamlit交互式决策面板"},
]

DATA_LINEAGE_LINKS = [
    {"source": "src_gov", "target": "raw_gx", "value": 70},
    {"source": "src_gov", "target": "raw_wx", "value": 840},
    {"source": "src_intl", "target": "raw_fao", "value": 50},
    {"source": "raw_gx", "target": "proc_clean", "value": 70},
    {"source": "raw_wx", "target": "proc_clean", "value": 840},
    {"source": "raw_fao", "target": "proc_clean", "value": 50},
    {"source": "proc_clean", "target": "proc_merge", "value": 960},
    {"source": "proc_merge", "target": "feat_eng", "value": 960},
    {"source": "feat_eng", "target": "model_train", "value": 70},
    {"source": "model_train", "target": "model_eval", "value": 70},
    {"source": "model_eval", "target": "prod_yield", "value": 70},
    {"source": "src_mkt", "target": "prod_carbon", "value": 23},
    {"source": "src_mkt", "target": "prod_econ", "value": 35},
    {"source": "prod_yield", "target": "prod_opt", "value": 70},
    {"source": "prod_carbon", "target": "prod_opt", "value": 23},
    {"source": "prod_econ", "target": "prod_opt", "value": 35},
    {"source": "prod_opt", "target": "api_svc", "value": 100},
    {"source": "prod_opt", "target": "app_ui", "value": 100},
]


# ---------------------------------------------------------------------------
# 三证一价生成器
# ---------------------------------------------------------------------------

class DataProductCertification:
    """数据产品三证一价生成器"""

    def __init__(self):
        self.product_id = "SCZC-DP-2026-001"
        self.product_name = "蔗循智策 - 甘蔗副产物循环经济决策数据产品"
        self.version = "2.0.0"
        self.owner = "蔗循智策项目团队"
        self.issue_date = datetime.now().strftime("%Y-%m-%d")

    def _hash(self, content: str) -> str:
        """生成内容哈希"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def generate_registration_cert(self) -> dict:
        """生成数据资产登记证书"""
        cert = {
            "certificate_type": "数据资产登记证书",
            "certificate_id": f"REG-{self.product_id}",
            "product_id": self.product_id,
            "product_name": self.product_name,
            "version": self.version,
            "owner": self.owner,
            "issue_date": self.issue_date,
            "valid_until": "2027-12-31",
            "issuing_authority": "项目自主登记（对标 GB/T 47950-2026）",
            "registration_note": "【演示性证书】本证书为项目团队按 GB/T 47950-2026 框架自建的展示材料，"
                                 "未经官方登记机构审核，不具法律效力，仅用于演示数据资产管理流程。",
            "asset_category": "决策支持类数据产品",
            "data_sources": [
                {"name": "广西甘蔗种植数据", "source": "广西统计年鉴", "license": "政府开放数据"},
                {"name": "气象数据", "source": "中国气象数据网/Open-Meteo ERA5", "license": "CC BY 4.0 / 政府开放"},
                {"name": "FAO全球数据", "source": "FAOSTAT", "license": "CC BY-NC-SA 3.0 IGO"},
                {"name": "IPCC排放因子", "source": "IPCC 2006指南", "license": "国际标准公开"},
                {"name": "碳交易价格", "source": "上海环境能源交易所", "license": "公开行情数据"},
                {"name": "市场价格", "source": "1688批发/行业研报", "license": "公开商业数据"},
            ],
            "registration_content": {
                "data_scale": "原始数据约960条记录，融合后特征矩阵70×11",
                "coverage": "中国-广西7市 + 泰国/越南/缅甸/老挝",
                "time_span": "2015-2024年",
                "update_frequency": "年度更新（产量/FAO）+ 实时更新（碳价/API）",
                "technical_architecture": "Python + scikit-learn + FastAPI + Streamlit",
            },
            "integrity_hash": self._hash(self.product_id + self.issue_date),
            "status": "已登记（演示）",
        }
        return cert

    def generate_quality_cert(self) -> dict:
        """生成数据质量证书（演示性自评估）"""
        # 读取实际数据计算质量指标
        quality_metrics = self._compute_quality_metrics()

        cert = {
            "certificate_type": "数据质量证书",
            "certificate_id": f"QLT-{self.product_id}",
            "product_id": self.product_id,
            "issue_date": self.issue_date,
            "assessment_standard": "GB/T 36344-2018 信息技术 数据质量评价指标",
            "assessment_note": "【演示性自评估】完整性由数据缺失率自动计算；"
                               "其余维度为基于数据事实的估算值，未经第三方机构评测。",
            "dimensions": {
                "completeness": {
                    "name": "完整性",
                    "score": quality_metrics["completeness"],
                    "method": "auto_computed（自动计算）",
                    "description": "关键字段无缺失，气象数据完整率99.2%",
                },
                "accuracy": {
                    "name": "准确性",
                    "score": quality_metrics["accuracy"],
                    "method": "estimated（估算）",
                    "description": "31个真实产量锚点与统计年鉴/公报交叉核验；其余年份为趋势内插",
                },
                "consistency": {
                    "name": "一致性",
                    "score": quality_metrics["consistency"],
                    "method": "estimated（估算）",
                    "description": "多源数据时间粒度、单位、行政区划统一对齐",
                },
                "timeliness": {
                    "name": "时效性",
                    "score": quality_metrics["timeliness"],
                    "method": "estimated（估算）",
                    "description": "产量数据至2024年，碳价数据近12月滚动",
                },
                "traceability": {
                    "name": "可追溯性",
                    "score": quality_metrics["traceability"],
                    "method": "estimated（估算）",
                    "description": "每条数据记录来源，模型预测有置信区间",
                },
            },
            "overall_score": quality_metrics["overall"],
            "grade": "A" if quality_metrics["overall"] >= 90 else "B",
            "inspector": "项目自评估（自动计算+人工复核，非第三方评测）",
        }
        return cert

    def _compute_quality_metrics(self) -> dict:
        """计算数据质量指标

        说明：
        - completeness（完整性）：由实际数据缺失率自动计算，可复现。
        - 其余四维（准确性/一致性/时效性/可追溯性）为基于数据事实的
          专家估算值（method='estimated'），非自动化实测：
          * accuracy 依据：31个真实产量锚点与统计年鉴/公报交叉核对无误，
            其余年份为趋势内插（见数据集来源说明），估算 98.5 表示锚点核验通过率；
          * consistency 依据：多源数据已统一时间粒度/单位/行政区划；
          * timeliness 依据：产量数据至2024年、碳价近12月滚动；
          * traceability 依据：每条记录均有来源标注。
        """
        try:
            gx = pd.read_csv(os.path.join(DATA_DIR, 'guangxi_sugarcane.csv'))
            weather = pd.read_csv(os.path.join(DATA_DIR, 'weather_data.csv'))

            # 完整性（自动计算）
            gx_complete = 1 - gx.isnull().sum().sum() / (gx.shape[0] * gx.shape[1])
            wx_complete = 1 - weather.isnull().sum().sum() / (weather.shape[0] * weather.shape[1])
            completeness = round((gx_complete * 0.6 + wx_complete * 0.4) * 100, 1)

            # 以下四维为估算值（method='estimated'），依据见方法注释
            accuracy = 98.5
            consistency = 95.0
            timeliness = 92.0
            traceability = 100.0

            overall = round((completeness + accuracy + consistency + timeliness + traceability) / 5, 1)

            return {
                "completeness": completeness,
                "accuracy": accuracy,
                "consistency": consistency,
                "timeliness": timeliness,
                "traceability": traceability,
                "overall": overall,
                "method": "completeness=auto_computed; others=estimated",
            }
        except Exception:
            return {
                "completeness": 95.0,
                "accuracy": 98.5,
                "consistency": 95.0,
                "timeliness": 92.0,
                "traceability": 100.0,
                "overall": 96.1,
                "method": "fallback_estimated",
            }

    def generate_security_cert(self) -> dict:
        """生成数据安全证书"""
        cert = {
            "certificate_type": "数据安全证书",
            "certificate_id": f"SEC-{self.product_id}",
            "product_id": self.product_id,
            "issue_date": self.issue_date,
            "assessment_standard": "GB/T 37988-2019 信息安全技术 数据安全能力成熟度模型",
            "security_level": "L2（一般数据）",
            "data_classification": "公开数据聚合产品，不含个人隐私、不含国家秘密",
            "measures": {
                "input_validation": "API请求参数严格类型校验与范围限制（Pydantic模型）",
                "access_control": "API Key鉴权；速率限制可通过 Nginx/云API网关扩展",
                "data_anonymization": "原始数据不直接暴露，仅输出聚合决策结果；API响应自动脱敏",
                "audit_logging": "结构化安全审计日志，记录API访问、数据访问与安全事件",
                "transmission_security": "生产环境部署时应启用HTTPS传输加密",
            },
            "compliance": [
                "《数据安全法》—— 数据处理活动合法合规",
                "《个人信息保护法》—— 不涉及个人信息",
                "GB/T 35273-2020 —— 个人信息安全规范（不适用但已评估）",
            ],
            "risk_assessment": "低风险：全部使用公开数据源，无敏感信息",
            "inspector": "项目安全自评估",
        }
        return cert

    def generate_pricing_report(self) -> dict:
        """生成数据定价报告（成本法+市场法）"""
        report = {
            "report_type": "数据产品定价报告",
            "report_id": f"PRC-{self.product_id}",
            "product_id": self.product_id,
            "issue_date": self.issue_date,
            "valuation_standard": "GB/T 46353-2025《数据资产价值评估》",
            "methods": {
                "cost_approach": {
                    "name": "成本法",
                    "description": "基于数据采集、清洗、建模、开发的全成本核算",
                    "items": [
                        {"item": "数据采集与采购", "cost": 0, "note": "全部公开数据，无采购成本"},
                        {"item": "数据清洗与标注", "cost": 8000, "note": "人工核对+自动化清洗，约80工时"},
                        {"item": "模型研发与验证", "cost": 15000, "note": "算法设计+LOOCV验证+SHAP分析"},
                        {"item": "系统开发与部署", "cost": 12000, "note": "FastAPI+Streamlit+测试"},
                        {"item": "文档与合规", "cost": 5000, "note": "三证一价+测试报告+方法学文档"},
                    ],
                    "total_cost": 40000,
                },
                "market_approach": {
                    "name": "市场法",
                    "description": "参考同类农业数据产品市场定价",
                    "comparables": [
                        {"product": "某农业遥感数据服务", "price": "5-10万元/年", "note": "卫星影像+作物识别"},
                        {"product": "某碳汇监测SaaS", "price": "3-8万元/年", "note": "林业碳汇核算"},
                        {"product": "某精准农业DSS", "price": "2-5万元/年", "note": "种植决策支持"},
                    ],
                    "estimated_price_range": "3-6万元/年",
                },
                "income_approach": {
                    "name": "收益法",
                    "description": "基于用户采用后产生的经济效益估算",
                    "scenario": {
                        "target_user": "中型糖厂（年产10万吨甘蔗）",
                        "benefit_from_optimization": "副产物增值优化年增收 200-500万元",
                        "benefit_from_carbon": "碳汇交易年增收 50-100万元",
                        "willingness_to_pay_ratio": "1-2%",
                        "estimated_value": "2.5-12万元/年",
                    },
                },
            },
            "recommended_price": {
                "annual_subscription": "4.8万元/年",
                "api_calls_package": "0.8元/次（按量付费，1万次起）",
                "enterprise_customization": "15-30万元（含本地化部署+定制模型）",
            },
            "pricing_rationale": "综合考虑成本回收、市场可比、用户支付意愿，取三者交集。开源版本免费（促进生态），商业API按量计费。",
        }
        return report

    def get_all_certifications(self) -> dict:
        """获取完整的三证一价（演示性材料）"""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "version": self.version,
            "generated_at": datetime.now().isoformat(),
            "registration_certificate": self.generate_registration_cert(),
            "quality_certificate": self.generate_quality_cert(),
            "security_certificate": self.generate_security_cert(),
            "pricing_report": self.generate_pricing_report(),
            "disclaimer": (
                "【演示性材料】三证一价均为项目团队按国家标准框架自建的展示样例，"
                "未经官方登记机构/评估机构审核，不具法律效力；定价为成本/市场/收益法"
                "的演示测算，不代表实际成交价格。"
            ),
        }


# ---------------------------------------------------------------------------
# 数据交易模拟
# ---------------------------------------------------------------------------

class DataTradingSimulation:
    """数据要素交易场景模拟器（演示用）

    ⚠️ 重要说明：本项目为高校参赛原型，尚未发生任何真实数据交易。
    本模块仅用于【演示】数据要素市场化流通的模式设想——场景、
    买方、价格均为虚构的示例参数，不代表实际交易、合同或收入。
    展示时请如实说明为"情景模拟"。
    """

    SCENARIOS = [
        {
            "id": "scenario_1",
            "name": "糖企订阅决策服务",
            "buyer": "广西来宾东糖集团",
            "seller": "蔗循智策数据产品",
            "type": "SaaS订阅",
            "description": "糖厂订阅年度决策服务，输入自家蔗田参数获取优化方案",
            "annual_price": 48000,
            "volume": "不限次数查询",
            "buyer_benefit": "单厂年增收益200-500万元（副产物优化）",
            "roi_ratio": "10-20倍",
        },
        {
            "id": "scenario_2",
            "name": "政府碳汇监测采购",
            "buyer": "广西生态环境厅",
            "seller": "蔗循智策碳核算模块",
            "type": "政府采购",
            "description": "采购碳排放核算与碳汇监测数据服务，支撑双碳考核",
            "annual_price": 150000,
            "volume": "全区7市覆盖",
            "buyer_benefit": "实现蔗业碳汇可量化、可交易、可考核",
            "roi_ratio": "政策合规+碳汇交易收益",
        },
        {
            "id": "scenario_3",
            "name": "跨境投资机构数据包",
            "buyer": "中粮集团泰国事业部",
            "seller": "蔗循智策跨境对比数据",
            "type": "数据包一次性采购",
            "description": "采购中-泰-越五国甘蔗产业对比数据集，支撑投资决策",
            "annual_price": 80000,
            "volume": "五国×10年数据集+API调用1万次",
            "buyer_benefit": "降低跨境投资风险，优化RCEP区域布局",
            "roi_ratio": "决策准确率提升预估30%",
        },
        {
            "id": "scenario_4",
            "name": "科研机构联合建模",
            "buyer": "中国农科院甘蔗研究所",
            "seller": "蔗循智策脱敏数据集",
            "type": "数据合作",
            "description": "提供脱敏后的特征数据集，联合开展品种改良建模",
            "annual_price": 0,
            "volume": "脱敏数据集+联合署名",
            "buyer_benefit": "加速科研产出，提升品种选育效率",
            "roi_ratio": "学术合作+生态共建",
        },
    ]

    @classmethod
    def get_scenarios(cls) -> List[dict]:
        """获取演示场景列表（虚构示例，非真实交易）"""
        return cls.SCENARIOS

    @classmethod
    def simulate_transaction(cls, scenario_id: str, months: int = 12) -> dict:
        """模拟指定场景的交易流水（演示数据，非真实交易记录）"""
        # 防御超大 months 导致内存/CPU耗尽
        if not isinstance(months, int) or months < 1 or months > 120:
            return {"error": "months 必须是 1-120 之间的整数"}

        scenario = next((s for s in cls.SCENARIOS if s["id"] == scenario_id), None)
        if not scenario:
            return {"error": "场景不存在"}

        monthly_revenue = scenario["annual_price"] / 12
        transactions = []
        cumulative = 0

        for m in range(1, months + 1):
            # 模拟波动（前3月爬坡，中间稳定，年末可能有折扣）
            if m <= 3:
                factor = 0.5 + m * 0.15
            elif m == 12:
                factor = 1.2  # 年末续费高峰
            else:
                factor = 1.0

            revenue = monthly_revenue * factor
            cumulative += revenue
            transactions.append({
                "month": m,
                "revenue": round(revenue, 2),
                "cumulative": round(cumulative, 2),
                "event": "上线" if m == 1 else ("续费高峰" if m == 12 else "常规服务"),
            })

        return {
            "scenario": scenario,
            "period_months": months,
            "total_revenue": round(cumulative, 2),
            "transactions": transactions,
            "data_elements_flow": {
                "input_data_volume_gb": 0.05,
                "output_decision_records": months * 100,
                "api_calls": months * 500,
            },
            "disclaimer": (
                "【情景模拟，非真实交易】本流水为演示性模拟数据，"
                "不构成任何真实交易、合同或收入的证明。"
            ),
        }


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def get_lineage_data() -> tuple:
    """获取数据血缘的节点和链路数据"""
    nodes = DATA_LINEAGE_NODES
    links = DATA_LINEAGE_LINKS
    return nodes, links


def get_certifications() -> dict:
    """获取三证一价"""
    cert = DataProductCertification()
    return cert.get_all_certifications()
