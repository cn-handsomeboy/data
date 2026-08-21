"""
用户验证闭环模块

构建虚拟但真实的用户验证案例，展示系统在不同场景下的实际应用效果。
所有数据基于真实产业参数推算，标注"模拟验证"以示区分。
"""

import json
import os
from datetime import datetime
from typing import List

# 云端持久化（GitHub 即存储）：未配置/失败时自动回退本地文件，不影响功能
try:
    from cloud_store import append_record as _cloud_append, read_json as _cloud_read
    from cloud_store import CLOUD_FEEDBACK_PATH as _CLOUD_FB
except Exception:
    def _cloud_append(*_a, **_k):
        return False

    def _cloud_read(*_a, **_k):
        return None

    _CLOUD_FB = "data/cloud_feedback.json"


# 数据目录（与 models.py 保持一致，避免循环导入）
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class UserValidationReport:
    """用户验证报告生成器"""

    # 三类典型用户验证案例
    CASES = [
        {
            "id": "UV-001",
            "user_type": "制糖企业",
            "org_name": "广西来宾东糖集团（模拟验证）",
            "org_scale": "中型糖厂，年处理甘蔗 30 万吨",
            "contact_role": "生产副总",
            "use_period": "2025/26 榨季（10个月）",
            "scenario": {
                "area_mu": 5000,
                "country": "China",
                "city": "来宾市",
                "before_scheme": "traditional",
                "adopted_scheme": "circular_advanced",
            },
            "before_after": {
                "annual_yield_tons": 29850,
                "before_net_benefit": 1290000,
                "after_net_benefit": 2870000,
                "before_carbon_tons": 5320,
                "after_carbon_tons": 1850,
                "carbon_revenue": 294000,
            },
            "satisfaction": {
                "overall": 4.6,
                "ease_of_use": 4.5,
                "result_accuracy": 4.4,
                "policy_value": 4.8,
                "cross_border_value": 3.5,  # 糖企对跨境需求低
            },
            "testimonial": (
                "系统推荐的'进阶循环'方案让我们把蔗渣从锅炉燃料转向造纸浆原料，"
                "单厂年增收益 158 万元。碳核算模块帮助我们申请了地方碳普惠试点，"
                "额外获得碳收益 29 万元。下一步考虑引入泰国甘蔗数据进行跨境对比。"
            ),
            "adoption_rate": 0.85,  # 85%的推荐方案被采纳
            "roi_months": 8,  # 投资回收期8个月
        },
        {
            "id": "UV-002",
            "user_type": "政府部门",
            "org_name": "崇左市糖业发展局（模拟验证）",
            "org_scale": "地级市行业管理部门",
            "contact_role": "产业发展科科长",
            "use_period": "2025年全年",
            "scenario": {
                "area_mu": 10000,
                "country": "China",
                "city": "崇左市",
                "before_scheme": "traditional",
                "adopted_scheme": "circular_optimal",
            },
            "before_after": {
                "annual_yield_tons": 59700,
                "before_net_benefit": 2580000,
                "after_net_benefit": 6390000,
                "before_carbon_tons": 10640,
                "after_carbon_tons": 3200,
                "carbon_revenue": 718000,
            },
            "satisfaction": {
                "overall": 4.8,
                "ease_of_use": 4.2,  # 政府用户认为专业术语偏多
                "result_accuracy": 4.7,
                "policy_value": 5.0,
                "cross_border_value": 4.5,
            },
            "testimonial": (
                "作为政府决策支撑工具，系统的价值在于'量化'——以前我们说'循环经济好'，"
                "现在可以拿出具体数字：崇左全市若推广最优循环方案，年增收益 3.8 亿元，"
                "减排 7.4 万吨 CO₂e。这份数据直接写进了我们的十四五糖业发展规划。"
            ),
            "adoption_rate": 0.60,  # 政府推广60%的示范点
            "roi_months": 0,  # 政府无需投资
        },
        {
            "id": "UV-003",
            "user_type": "跨境投资者",
            "org_name": "中粮集团泰国事业部（模拟验证）",
            "org_scale": "跨国农业投资部门",
            "contact_role": "投资分析师",
            "use_period": "2025年Q2-Q3（项目尽职调查期）",
            "scenario": {
                "area_mu": 2000,
                "country": "Thailand",
                "city": "N/A",
                "before_scheme": "traditional",
                "adopted_scheme": "circular_basic",
            },
            "before_after": {
                "annual_yield_tons": 6480,
                "before_net_benefit": 420000,
                "after_net_benefit": 798000,
                "before_carbon_tons": 2180,
                "after_carbon_tons": 1450,
                "carbon_revenue": 62000,
            },
            "satisfaction": {
                "overall": 4.4,
                "ease_of_use": 4.3,
                "result_accuracy": 4.0,  # FAO数据精度有限
                "policy_value": 3.8,
                "cross_border_value": 5.0,  # 跨境投资者最看重这个功能
            },
            "testimonial": (
                "跨境对比功能是我们使用这个系统的核心原因。"
                "通过中-泰-越五国数据协同，我们发现泰国甘蔗单产比广西低 45%，"
                "但劳动力成本也低 30%。系统帮助我们在 RCEP 框架下优化了投资组合："
                "保留泰国粗加工环节，将深加工和碳汇交易放在广西。"
            ),
            "adoption_rate": 0.70,
            "roi_months": 14,
        },
    ]

    @classmethod
    def get_cases(cls) -> List[dict]:
        return cls.CASES

    @classmethod
    def get_summary(cls) -> dict:
        """生成验证汇总统计"""
        cases = cls.CASES
        total_users = len(cases)
        avg_satisfaction = sum(c["satisfaction"]["overall"] for c in cases) / total_users
        avg_adoption = sum(c["adoption_rate"] for c in cases) / total_users
        total_before_benefit = sum(c["before_after"]["before_net_benefit"] for c in cases)
        total_after_benefit = sum(c["before_after"]["after_net_benefit"] for c in cases)
        total_carbon_reduction = sum(
            c["before_after"]["before_carbon_tons"] - c["before_after"]["after_carbon_tons"]
            for c in cases
        )
        return {
            "total_users": total_users,
            "avg_satisfaction": round(avg_satisfaction, 2),
            "avg_adoption_rate": round(avg_adoption * 100, 1),
            "total_before_benefit": total_before_benefit,
            "total_after_benefit": total_after_benefit,
            "total_increase": total_after_benefit - total_before_benefit,
            "increase_rate": round((total_after_benefit / total_before_benefit - 1) * 100, 1),
            "total_carbon_reduction_tons": total_carbon_reduction,
            "generated_at": datetime.now().isoformat(),
            "disclaimer": "以上为用户模拟验证数据，基于真实产业参数推算，用于展示系统应用效果。",
        }

    @classmethod
    def get_satisfaction_breakdown(cls) -> List[dict]:
        """获取满意度维度拆分"""
        cases = cls.CASES
        dimensions = ["overall", "ease_of_use", "result_accuracy", "policy_value", "cross_border_value"]
        dim_names = {
            "overall": "总体满意度",
            "ease_of_use": "易用性",
            "result_accuracy": "结果准确性",
            "policy_value": "政策参考价值",
            "cross_border_value": "跨境对比价值",
        }
        result = []
        for dim in dimensions:
            scores = [c["satisfaction"][dim] for c in cases]
            result.append({
                "dimension": dim_names[dim],
                "enterprise": round(scores[0], 2),
                "government": round(scores[1], 2),
                "investor": round(scores[2], 2),
                "average": round(sum(scores) / len(scores), 2),
            })
        return result


class FeedbackCollector:
    """用户反馈收集器 —— 实现预测→实际→偏差分析→模型校准的完整闭环

    支持评委/用户提交真实种植数据，系统自动对比预测值并计算偏差率，
    为模型迭代优化提供数据支撑。
    """

    FEEDBACK_FILE = os.path.join(_DATA_DIR, 'user_feedback.json')

    @classmethod
    def _load_feedbacks(cls):
        """加载已有反馈"""
        if os.path.exists(cls.FEEDBACK_FILE):
            with open(cls.FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    @classmethod
    def _save_feedbacks(cls, feedbacks):
        """保存反馈到JSON"""
        os.makedirs(os.path.dirname(cls.FEEDBACK_FILE), exist_ok=True)
        with open(cls.FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)

    @classmethod
    def submit_feedback(cls, predicted_yield, actual_yield, predicted_benefit,
                        actual_benefit, city, country, user_type="其他", notes=""):
        """提交用户实际数据，自动计算偏差"""
        from data_security import InputValidator

        # 输入安全校验
        city = InputValidator.validate_city(city)
        notes = InputValidator.sanitize_string(notes, max_length=500)
        notes = InputValidator.redact_pii(notes)
        user_type = InputValidator.sanitize_string(user_type, max_length=50)
        country = InputValidator.sanitize_string(country, max_length=50)

        # 业务合法性校验：数值范围与边界
        predicted_yield = InputValidator.validate_numeric(
            predicted_yield, min_val=0.01, max_val=50.0, field_name="预测单产"
        )
        actual_yield = InputValidator.validate_numeric(
            actual_yield, min_val=0.0, max_val=50.0, field_name="实际单产"
        )
        predicted_benefit = InputValidator.validate_numeric(
            predicted_benefit, min_val=0.01, max_val=1e9, field_name="预测净收益"
        )
        actual_benefit = InputValidator.validate_numeric(
            actual_benefit, min_val=0.0, max_val=1e9, field_name="实际净收益"
        )

        feedbacks = cls._load_feedbacks()

        fb = {
            "id": f"FB-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(feedbacks)+1:03d}",
            "timestamp": datetime.now().isoformat(),
            "city": city,
            "country": InputValidator.sanitize_string(country, max_length=50),
            "user_type": user_type,
            "predicted_yield": float(predicted_yield),
            "actual_yield": float(actual_yield),
            "yield_error_pct": round(
                (actual_yield - predicted_yield) / predicted_yield * 100, 2
            ),
            "predicted_benefit": float(predicted_benefit),
            "actual_benefit": float(actual_benefit),
            "benefit_error_pct": round(
                (actual_benefit - predicted_benefit) / predicted_benefit * 100, 2
            ),
            "notes": notes,
        }
        feedbacks.append(fb)
        cls._save_feedbacks(feedbacks)
        # 云端持久化（GitHub）：失败仅记日志，不影响本地主流程
        _cloud_append(_CLOUD_FB, fb)
        return fb

    @classmethod
    def get_feedbacks(cls):
        """获取所有反馈（本地 + 云端合并，按 id 去重，云端优先在前）"""
        merged = {}
        cloud = _cloud_read(_CLOUD_FB)
        if isinstance(cloud, list):
            for f in cloud:
                if isinstance(f, dict) and f.get("id"):
                    merged[f["id"]] = f
        for f in cls._load_feedbacks():
            if isinstance(f, dict) and f.get("id"):
                merged[f["id"]] = f
        return list(merged.values())

    @classmethod
    def get_validation_stats(cls):
        """获取验证统计（预测 vs 实际）"""
        feedbacks = cls._load_feedbacks()
        if not feedbacks:
            return {
                "total_feedbacks": 0,
                "avg_yield_error_pct": None,
                "avg_benefit_error_pct": None,
                "yield_mape": None,
                "benefit_mape": None,
                "calibration_needed": False,
            }

        yield_errors = [abs(f["yield_error_pct"]) for f in feedbacks]
        benefit_errors = [abs(f["benefit_error_pct"]) for f in feedbacks]

        return {
            "total_feedbacks": len(feedbacks),
            "avg_yield_error_pct": round(
                sum(f["yield_error_pct"] for f in feedbacks) / len(feedbacks), 2
            ),
            "avg_benefit_error_pct": round(
                sum(f["benefit_error_pct"] for f in feedbacks) / len(feedbacks), 2
            ),
            "yield_mape": round(sum(yield_errors) / len(yield_errors), 2),
            "benefit_mape": round(sum(benefit_errors) / len(benefit_errors), 2),
            "calibration_needed": (sum(yield_errors) / len(yield_errors)) > 15.0,
            "latest_feedback": feedbacks[-1],
        }
