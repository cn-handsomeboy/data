"""
数据安全核心模块

将数据安全从文档声明下沉到代码实现，提供：
1. 数据分类分级管理（自动识别+人工标注）
2. 数据脱敏（掩码/截断/泛化）
3. 数据完整性校验（SHA-256）
4. 访问审计日志（结构化记录）
5. 输入安全过滤（防注入/防越界）
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
SECURITY_LOG = os.path.join(os.path.dirname(__file__), 'logs', 'security_audit.log')
os.makedirs(os.path.dirname(SECURITY_LOG), exist_ok=True)

security_logger = logging.getLogger('sugarcane_security')
security_logger.setLevel(logging.INFO)
if not security_logger.handlers:
    fh = logging.FileHandler(SECURITY_LOG, encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    security_logger.addHandler(fh)
    # 同时输出到控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('[SECURITY] %(message)s'))
    security_logger.addHandler(ch)

# ---------------------------------------------------------------------------
# 速率限制说明（无外部依赖实现）
# ---------------------------------------------------------------------------
# 生产环境建议在反向代理或 API 网关层配置速率限制，例如：
# - Nginx: limit_req_zone / limit_req
# - Traefik: middleware rateLimit
# - 云 API 网关：按 IP / API Key 限流
# 本模块通过审计日志暴露请求元数据（IP、API Key、时间戳），便于上层限流。
# 客户端可通过响应头 X-RateLimit-* 感知限制状态（需网关填充）。

# ---------------------------------------------------------------------------
# 1. 数据分类分级枚举
# ---------------------------------------------------------------------------
class DataClassification(Enum):
    PUBLIC = ("公开数据", 1, "可完全公开，无安全限制")
    INTERNAL = ("内部数据", 2, "需授权访问，脱敏后可公开")
    CONFIDENTIAL = ("敏感数据", 3, "严格管控，禁止出境")

    def __init__(self, label_cn: str, level: int, description: str):
        self.label_cn = label_cn
        self.level = level
        self.description = description


# ---------------------------------------------------------------------------
# 2. 数据分类分级管理器
# ---------------------------------------------------------------------------
class DataClassifier:
    """
    数据分类分级管理器

    对标：GB/T 47949-2026《资产管理 数据资产分类与代码》
    """

    # 本项目数据资产注册表
    REGISTRY = {
        # 数据集名称: (分类, 脱敏策略, 来源)
        "guangxi_sugarcane.csv": (DataClassification.INTERNAL, "aggregation", "广西统计年鉴"),
        "weather_data.csv": (DataClassification.PUBLIC, "none", "中国气象数据网"),
        "fao_global.csv": (DataClassification.PUBLIC, "none", "FAOSTAT"),
        "ipcc_factors.csv": (DataClassification.PUBLIC, "none", "IPCC 2006"),
        "carbon_price.csv": (DataClassification.PUBLIC, "none", "上海环交所"),
        "byproduct_params.csv": (DataClassification.INTERNAL, "generalization", "学术文献"),
        "market_prices.csv": (DataClassification.PUBLIC, "none", "1688批发/行业研报"),
        # 2026年新增：东盟跨境与气象扩充数据集（公开来源：FAOSTAT/Open-Meteo ERA5再分析）
        "weather_data_expanded.csv": (DataClassification.PUBLIC, "none", "Open-Meteo ERA5再分析"),
        "weather_data_asean.csv": (DataClassification.PUBLIC, "none", "Open-Meteo ERA5再分析"),
        "asean_yield_weather.csv": (DataClassification.PUBLIC, "none", "FAOSTAT/Open-Meteo ERA5再分析"),
        "asean_climate_normals.json": (DataClassification.PUBLIC, "none", "Open-Meteo ERA5再分析"),
    }

    @classmethod
    def classify(cls, dataset_name: str) -> Dict[str, Any]:
        """获取数据集分类信息"""
        classification, strategy, source = cls.REGISTRY.get(
            dataset_name,
            (DataClassification.INTERNAL, "aggregation", "未知来源")
        )
        return {
            "dataset": dataset_name,
            "classification": classification.name,
            "classification_cn": classification.label_cn,
            "level": classification.level,
            "description": classification.description,
            "masking_strategy": strategy,
            "source": source,
        }

    @classmethod
    def list_all(cls) -> List[Dict[str, Any]]:
        """列出所有数据资产分类"""
        return [cls.classify(name) for name in cls.REGISTRY.keys()]

    @classmethod
    def check_cross_border_allowed(cls, dataset_name: str, target_country: str) -> bool:
        """
        检查数据是否允许跨境流通

        规则：
        - 1级（公开）：允许
        - 2级（内部）：农业非敏感数据，RCEP框架下允许
        - 3级（敏感）：禁止
        """
        info = cls.classify(dataset_name)
        level = info["level"]
        if level == 1:
            return True
        elif level == 2:
            # 农业产量、气象等非敏感数据，RCEP允许
            if target_country in ["Thailand", "Vietnam", "Myanmar", "Laos"]:
                security_logger.info(
                    f"跨境授权: {dataset_name} -> {target_country} (RCEP农业数据白名单)"
                )
                return True
            return False
        else:
            return False


# ---------------------------------------------------------------------------
# 3. 数据脱敏引擎
# ---------------------------------------------------------------------------
class DataMasker:
    """
    数据脱敏引擎

    策略：
    - none: 不脱敏
    - aggregation: 聚合级数据（市级），已天然脱敏
    - generalization: 泛化（保留范围，隐藏精确值）
    - masking: 掩码（部分隐藏）
    """

    @staticmethod
    def mask_value(value: Any, strategy: str = "masking") -> Any:
        """对单个值脱敏"""
        if pd.isna(value):
            return value

        if strategy == "none":
            return value

        elif strategy == "aggregation":
            # 聚合数据已脱敏，直接返回
            return value

        elif strategy == "generalization":
            # 数值泛化：保留数量级，隐藏精确值
            if isinstance(value, (int, float)):
                if value == 0:
                    return 0
                magnitude = 10 ** (len(str(int(abs(value)))) - 1)
                return round(value / magnitude) * magnitude
            return value

        elif strategy == "masking":
            # 字符串掩码：保留前20%和后20%
            s = str(value)
            if len(s) <= 4:
                return "*" * len(s)
            show = max(1, len(s) // 5)
            return s[:show] + "*" * (len(s) - 2 * show) + s[-show:]

        return value

    @classmethod
    def mask_dataframe(
        cls,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        strategy: str = "masking"
    ) -> pd.DataFrame:
        """对DataFrame脱敏"""
        df_masked = df.copy()
        cols = columns or df.columns.tolist()
        for col in cols:
            if col in df_masked.columns:
                df_masked[col] = df_masked[col].apply(
                    lambda x: cls.mask_value(x, strategy)
                )
        return df_masked

    @classmethod
    def _redact_value(cls, value: Any) -> Any:
        """递归脱敏单个值：字符串中的 PII + 嵌套结构遍历"""
        if isinstance(value, str):
            return InputValidator.redact_pii(value)
        if isinstance(value, dict):
            return cls._redact_nested(value)
        if isinstance(value, list):
            return [cls._redact_value(item) for item in value]
        return value

    @classmethod
    def _redact_nested(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """递归遍历字典，对字符串值脱敏 PII"""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = InputValidator.redact_pii(value)
            elif isinstance(value, dict):
                result[key] = cls._redact_nested(value)
            elif isinstance(value, list):
                result[key] = [cls._redact_value(item) for item in value]
            else:
                result[key] = value
        return result

    @classmethod
    def mask_api_response(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        对API响应脱敏（隐藏内部计算细节，递归处理嵌套列表与字符串PII）
        """
        masked = json.loads(json.dumps(data, default=str))  # 深拷贝，兼容非序列化对象

        # 隐藏敏感内部字段
        sensitive_keys = ["_internal", "debug", "raw_model_output", "training_data"]
        for key in sensitive_keys:
            if key in masked:
                masked[key] = "[REDACTED]"

        # 对输入参数进行掩码（保护用户隐私）
        if "input" in masked and isinstance(masked["input"], dict):
            user_input = masked["input"]
            # 地理位置信息部分掩码
            if "city" in user_input:
                city = user_input["city"]
                if isinstance(city, str) and len(city) > 2:
                    user_input["city"] = city[:2] + "*" * (len(city) - 2)
            # 自由文本备注中的 PII 自动脱敏
            if "notes" in user_input and isinstance(user_input["notes"], str):
                user_input["notes"] = InputValidator.redact_pii(user_input["notes"])

        # 对反馈字段中的备注进行 PII 脱敏
        if "feedback" in masked and isinstance(masked["feedback"], dict):
            fb = masked["feedback"]
            if "notes" in fb and isinstance(fb["notes"], str):
                fb["notes"] = InputValidator.redact_pii(fb["notes"])

        # 递归扫描整个响应，对任意嵌套字符串中的 PII 进行最终脱敏
        masked = cls._redact_nested(masked)

        return masked


# ---------------------------------------------------------------------------
# 4. 数据完整性校验
# ---------------------------------------------------------------------------
class DataIntegrityChecker:
    """
    数据完整性校验器

    功能：
    - 计算文件SHA-256哈希
    - 验证文件是否被篡改
    - 生成数据资产指纹
    """

    HASH_RECORD = os.path.join(os.path.dirname(__file__), 'data', '.file_hashes.json')

    @classmethod
    def compute_hash(cls, file_path: str) -> str:
        """计算文件SHA-256哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @classmethod
    def record_hashes(cls, data_dir: Optional[str] = None) -> Dict[str, str]:
        """记录所有数据文件的哈希值"""
        data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
        hashes = {}
        for fname in os.listdir(data_dir):
            if fname.endswith('.csv'):
                fpath = os.path.join(data_dir, fname)
                hashes[fname] = cls.compute_hash(fpath)

        with open(cls.HASH_RECORD, 'w', encoding='utf-8') as f:
            json.dump(hashes, f, ensure_ascii=False, indent=2)

        security_logger.info(f"已记录 {len(hashes)} 个数据文件哈希指纹")
        return hashes

    @classmethod
    def verify_integrity(cls, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        验证数据完整性

        返回：
        {
            "status": "ok" | "warning" | "error",
            "checked": 检查文件数,
            "passed": 通过数,
            "failed": 失败列表 [{"file": ..., "expected": ..., "actual": ...}],
            "missing": 缺失文件列表
        }
        """
        data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')

        if not os.path.exists(cls.HASH_RECORD):
            # 首次运行，自动记录
            cls.record_hashes(data_dir)
            return {
                "status": "ok",
                "message": "首次运行，已自动生成哈希指纹",
                "checked": 0, "passed": 0, "failed": [], "missing": []
            }

        with open(cls.HASH_RECORD, 'r', encoding='utf-8') as f:
            recorded = json.load(f)

        checked, passed, failed, missing = 0, 0, [], []

        for fname, expected_hash in recorded.items():
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                missing.append(fname)
                security_logger.warning(f"数据文件缺失: {fname}")
                continue

            checked += 1
            actual_hash = cls.compute_hash(fpath)
            if actual_hash == expected_hash:
                passed += 1
            else:
                failed.append({
                    "file": fname,
                    "expected": expected_hash[:16] + "...",
                    "actual": actual_hash[:16] + "..."
                })
                security_logger.error(f"数据文件被篡改: {fname}")

        status = "ok" if (failed == [] and missing == []) else "error"
        if missing:
            status = "error"

        return {
            "status": status,
            "checked": checked,
            "passed": passed,
            "failed": failed,
            "missing": missing,
        }


# ---------------------------------------------------------------------------
# 5. 访问审计日志
# ---------------------------------------------------------------------------
class AuditLogger:
    """
    访问审计日志器

    记录所有数据访问行为，支持：
    - API调用审计
    - 数据文件访问审计
    - 跨境数据流通审计
    """

    @staticmethod
    def log_api_access(
        endpoint: str,
        client_ip: str,
        api_key: str,
        params: Dict[str, Any],
        status_code: int,
        response_size: int,
        duration_ms: float,
        country: Optional[str] = None
    ):
        """记录API访问"""
        # 对API Key脱敏
        masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else "****"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "api_access",
            "endpoint": endpoint,
            "client_ip": client_ip,
            "api_key": masked_key,
            "params": {k: v for k, v in params.items() if k not in ["password", "secret"]},
            "status_code": status_code,
            "response_size_bytes": response_size,
            "duration_ms": round(duration_ms, 2),
            "country": country,
        }
        security_logger.info(f"API_ACCESS | {json.dumps(log_entry, ensure_ascii=False)}")

    @staticmethod
    def log_data_access(
        dataset_name: str,
        operation: str,
        user: str,
        rows_accessed: Optional[int] = None,
        cross_border: bool = False,
        target_country: Optional[str] = None
    ):
        """记录数据访问"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "data_access",
            "dataset": dataset_name,
            "operation": operation,
            "user": user,
            "rows_accessed": rows_accessed,
            "cross_border": cross_border,
            "target_country": target_country,
            "allowed": DataClassifier.check_cross_border_allowed(dataset_name, target_country or ""),
        }
        security_logger.info(f"DATA_ACCESS | {json.dumps(log_entry, ensure_ascii=False)}")

    @staticmethod
    def log_security_event(
        event_type: str,
        severity: str,
        description: str,
        details: Optional[Dict] = None
    ):
        """记录安全事件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "security",
            "type": event_type,
            "severity": severity,
            "description": description,
            "details": details or {},
        }
        if severity in ["HIGH", "CRITICAL"]:
            security_logger.error(f"SECURITY_ALERT | {json.dumps(log_entry, ensure_ascii=False)}")
        else:
            security_logger.warning(f"SECURITY_EVENT | {json.dumps(log_entry, ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# 6. 输入安全过滤
# ---------------------------------------------------------------------------
class InputValidator:
    """
    输入安全验证器

    防止：
    - SQL注入（虽然用pandas，但防范路径遍历）
    - 路径遍历攻击
    - 超大输入导致DoS
    - 恶意字符串注入
    """

    # 危险模式
    DANGEROUS_PATTERNS = [
        r"\.\./",           # 路径遍历
        r"\.\.\\",          # Windows路径遍历
        r"[<>\"'\`]|script|javascript|onerror|onload",  # XSS尝试
        r"DROP\s+|DELETE\s+|INSERT\s+|UPDATE\s+",  # SQL关键词（防范式）
    ]

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 200) -> str:
        """清理字符串输入"""
        if not isinstance(value, str):
            return str(value)[:max_length]

        # 截断超长输入
        value = value[:max_length]

        # 检测危险模式
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                security_logger.warning(f"检测到危险输入模式: {pattern}")
                # 替换危险字符
                value = re.sub(pattern, "[BLOCKED]", value, flags=re.IGNORECASE)

        return value.strip()

    @classmethod
    def validate_numeric(
        cls,
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        field_name: str = "unknown"
    ) -> float:
        """验证数值输入"""
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"字段 {field_name} 必须为数值")

        if min_val is not None and num < min_val:
            raise ValueError(f"字段 {field_name} 不能小于 {min_val}")
        if max_val is not None and num > max_val:
            raise ValueError(f"字段 {field_name} 不能大于 {max_val}")

        return num

    @classmethod
    def validate_city(cls, city: str, allowed_cities: Optional[List[str]] = None) -> str:
        """验证城市名称（防止任意路径）"""
        allowed = allowed_cities or [
            "崇左市", "来宾市", "南宁市", "柳州市",
            "百色市", "河池市", "防城港市"
        ]
        city = cls.sanitize_string(city, max_length=50)
        if city not in allowed:
            raise ValueError(f"不支持的城市: {city}")
        return city

    @classmethod
    def redact_pii(cls, text: str) -> str:
        """
        检测并脱敏常见个人身份信息（PII）

        包括：手机号、中国大陆身份证号、邮箱地址。
        用于用户反馈备注等自由文本字段的自动保护。
        """
        if not isinstance(text, str):
            text = str(text)

        # 手机号（中国大陆 11 位，1[3-9]开头）
        text = re.sub(
            r'(?<![\d])1[3-9]\d{9}(?![\d])',
            lambda m: m.group(0)[:3] + '****' + m.group(0)[-4:],
            text
        )
        # 身份证号（15或18位，简单宽松匹配）
        text = re.sub(
            r'\b\d{15}|\d{17}[\dXx]\b',
            lambda m: m.group(0)[:4] + '*' * (len(m.group(0)) - 8) + m.group(0)[-4:],
            text
        )
        # 邮箱
        text = re.sub(
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
            lambda m: m.group(0).split('@')[0][:2] + '***@' + m.group(0).split('@')[1],
            text
        )
        return text


# ---------------------------------------------------------------------------
# 7. 数据安全总控（一键检查）
# ---------------------------------------------------------------------------
class SecurityManager:
    """
    数据安全总控

    提供一键安全体检功能。
    """

    @classmethod
    def _check_api_key_strength(cls) -> Dict[str, Any]:
        """检查 API Key 是否为开发阶段随机密钥（生产环境必须替换）"""
        import os
        api_key = os.environ.get("SUGARCANE_API_KEY", "")
        # 常见弱密钥/示例密钥模式
        weak_patterns = [
            api_key.startswith("dev-"),
            api_key.startswith("change-me"),
            api_key.lower() in ["", "123456", "password", "admin"],
            len(api_key) < 16,
        ]
        is_weak = any(weak_patterns)
        return {
            "name": "API Key 强度检查",
            "status": "warning" if is_weak else "ok",
            "detail": (
                "检测到正在使用开发/弱密钥，生产环境请设置强随机字符串"
                if is_weak else "API Key 长度与格式符合基本要求"
            ),
        }

    @classmethod
    def full_security_check(cls) -> Dict[str, Any]:
        """执行完整安全体检"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # 1. 数据完整性检查
        integrity = DataIntegrityChecker.verify_integrity()
        results["checks"]["data_integrity"] = {
            "name": "数据完整性校验",
            "status": integrity["status"],
            "detail": f"通过 {integrity['passed']}/{integrity['checked']} 项",
        }

        # 2. 数据分类分级检查
        registry = DataClassifier.list_all()
        all_classified = all(r["classification"] != "UNKNOWN" for r in registry)
        results["checks"]["classification"] = {
            "name": "数据分类分级",
            "status": "ok" if all_classified else "warning",
            "detail": f"已分类 {len(registry)} 个数据集",
        }

        # 3. 日志系统检查
        log_exists = os.path.exists(SECURITY_LOG)
        results["checks"]["audit_log"] = {
            "name": "审计日志",
            "status": "ok" if log_exists else "warning",
            "detail": "审计日志正常运行" if log_exists else "审计日志未初始化",
        }

        # 4. 哈希记录检查
        hash_exists = os.path.exists(DataIntegrityChecker.HASH_RECORD)
        results["checks"]["hash_record"] = {
            "name": "哈希指纹记录",
            "status": "ok" if hash_exists else "warning",
            "detail": "已生成数据指纹" if hash_exists else "未生成数据指纹",
        }

        # 5. 跨境合规检查
        cross_border_ok = all(
            DataClassifier.check_cross_border_allowed(item["dataset"], "Thailand")
            for item in registry
        )
        results["checks"]["cross_border"] = {
            "name": "跨境数据合规",
            "status": "ok" if cross_border_ok else "warning",
            "detail": "RCEP框架下农业数据白名单机制" if cross_border_ok else "存在不允许跨境的数据集",
        }

        # 6. API Key 强度检查
        results["checks"]["api_key"] = cls._check_api_key_strength()

        # 总体评分
        ok_count = sum(1 for c in results["checks"].values() if c["status"] == "ok")
        total_count = len(results["checks"])
        results["overall_score"] = int((ok_count / total_count) * 100)
        results["overall_status"] = "secure" if results["overall_score"] >= 80 else "warning"

        return results


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
def get_security_status() -> Dict[str, Any]:
    """获取当前安全状态（供API/UI调用）"""
    return SecurityManager.full_security_check()


def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """脱敏敏感数据"""
    return DataMasker.mask_api_response(data)


def log_access(endpoint: str, **kwargs):
    """记录访问日志"""
    AuditLogger.log_api_access(endpoint=endpoint, **kwargs)


if __name__ == "__main__":
    # 自测
    print("=" * 60)
    print("数据安全模块自测")
    print("=" * 60)

    # 1. 分类分级
    print("\n[1] 数据分类分级:")
    for item in DataClassifier.list_all():
        print(f"  {item['dataset']}: {item['classification_cn']} (Lv.{item['level']})")

    # 2. 脱敏
    print("\n[2] 数据脱敏测试:")
    print(f"  原始: 崇左市 -> 脱敏: {DataMasker.mask_value('崇左市', 'masking')}")
    print(f"  原始: 123.456 -> 泛化: {DataMasker.mask_value(123.456, 'generalization')}")

    # 3. 完整性
    print("\n[3] 数据完整性:")
    result = DataIntegrityChecker.verify_integrity()
    print(f"  状态: {result['status']}, 通过: {result['passed']}/{result['checked']}")

    # 4. 安全体检
    print("\n[4] 安全体检:")
    check = SecurityManager.full_security_check()
    print(f"  总分: {check['overall_score']}/100")
    for name, detail in check["checks"].items():
        print(f"  [{detail['status'].upper()}] {detail['name']}: {detail['detail']}")

    print("\n" + "=" * 60)
