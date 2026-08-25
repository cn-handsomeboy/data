"""
核心算法模型模块
包含：产量预测、副产物估算、碳排放核算、经济效益计算、多目标优化
"""

import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from typing import Any, Optional

# ---- 可选高级模型 ----
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

# ---------------------------------------------------------------------------
# 日志系统
# ---------------------------------------------------------------------------
logger = logging.getLogger('sugarcane_decision')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '[%(levelname)s] %(name)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 抑制sklearn已知无害警告（不影响结果，仅减少日志噪音）
# 注意：仅精确匹配已知无害警告，不使用 module 级地毯式过滤，
# 以免吞掉 ConvergenceWarning 等重要诊断信息
import warnings
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def _load_config():
    """加载 JSON 配置文件"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    logger.warning("config.json 不存在，使用默认值")
    return {}


CONFIG = _load_config()
MODEL_CFG = CONFIG.get('model', {})
OPT_CFG = CONFIG.get('optimization', {})
FALLBACK_CFG = CONFIG.get('fallback', {})
COST_CFG = CONFIG.get('processing_costs', {})
COAL_CFG = CONFIG.get('coal', {})
DIESEL_CFG = CONFIG.get('diesel_emission', {})


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _safe_get(df, column, default=0.0):
    """安全地从DataFrame中获取单个值，空结果返回默认值并打印警告"""
    if len(df) == 0:
        logger.warning("DataFrame为空，列'%s'返回默认值 %s", column, default)
        return default
    return float(df[column].values[0])


def _safe_clone(model):
    """安全克隆模型实例（用于 LOOCV 每个 fold 重新实例化）"""
    if not hasattr(model, 'get_params'):
        return model.__class__()
    params = model.get_params()
    return model.__class__(**params)


def _check_numeric(errors, name, value, low, high):
    """通用数值边界校验辅助函数（防御 NaN/None/非数值）"""
    if value is None:
        errors.append(f"{name} 不能为空")
        return
    if not isinstance(value, (int, float)):
        errors.append(f"{name} 必须为数值")
        return
    if pd.isna(value):
        errors.append(f"{name} 不能为 NaN")
        return
    if value < low or value > high:
        errors.append(f"{name} 应在 [{low}, {high}] 范围内，当前为 {value}")


def _validate_predict_inputs(avg_temp, precipitation, sunshine, city, year=None):
    """产量预测接口入参边界校验"""
    errors = []
    _check_numeric(errors, "生长季均温（℃）", avg_temp, 10.0, 45.0)
    _check_numeric(errors, "生长季累计降水（mm）", precipitation, 0.0, 5000.0)
    _check_numeric(errors, "生长季累计日照（h）", sunshine, 0.0, 3000.0)

    allowed_cities = {
        '崇左市', '来宾市', '南宁市', '柳州市',
        '百色市', '河池市', '防城港市'
    }
    if city not in allowed_cities:
        errors.append(f"不支持的城市: {city}")

    if year is not None and not isinstance(year, (int, float)):
        errors.append("年份必须为数值")

    if errors:
        raise ValueError("; ".join(errors))


def _validate_decision_inputs(area_mu, avg_temp, precipitation, sunshine,
                              fertilizer_n_kg, diesel_l, electricity_kwh,
                              carbon_price, country, city):
    """统一的决策入参边界校验（防御非法/极端输入导致荒谬输出）"""
    errors = []

    _check_numeric(errors, "种植面积（亩）", area_mu, 0.0, 100000.0)
    _check_numeric(errors, "生长季均温（℃）", avg_temp, 10.0, 45.0)
    _check_numeric(errors, "生长季累计降水（mm）", precipitation, 0.0, 5000.0)
    _check_numeric(errors, "生长季累计日照（h）", sunshine, 0.0, 3000.0)

    # 农资与能源投入按亩均校验，兼顾大小面积场景
    _check_numeric(errors, "氮肥用量（kg N）", fertilizer_n_kg,
                   0.0, 220.0 * max(area_mu, 1.0))
    _check_numeric(errors, "柴油用量（L）", diesel_l,
                   0.0, 50.0 * max(area_mu, 1.0))
    _check_numeric(errors, "电力用量（kWh）", electricity_kwh,
                   0.0, 500.0 * max(area_mu, 1.0))

    if carbon_price is not None:
        _check_numeric(errors, "碳价（元/吨）", carbon_price, 0.0, 10000.0)

    allowed_countries = {'China', 'Thailand', 'Vietnam', 'Myanmar', 'Laos'}
    if country is None or country not in allowed_countries:
        errors.append(f"不支持的国家: {country}")

    allowed_cities = {
        '崇左市', '来宾市', '南宁市', '柳州市',
        '百色市', '河池市', '防城港市'
    }
    if city is None or city not in allowed_cities:
        errors.append(f"不支持的城市: {city}")

    if errors:
        raise ValueError("; ".join(errors))


def load_data():
    """加载所有数据集（优先使用扩展后的气象数据）"""
    gx = pd.read_csv(os.path.join(DATA_DIR, 'guangxi_sugarcane.csv'))
    # 优先使用扩展气象数据（含更多变量，更多年份）
    expanded_path = os.path.join(DATA_DIR, 'weather_data_expanded.csv')
    if os.path.exists(expanded_path):
        weather = pd.read_csv(expanded_path)
        logger.info("使用扩展气象数据: %d 条, %d-%d, %d个变量",
                     len(weather), weather['year'].min(), weather['year'].max(),
                     len(weather.columns))
    else:
        weather = pd.read_csv(os.path.join(DATA_DIR, 'weather_data.csv'))
        logger.info("使用原始气象数据: %d 条", len(weather))
    fao = pd.read_csv(os.path.join(DATA_DIR, 'fao_global.csv'))
    ipcc = pd.read_csv(os.path.join(DATA_DIR, 'ipcc_factors.csv'))
    carbon = pd.read_csv(os.path.join(DATA_DIR, 'carbon_price.csv'))
    byproduct = pd.read_csv(os.path.join(DATA_DIR, 'byproduct_params.csv'))
    market = pd.read_csv(os.path.join(DATA_DIR, 'market_prices.csv'))
    return gx, weather, fao, ipcc, carbon, byproduct, market


# 各市生长季气候基准值（2010-2024年均值），用于预测时自动填充缺失特征
# 来源：Open-Meteo ERA5 再分析数据
CITY_CLIMATE_NORMALS = {
    '崇左市': {'humidity_pct': 78.3, 'pressure_hpa': 1005.2, 'wind_speed_ms': 1.89,
              'evapotranspiration_mm': 745.5, 'soil_temp_c': 25.8, 'soil_moisture_m3m3': 0.328},
    '来宾市': {'humidity_pct': 76.8, 'pressure_hpa': 1002.7, 'wind_speed_ms': 2.05,
              'evapotranspiration_mm': 765.2, 'soil_temp_c': 25.3, 'soil_moisture_m3m3': 0.335},
    '南宁市': {'humidity_pct': 80.1, 'pressure_hpa': 1004.5, 'wind_speed_ms': 1.78,
              'evapotranspiration_mm': 720.8, 'soil_temp_c': 26.1, 'soil_moisture_m3m3': 0.342},
    '柳州市': {'humidity_pct': 77.5, 'pressure_hpa': 1000.8, 'wind_speed_ms': 2.12,
              'evapotranspiration_mm': 738.6, 'soil_temp_c': 24.9, 'soil_moisture_m3m3': 0.330},
    '百色市': {'humidity_pct': 75.6, 'pressure_hpa': 1003.9, 'wind_speed_ms': 1.95,
              'evapotranspiration_mm': 782.3, 'soil_temp_c': 26.4, 'soil_moisture_m3m3': 0.318},
    '河池市': {'humidity_pct': 78.9, 'pressure_hpa': 1001.2, 'wind_speed_ms': 1.82,
              'evapotranspiration_mm': 710.4, 'soil_temp_c': 24.6, 'soil_moisture_m3m3': 0.340},
    '防城港市': {'humidity_pct': 82.3, 'pressure_hpa': 1006.8, 'wind_speed_ms': 2.45,
               'evapotranspiration_mm': 695.1, 'soil_temp_c': 26.7, 'soil_moisture_m3m3': 0.355},
}


# ---------------------------------------------------------------------------
# 模型热加载/预加载辅助（减少 API 冷启动时间）
# ---------------------------------------------------------------------------
_warmed_system: Optional[Any] = None


def warm_start_models(force_retrain: bool = False) -> Any:
    """预加载/热启动决策系统，减少 API 首次调用延迟

    优先加载已保存的模型文件；加载失败或强制重训练时重新训练。
    返回预热的系统实例，可被 API 层复用。
    """
    global _warmed_system
    if _warmed_system is not None and not force_retrain:
        return _warmed_system

    system = SugarcaneDecisionSystem()
    loaded = False
    if not force_retrain:
        try:
            loaded = system.yield_predictor.load_model()
            if loaded:
                logger.info("模型热加载成功: %s",
                            system.yield_predictor.metrics.get('model_name', 'unknown'))
        except Exception as e:
            logger.warning("模型热加载失败: %s，将重新训练", e)

    if not loaded:
        try:
            gx, weather, _, _, _, _, _ = load_data()
            system.yield_predictor.train(gx, weather, model_type='auto')
            logger.info("模型热训练完成: %s",
                        system.yield_predictor.metrics.get('model_name', 'unknown'))
        except Exception as e:
            logger.error("模型热启动失败: %s", e)
            raise

    _warmed_system = system
    return system


def get_default_carbon_price(country='China'):
    """从历史碳价数据计算智能默认值（近12个月均价）

    中国：上海环境能源交易所全国碳市场CEA收盘价。
    东盟国家：使用 market_prices.csv 中对应国家的碳信用估算均价。
    若数据不可用，返回配置中的 fallback 值。
    """
    # 东盟国家：从市场数据读取碳信用估算价格
    if country != 'China':
        try:
            market_path = os.path.join(DATA_DIR, 'market_prices.csv')
            if os.path.exists(market_path):
                market_df = pd.read_csv(market_path)
                row = market_df[
                    (market_df['country'] == country) &
                    (market_df['product_name'] == 'carbon_credit')
                ]
                if len(row) > 0:
                    price = float(row['price_avg_yuan_per_ton'].values[0])
                    logger.info("%s 默认碳信用价格: %.2f 元/吨", country, price)
                    return round(price, 2)
        except Exception as e:
            logger.warning("读取 %s 碳信用价格失败: %s，使用 fallback", country, e)
        # 东盟兜底
        return CONFIG.get('fallback', {}).get('carbon_price_asean', 40.0)

    carbon_path = os.path.join(DATA_DIR, 'carbon_price.csv')
    if not os.path.exists(carbon_path):
        return CONFIG.get('fallback', {}).get('carbon_price', 85.0)

    try:
        carbon_df = pd.read_csv(carbon_path)
        carbon_df['date'] = pd.to_datetime(carbon_df['date'])
        recent = carbon_df[
            carbon_df['date'] >= carbon_df['date'].max() - pd.DateOffset(months=12)
        ]
        if len(recent) > 0:
            avg_price = float(recent['close_price'].mean())
            logger.info("近12个月中国碳价均价: %.2f 元/吨 (样本数: %d)",
                        avg_price, len(recent))
            return round(avg_price, 2)
    except Exception as e:
        logger.warning("碳价数据读取失败: %s，使用默认值", e)

    return 85.0


# ===================================================================
# 模型类
# ===================================================================

class YieldPredictor:
    """甘蔗产量预测模型（Ridge回归 + LOOCV交叉验证 + 城市哑变量）

    特征设计（8个）：
    - 气象变量：avg_temp_c, precipitation_mm, sunshine_hours（气候波动，用户输入）
    - 城市哑变量：city_来宾市, city_南宁市, city_柳州市（崇左市为基准）
      捕获各市的土壤类型、品种结构、管理水平等系统性差异
    - 年份趋势：year（捕获品种改良、技术进步带来的产量趋势）
    - 城市种植面积：planting_area_wan_mu（全市甘蔗总面积，万亩）
      城市级上下文变量，反映产区规模与产业集中度，由系统根据城市自动查表填写

    小样本场景使用 Ridge 回归 + Leave-One-Out 交叉验证。
    """

    # 城市列表（用于 one-hot 编码，崇左市为基准/参考类别）
    CITY_DUMMIES = ['来宾市', '南宁市', '柳州市', '百色市', '河池市', '防城港市']

    # 各市历史种植面积（万亩），用于预测时自动填充城市上下文
    CITY_AREA_WAN_MU = {
        '崇左市': 417.0, '来宾市': 179.0,
        '南宁市': 195.0, '柳州市': 99.0,
        '百色市': 86.0, '河池市': 91.0,
        '防城港市': 63.0,
    }

    def __init__(self):
        self.model = None
        self.scaler = None       # StandardScaler
        self._X_train = None     # 训练特征（bootstrap用）
        self._y_train = None     # 训练标签（bootstrap用）
        self.fallback_yield = FALLBACK_CFG.get('default_yield_per_mu_tons', 6.0)
        # 基础气象特征（用户输入3个 + 扩展6个自动填充）
        self.features = ['avg_temp_c', 'precipitation_mm', 'sunshine_hours',
                         'humidity_pct', 'pressure_hpa', 'wind_speed_ms',
                         'evapotranspiration_mm', 'soil_temp_c', 'soil_moisture_m3m3']
        self.features += [f'city_{c}' for c in self.CITY_DUMMIES]
        self.features += ['year', 'planting_area_wan_mu']
        self.target = 'yield_per_mu_tons'
        self._trained = False
        self._train_metrics = None
        self.model_comparison = None
        self.shap_explainer = None   # SHAP解释器
        self.shap_values_train = None  # 训练集SHAP值
        self.shap_summary = None     # SHAP汇总统计

    def _train_evaluate_model(self, model, X, y, model_name):
        """使用固定保守超参 + LOOCV 训练并评估单个模型。

        设计说明：小样本（70 样本）场景下不做超参数搜索（GridSearchCV），
        原因有二：
        (1) 超参搜索与 LOOCV 共用同一份数据会造成信息泄漏，使 R² 被系统性
            高估（GridSearchCV 在全部样本上选参后再用 LOOCV 评估属典型泄漏）；
        (2) 小样本下超参搜索本身方差极大，收益有限，用保守默认超参反而更稳健。
        因此采用预先设定的保守超参，保证 LOOCV 评估结果的无偏性。
        """
        best_model = model

        # ---- LOOCV 评估（无超参搜索，无信息泄漏）----
        # 注意：标准化参数必须每折用训练子集重新拟合，否则验证样本会参与
        # 均值/标准差的估计，导致与超参搜索泄漏同类型的"标准化泄漏"。
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X):
            X_tr = X.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            y_te = y.iloc[test_idx]

            # 每折在训练子集上独立标准化，验证子集只 transform（不参与 fit）
            fold_scaler = StandardScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_tr)
            X_te_scaled = fold_scaler.transform(X_te)

            # 每个 fold 用相同的固定超参重新训练
            fold_model = _safe_clone(best_model)
            fold_model.fit(X_tr_scaled, y_tr)
            y_pred.append(fold_model.predict(X_te_scaled)[0])
            y_true.append(y_te.values[0])

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 最终在全量数据上训练（供线上预测使用）。
        # 此处标准化用 train() 中已就绪的 self.scaler（全量拟合，
        # 与预测阶段同源，属正常操作，无泄漏问题）。
        X_global = self.scaler.transform(X)
        best_model.fit(X_global, y)

        return {
            'model_name': model_name,
            'model': best_model,
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'loocv_samples': len(y_true),
        }

    def train(self, gx_df, weather_df, model_type='ridge'):
        """训练产量预测模型（Ridge回归 + LOOCV）。

        使用城市-年份细粒度数据训练，Leave-One-Out 交叉验证评估。
        支持多模型对比和自动选择最优模型。

        气象数据使用甘蔗生长季（5-10月）聚合，并按城市匹配。
        文献依据：甘蔗分蘖-伸长期（5-9月）和成熟初期（10月）
        的水热条件是决定单产的关键期。

        Args:
            gx_df: 广西甘蔗产量数据
            weather_df: 气象数据（含 city 列用于城市匹配）
            model_type: 'ridge'（默认）| 'rf' | 'auto'
        """
        # ================================================================
        # 甘蔗生长季（5-10月）气象聚合
        # 理由：广西甘蔗一般在2-3月下种，5月进入分蘖-伸长期，
        # 5-9月是水热需求最大的关键期，10月进入成熟初期。
        # 用生长季而非全年的聚合值能更精准地捕获气象对产量的边际效应。
        # ================================================================
        gs = weather_df[weather_df['month'].between(5, 10)].copy()

        # 聚合所有气象变量：温度/降水/日照取合计，其余取均值
        agg_dict = {
            'avg_temp_c': 'mean',
            'precipitation_mm': 'sum',
            'sunshine_hours': 'sum',
        }
        # 新增扩展变量（生长季均值，天然连续变量）
        for col in ['humidity_pct', 'pressure_hpa', 'wind_speed_ms',
                     'evapotranspiration_mm', 'soil_temp_c', 'soil_moisture_m3m3']:
            if col in gs.columns:
                agg_dict[col] = 'mean'
        weather_yearly = gs.groupby(['year', 'city']).agg(agg_dict).reset_index()

        # 按年份+城市匹配（每个城市使用自己的气象站数据）
        merged = pd.merge(gx_df, weather_yearly, on=['year', 'city'], how='inner')
        logger.info("合并后训练样本数: %d（城市-年份细粒度, 生长季气象）", len(merged))

        # 城市 one-hot 编码（崇左市为基准/参考类别，不生成哑变量列）
        for c in self.CITY_DUMMIES:
            merged[f'city_{c}'] = (merged['city'] == c).astype(int)

        # 年份直接作为特征（捕获技术进步趋势）
        merged['year'] = merged['year'].astype(float)

        # 种植面积：CSV中planting_area_mu已是万亩口径，直接映射
        merged['planting_area_wan_mu'] = merged['planting_area_mu']

        # ================================================================
        # 交互特征：气象因子之间的非线性耦合
        # 理由：高温+高湿（湿热胁迫）、高温+强日照（蒸腾加速），
        # 这些交互效应在单因子线性模型中无法捕获。
        # ================================================================
        for feat_a, feat_b in [('avg_temp_c', 'precipitation_mm'),
                                ('avg_temp_c', 'sunshine_hours'),
                                ('precipitation_mm', 'sunshine_hours')]:
            merged[f'{feat_a}_x_{feat_b}'] = merged[feat_a] * merged[feat_b]

        # 更新特征列表
        interaction_features = ['avg_temp_c_x_precipitation_mm',
                                'avg_temp_c_x_sunshine_hours',
                                'precipitation_mm_x_sunshine_hours']
        # 防御性过滤：仅保留合并后确实存在的特征列。
        # 数据源含扩展气象变量（9 维）时全量进入；若退化为仅温/湿/日照的旧数据，
        # 自动剔除缺失列，避免 DataFrame 选列 KeyError，同时保证训练-预测口径一致。
        all_features = [
            f for f in (self.features + interaction_features)
            if f in merged.columns
        ]

        min_samples = MODEL_CFG.get('min_samples_for_training', 10)
        if len(merged) < min_samples:
            self.fallback_yield = float(merged[self.target].mean())
            self._trained = False
            self._train_metrics = {'mse': 0.0, 'r2': float('nan'), 'fallback': True}
            logger.warning(
                "样本量不足 (%d < %d)，使用历史均值 %.2f 吨/亩作为fallback",
                len(merged), min_samples, self.fallback_yield
            )
            return self._train_metrics

        X = merged[all_features].copy()
        y = merged[self.target].copy()

        # 存储训练数据的产量范围（用于预测约束，避免极端输入导致荒谬预测）
        # 采用 2.5%~97.5% 分位数而非 min/max，避免单个离群值影响约束边界；
        # 同时保留 min/max 作为兜底（样本过少时分位数退化）。
        self._train_yield_min = float(y.min())
        self._train_yield_max = float(y.max())
        self._train_yield_q025 = float(np.percentile(y, 2.5))
        self._train_yield_q975 = float(np.percentile(y, 97.5))

        # ---- 特征标准化 ----
        self.scaler = StandardScaler()
        X_scaled_arr = self.scaler.fit_transform(X)

        self.active_features = all_features
        X_final = pd.DataFrame(X_scaled_arr, columns=all_features, index=X.index)

        # 记录扩展气候变量的训练均值，供 predict 时填充缺失输入。
        # 原因：CITY_CLIMATE_NORMALS 中个别变量单位与训练聚合口径不一致
        # （如 evapotranspiration 年累计 vs 生长季月均值），直接用会造成
        # 预测特征离群、模型外推爆炸。改用训练均值可保证预测-训练口径一致。
        self._ext_feature_means = {
            col: float(merged[col].mean())
            for col in ['humidity_pct', 'pressure_hpa', 'wind_speed_ms',
                        'evapotranspiration_mm', 'soil_temp_c', 'soil_moisture_m3m3']
            if col in merged.columns
        }

        # 存储训练数据（bootstrap用）
        self._X_train = X_final.copy()
        self._y_train = y.copy()
        # 存储训练年份范围：predict 时 year=None 默认取训练集最近年份，
        # 避免超出训练区间的年份线性外推导致荒谬预测（Ridge 年份特征外推风险）
        self._train_year_min = int(merged['year'].min())
        self._train_year_max = int(merged['year'].max())
        self._city_area_from_training = {}
        for c in merged['city'].unique():
            recent = merged[merged['city'] == c].sort_values('year').tail(3)
            self._city_area_from_training[c] = float(recent['planting_area_wan_mu'].mean())

        # ---- 候选模型（小样本下精简为2个，覆盖"可解释"与"精度"两类）----
        # ridge：线性可解释基线；gbrt：非线性精度模型（固定保守超参，避免过拟合）
        candidates = {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'gbrt': GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.05,
                min_samples_leaf=2, random_state=42
            ),
        }

        if model_type == 'auto':
            results = []
            for name, m in candidates.items():
                result = self._train_evaluate_model(m, X.copy(), y.copy(), name)
                results.append(result)

            results.sort(key=lambda r: r['r2'], reverse=True)
            best = results[0]
            self.model = best['model']
            self._trained = True
            self._train_metrics = {
                'mse': best['mse'], 'rmse': best['rmse'], 'mae': best['mae'],
                'r2': best['r2'], 'fallback': False,
                'model_name': best['model_name'],
                'loocv_samples': best['loocv_samples'],
            }
            self.model_comparison = {
                r['model_name']: {
                    'r2': r['r2'], 'rmse': r['rmse'], 'mae': r['mae']
                } for r in results
            }
            logger.info(
                "模型对比完成，最优模型: %s (R²=%.4f)",
                best['model_name'], best['r2']
            )
        else:
            model = candidates.get(model_type, candidates['ridge'])
            result = self._train_evaluate_model(model, X.copy(), y.copy(), model_type)
            self.model = result['model']
            self._trained = True
            self._train_metrics = {
                'mse': result['mse'], 'rmse': result['rmse'],
                'mae': result['mae'], 'r2': result['r2'],
                'fallback': False, 'model_name': result['model_name'],
                'loocv_samples': result['loocv_samples'],
            }
            logger.info(
                "产量预测模型训练完成 - %s, R²=%.4f, RMSE=%.4f",
                model_type, result['r2'], result['rmse']
            )

        # ---- 特征重要性 ----
        self.feature_importance = {}
        try:
            # 使用RF模型自带的特征重要性（如果选了RF）
            rf_model = None
            if hasattr(self.model, 'feature_importances_'):
                rf_model = self.model
            else:
                # 否则训练一个RF用于特征重要性分析
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_model.fit(X_final, y)

            importances = rf_model.feature_importances_
            # 只展示top 15（多项式特征可能很多）
            top_indices = np.argsort(importances)[-15:]
            for i in top_indices:
                feat = self.active_features[i]
                self.feature_importance[feat] = {
                    'mean': float(importances[i]),
                    'std': 0.0
                }
            top3 = sorted(self.feature_importance.items(),
                          key=lambda x: -x[1]['mean'])[:3]
            logger.info("特征重要性 top3: %s",
                        [(f, round(v['mean'], 4)) for f, v in top3])
        except Exception as e:
            logger.warning("特征重要性计算失败: %s", e)

        # ---- SHAP 可解释性分析 ----
        if HAS_SHAP and self.model is not None:
            try:
                self._build_shap_explainer(X_final, y)
            except Exception as e:
                logger.warning("SHAP分析失败: %s", e)

        # 保存模型（权限失败时优雅降级，不影响运行）
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, 'yield_predictor.pkl')
            payload = {
                'model': self.model,
                'scaler': self.scaler,
                'fallback_yield': self.fallback_yield,
                'trained': self._trained,
                'metrics': self._train_metrics,
                'model_comparison': self.model_comparison,
                'active_features': getattr(self, 'active_features', self.features),
                'city_area_from_training': getattr(self, '_city_area_from_training', {}),
                'feature_importance': getattr(self, 'feature_importance', None),
                'shap_summary': getattr(self, 'shap_summary', None),
                # 保存训练数据，用于加载后重建 SHAP / Bootstrap CI
                '_X_train': getattr(self, '_X_train', None),
                '_y_train': getattr(self, '_y_train', None),
                '_train_yield_min': getattr(self, '_train_yield_min', None),
                '_train_yield_max': getattr(self, '_train_yield_max', None),
                '_train_yield_q025': getattr(self, '_train_yield_q025', None),
                '_train_yield_q975': getattr(self, '_train_yield_q975', None),
                '_train_year_min': getattr(self, '_train_year_min', None),
                '_train_year_max': getattr(self, '_train_year_max', None),
                '_ext_feature_means': getattr(self, '_ext_feature_means', {}),
                # 安全元数据：依赖版本与序列化格式版本
                '_deps_version': {
                    'sklearn': sklearn.__version__,
                    'pandas': pd.__version__,
                    'numpy': np.__version__,
                },
                '_model_format_version': 1,
            }
            with open(model_path, 'wb') as f:
                pickle.dump(payload, f)
            # 生成模型文件哈希指纹，防止篡改
            hash_path = os.path.join(MODELS_DIR, 'yield_predictor.hash')
            with open(hash_path, 'w', encoding='utf-8') as f:
                f.write(self._compute_model_hash(model_path))
            logger.info("模型已保存并生成哈希指纹: %s", hash_path)
        except PermissionError as e:
            logger.warning("模型保存权限被拒绝: %s。本次运行使用内存模型，不影响决策。", e)
        except Exception as e:
            logger.warning("模型保存失败: %s。本次运行使用内存模型。", e)

        return self._train_metrics

    @staticmethod
    def _compute_model_hash(file_path: str) -> str:
        """计算模型文件 SHA-256 哈希"""
        import hashlib
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def load_model(self):
        """加载已训练模型（兼容新旧两种格式），带完整性校验"""
        model_path = os.path.join(MODELS_DIR, 'yield_predictor.pkl')
        if not os.path.exists(model_path):
            return False

        # 完整性校验：若存在哈希记录则比对
        hash_path = os.path.join(MODELS_DIR, 'yield_predictor.hash')
        if os.path.exists(hash_path):
            with open(hash_path, 'r', encoding='utf-8') as f:
                expected_hash = f.read().strip()
            actual_hash = self._compute_model_hash(model_path)
            if expected_hash != actual_hash:
                logger.error(
                    "模型文件完整性校验失败：可能被篡改。expected=%s... actual=%s...",
                    expected_hash[:16], actual_hash[:16]
                )
                return False
            logger.info("模型文件完整性校验通过")

        try:
            with open(model_path, 'rb') as f:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    data = pickle.load(f)  # 已做完整性校验；后续建议迁移到 joblib/ONNX
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError, ImportError) as e:
            logger.warning(
                "模型文件损坏或依赖缺失/不兼容，无法加载: %s。将重新训练模型。", e
            )
            return False
        except Exception as e:
            logger.warning("模型文件读取失败: %s。将重新训练模型。", e)
            return False

        # 兼容旧格式（直接存储sklearn模型对象）
        if hasattr(data, 'predict'):
            self.model = data
            self._trained = True
            logger.info("模型已加载（旧格式）")
            return True

        # 新格式（dict）
        if isinstance(data, dict):
            self.model = data.get('model')
            self.scaler = data.get('scaler')
            self.fallback_yield = data.get(
                'fallback_yield', self.fallback_yield)
            self._trained = data.get('trained', self.model is not None)
            if 'active_features' in data:
                self.active_features = data['active_features']
            if 'metrics' in data:
                self._train_metrics = data['metrics']
            if 'model_comparison' in data:
                self.model_comparison = data['model_comparison']
            if 'city_area_from_training' in data:
                self._city_area_from_training = data['city_area_from_training']
            if 'feature_importance' in data:
                self.feature_importance = data['feature_importance']
            if 'shap_summary' in data:
                self.shap_summary = data['shap_summary']
            # 恢复训练数据（用于 Bootstrap CI 和 SHAP 重建）
            if '_X_train' in data and data['_X_train'] is not None:
                self._X_train = data['_X_train']
            if '_y_train' in data and data['_y_train'] is not None:
                self._y_train = data['_y_train']
            if '_train_yield_min' in data and data['_train_yield_min'] is not None:
                self._train_yield_min = data['_train_yield_min']
            if '_train_yield_max' in data and data['_train_yield_max'] is not None:
                self._train_yield_max = data['_train_yield_max']
            if '_train_yield_q025' in data and data['_train_yield_q025'] is not None:
                self._train_yield_q025 = data['_train_yield_q025']
            if '_train_yield_q975' in data and data['_train_yield_q975'] is not None:
                self._train_yield_q975 = data['_train_yield_q975']
            if '_train_year_min' in data and data['_train_year_min'] is not None:
                self._train_year_min = data['_train_year_min']
            if '_train_year_max' in data and data['_train_year_max'] is not None:
                self._train_year_max = data['_train_year_max']
            if '_ext_feature_means' in data and data['_ext_feature_means']:
                self._ext_feature_means = data['_ext_feature_means']
            # 加载后重建 SHAP explainer（需要训练数据）
            if HAS_SHAP and self.shap_summary is not None and self._X_train is not None:
                try:
                    self._build_shap_explainer(self._X_train, self._y_train)
                    logger.info("SHAP explainer 已重建")
                except Exception as e:
                    logger.warning("加载后重建 SHAP explainer 失败: %s", e)
            # 依赖版本校验
            saved_deps = data.get('_deps_version')
            if saved_deps:
                current_deps = {
                    'sklearn': sklearn.__version__,
                    'pandas': pd.__version__,
                    'numpy': np.__version__,
                }
                mismatches = [
                    f"{k}: saved={saved_deps.get(k)} current={current_deps[k]}"
                    for k in current_deps
                    if saved_deps.get(k) != current_deps[k]
                ]
                if mismatches:
                    logger.warning(
                        "模型依赖版本不一致，可能影响预测结果: %s",
                        "; ".join(mismatches)
                    )

            model_name = self._train_metrics.get('model_name', 'unknown') if self._train_metrics else 'unknown'
            logger.info("模型已加载: %s, fallback_yield=%.2f", model_name, self.fallback_yield)
            return True

        logger.warning("未知的模型文件格式")
        return False

    def _build_features_row(self, avg_temp, precipitation, sunshine, city, year=None):
        """构建预测特征行（DRY原则：predict()和predict_with_ci()共用）

        Args:
            avg_temp: 生长季均温（℃）
            precipitation: 生长季累计降水（mm）
            sunshine: 生长季累计日照时数（h）
            city: 城市名称
            year: 预测年份，None时使用当前年份
        Returns:
            dict: 特征字典
        """
        from datetime import datetime
        row = {
            'avg_temp_c': avg_temp,
            'precipitation_mm': precipitation,
            'sunshine_hours': sunshine,
        }
        for c in self.CITY_DUMMIES:
            row[f'city_{c}'] = 1 if city == c else 0
        row['year'] = float(year) if year is not None else float(
            getattr(self, '_train_year_max', datetime.now().year)
        )
        if hasattr(self, '_city_area_from_training') and city in self._city_area_from_training:
            row['planting_area_wan_mu'] = self._city_area_from_training[city]
        else:
            row['planting_area_wan_mu'] = self.CITY_AREA_WAN_MU.get(city, 200.0)
        # 交互特征
        row['avg_temp_c_x_precipitation_mm'] = row['avg_temp_c'] * row['precipitation_mm']
        row['avg_temp_c_x_sunshine_hours'] = row['avg_temp_c'] * row['sunshine_hours']
        row['precipitation_mm_x_sunshine_hours'] = row['precipitation_mm'] * row['sunshine_hours']
        # 扩展气象变量：用户仅输入温度/降水/日照，其余 6 个变量按训练均值自动填充。
        # 训练时这些列已进入特征集，预测时必须补齐，否则 DataFrame 选列会 KeyError。
        # 优先用训练均值（口径与训练一致），无训练均值时回退到该市气候基准值。
        ext_means = getattr(self, '_ext_feature_means', {})
        normals = CITY_CLIMATE_NORMALS.get(city, {})
        for feat in ['humidity_pct', 'pressure_hpa', 'wind_speed_ms',
                     'evapotranspiration_mm', 'soil_temp_c', 'soil_moisture_m3m3']:
            if feat in ext_means:
                row[feat] = ext_means[feat]
            else:
                row[feat] = normals.get(feat, 0.0)
        return row

    def predict(self, avg_temp, precipitation, sunshine, city='崇左市',
                year=None):
        """预测单产（生长季气象+城市哑变量+年份+城市面积上下文，模型不可用时使用fallback）

        用户输入：avg_temp, precipitation, sunshine, city
        （注意：此处输入应为生长季5-10月聚合值，与训练时口径一致）
        系统自动填充：year（当前年份）、planting_area_wan_mu（城市历史值）

        Args:
            avg_temp: 生长季均温（℃）
            precipitation: 生长季累计降水（mm）
            sunshine: 生长季累计日照时数（h）
            city: 城市名称（崇左市/来宾市/南宁市/柳州市/百色市/河池市/防城港市）
            year: 预测年份，None时使用当前年份

        Returns:
            float: 预测单产（吨/亩）
        """
        # 入参边界校验（直接调用预测接口时的防御）
        _validate_predict_inputs(avg_temp, precipitation, sunshine, city, year)

        # 尝试加载模型
        if not self._trained and self.model is None:
            if not self.load_model():
                logger.info("模型未训练，使用fallback值: %.2f 吨/亩",
                            self.fallback_yield)
                return self.fallback_yield

        if self.model is None:
            logger.info("模型为空，使用fallback值: %.2f 吨/亩",
                        self.fallback_yield)
            return self.fallback_yield

        # 构建特征向量（使用共享方法）
        row = self._build_features_row(avg_temp, precipitation, sunshine, city, year)

        # 使用训练时的特征列表
        feature_cols = getattr(self, 'active_features', self.features)
        X = pd.DataFrame([row])[feature_cols]

        # 标准化
        if self.scaler is not None:
            X_arr = self.scaler.transform(X)
        else:
            X_arr = X.values

        predicted = float(self.model.predict(X_arr)[0])

        # 保存原始预测值（约束前），用于置信区间
        self._last_raw_prediction = predicted

        # 约束预测值在训练数据的分位数范围内，防止极端外推导致荒谬结果。
        # 说明：这是"工程外推防护"而非统计推断，仅用于拦截明显不合理的外推值；
        # 约束后的值可能与 LOOCV 评估口径（未约束）存在差异，属预期行为。
        # 优先用 2.5%~97.5% 分位数（更稳健），无分位数时回退到 min/max。
        lo = getattr(self, '_train_yield_q025', getattr(self, '_train_yield_min', 3.87))
        hi = getattr(self, '_train_yield_q975', getattr(self, '_train_yield_max', 6.74))
        if predicted < lo or predicted > hi:
            logger.info("预测值 %.2f 超出约束范围 [%.2f, %.2f]，已约束至边界值",
                        predicted, lo, hi)
            predicted = max(lo, min(hi, predicted))
        return predicted

    def predict_with_ci(self, avg_temp, precipitation, sunshine,
                        city='崇左市', year=None, n_bootstrap=200):
        """预测单产并返回 Bootstrap 95% 置信区间

        Args:
            n_bootstrap: bootstrap 重采样次数

        Returns:
            {'point': float, 'ci_lower': float, 'ci_upper': float}
        """
        # 入参边界校验（predict 内部也会校验，此处显式调用增强可读性）
        _validate_predict_inputs(avg_temp, precipitation, sunshine, city, year)
        point = self.predict(avg_temp, precipitation, sunshine, city, year)

        # 如果没有训练数据，fallback到RMSE-based CI
        if self._X_train is None or self._y_train is None or self.model is None:
            if self._train_metrics and not self._train_metrics.get('fallback'):
                sigma = self._train_metrics.get('rmse', 0.3)
                return {
                    'point': point,
                    'ci_lower': point - 1.96 * sigma,
                    'ci_upper': point + 1.96 * sigma,
                    'method': 'rmse_based'
                }
            return {'point': point, 'ci_lower': point, 'ci_upper': point,
                    'method': 'none'}

        # Bootstrap
        n = len(self._X_train)
        predictions = []
        rng = np.random.RandomState(42)

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_boot = self._X_train.iloc[idx]
            y_boot = self._y_train.iloc[idx]

            try:
                m = _safe_clone(self.model)
                m.fit(X_boot, y_boot)

                # 构建预测特征（使用共享方法）
                row = self._build_features_row(avg_temp, precipitation, sunshine, city, year)

                feature_cols = getattr(self, 'active_features', self.features)
                X_pred = pd.DataFrame([row])[feature_cols]

                if self.scaler is not None:
                    X_arr = self.scaler.transform(X_pred)
                else:
                    X_arr = X_pred.values

                pred = float(m.predict(X_arr)[0])
                # 与 predict() 保持一致的工程约束：bootstrap 预测也应用
                # 训练数据分位数边界，避免离群 bootstrap 预测拉宽 CI 到荒谬区间
                lo_c = getattr(self, '_train_yield_q025',
                               getattr(self, '_train_yield_min', 3.0))
                hi_c = getattr(self, '_train_yield_q975',
                               getattr(self, '_train_yield_max', 7.0))
                pred = max(lo_c, min(hi_c, pred))
                predictions.append(pred)
            except Exception:
                continue

        if len(predictions) < 50:
            return {'point': point, 'ci_lower': point, 'ci_upper': point,
                    'method': 'insufficient_bootstrap'}

        predictions = np.array(predictions)
        ci_lower = float(np.percentile(predictions, 2.5))
        ci_upper = float(np.percentile(predictions, 97.5))

        # 退化检测:若 bootstrap 预测被分位数护栏压缩为单点(下界==上界),
        # 区间已无信息量,回退到 RMSE-based 近似并如实标注,避免输出
        # 名义上为 bootstrap 实则退化的假区间(曾出现 lower==upper 的情况)。
        if ci_lower >= ci_upper:
            sigma = 0.3
            if self._train_metrics and not self._train_metrics.get('fallback'):
                sigma = self._train_metrics.get('rmse', 0.3)
            return {
                'point': point,
                'ci_lower': point - 1.96 * sigma,
                'ci_upper': point + 1.96 * sigma,
                'bootstrap_mean': float(np.mean(predictions)),
                'bootstrap_std': float(np.std(predictions)),
                'n_bootstrap': len(predictions),
                'method': 'rmse_fallback_degraded'
            }

        return {
            'point': point,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_mean': float(np.mean(predictions)),
            'bootstrap_std': float(np.std(predictions)),
            'n_bootstrap': len(predictions),
            'method': 'bootstrap'
        }

    def _build_shap_explainer(self, X_train, y_train):
        """构建SHAP解释器并计算训练集SHAP值

        使用shap通用Explainer接口，自动适配模型类型：
        - 树模型（RF/GBRT/XGB）：自动检测使用TreeExplainer
        - 线性模型（Ridge/ElasticNet）：自动检测使用LinearExplainer
        """
        if not HAS_SHAP or self.model is None:
            return

        try:
            # 使用新版通用Explainer接口（自动检测模型类型）
            self.shap_explainer = shap.Explainer(self.model, X_train.values)
            shap_values = self.shap_explainer(X_train.values)

            # 处理不同返回格式
            if hasattr(shap_values, 'values'):
                sv = shap_values.values
            else:
                sv = np.array(shap_values)

            # 计算特征级别的SHAP汇总统计
            self.shap_values_train = sv
            feature_cols = getattr(self, 'active_features', self.features)

            self.shap_summary = {}
            for i, feat in enumerate(feature_cols):
                if i < sv.shape[1]:
                    vals = sv[:, i]
                    self.shap_summary[feat] = {
                        'mean_abs_shap': float(np.mean(np.abs(vals))),
                        'std_shap': float(np.std(vals)),
                        'top_positive': float(np.percentile(vals, 95)),
                        'top_negative': float(np.percentile(vals, 5)),
                    }

            # 记录top3最重要特征
            top3 = sorted(self.shap_summary.items(),
                          key=lambda x: -x[1]['mean_abs_shap'])[:3]
            logger.info("SHAP特征重要性 top3: %s",
                        [(f, round(v['mean_abs_shap'], 4)) for f, v in top3])
        except Exception as e:
            logger.warning("SHAP explainer构建失败: %s", e)
            self.shap_explainer = None
            self.shap_summary = None

    def explain_shap(self, avg_temp, precipitation, sunshine,
                     city='崇左市', year=None):
        """对单次预测进行SHAP解释

        Returns:
            dict: {
                'feature_names': [...],
                'shap_values': [...],
                'base_value': float,
                'prediction': float,
                'top_positive': [(feature, value), ...],
                'top_negative': [(feature, value), ...]
            }
        """
        # 入参边界校验
        _validate_predict_inputs(avg_temp, precipitation, sunshine, city, year)

        if not HAS_SHAP or self.shap_explainer is None:
            return {'error': 'SHAP不可用或未训练'}

        try:
            row = self._build_features_row(avg_temp, precipitation, sunshine, city, year)
            feature_cols = getattr(self, 'active_features', self.features)
            X_pred = pd.DataFrame([row])[feature_cols]

            if self.scaler is not None:
                X_arr = self.scaler.transform(X_pred)
            else:
                X_arr = X_pred.values

            # 计算SHAP值（使用通用Explainer接口）
            shap_out = self.shap_explainer(X_arr)
            if hasattr(shap_out, 'values'):
                sv = shap_out.values
            else:
                sv = np.array(shap_out)
            sv = np.array(sv).flatten()

            # 获取base_value
            base_val = 0.0
            if hasattr(self.shap_explainer, 'expected_value'):
                ev = self.shap_explainer.expected_value
                base_val = float(ev[0] if isinstance(ev, np.ndarray) else ev)
            elif hasattr(self.shap_explainer, 'base_value'):
                bv = self.shap_explainer.base_value
                base_val = float(bv[0] if isinstance(bv, np.ndarray) else bv)

            pred = float(self.model.predict(X_arr)[0])

            # 排序正负贡献
            pairs = list(zip(feature_cols, sv))
            top_pos = sorted([(f, float(v)) for f, v in pairs if v > 0],
                             key=lambda x: -x[1])[:3]
            top_neg = sorted([(f, float(v)) for f, v in pairs if v < 0],
                             key=lambda x: x[1])[:3]

            return {
                'feature_names': feature_cols,
                'shap_values': [float(v) for v in sv],
                'base_value': base_val,
                'prediction': pred,
                'top_positive': top_pos,
                'top_negative': top_neg,
            }
        except Exception as e:
            logger.warning("SHAP解释失败: %s", e)
            return {'error': str(e)}

    @property
    def metrics(self):
        """返回最近一次训练的指标"""
        return self._train_metrics


class ByproductEstimator:
    """副产物产量估算"""

    def __init__(self):
        self.byproduct_df = pd.read_csv(os.path.join(DATA_DIR, 'byproduct_params.csv'))

    def estimate(self, sugarcane_yield_tons):
        """
        根据甘蔗产量估算各类副产物产量

        Args:
            sugarcane_yield_tons: 甘蔗产量（吨）

        Returns:
            dict: 各类副产物产量
        """
        byproducts = {}
        # 按 byproduct_name 分组取第一条记录（coefficient 对同一副产物一致）
        # 先按 coefficient 降序排序确保稳定性，再 drop_duplicates
        df_sorted = self.byproduct_df.sort_values(
            ['byproduct_name', 'production_coefficient_tons_per_ton_cane'],
            ascending=[True, False]
        )
        for _, row in df_sorted.drop_duplicates('byproduct_name').iterrows():
            name = row['byproduct_name']
            coeff = row['production_coefficient_tons_per_ton_cane']
            byproducts[name] = {
                'quantity': sugarcane_yield_tons * coeff,
                'coefficient': coeff
            }
        return byproducts


class CarbonCalculator:
    """碳排放核算（IPCC系数法，GWP从数据文件动态读取）"""

    def __init__(self):
        self.ipcc_df = pd.read_csv(os.path.join(DATA_DIR, 'ipcc_factors.csv'))

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _get_gwp(self, gas_pattern, default, source=None):
        """从IPCC数据中按气体名称模式读取GWP100值

        Args:
            gas_pattern: 气体名称模式（'CO2'/'CH4'/'N2O'）
            default: 未命中时的兜底值
            source: 可选，精确指定排放源（emission_source），
                    避免同气体多行时取错（如化石源/生物源CH4 GWP不同）
        """
        row = self.ipcc_df[
            self.ipcc_df['emission_factor_unit'].str.contains(gas_pattern, na=False)
        ]
        if source is not None:
            row = row[row['emission_source'] == source]
        return _safe_get(row, 'gwp_100year', default=default)

    def _get_emission_factor(self, source, default=0.0):
        """按排放源名称读取排放因子"""
        row = self.ipcc_df[self.ipcc_df['emission_source'] == source]
        return _safe_get(row, 'emission_factor_value', default=default)

    def _get_emission_factors_by_source(self, source):
        """按排放源名称读取所有温室气体因子（返回 {gas: {factor, gwp}}）"""
        subset = self.ipcc_df[self.ipcc_df['emission_source'] == source]
        result = {}
        for _, row in subset.iterrows():
            unit = row['emission_factor_unit']
            if 'CO2' in unit:
                gas = 'CO2'
            elif 'CH4' in unit:
                gas = 'CH4'
            elif 'N2O' in unit:
                gas = 'N2O'
            else:
                continue
            result[gas] = {
                'factor': float(row['emission_factor_value']),
                'gwp': float(row['gwp_100year'])
            }
        return result

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------
    def calculate_landfill_emission(self, filter_mud_quantity_tons):
        """计算滤泥填埋的CH4排放（厌氧分解）

        滤泥富含有机质（碳含量约200 kg/吨，水分75%），在填埋场厌氧条件下
        产生CH4。采用IPCC一级降解有机碳（DOC）法简化估算。
        """
        # 干物质 = 25%（水分75%），有机碳 = 干物质 × 20%
        dry_mass = filter_mud_quantity_tons * 0.25
        organic_carbon = dry_mass * 0.20
        # DOC可降解比例 50%，CH4生成潜力 60%
        doc = organic_carbon * 0.50
        ch4_potential = doc * 0.60 * (16 / 12)  # C→CH4 质量转换
        # 滤泥填埋为厌氧生物分解，属生物源CH4（IPCC AR6 GWP=27，
        # 区别于化石源CH4 GWP=29.8；引用sugarcane_burning_field行的生物源CH4值）
        ch4_gwp = self._get_gwp('CH4', default=27, source='sugarcane_burning_field')
        return {
            'ch4_kg': ch4_potential,
            'co2_equivalent_kg': ch4_potential * ch4_gwp
        }

    def calculate_soil_carbon_sequestration(self, area_mu, practice='cover_crop'):
        """计算土壤碳封存效益（蔗叶还田/覆盖作物）

        IPCC 2019 Refinement 指出，将作物残留物还田或种植覆盖作物
        可增加土壤有机碳（SOC）。此方法估算年土壤固碳量。

        Args:
            area_mu: 蔗田面积（亩）
            practice: 'cover_crop' | 'no_till'

        Returns:
            dict: {'sequestration_kg_co2': ..., 'area_ha': ...}
        """
        area_ha = area_mu / 15.0  # 亩→公顷
        source_map = {
            'cover_crop': 'soil_carbon_sequestration_cover_crop',
            'no_till': 'soil_carbon_sequestration_no_till',
        }
        source = source_map.get(practice, 'soil_carbon_sequestration_cover_crop')
        factor = self._get_emission_factor(source, default=-300)

        sequestration = area_ha * abs(factor)  # 固碳量均为正值表示减排
        return {
            'sequestration_kg_co2': sequestration,
            'area_ha': area_ha,
            'practice': practice
        }

    def calculate_burning_emission(self, leaf_quantity_tons):
        """计算蔗叶焚烧碳排放"""
        factors = self._get_emission_factors_by_source('sugarcane_burning_field')

        co2 = leaf_quantity_tons * factors.get('CO2', {}).get('factor', 0.0)
        ch4 = leaf_quantity_tons * factors.get('CH4', {}).get('factor', 0.0)
        n2o = leaf_quantity_tons * factors.get('N2O', {}).get('factor', 0.0)

        ch4_gwp = factors.get('CH4', {}).get('gwp', 28)
        n2o_gwp = factors.get('N2O', {}).get('gwp', 298)

        return {
            'co2_kg': co2,
            'ch4_kg': ch4,
            'n2o_kg': n2o,
            'co2_equivalent_kg': co2 + ch4 * ch4_gwp + n2o * n2o_gwp
        }

    def calculate_biomass_substitution(self, leaf_quantity_tons,
                                       coal_price_per_ton=None,
                                       co_firing_ratio=1.0):
        """计算生物质替代煤炭的减排量

        Args:
            leaf_quantity_tons: 蔗叶量（吨）
            coal_price_per_ton: 煤价（元/吨），None时从配置读取
            co_firing_ratio: 掺烧比例（0-1），1.0=100%理论替代，
                            0.3=30%行业标准掺烧
        """
        if coal_price_per_ton is None:
            coal_price_per_ton = COAL_CFG.get('price_per_ton', 900)
        substitution_rate = COAL_CFG.get('substitution_rate', 0.8)

        factor_val = self._get_emission_factor('biomass_pellet_substitution_coal')
        carbon_reduction = leaf_quantity_tons * abs(factor_val) * co_firing_ratio

        return {
            'carbon_reduction_kg': carbon_reduction,
            'coal_substitution_value': (
                leaf_quantity_tons * coal_price_per_ton *
                substitution_rate * co_firing_ratio
            ),
            'co_firing_ratio': co_firing_ratio
        }

    def calculate_full_chain(self, sugarcane_yield_tons, fertilizer_n_kg,
                             diesel_l, electricity_kwh, country='China',
                             fertilizer_p2o5_kg=0, fertilizer_k2o_kg=0):
        """
        计算全链条碳排放（IPCC AR6 GWP值）

        Args:
            sugarcane_yield_tons: 甘蔗产量（吨）
            fertilizer_n_kg: 氮肥施用量（kg N）
            diesel_l: 柴油使用量（L）
            electricity_kwh: 电力使用量（kWh）
            country: 国家（影响电网排放因子）
            fertilizer_p2o5_kg: 磷肥施用量（kg P2O5），默认0
            fertilizer_k2o_kg: 钾肥施用量（kg K2O），默认0
        """
        # GWP值按排放源精确匹配（避免同气体多行取错）
        # 化肥N₂O：IPCC 2006 Vol.4 Ch.11（AR6 GWP=273）
        n2o_gwp = self._get_gwp('N2O', default=273, source='N_fertilizer_application')
        # 柴油CH₄：化石源（AR6 GWP=29.8）
        ch4_gwp = self._get_gwp('CH4', default=29.8, source='diesel_fuel_combustion')
        # 柴油N₂O：AR6 GWP=273
        n2o_gwp_diesel = self._get_gwp('N2O', default=273, source='diesel_fuel_combustion')

        # 柴油排放因子（从配置文件读取）
        co2_per_l = DIESEL_CFG.get('co2_per_liter', 2.68)
        ch4_per_l = DIESEL_CFG.get('ch4_per_liter', 0.0002)
        n2o_per_l = DIESEL_CFG.get('n2o_per_liter', 0.0001)

        # 种植环节：化肥N2O排放
        # IPCC标准公式：N2O_emissions = N_input × EF × (44/28) × GWP
        # (44/28) = 1.571 是N2O-N到N2O的分子量转换系数
        n_ef = self._get_emission_factor('N_fertilizer_application', default=0.01)
        N2O_N_TO_N2O = 44.0 / 28.0  # IPCC Tier 1 分子量转换
        n2o_from_fertilizer = fertilizer_n_kg * n_ef * N2O_N_TO_N2O * n2o_gwp

        # 种植环节：磷肥和钾肥生产碳排放
        p_ef = self._get_emission_factor('P_fertilizer_application', default=0.2)
        co2_from_p_fertilizer = fertilizer_p2o5_kg * p_ef
        k_ef = self._get_emission_factor('K_fertilizer_application', default=0.15)
        co2_from_k_fertilizer = fertilizer_k2o_kg * k_ef

        # 机械作业：柴油燃烧
        co2_from_diesel = diesel_l * co2_per_l
        ch4_from_diesel = diesel_l * ch4_per_l * ch4_gwp
        n2o_from_diesel = diesel_l * n2o_per_l * n2o_gwp_diesel

        # 加工环节：电力消耗
        grid_factor = self._get_emission_factor(
            f'electricity_grid_{country}', default=0.57
        )
        co2_from_electricity = electricity_kwh * grid_factor

        total = (n2o_from_fertilizer + co2_from_p_fertilizer +
                 co2_from_k_fertilizer + co2_from_diesel +
                 ch4_from_diesel + n2o_from_diesel + co2_from_electricity)

        return {
            'planting': n2o_from_fertilizer + co2_from_p_fertilizer + co2_from_k_fertilizer,
            'mechanization': co2_from_diesel + ch4_from_diesel + n2o_from_diesel,
            'processing': co2_from_electricity,
            'total_kg': total,
            'total_tons': total / 1000
        }


class EconomicCalculator:
    """经济效益计算"""

    def __init__(self):
        self.market_df = pd.read_csv(os.path.join(DATA_DIR, 'market_prices.csv'))
        self.biomass_cost = COST_CFG.get('biomass_pellet_per_ton', 200)
        self.organic_cost = COST_CFG.get('organic_fertilizer_per_ton', 150)
        self.deep_cost = COST_CFG.get('deep_processed_per_ton', 500)
        self.landfill_cost = COST_CFG.get('landfill_per_ton', 100)
        self.boiler_value = COST_CFG.get('boiler_fuel_per_ton', 50)

    def _get_market_price(self, country, product_name):
        """安全获取市场价格"""
        price_row = self.market_df[
            (self.market_df['country'] == country) &
            (self.market_df['product_name'] == product_name)
        ]
        return _safe_get(price_row, 'price_avg_yuan_per_ton', default=0.0)

    def calculate_byproduct_value(self, byproduct_quantities, country='China'):
        """
        计算副产物各类利用方式的经济价值

        Args:
            byproduct_quantities: dict, 副产物产量
            country: str, 国家
        """
        results = {}

        for bp_name, bp_data in byproduct_quantities.items():
            quantity = bp_data['quantity']

            if bp_name == 'sugarcane_leaf':
                pellet_price = self._get_market_price(country, 'biomass_pellet')
                feed_price = self._get_market_price(country, 'sugarcane_leaf_animal_feed')
                results['sugarcane_leaf'] = {
                    'quantity': quantity,
                    'burn': {'revenue': 0.0, 'cost': 0.0},
                    'biomass_pellet': {
                        'revenue': quantity * pellet_price,
                        'cost': quantity * self.biomass_cost
                    },
                    'animal_feed': {
                        'revenue': quantity * (feed_price if feed_price > 0 else 280),
                        'cost': quantity * 50  # 青贮/氨化处理成本
                    }
                }

            elif bp_name == 'filter_mud':
                organic_price = self._get_market_price(country, 'organic_fertilizer')
                results['filter_mud'] = {
                    'quantity': quantity,
                    'landfill': {'revenue': -quantity * self.landfill_cost, 'cost': 0.0},
                    'organic_fertilizer': {
                        'revenue': quantity * organic_price,
                        'cost': quantity * self.organic_cost
                    }
                }

            elif bp_name == 'molasses':
                direct_price = self._get_market_price(country, 'molasses_direct')
                deep_price = self._get_market_price(country, 'molasses_deep_processed')
                results['molasses'] = {
                    'quantity': quantity,
                    'direct_sale': {'revenue': quantity * direct_price, 'cost': 0.0},
                    'deep_processed': {
                        'revenue': quantity * deep_price,
                        'cost': quantity * self.deep_cost
                    }
                }

            elif bp_name == 'sugarcane_top':
                top_feed_price = self._get_market_price(country, 'sugarcane_top_animal_feed')
                results['sugarcane_top'] = {
                    'quantity': quantity,
                    'burn': {'revenue': 0.0, 'cost': 0.0},
                    'animal_feed': {
                        'revenue': quantity * (top_feed_price if top_feed_price > 0 else 280),
                        'cost': quantity * 50  # 青贮/氨化处理成本
                    }
                }

            elif bp_name == 'bagasse':
                pulp_price = self._get_market_price(country, 'bagasse_pulp_paper')
                wood_price = self._get_market_price(country, 'bagasse_plywood')
                biogas_price = self._get_market_price(country, 'bagasse_biogas')
                tableware_price = self._get_market_price(country, 'bagasse_tableware')
                results['bagasse'] = {
                    'quantity': quantity,
                    'boiler_fuel': {
                        'revenue': quantity * self.boiler_value,
                        'cost': 0.0
                    },
                    'pulp_paper': {
                        'revenue': quantity * (pulp_price if pulp_price > 0 else 650),
                        'cost': quantity * 120
                    },
                    'plywood': {
                        'revenue': quantity * (wood_price if wood_price > 0 else 500),
                        'cost': quantity * 100
                    },
                    'biogas': {
                        'revenue': quantity * (biogas_price if biogas_price > 0 else 280),
                        'cost': quantity * 80
                    },
                    'tableware': {
                        'revenue': quantity * (tableware_price if tableware_price > 0 else 1500),
                        'cost': quantity * 800  # 蔗渣浆加工成本(制浆+清洗+能源，参考兴桂纸业2024)
                    }
                }

        return results

    def calculate_net_benefit(self, economic_results):
        """计算五类方案的净收益

        传统模式(traditional):
            蔗叶焚烧 + 滤泥填埋 + 糖蜜直接出售 + 蔗渣锅炉燃料

        改良传统(improved_traditional):
            蔗叶饲料化 + 滤泥填埋 + 糖蜜直接出售 + 蔗渣锅炉燃料
            （低投入改良，适合小农户过渡）

        基础循环(circular_basic):
            蔗叶饲料化 + 滤泥有机肥 + 糖蜜直接出售 + 蔗渣沼气
            （低投入循环利用，适合小农户）

        进阶循环(circular_advanced):
            蔗叶生物质颗粒 + 滤泥有机肥 + 糖蜜深加工 + 蔗渣造纸浆
            （中等投入，适合合作社）

        最优循环(circular_optimal):
            蔗叶生物质颗粒替代煤炭 + 滤泥有机肥 + 糖蜜深加工 + 蔗渣环保餐具
            （最高附加值循环利用，适合糖企，对标来宾27亿环保餐具产业）
        """
        schemes = {
            'traditional': 0.0,
            'improved_traditional': 0.0,
            'circular_basic': 0.0,
            'circular_advanced': 0.0,
            'circular_optimal': 0.0
        }

        for bp_name, bp_data in economic_results.items():
            if bp_name == 'sugarcane_leaf':
                schemes['traditional'] += (
                    bp_data['burn']['revenue'] - bp_data['burn']['cost'])
                schemes['improved_traditional'] += (
                    bp_data['animal_feed']['revenue'] -
                    bp_data['animal_feed']['cost'])
                schemes['circular_basic'] += (
                    bp_data['animal_feed']['revenue'] -
                    bp_data['animal_feed']['cost'])
                schemes['circular_advanced'] += (
                    bp_data['biomass_pellet']['revenue'] -
                    bp_data['biomass_pellet']['cost'])
                schemes['circular_optimal'] += (
                    bp_data['biomass_pellet']['revenue'] -
                    bp_data['biomass_pellet']['cost'])

            elif bp_name == 'filter_mud':
                schemes['traditional'] += (
                    bp_data['landfill']['revenue'] - bp_data['landfill']['cost'])
                schemes['improved_traditional'] += (
                    bp_data['landfill']['revenue'] - bp_data['landfill']['cost'])
                schemes['circular_basic'] += (
                    bp_data['organic_fertilizer']['revenue'] -
                    bp_data['organic_fertilizer']['cost'])
                schemes['circular_advanced'] += (
                    bp_data['organic_fertilizer']['revenue'] -
                    bp_data['organic_fertilizer']['cost'])
                schemes['circular_optimal'] += (
                    bp_data['organic_fertilizer']['revenue'] -
                    bp_data['organic_fertilizer']['cost'])

            elif bp_name == 'molasses':
                schemes['traditional'] += (
                    bp_data['direct_sale']['revenue'] -
                    bp_data['direct_sale']['cost'])
                schemes['improved_traditional'] += (
                    bp_data['direct_sale']['revenue'] -
                    bp_data['direct_sale']['cost'])
                schemes['circular_basic'] += (
                    bp_data['direct_sale']['revenue'] -
                    bp_data['direct_sale']['cost'])
                schemes['circular_advanced'] += (
                    bp_data['deep_processed']['revenue'] -
                    bp_data['deep_processed']['cost'])
                schemes['circular_optimal'] += (
                    bp_data['deep_processed']['revenue'] -
                    bp_data['deep_processed']['cost'])

            elif bp_name == 'sugarcane_top':
                # 蔗梢：传统=焚烧（零收益），其余方案=饲料化
                schemes['traditional'] += (
                    bp_data['burn']['revenue'] - bp_data['burn']['cost'])
                schemes['improved_traditional'] += (
                    bp_data['animal_feed']['revenue'] -
                    bp_data['animal_feed']['cost'])
                schemes['circular_basic'] += (
                    bp_data['animal_feed']['revenue'] -
                    bp_data['animal_feed']['cost'])
                schemes['circular_advanced'] += (
                    bp_data['animal_feed']['revenue'] -
                    bp_data['animal_feed']['cost'])
                schemes['circular_optimal'] += (
                    bp_data['animal_feed']['revenue'] -
                    bp_data['animal_feed']['cost'])

            elif bp_name == 'bagasse':
                schemes['traditional'] += (
                    bp_data['boiler_fuel']['revenue'] -
                    bp_data['boiler_fuel']['cost'])
                schemes['improved_traditional'] += (
                    bp_data['boiler_fuel']['revenue'] -
                    bp_data['boiler_fuel']['cost'])
                schemes['circular_basic'] += (
                    bp_data['biogas']['revenue'] -
                    bp_data['biogas']['cost'])
                schemes['circular_advanced'] += (
                    bp_data['pulp_paper']['revenue'] -
                    bp_data['pulp_paper']['cost'])
                schemes['circular_optimal'] += (
                    bp_data['tableware']['revenue'] -
                    bp_data['tableware']['cost'])

        return schemes


class OptimizationEngine:
    """多目标优化引擎

    通过依赖注入接收 CarbonCalculator 和 EconomicCalculator，
    避免重复实例化。
    """

    def __init__(self, carbon_calc=None, economic_calc=None):
        self.carbon_calc = carbon_calc or CarbonCalculator()
        self.economic_calc = economic_calc or EconomicCalculator()
        self.benefit_weight = OPT_CFG.get('benefit_weight', 0.7)
        self.carbon_weight = OPT_CFG.get('carbon_weight', 0.3)

    def optimize(self, sugarcane_yield_tons, byproduct_quantities,
                 carbon_price=85, country='China', co_firing_ratio=1.0,
                 benefit_weight=None, carbon_weight=None,
                 carbon_trading_scenario='energy_only'):
        """
        多目标优化：收益最大化 + 碳排放最小化

        Args:
            co_firing_ratio: 生物质掺烧比（0-1），1.0=理论最优，0.3=行业标准
            benefit_weight: 收益权重（0-1），None时使用配置默认值
            carbon_weight: 碳权重（0-1），None时使用配置默认值
            carbon_trading_scenario: 'energy_only'=仅能源排放可交易,
                                    'future_agriculture'=假设农业纳入碳市场

        Returns:
            dict: {'optimal': {...}, 'all_schemes': [...], 'weights': {...}}
        """
        # 参数边界校验（防御非法权重/掺烧比）
        if not isinstance(co_firing_ratio, (int, float)) or not (0.0 <= co_firing_ratio <= 1.0):
            raise ValueError("co_firing_ratio 必须在 [0, 1] 之间")

        # 权重支持外部传入（用于前端滑块实时调节）
        bw = benefit_weight if benefit_weight is not None else self.benefit_weight
        cw = carbon_weight if carbon_weight is not None else self.carbon_weight

        if bw is not None and (not isinstance(bw, (int, float)) or not (0.0 <= bw <= 1.0)):
            raise ValueError("benefit_weight 必须在 [0, 1] 之间")
        if cw is not None and (not isinstance(cw, (int, float)) or not (0.0 <= cw <= 1.0)):
            raise ValueError("carbon_weight 必须在 [0, 1] 之间")

        # 归一化确保 bw + cw = 1.0（当两者均显式传入时）
        if benefit_weight is not None and carbon_weight is not None:
            total = bw + cw
            if total > 0:
                bw, cw = bw / total, cw / total

        # 经济效益
        economic = self.economic_calc.calculate_byproduct_value(
            byproduct_quantities, country)
        net_benefit = self.economic_calc.calculate_net_benefit(economic)

        # 获取各副产物产量
        leaf_qty = byproduct_quantities.get('sugarcane_leaf', {}).get('quantity', 0)
        top_qty = byproduct_quantities.get('sugarcane_top', {}).get('quantity', 0)
        bagasse_qty = byproduct_quantities.get('bagasse', {}).get('quantity', 0)
        mud_qty = byproduct_quantities.get('filter_mud', {}).get('quantity', 0)
        molasses_qty = byproduct_quantities.get('molasses', {}).get('quantity', 0)

        # 碳排放基础计算
        burning = self.carbon_calc.calculate_burning_emission(leaf_qty)
        top_burning = self.carbon_calc.calculate_burning_emission(top_qty)
        substitution = self.carbon_calc.calculate_biomass_substitution(
            leaf_qty, co_firing_ratio=co_firing_ratio)
        landfill = self.carbon_calc.calculate_landfill_emission(mud_qty)

        # 五方案差异化碳排放（kg CO2e）
        # 设计原则：经济收益越高，碳排放越低（循环经济双赢）
        # 蔗梢口径与经济计算一致：传统=焚烧（计入碳排），其余方案=饲料化（残留+加工能耗）
        #
        # 【情景假设参数说明】
        # 焚烧/填埋/生物质替代三项有文献依据（Andreae 2001 / IPCC DOC法 / Cherubini 2009）；
        # 但下列"残留比例"与"加工能耗"系数（0.10~0.20残留、30~300 kgCO2e/吨能耗）
        # 无公开实测文献值，为基于行业经验的情景假设，仅用于五方案的相对排序演示，
        # 不影响"循环经济减排"的定性结论。如需精确核算可替换为实测值。
        RESIDUAL_BURN_RATIO_FEED = 0.10   # 饲料化后残留焚烧/腐解比例（情景假设）
        RESIDUAL_BURN_RATIO_SILAGE = 0.20 # 青贮化后残留比例（情景假设）
        RESIDUAL_LANDFILL_ORGANIC = 0.15  # 有机肥化后残留填埋当量（情景假设）
        RESIDUAL_LANDFILL_BASIC = 0.20    # 基础循环残留填埋当量（情景假设）
        PROC_ENERGY_SILAGE = 50.0         # 青贮/氨化能耗 kgCO2e/吨（情景假设）
        PROC_ENERGY_FEED = 30.0           # 饲料加工能耗 kgCO2e/吨（情景假设）
        PROC_ENERGY_PULP = 150.0          # 造纸浆加工能耗 kgCO2e/吨（情景假设）
        PROC_ENERGY_TABLEWARE = 300.0     # 环保餐具加工能耗 kgCO2e/吨（情景假设）
        PROC_ENERGY_MOLASSES = 80.0       # 糖蜜深加工能耗 kgCO2e/吨（情景假设）
        BIOGAS_SUBSTITUTION_RATIO = 0.30  # 沼气替代能源比例（情景假设）

        carbon_map = {
            'traditional': (
                burning['co2_equivalent_kg']           # 蔗叶全额焚烧
                + top_burning['co2_equivalent_kg']     # 蔗梢全额焚烧
                + landfill['co2_equivalent_kg']        # 滤泥填埋CH4
            ),
            'improved_traditional': (
                RESIDUAL_BURN_RATIO_SILAGE * (burning['co2_equivalent_kg'] + top_burning['co2_equivalent_kg'])
                + landfill['co2_equivalent_kg']        # 滤泥仍填埋
                + PROC_ENERGY_SILAGE * (leaf_qty + top_qty)  # 青贮/氨化加工能耗
            ),
            'circular_basic': (
                RESIDUAL_BURN_RATIO_FEED * (burning['co2_equivalent_kg'] + top_burning['co2_equivalent_kg'])
                + RESIDUAL_LANDFILL_BASIC * landfill['co2_equivalent_kg']  # 有机肥仍有少量排放
                - BIOGAS_SUBSTITUTION_RATIO * substitution['carbon_reduction_kg']  # 沼气替代部分能源
                + PROC_ENERGY_FEED * (leaf_qty + top_qty)  # 饲料加工
            ),
            'circular_advanced': (
                -0.60 * substitution['carbon_reduction_kg']  # 颗粒替代60%煤炭
                + RESIDUAL_LANDFILL_ORGANIC * landfill['co2_equivalent_kg']  # 有机肥
                + PROC_ENERGY_PULP * bagasse_qty             # 造纸浆加工能耗
                + PROC_ENERGY_MOLASSES * molasses_qty        # 糖蜜深加工能耗
                + PROC_ENERGY_FEED * top_qty                 # 蔗梢饲料化加工能耗
            ),
            'circular_optimal': (
                -substitution['carbon_reduction_kg']   # 颗粒全额替代煤炭
                + RESIDUAL_LANDFILL_ORGANIC * landfill['co2_equivalent_kg']  # 有机肥
                + PROC_ENERGY_TABLEWARE * bagasse_qty  # 环保餐具加工能耗
                + PROC_ENERGY_MOLASSES * molasses_qty  # 糖蜜深加工能耗
                + PROC_ENERGY_FEED * top_qty           # 蔗梢饲料化加工能耗
            )
        }

        scheme_names = [
            'traditional', 'improved_traditional', 'circular_basic',
            'circular_advanced', 'circular_optimal'
        ]
        benefits = [net_benefit[s] for s in scheme_names]
        carbons = [carbon_map[s] for s in scheme_names]

        # ---- 第一遍：先算碳交易收益与综合收益（total_benefit）----
        # 综合收益 = 净收益 + 碳交易收益（含碳价的最终经济口径）
        intermediates = []
        for scheme_name, benefit, carbon_emission in zip(
                scheme_names, benefits, carbons):

            if carbon_trading_scenario == 'energy_only':
                # 仅能源相关CO₂可交易（现实情景：农业N₂O、生物源CH₄目前不在CEA碳市场）
                # 传统模式 = 焚烧+填埋，全部为农业/生物源排放，可交易排放=0；
                # 其余方案含加工能耗（能源相关），按情景假设比例估算可交易份额。
                # 【情景假设】以下能源相关占比系数（0.4~0.8）无公开实测值，
                # 为基于加工能耗占比的近似估计，仅用于碳交易收益的情景演示。
                TRADABLE_RATIO = {
                    'traditional': 0.0,             # 焚烧+填埋，无能源相关排放
                    'improved_traditional': 0.8,    # 青贮加工能耗为主（情景假设）
                    'circular_basic': 0.6,          # 饲料+沼气（情景假设）
                    'circular_advanced': 0.5,       # 造纸浆+深加工（情景假设）
                    'circular_optimal': 0.4,        # 餐具+深加工（情景假设）
                }
                tradable_emission = carbon_emission * TRADABLE_RATIO.get(
                    scheme_name, 0.0)
            else:  # future_agriculture
                # 情景分析：假设未来碳市场扩展到农业，全部排放可交易
                tradable_emission = carbon_emission

            carbon_revenue = -(tradable_emission / 1000) * carbon_price
            total_benefit = benefit + carbon_revenue

            intermediates.append({
                'name': scheme_name,
                'net_benefit': benefit,
                'total_benefit': total_benefit,
                'carbon_emission_kg': carbon_emission,
                'carbon_revenue': carbon_revenue,
                'tradable_emission_kg': tradable_emission,
            })

        # ---- 第二遍：标准化评分与排序 ----
        # 收益维度统一采用综合收益（total_benefit），保证排序口径与展示口径一致
        total_benefits = [it['total_benefit'] for it in intermediates]
        min_b, max_b = min(total_benefits), max(total_benefits)
        min_c, max_c = min(carbons), max(carbons)

        schemes = []
        for it in intermediates:
            benefit_v = it['total_benefit']
            carbon_emission = it['carbon_emission_kg']

            if max_b != min_b:
                benefit_score = (benefit_v - min_b) / (max_b - min_b)
            else:
                benefit_score = 0.5

            if max_c != min_c:
                carbon_score = (max_c - carbon_emission) / (max_c - min_c)
            else:
                carbon_score = 0.5

            benefit_score = max(0.0, min(1.0, benefit_score))
            carbon_score = max(0.0, min(1.0, carbon_score))

            total_score = (bw * benefit_score + cw * carbon_score)

            it['benefit_score'] = benefit_score
            it['carbon_score'] = carbon_score
            it['total_score'] = total_score
            schemes.append(it)

        schemes.sort(key=lambda x: x['total_score'], reverse=True)

        # 附加碳排放明细
        for s in schemes:
            if s['name'] == 'traditional':
                s['landfill_ch4_kg'] = landfill['co2_equivalent_kg']
            else:
                s['landfill_ch4_kg'] = 0.0

        return {
            'optimal': schemes[0],
            'all_schemes': schemes,
            'landfill_emission': landfill,
            'weights': {'benefit': bw, 'carbon': cw},
            'carbon_trading_scenario': carbon_trading_scenario
        }


class SugarcaneDecisionSystem:
    """甘蔗副产物循环经济决策系统主类

    采用依赖注入：CarbonCalculator 和 EconomicCalculator 实例在
    主系统和 OptimizationEngine 之间共享。

    跨境产量预测策略：
    - China: 使用基于广西气象-产量数据训练的回归模型
    - Thailand/Vietnam/Myanmar/Laos: 使用FAO历史统计均值（吨/公顷→吨/亩换算）
      因缺乏城市级气象-产量配对数据，采用FAO十年均值作为统计基准
    """

    # FAO吨/公顷 → 吨/亩换算系数 (1公顷=15亩)
    HA_TO_MU = 1.0 / 15.0

    def __init__(self):
        self.yield_predictor = YieldPredictor()
        self.byproduct_estimator = ByproductEstimator()
        # 共享实例
        self.carbon_calculator = CarbonCalculator()
        self.economic_calculator = EconomicCalculator()
        self.optimizer = OptimizationEngine(
            carbon_calc=self.carbon_calculator,
            economic_calc=self.economic_calculator
        )
        # FAO跨境产量基准（吨/亩）
        self._fao_yield_baseline = self._load_fao_yield_baseline()

    def _load_fao_yield_baseline(self):
        """从FAO数据加载各国历史平均单产（吨/亩）及不确定性区间"""
        try:
            fao = pd.read_csv(os.path.join(DATA_DIR, 'fao_global.csv'))
            baseline = {}
            for country in ['China', 'Thailand', 'Vietnam', 'Myanmar', 'Laos']:
                sub = fao[fao['country'] == country]
                vals = sub['yield_per_ha_tons'].values * self.HA_TO_MU
                baseline[country] = {
                    'mean': round(float(vals.mean()), 2),
                    'std': round(float(vals.std()), 2),
                    'min': round(float(vals.min()), 2),
                    'max': round(float(vals.max()), 2),
                }
            logger.info("FAO产量基准(吨/亩): %s",
                        {k: '%.2f±%.2f' % (v['mean'], v['std'])
                         for k, v in baseline.items()})
            self._fao_baseline_detail = baseline
            return {k: v['mean'] for k, v in baseline.items()}
        except Exception as e:
            logger.warning("FAO数据加载失败: %s", e)
            return {'China': 5.5, 'Thailand': 3.3, 'Vietnam': 4.0, 'Myanmar': 4.0, 'Laos': 3.0}

    def train_models(self, model_type: str = 'auto') -> dict:
        """训练所有模型，自动选择最优算法

        Args:
            model_type: 'ridge' | 'rf' | 'gbrt' | 'auto'（默认，自动对比选择）
        """
        gx, weather, _, _, _, _, _ = load_data()
        return self.yield_predictor.train(gx, weather, model_type=model_type)

    def run_decision(self, area_mu: float, avg_temp: float,
                     precipitation: float, sunshine: float,
                     fertilizer_n_kg: float, diesel_l: float,
                     electricity_kwh: float, carbon_price: float = None,
                     country: str = 'China', city: str = '崇左市',
                     scenario: str = 'optimal',
                     benefit_weight: float = None,
                     carbon_weight: float = None,
                     carbon_trading_scenario: str = 'energy_only') -> dict:
        """
        运行完整决策流程

        Args:
            scenario: 'optimal'(100%煤炭替代) | 'realistic'(30%掺烧,行业标准)
            benefit_weight: 收益权重（0-1），None时使用默认
            carbon_weight: 碳权重（0-1），None时使用默认
            carbon_trading_scenario: 'energy_only' | 'future_agriculture'
        """
        _validate_decision_inputs(
            area_mu, avg_temp, precipitation, sunshine,
            fertilizer_n_kg, diesel_l, electricity_kwh,
            carbon_price, country, city
        )

        if carbon_price is None:
            carbon_price = get_default_carbon_price(country)

        co_firing = 1.0 if scenario == 'optimal' else 0.3

        # 1. 预测产量
        yield_ci = None
        if country == 'China':
            yield_per_mu = self.yield_predictor.predict(
                avg_temp, precipitation, sunshine, city=city)
            yield_source = 'model'
            try:
                ci_result = self.yield_predictor.predict_with_ci(
                    avg_temp, precipitation, sunshine, city=city, n_bootstrap=200)
                yield_ci = {
                    'lower': ci_result['ci_lower'],
                    'upper': ci_result['ci_upper'],
                    'method': ci_result.get('method', 'bootstrap_200')
                }
            except Exception:
                pass
        else:
            fao_info = self._fao_baseline_detail.get(country, {})
            yield_per_mu = self._fao_yield_baseline.get(country, 5.0)
            yield_source = 'fao_statistical_average'
            yield_ci = {
                'lower': fao_info.get('min', yield_per_mu * 0.8),
                'upper': fao_info.get('max', yield_per_mu * 1.2),
                'std': fao_info.get('std', yield_per_mu * 0.1),
                'method': 'fao_10yr_range'
            }
        total_yield = yield_per_mu * area_mu

        # 2-5 同前
        byproducts = self.byproduct_estimator.estimate(total_yield)
        carbon_emission = self.carbon_calculator.calculate_full_chain(
            total_yield, fertilizer_n_kg, diesel_l, electricity_kwh, country)
        economic = self.economic_calculator.calculate_byproduct_value(
            byproducts, country)
        net_benefit = self.economic_calculator.calculate_net_benefit(economic)
        optimization = self.optimizer.optimize(
            total_yield, byproducts, carbon_price, country,
            co_firing_ratio=co_firing,
            benefit_weight=benefit_weight,
            carbon_weight=carbon_weight,
            carbon_trading_scenario=carbon_trading_scenario)

        logger.info(
            "决策完成: country=%s, yield=%.2f, optimal=%s, benefit=%.2f, "
            "weights=(%.2f, %.2f), scenario=%s",
            country, yield_per_mu,
            optimization['optimal']['name'],
            optimization['optimal']['net_benefit'],
            optimization['weights']['benefit'],
            optimization['weights']['carbon'],
            scenario
        )

        return {
            'area_mu': area_mu,
            'yield_per_mu': yield_per_mu,
            'yield_source': yield_source,
            'yield_ci': yield_ci,
            'total_yield': total_yield,
            'byproducts': byproducts,
            'carbon_emission': carbon_emission,
            'economic': economic,
            'net_benefit': net_benefit,
            'optimization': optimization,
            'scenario': scenario
        }


# ===================================================================
# 模块入口
# ===================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    system = SugarcaneDecisionSystem()

    logger.info("正在训练模型...")
    metrics = system.train_models()
    logger.info("模型训练完成，R²=%.4f", metrics['r2'])

    logger.info("运行决策示例...")
    result = system.run_decision(
        area_mu=10,
        avg_temp=28.5,
        precipitation=2200,
        sunshine=900,
        fertilizer_n_kg=150,
        diesel_l=50,
        electricity_kwh=500,
        carbon_price=85,
        country='China'
    )

    print(f"\n预测单产: {result['yield_per_mu']:.2f} 吨/亩")
    print(f"总产量: {result['total_yield']:.2f} 吨")
    print(f"碳排放: {result['carbon_emission']['total_tons']:.2f} 吨CO2")
    print(f"最优方案: {result['optimization']['optimal']['name']}")
    print(f"净收益: {result['optimization']['optimal']['net_benefit']:.2f} 元")
