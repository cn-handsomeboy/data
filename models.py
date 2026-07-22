"""
核心算法模型模块
包含：产量预测、副产物估算、碳排放核算、经济效益计算、多目标优化
"""

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              StackingRegressor)
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

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
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(levelname)s] %(name)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 抑制sklearn已知无害警告（不影响结果，仅减少日志噪音）
import warnings
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


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


def load_data():
    """加载所有数据集"""
    gx = pd.read_csv(os.path.join(DATA_DIR, 'guangxi_sugarcane.csv'))
    weather = pd.read_csv(os.path.join(DATA_DIR, 'weather_data.csv'))
    fao = pd.read_csv(os.path.join(DATA_DIR, 'fao_global.csv'))
    ipcc = pd.read_csv(os.path.join(DATA_DIR, 'ipcc_factors.csv'))
    carbon = pd.read_csv(os.path.join(DATA_DIR, 'carbon_price.csv'))
    byproduct = pd.read_csv(os.path.join(DATA_DIR, 'byproduct_params.csv'))
    market = pd.read_csv(os.path.join(DATA_DIR, 'market_prices.csv'))
    return gx, weather, fao, ipcc, carbon, byproduct, market


def get_default_carbon_price():
    """从历史碳价数据计算智能默认值（近12个月均价）

    数据来源：上海环境能源交易所全国碳市场CEA收盘价。
    若数据不可用，返回配置中的 fallback 值。
    """
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
            logger.info("近12个月碳价均价: %.2f 元/吨 (样本数: %d)",
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
        self.features = ['avg_temp_c', 'precipitation_mm', 'sunshine_hours']
        self.features += [f'city_{c}' for c in self.CITY_DUMMIES]
        self.features += ['year', 'planting_area_wan_mu']
        self.target = 'yield_per_mu_tons'
        self._trained = False
        self._train_metrics = None
        self.model_comparison = None

    def _train_single_model(self, model, X, y, model_name):
        """使用 LOOCV + GridSearchCV 训练单个模型并返回指标"""
        # ---- 超参数网格 ----
        n_est = MODEL_CFG.get('n_estimators', 100)
        param_grids = {
            'ridge': {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]},
            'rf': {
                'n_estimators': [max(30, n_est // 3), n_est // 2, n_est],
                'max_depth': [2, 3, 4, 5],
                'min_samples_leaf': [1, 2, 3]
            },
            'gbrt': {
                'n_estimators': [max(30, n_est // 3), n_est // 2, n_est],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_leaf': [1, 2, 3]
            }
        }

        pg = param_grids.get(model_name, {})
        best_model = model

        # ElasticNetCV has built-in CV, no GridSearchCV needed
        if model_name == 'elasticnet':
            try:
                best_model.fit(X, y)
                logger.info("  elasticnet CV: alpha=%.4f, l1_ratio=%.2f",
                            best_model.alpha_, best_model.l1_ratio_)
            except Exception as e:
                logger.warning("  elasticnet failed: %s", e)
                best_model = model
                best_model.fit(X, y)
        elif pg:
            try:
                gs = GridSearchCV(
                    model, pg, cv=min(5, len(X)),
                    scoring='neg_mean_squared_error', n_jobs=1
                )
                gs.fit(X, y)
                best_model = gs.best_estimator_
                logger.info("  %s GridSearchCV best params: %s",
                            model_name, gs.best_params_)
            except Exception as e:
                logger.warning("  %s GridSearchCV failed: %s, using defaults", model_name, e)
                best_model = model
                best_model.fit(X, y)
        else:
            best_model.fit(X, y)

        # ---- LOOCV 评估 ----
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X):
            X_tr = X.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            y_te = y.iloc[test_idx]

            # 对每个fold重新训练（使用最佳超参数重新实例化）
            if hasattr(best_model, 'get_params'):
                fold_model = best_model.__class__(**best_model.get_params())
            else:
                fold_model = best_model.__class__()
            fold_model.fit(X_tr, y_tr)
            y_pred.append(fold_model.predict(X_te)[0])
            y_true.append(y_te.values[0])

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 最终在全量数据上训练
        best_model.fit(X, y)

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

        weather_yearly = gs.groupby(['year', 'city']).agg({
            'avg_temp_c': 'mean',              # 生长季均温
            'precipitation_mm': 'sum',          # 生长季累计降水
            'sunshine_hours': 'sum'             # 生长季累计日照
        }).reset_index()

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
        all_features = self.features + interaction_features

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

        # ---- 特征标准化 ----
        self.scaler = StandardScaler()
        X_scaled_arr = self.scaler.fit_transform(X)

        self.active_features = all_features
        X_final = pd.DataFrame(X_scaled_arr, columns=all_features, index=X.index)

        # 存储训练数据（bootstrap用）
        self._X_train = X_final.copy()
        self._y_train = y.copy()
        self._city_area_from_training = {}
        for c in merged['city'].unique():
            recent = merged[merged['city'] == c].sort_values('year').tail(3)
            self._city_area_from_training[c] = float(recent['planting_area_wan_mu'].mean())

        # ---- 候选模型 ----
        candidates = {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'rf': RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
            'gbrt': GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42
            ),
            'elasticnet': ElasticNetCV(cv=5, random_state=42, max_iter=10000,
                                       alphas=None, l1_ratio=[.1, .3, .5, .7, .9]),
        }

        if model_type == 'auto':
            results = []
            for name, m in candidates.items():
                result = self._train_single_model(m, X_final.copy(), y.copy(), name)
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
            result = self._train_single_model(model, X_final.copy(), y.copy(), model_type)
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

        # ---- RepeatedKFold 稳健评估 ----
        try:
            rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
            rkf_scores = []
            for train_idx, test_idx in rkf.split(X_final):
                X_tr, X_te = X_final.iloc[train_idx], X_final.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                m = self.model.__class__(**self.model.get_params()) if hasattr(self.model, 'get_params') else self.model.__class__()
                m.fit(X_tr, y_tr)
                yp = m.predict(X_te)
                rkf_scores.append(r2_score(y_te, yp))
            self._train_metrics['r2_repeated_kfold_mean'] = float(np.mean(rkf_scores))
            self._train_metrics['r2_repeated_kfold_std'] = float(np.std(rkf_scores))
            logger.info("RepeatedKFold (5x10): R²=%.4f ± %.4f",
                        self._train_metrics['r2_repeated_kfold_mean'],
                        self._train_metrics['r2_repeated_kfold_std'])
        except Exception as e:
            logger.warning("RepeatedKFold失败: %s", e)

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

        # ---- Stacking Ensemble (学术最佳实践) ----
        self.stacking_model = None
        self.stacking_metrics = None
        try:
            base_estimators = [
                ('ridge', Ridge(alpha=0.01, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=50, max_depth=4,
                     min_samples_leaf=2, random_state=42)),
            ]
            stacking = StackingRegressor(
                estimators=base_estimators,
                final_estimator=Ridge(alpha=1.0, random_state=42),
                cv=5
            )
            stacking.fit(X_final, y)

            # LOOCV评估Stacking
            y_true_s, y_pred_s = [], []
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(X_final):
                X_tr = X_final.iloc[train_idx]
                X_te = X_final.iloc[test_idx]
                y_tr = y.iloc[train_idx]
                y_te = y.iloc[test_idx]
                st = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=Ridge(alpha=1.0, random_state=42),
                    cv=5
                )
                st.fit(X_tr, y_tr)
                y_pred_s.append(st.predict(X_te)[0])
                y_true_s.append(y_te.values[0])

            s_r2 = r2_score(y_true_s, y_pred_s)
            s_rmse = np.sqrt(mean_squared_error(y_true_s, y_pred_s))
            self.stacking_model = stacking
            self.stacking_metrics = {
                'r2': float(s_r2), 'rmse': float(s_rmse),
                'loocv_samples': len(y_true_s)
            }
            logger.info("Stacking Ensemble LOOCV: R²=%.4f, RMSE=%.4f", s_r2, s_rmse)
        except Exception as e:
            logger.warning("Stacking Ensemble失败: %s", e)

        # 保存模型
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, 'yield_predictor.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'fallback_yield': self.fallback_yield,
                'trained': self._trained,
                'metrics': self._train_metrics,
                'model_comparison': self.model_comparison,
                'active_features': getattr(self, 'active_features', self.features),
                'city_area_from_training': getattr(self, '_city_area_from_training', {}),
                'feature_importance': getattr(self, 'feature_importance', None),
                'stacking_metrics': getattr(self, 'stacking_metrics', None),
            }, f)

        return self._train_metrics

    def load_model(self):
        """加载已训练模型（兼容新旧两种格式）"""
        model_path = os.path.join(MODELS_DIR, 'yield_predictor.pkl')
        if not os.path.exists(model_path):
            return False

        with open(model_path, 'rb') as f:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                data = pickle.load(f)  # 本地可信文件，加载后即时验证格式

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
            if 'stacking_metrics' in data:
                self.stacking_metrics = data['stacking_metrics']
            logger.info("模型已加载，fallback=%.2f", self.fallback_yield)
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
        row['year'] = float(year) if year is not None else float(datetime.now().year)
        if hasattr(self, '_city_area_from_training') and city in self._city_area_from_training:
            row['planting_area_wan_mu'] = self._city_area_from_training[city]
        else:
            row['planting_area_wan_mu'] = self.CITY_AREA_WAN_MU.get(city, 200.0)
        # 交互特征
        row['avg_temp_c_x_precipitation_mm'] = row['avg_temp_c'] * row['precipitation_mm']
        row['avg_temp_c_x_sunshine_hours'] = row['avg_temp_c'] * row['sunshine_hours']
        row['precipitation_mm_x_sunshine_hours'] = row['precipitation_mm'] * row['sunshine_hours']
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

        # 约束在历史分位数范围内（基于广西统计年鉴 2015-2024 年数据，覆盖全部7市）
        lo = 3.87
        hi = 6.74
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
                m = self.model.__class__(**self.model.get_params()) if hasattr(
                    self.model, 'get_params') else self.model.__class__()
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
                predictions.append(pred)
            except Exception:
                continue

        if len(predictions) < 50:
            return {'point': point, 'ci_lower': point, 'ci_upper': point,
                    'method': 'insufficient_bootstrap'}

        predictions = np.array(predictions)
        return {
            'point': point,
            'ci_lower': float(np.percentile(predictions, 2.5)),
            'ci_upper': float(np.percentile(predictions, 97.5)),
            'bootstrap_mean': float(np.mean(predictions)),
            'bootstrap_std': float(np.std(predictions)),
            'n_bootstrap': len(predictions),
            'method': 'bootstrap'
        }

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
    def _get_gwp(self, gas_pattern, default):
        """从IPCC数据中按气体名称模式读取GWP100值"""
        row = self.ipcc_df[
            self.ipcc_df['emission_factor_unit'].str.contains(gas_pattern, na=False)
        ]
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
        ch4_gwp = self._get_gwp('CH4', default=28)
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
        n2o_gwp = self._get_gwp('N2O', default=273)
        ch4_gwp = self._get_gwp('CH4', default=27)

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
        n2o_from_diesel = diesel_l * n2o_per_l * n2o_gwp

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
        """计算三类方案的净收益

        传统模式(traditional):
            蔗叶焚烧 + 滤泥填埋 + 糖蜜直接出售 + 蔗渣锅炉燃料

        基础循环(circular_basic):
            蔗叶饲料化 + 滤泥有机肥 + 糖蜜直接出售 + 蔗渣锅炉燃料
            （低投入循环利用，适合小农户）

        最优循环(circular_optimal):
            蔗叶生物质颗粒 + 滤泥有机肥 + 糖蜜深加工 + 蔗渣环保餐具
            （最高附加值循环利用，适合糖企/合作社，对标来宾27亿环保餐具产业）
        """
        schemes = {
            'traditional': 0.0,
            'circular_basic': 0.0,
            'circular_optimal': 0.0
        }

        for bp_name, bp_data in economic_results.items():
            if bp_name == 'sugarcane_leaf':
                schemes['traditional'] += (
                    bp_data['burn']['revenue'] - bp_data['burn']['cost'])
                schemes['circular_basic'] += (
                    bp_data['animal_feed']['revenue'] -
                    bp_data['animal_feed']['cost'])
                schemes['circular_optimal'] += (
                    bp_data['biomass_pellet']['revenue'] -
                    bp_data['biomass_pellet']['cost'])

            elif bp_name == 'filter_mud':
                schemes['traditional'] += (
                    bp_data['landfill']['revenue'] - bp_data['landfill']['cost'])
                schemes['circular_basic'] += (
                    bp_data['organic_fertilizer']['revenue'] -
                    bp_data['organic_fertilizer']['cost'])
                schemes['circular_optimal'] += (
                    bp_data['organic_fertilizer']['revenue'] -
                    bp_data['organic_fertilizer']['cost'])

            elif bp_name == 'molasses':
                schemes['traditional'] += (
                    bp_data['direct_sale']['revenue'] -
                    bp_data['direct_sale']['cost'])
                schemes['circular_basic'] += (
                    bp_data['direct_sale']['revenue'] -
                    bp_data['direct_sale']['cost'])
                schemes['circular_optimal'] += (
                    bp_data['deep_processed']['revenue'] -
                    bp_data['deep_processed']['cost'])

            elif bp_name == 'bagasse':
                schemes['traditional'] += (
                    bp_data['boiler_fuel']['revenue'] -
                    bp_data['boiler_fuel']['cost'])
                schemes['circular_basic'] += (
                    bp_data['boiler_fuel']['revenue'] -
                    bp_data['boiler_fuel']['cost'])
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
                 carbon_price=85, country='China', co_firing_ratio=1.0):
        """
        多目标优化：收益最大化 + 碳排放最小化

        Args:
            co_firing_ratio: 生物质掺烧比（0-1），1.0=理论最优，0.3=行业标准

        Returns:
            dict: {'optimal': {...}, 'all_schemes': [...]}
        """
        # 经济效益
        economic = self.economic_calc.calculate_byproduct_value(
            byproduct_quantities, country)
        net_benefit = self.economic_calc.calculate_net_benefit(economic)

        # 碳排放
        leaf_qty = byproduct_quantities.get('sugarcane_leaf', {}).get('quantity', 0)
        burning = self.carbon_calc.calculate_burning_emission(leaf_qty)
        substitution = self.carbon_calc.calculate_biomass_substitution(
            leaf_qty, co_firing_ratio=co_firing_ratio)

        # 循环基础方案碳排放 = 避免焚烧排放 × 90%（扣除10%基础加工能耗）
        # 方法依据：基础循环（颗粒燃料+有机肥+直销）避免了田间焚烧，
        # 但颗粒压制、运输等环节仍有少量化石能源消耗，按生物质全生命周期
        # LCA研究（Cherubini 2009, IPCC 2011），加工能耗约占减排量的8-15%
        circular_basic_carbon = -burning['co2_equivalent_kg'] * 0.9
        carbon_map = {
            'traditional': burning['co2_equivalent_kg'],
            'circular_basic': circular_basic_carbon,
            'circular_optimal': -substitution['carbon_reduction_kg']
        }

        scheme_names = ['traditional', 'circular_basic', 'circular_optimal']
        benefits = [net_benefit[s] for s in scheme_names]
        carbons = [carbon_map[s] for s in scheme_names]

        # Min-max 标准化
        min_b, max_b = min(benefits), max(benefits)
        min_c, max_c = min(carbons), max(carbons)

        schemes = []
        for scheme_name, benefit, carbon_emission in zip(
                scheme_names, benefits, carbons):

            # 碳交易收益（仅覆盖能源相关排放，不含农业N₂O）
            # 排放>0 需购买配额（负收益），排放<0 获得碳信用（正收益）
            carbon_revenue = -(carbon_emission / 1000) * carbon_price
            total_benefit = benefit + carbon_revenue  # 含碳收益的综合净收益

            if max_b != min_b:
                benefit_score = (benefit - min_b) / (max_b - min_b)
            else:
                benefit_score = 0.5

            if max_c != min_c:
                carbon_score = (max_c - carbon_emission) / (max_c - min_c)
            else:
                carbon_score = 0.5

            benefit_score = max(0.0, min(1.0, benefit_score))
            carbon_score = max(0.0, min(1.0, carbon_score))

            total_score = (self.benefit_weight * benefit_score +
                           self.carbon_weight * carbon_score)

            schemes.append({
                'name': scheme_name,
                'net_benefit': benefit,
                'total_benefit': total_benefit,
                'carbon_emission_kg': carbon_emission,
                'carbon_revenue': carbon_revenue,
                'total_score': total_score,
                'benefit_score': benefit_score,
                'carbon_score': carbon_score
            })

        schemes.sort(key=lambda x: x['total_score'], reverse=True)

        # 滤泥填埋碳排放（仅traditional方案涉及）
        filter_mud_qty = byproduct_quantities.get('filter_mud', {}).get('quantity', 0)
        landfill = self.carbon_calc.calculate_landfill_emission(filter_mud_qty)

        # 附加碳排放明细到每个方案
        for s in schemes:
            if s['name'] == 'traditional':
                s['landfill_ch4_kg'] = landfill['co2_equivalent_kg']
            else:
                s['landfill_ch4_kg'] = 0.0

        return {
            'optimal': schemes[0],
            'all_schemes': schemes,
            'landfill_emission': landfill
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
        self._fao_yield基准 = self._load_fao_yield_baseline()

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
                     scenario: str = 'optimal') -> dict:
        """
        运行完整决策流程

        Args:
            scenario: 'optimal'(100%煤炭替代) | 'realistic'(30%掺烧,行业标准)
        """
        if carbon_price is None:
            carbon_price = get_default_carbon_price()

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
                    'method': 'bootstrap_200'
                }
            except Exception:
                pass
        else:
            fao_info = self._fao_baseline_detail.get(country, {})
            yield_per_mu = self._fao_yield基准.get(country, 5.0)
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
            co_firing_ratio=co_firing)

        logger.info(
            "决策完成: country=%s, yield=%.2f, optimal=%s, benefit=%.2f, scenario=%s",
            country, yield_per_mu,
            optimization['optimal']['name'],
            optimization['optimal']['net_benefit'],
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
        sunshine=3500,
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
