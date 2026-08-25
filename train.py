"""
蔗循智策 — 模型训练入口（独立脚本）
====================================
用途：训练产量预测模型并保存到 models/ 目录，与推理（app.py / api.py / test.py）
分离，明确"训练→保存→推理"边界。

与推理的边界约定：
- 训练：运行本脚本，读取 data/ 数据集 -> 训练 -> 保存 models/yield_predictor.pkl + .hash
- 推理：app.py / api.py 通过 warm_start_models() 加载已保存模型，不触发训练

用法：
    python train.py                 # 自动选择最优模型（Ridge/GBRT 对比）
    python train.py --model-type ridge   # 指定模型：ridge / gbrt / auto
    python train.py --force         # 强制重训（默认若模型已存在则跳过）

复现性：
    - 固定随机种子 random_state=42（见 config.json model.random_state）
    - LOOCV 评估，无超参搜索，无信息泄漏
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('train')

# 保证从项目根目录运行（无论从何处调用本脚本）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'yield_predictor.pkl')


def main():
    parser = argparse.ArgumentParser(description='蔗循智策模型训练入口')
    parser.add_argument('--model-type', default='auto',
                        choices=['ridge', 'gbrt', 'auto'],
                        help='模型类型：ridge / gbrt / auto（自动对比选择最优，默认）')
    parser.add_argument('--force', action='store_true',
                        help='强制重训（默认：模型已存在时跳过）')
    args = parser.parse_args()

    from models import load_data, SugarcaneDecisionSystem

    if os.path.exists(MODEL_PATH) and not args.force:
        logger.info('模型已存在: %s，跳过训练（如需重训请加 --force）', MODEL_PATH)
        logger.info('提示：推理入口见 app.py / api.py 的 warm_start_models()')
        return

    logger.info('加载数据集...')
    gx, weather, *_ = load_data()
    logger.info('训练样本: %s 条（%d市×%d年，城市-年份细粒度）',
                len(gx), gx['city'].nunique(), gx['year'].nunique())

    system = SugarcaneDecisionSystem()
    logger.info('开始训练（model_type=%s, LOOCV评估）...', args.model_type)
    metrics = system.train_models(model_type=args.model_type)

    r2 = metrics.get('r2', float('nan'))
    rmse = metrics.get('rmse', float('nan'))
    mae = metrics.get('mae', float('nan'))
    logger.info('训练完成: model=%s, LOOCV-R²=%.4f, RMSE=%.4f, MAE=%.4f, 样本=%s',
                metrics.get('model_name'), r2, rmse, mae,
                metrics.get('loocv_samples'))

    model_file = os.path.join(PROJECT_DIR, 'models', 'yield_predictor.pkl')
    hash_file = os.path.join(PROJECT_DIR, 'models', 'yield_predictor.hash')
    logger.info('模型已保存: %s (+ %s)', model_file, hash_file)
    logger.info('推理入口: 运行 app.py（Web）或 api.py（REST API），自动加载本模型')


if __name__ == '__main__':
    main()
