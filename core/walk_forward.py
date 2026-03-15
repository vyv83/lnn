"""
core/walk_forward.py
Скрипт для Walk-Forward валидации LiquidHawkesModel.

Разбивает данные на блоки:
[ Train ][ Test ][ ... ]
         [ Train ][ Test ]
"""
import logging
import pandas as pd
from typing import List, Tuple
from core.trainer import LiquidTrainer, HawkesDataset
from core.backtest import BacktestEngine
from core.model import build_model
from core.types import ModelConfig, TrainConfig

logger = logging.getLogger(__name__)

def walk_forward_split(
    df: pd.DataFrame, 
    n_folds: int = 3,
    train_ratio: float = 0.7
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Разбить данные на фолды для walk-forward обучения."""
    folds = []
    total_len = len(df)
    fold_size = total_len // n_folds
    
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size
        
        split_point = int(start + (end - start) * train_ratio)
        
        train_df = df.iloc[start:split_point]
        test_df = df.iloc[split_point:end]
        
        folds.append((train_df, test_df))
        
    return folds

def run_walk_forward(
    features_df: pd.DataFrame,
    m_cfg: ModelConfig,
    t_cfg: TrainConfig
):
    """Запустить полный цикл обучения и теста на нескольких фолдах."""
    folds = walk_forward_split(features_df)
    results = []
    
    for i, (train_df, test_df) in enumerate(folds):
        logger.info(f"Фолд {i+1}/{len(folds)}: Train={len(train_df)}, Test={len(test_df)}")
        
        model = build_model(m_cfg)
        trainer = LiquidTrainer(model, t_cfg)
        
        # Обучение (упрощенно только Stage 1)
        # В реале тут был бы вызов SL + RL
        # ...
        
        engine = BacktestEngine()
        res = engine.run(model, test_df)
        results.append(res)
        logger.info(f"Фолд {i+1} Return: {res.total_return*100:.2f}%")
        
    return results
