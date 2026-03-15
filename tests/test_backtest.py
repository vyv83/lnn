"""
tests/test_backtest.py
Тесты для core/backtest.py.
"""
import pytest
import torch
import pandas as pd
import numpy as np
from core.backtest import BacktestEngine
from core.model import LiquidHawkesModel
from core.types import ModelConfig
from core.features import FEATURE_COLS

@pytest.fixture
def sample_data():
    """Синтетические данные для бэктеста."""
    N = 1000
    df = pd.DataFrame({
        "timestamp_us": np.arange(N) * 1000000,
        "price": 60000 + np.cumsum(np.random.randn(N) * 10),
        "dt_sec": np.ones(N) * 1.0,
    })
    for col in FEATURE_COLS:
        df[col] = np.random.randn(N)
    return df

def test_backtest_engine_run(sample_data):
    """Проверка базового прогона бэктеста."""
    m_cfg = ModelConfig(cfc_neurons=8, cfc_motor=4)
    model = LiquidHawkesModel(m_cfg)
    
    engine = BacktestEngine(initial_balance=1000.0)
    result = engine.run(model, sample_data)
    
    assert len(result.equity) == len(sample_data)
    assert result.equity[0] == 1000.0
    assert isinstance(result.total_return, float)
    assert isinstance(result.trades, list)

def test_backtest_metrics(sample_data):
    """Проверка расчета метрик."""
    engine = BacktestEngine()
    # Создаем фиктивный equity
    equity = np.array([100, 110, 105, 120, 115])
    metrics = engine._calculate_metrics(equity, [])
    
    assert metrics.total_return == pytest.approx(0.15)
    assert metrics.max_drawdown > 0
    assert metrics.calmar > 0
