"""
tests/test_trainer.py
Тесты для core/trainer.py.
"""
import pytest
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from core.trainer import HawkesDataset, LiquidTrainer
from core.model import LiquidHawkesModel
from core.types import TrainConfig, ModelConfig
from core.features import FEATURE_COLS

@pytest.fixture
def sample_features():
    """Синтетические признаки для тестов."""
    N = 2000
    df = pd.DataFrame({
        "event_type": np.random.randint(0, 4, N),
        "side_int": np.random.randint(0, 2, N),
        "price": np.random.uniform(60000, 70000, N),
        "amount": np.random.uniform(0.001, 0.1, N),
        "dt_us": np.random.randint(100, 5000, N),
        "dt_sec": np.random.uniform(0.0001, 0.005, N)
    })
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.random.randn(N)
    return df

def test_dataset_output(sample_features):
    """Проверка форм данных из Dataset."""
    seq_len = 128
    ds = HawkesDataset(sample_features, seq_len=seq_len)
    
    x, dt, y = ds[0]
    
    assert x.shape == (seq_len, len(FEATURE_COLS))
    assert dt.shape == (seq_len,)
    assert y.shape == (seq_len, 3) # buy_trade, sell_trade, liq
    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()

def test_trainer_sl_step(sample_features):
    """Проверка одного шага обучения SL."""
    m_cfg = ModelConfig(cfc_neurons=16, cfc_motor=8)
    model = LiquidHawkesModel(m_cfg)
    t_cfg = TrainConfig(phase1_lr=1e-3, batch_size=2)
    
    trainer = LiquidTrainer(model, t_cfg)
    ds = HawkesDataset(sample_features, seq_len=64)
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    
    # Запомним веса
    old_params = [p.clone() for p in model.parameters()]
    
    loss = trainer.train_epoch_sl(dl)
    
    assert loss > 0
    # Проверим что веса обновились
    new_params = list(model.parameters())
    any_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(old_params, new_params))
    assert any_changed
