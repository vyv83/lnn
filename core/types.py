"""
core/types.py
Контракты данных (dataclass-ы) для всего проекта.
Этот файл не импортирует ничего из других модулей проекта.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Event:
    """Одно событие на бирже (тик, стакан, дериватив, ликвидация)."""
    timestamp_us: int        # микросекунды с epoch
    event_type: int          # 0=trade, 1=bid_update, 2=ask_update, 3=deriv, 4=liq
    price: float
    amount: float
    side: int                # 0=buy, 1=sell
    dt_us: int = 0           # микросекунды с предыдущего события


@dataclass
class Signal:
    """Выход модели на одном шаге."""
    action: float            # [-1, +1] позиция (отрицательная = шорт)
    confidence: float        # [0, 1] уверенность модели
    intensity_buy: float     # λ_buy: интенсивность buy-ордеров
    intensity_sell: float    # λ_sell: интенсивность sell-ордеров
    intensity_cancel: float  # λ_cancel: интенсивность отмен


@dataclass
class Trade:
    """Совершённая сделка в бэктесте или live-торговле."""
    step: int
    timestamp_us: int
    side: str                # 'long' | 'short' | 'close'
    size: float
    price: float
    commission: float
    pnl: float
    confidence: float


@dataclass
class BacktestResult:
    """Результат одного прогона бэктестинга."""
    equity: np.ndarray
    trades: list[Trade]
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    total_trades: int


@dataclass
class ModelConfig:
    """Гиперпараметры модели."""
    input_size: int = 17       # 17 фичей: 5 trade + 8 book + 4 derivatives
    cfc_neurons: int = 32      # Общее кол-во нейронов CfC/NCP
    cfc_motor: int = 8         # Моторные нейроны (выход CfC)
    backbone_units: int = 64   # Размер backbone MLP внутри CfC
    backbone_layers: int = 1   # Слоёв в backbone
    seq_len: int = 512         # Длина последовательности событий
    horizon: int = 100         # Горизонт предсказания (событий вперёд)


@dataclass
class TrainConfig:
    """Параметры обучения."""
    phase1_epochs: int = 20
    phase1_lr: float = 1e-3
    phase2_epochs: int = 10
    phase2_lr: float = 1e-4
    batch_size: int = 32
    commission: float = 0.0004
    confidence_threshold: float = 0.5


@dataclass
class DataConfig:
    """Параметры источника данных."""
    exchange: str = "binance-futures"
    symbol: str = "BTCUSDT"
    data_dir: str = "./data/cache"
    feature_window: int = 500
    train_ratio: float = 0.8


if __name__ == "__main__":
    # Демонстрация контрактов
    evt = Event(timestamp_us=1_000_000, event_type=0, price=65000.5, amount=0.01, side=0, dt_us=150)
    sig = Signal(action=0.7, confidence=0.85, intensity_buy=2.3, intensity_sell=1.1, intensity_cancel=0.5)
    cfg = ModelConfig()
    print(f"Event:       {evt}")
    print(f"Signal:      {sig}")
    print(f"ModelConfig: {cfg}")
    print("core/types.py OK")
