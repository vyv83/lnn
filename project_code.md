# Full Project Source Code - Liquid Hawkes

## Directory Tree

```
lnn/
├── config.toml
├── core/
│   ├── __init__.py
│   ├── backtest.py
│   ├── config.py
│   ├── events.py
│   ├── features.py
│   ├── model.py
│   ├── trainer.py
│   ├── types.py
│   └── walk_forward.py
├── pyproject.toml
└── ui/
    ├── __init__.py
    ├── app.py
    └── pages/
        ├── 2_📥_Data.py
        ├── 3_🧠_Model.py
        └── 4_📈_Backtest.py
```

---

### [pyproject.toml](pyproject.toml)
```toml
[project]
name = "liquid-hawkes"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "ncps>=0.0.7",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "plotly>=5.18",
    "streamlit>=1.38",
    "tardis-dev>=2.0",
    "pytest>=7.0",
]

[project.scripts]
liquid-ui = "ui.app:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["core", "data", "ui"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

### [config.toml](config.toml)
```toml
[data]
exchange = "binance-futures"
symbol = "BTCUSDT"
data_dir = "./data/cache"
feature_window = 500
train_ratio = 0.8

[model]
input_size = 17
cfc_neurons = 128
cfc_motor = 8
backbone_units = 64
backbone_layers = 1
seq_len = 512
horizon = 100

[training]
phase1_epochs = 30
phase1_lr = 0.0005
phase2_epochs = 10
phase2_lr = 0.00005
batch_size = 1024

[trading]
commission = 0.0004
slippage = 0.00001
confidence_threshold = 0.5
min_trade_delta = 0.05

[walk_forward]
train_window = 300000
test_window = 100000
step_size = 100000
max_folds = 10

[ui]
refresh_interval_sec = 2
chart_height = 400
theme = "dark"
```

### [core/types.py](core/types.py)
```python
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
```

### [core/config.py](core/config.py)
```python
"""
core/config.py
Загружает config.toml и возвращает типизированные объекты конфигурации.
Использует стандартный tomllib (Python 3.11+) — никаких доп. зависимостей.
"""
import tomllib
import logging
from pathlib import Path
from typing import Optional, Tuple

from core.types import ModelConfig, TrainConfig, DataConfig

logger = logging.getLogger(__name__)

# Путь к конфигу по умолчанию — ищем снизу вверх от текущей директории
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def load_raw(config_path: Optional[Path] = None) -> dict:
    """Загрузить сырой словарь из config.toml."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    logger.info("Конфиг загружен из %s", path)
    return data


def load_model_config(config_path: Optional[Path] = None) -> ModelConfig:
    """Загрузить секцию [model] из config.toml."""
    raw = load_raw(config_path)
    section = raw.get("model", {})
    return ModelConfig(
        input_size=section.get("input_size", 17),
        cfc_neurons=section.get("cfc_neurons", 32),
        cfc_motor=section.get("cfc_motor", 8),
        backbone_units=section.get("backbone_units", 64),
        backbone_layers=section.get("backbone_layers", 1),
        seq_len=section.get("seq_len", 512),
        horizon=section.get("horizon", 100),
    )


def load_train_config(config_path: Optional[Path] = None) -> TrainConfig:
    """Загрузить секции [training] + [trading] из config.toml."""
    raw = load_raw(config_path)
    tr = raw.get("training", {})
    td = raw.get("trading", {})
    return TrainConfig(
        phase1_epochs=tr.get("phase1_epochs", 20),
        phase1_lr=tr.get("phase1_lr", 1e-3),
        phase2_epochs=tr.get("phase2_epochs", 10),
        phase2_lr=tr.get("phase2_lr", 1e-4),
        batch_size=tr.get("batch_size", 32),
        commission=td.get("commission", 0.0004),
        confidence_threshold=td.get("confidence_threshold", 0.5),
    )


def load_data_config(config_path: Optional[Path] = None) -> DataConfig:
    """Загрузить секцию [data] из config.toml."""
    raw = load_raw(config_path)
    section = raw.get("data", {})
    return DataConfig(
        exchange=section.get("exchange", "binance-futures"),
        symbol=section.get("symbol", "BTCUSDT"),
        data_dir=section.get("data_dir", "./data/cache"),
        feature_window=section.get("feature_window", 500),
        train_ratio=section.get("train_ratio", 0.8),
    )


def load_config(config_path: Optional[Path] = None) -> Tuple[ModelConfig, TrainConfig, DataConfig]:
    """Загрузить все конфигурации разом."""
    return (
        load_model_config(config_path),
        load_train_config(config_path),
        load_data_config(config_path)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_cfg = load_model_config()
    train_cfg = load_train_config()
    data_cfg = load_data_config()
    print(f"ModelConfig: {model_cfg}")
    print(f"TrainConfig: {train_cfg}")
    print(f"DataConfig:  {data_cfg}")
    print("core/config.py OK")
```

### [core/__init__.py](core/__init__.py)
```python
"""Пустой файл — core-пакет."""
```

### [core/events.py](core/events.py)
```python
"""
core/events.py
Парсинг 4 типов CSV от Tardis.dev и объединение в единый
хронологический поток событий.

Источники:
  - trades           → event_type=0 (trade)
  - book_snapshot_5  → event_type=1 (book update)
  - derivative_ticker → event_type=2 (derivative)
  - liquidations     → event_type=3 (liquidation)

Выход: список Event, отсортированный по timestamp_us, с вычисленным dt_us.
Сохраняется в Parquet для быстрого повторного чтения.
"""
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from core.types import Event

logger = logging.getLogger(__name__)


# ─── Константы event_type ────────────────────────────────────────────────────
ET_TRADE = 0
ET_BOOK  = 1
ET_DERIV = 2
ET_LIQ   = 3


# ─── Парсеры отдельных CSV ───────────────────────────────────────────────────

def _parse_trades(path: Path) -> pd.DataFrame:
    """Парсинг файла trades CSV.gz → унифицированный DataFrame."""
    df = pd.read_csv(path, compression="gzip")
    df = df[["timestamp", "side", "price", "amount"]].copy()
    df["event_type"] = ET_TRADE
    df["side_int"] = (df["side"] == "sell").astype(int)
    df = df.rename(columns={"timestamp": "timestamp_us"})
    df["timestamp_us"] = df["timestamp_us"].astype(np.int64)
    df["price"] = df["price"].astype(float)
    df["amount"] = df["amount"].astype(float)
    return df[["timestamp_us", "event_type", "price", "amount", "side_int"]]


def _parse_book(path: Path) -> pd.DataFrame:
    """Парсинг book_snapshot_5 → берём лучший bid/ask как price/amount."""
    df = pd.read_csv(path, compression="gzip")
    # Лучший bid (bid[0]) и ask[0] — для unified stream используем mid как price
    bid_p = df["bids[0].price"].astype(float)
    ask_p = df["asks[0].price"].astype(float)
    bid_v = df["bids[0].amount"].astype(float)
    ask_v = df["asks[0].amount"].astype(float)
    out = pd.DataFrame({
        "timestamp_us": df["timestamp"].astype(np.int64),
        "event_type": ET_BOOK,
        "price": (bid_p + ask_p) / 2.0,   # mid price
        "amount": bid_v + ask_v,           # суммарный объём L1
        "side_int": 1,                     # book — нейтральный
    })
    return out


def _parse_deriv(path: Path) -> pd.DataFrame:
    """Парсинг derivative_ticker → mark_price как price."""
    df = pd.read_csv(path, compression="gzip")
    out = pd.DataFrame({
        "timestamp_us": df["timestamp"].astype(np.int64),
        "event_type": ET_DERIV,
        "price": df["mark_price"].astype(float),
        "amount": df["open_interest"].astype(float).fillna(0.0),
        "side_int": 1,
    })
    return out.dropna(subset=["price"])


def _parse_liquidations(path: Path) -> pd.DataFrame:
    """Парсинг liquidations → size как amount."""
    df = pd.read_csv(path, compression="gzip")
    if df.empty:
        return pd.DataFrame(columns=["timestamp_us", "event_type", "price", "amount", "side_int"])
    out = pd.DataFrame({
        "timestamp_us": df["timestamp"].astype(np.int64),
        "event_type": ET_LIQ,
        "price": df["price"].astype(float),
        "amount": df["amount"].astype(float),
        "side_int": (df["side"] == "sell").astype(int),
    })
    return out


# ─── Основная функция ────────────────────────────────────────────────────────

def build_event_stream(
    data_dir: str | Path,
    date: str,
    exchange: str = "binance-futures",
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """
    Объединить все 4 потока в единый хронологический DataFrame событий.

    Args:
        data_dir: Папка с CSV.gz файлами от Tardis.dev
        date:     Дата в формате YYYY-MM-DD
        exchange: Идентификатор биржи
        symbol:   Тикер

    Returns:
        DataFrame с колонками:
          timestamp_us, event_type, price, amount, side_int, dt_us
        Отсортированный по timestamp_us.
    """
    data_dir = Path(data_dir)
    # Tardis.dev сохраняет файлы в структуру: exchange/data_type/YYYY-MM-DD.csv.gz
    ex = exchange
    sym = symbol.lower()

    def _find(data_type: str) -> Path | None:
        pattern = f"{ex}/{data_type}/{sym}_{date}*.csv.gz"
        files = list(data_dir.glob(pattern))
        if not files:
            # Попробуем без жесткой структуры, просто ищем по названию
            pattern2 = f"**/*{data_type}*{date}*.csv.gz"
            files = list(data_dir.glob(pattern2))
        return files[0] if files else None

    parsers = {
        "trades": _parse_trades,
        "book_snapshot_5": _parse_book,
        "derivative_ticker": _parse_deriv,
        "liquidations": _parse_liquidations,
    }

    frames = []
    for data_type, parser in parsers.items():
        path = _find(data_type)
        if path is None:
            logger.warning("Файл не найден: %s для даты %s", data_type, date)
            continue
        logger.info("Парсинг %s (%s)...", data_type, path.name)
        df = parser(path)
        logger.info("  → %d строк", len(df))
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"Нет данных в {data_dir} за {date}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp_us").reset_index(drop=True)

    # Вычисляем dt в микросекундах между последовательными событиями
    combined["dt_us"] = combined["timestamp_us"].diff().fillna(0).astype(np.int64)
    combined.loc[combined["dt_us"] < 0, "dt_us"] = 0

    logger.info(
        "Поток событий: %d событий | %.1f часов данных",
        len(combined),
        (combined["timestamp_us"].iloc[-1] - combined["timestamp_us"].iloc[0]) / 3_600_000_000,
    )
    return combined


def stream_to_events(df: pd.DataFrame) -> Iterator[Event]:
    """Конвертировать DataFrame в поток объектов Event (генератор)."""
    for row in df.itertuples(index=False):
        yield Event(
            timestamp_us=int(row.timestamp_us),
            event_type=int(row.event_type),
            price=float(row.price),
            amount=float(row.amount),
            side=int(row.side_int),
            dt_us=int(row.dt_us),
        )


def save_event_stream(df: pd.DataFrame, output_path: str | Path) -> None:
    """Сохранить поток событий в Parquet для быстрого повторного чтения."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Сохранено: %s (%.1f MB)", output_path, size_mb)


def load_event_stream(path: str | Path) -> pd.DataFrame:
    """Загрузить поток событий из Parquet."""
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    DATA_DIR = "./data/cache"
    DATE = "2026-03-01"

    df = build_event_stream(DATA_DIR, DATE)
    print(f"\nВсего событий: {len(df):,}")
    print(f"Типы:          {df['event_type'].value_counts().to_dict()}")
    print(f"dt_us min/max: {df['dt_us'].min()} / {df['dt_us'].max()}")
    print(df.head(10))

    save_event_stream(df, f"./data/cache/events_{DATE}.parquet")
    print("\ncore/events.py OK")

### [core/features.py](core/features.py)
```python
"""
core/features.py
Feature engineering: 17 признаков из потока событий.

Слой 1 — Trade Flow (5 фичей):
  [0]  log_price       — log(price)
  [1]  log_amount      — log(amount + ε)
  [2]  side            — 0=buy, 1=sell
  [3]  log_dt          — log(dt_сек + ε)  ← КЛЮЧ для CfC timespans
  [4]  trade_imbalance — скользящий (buy-vol - sell-vol) / total за window

Слой 2 — Book State (8 фичей):
  [5]  spread          — ask[0].price - bid[0].price
  [6]  mid_price       — (bid[0] + ask[0]) / 2
  [7]  imbalance_L1    — (bid[0].vol - ask[0].vol) / total ← сильный предиктор
  [8]  imbalance_L5    — (sum bid[0:5] - sum ask[0:5]) / total
  [9]  log_bid_depth5  — log(суммарный bid объём 5 уровней)
  [10] log_ask_depth5  — log(суммарный ask объём 5 уровней)
  [11] book_skew       — 0.6*imbalance_L1 + 0.4*imbalance_L5
  [12] spread_velocity — diff(spread)

Слой 3 — Derivatives Context (4 фичи):
  [13] funding_rate
  [14] oi_change       — pct_change(open_interest) clipped [-0.1, 0.1]
  [15] liq_intensity   — log(rolling sum liquidation amount за window)
  [16] basis           — (mark_price - index_price) / index_price

Нормализация: z-score (mean/std на train, apply на test).
dt передаётся ОТДЕЛЬНО как timespans в CfC (не входит в матрицу фичей).
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "log_price", "log_amount", "side", "log_dt", "trade_imbalance",   # 5
    "spread", "mid_price", "imbalance_L1", "imbalance_L5",             # 4
    "log_bid_depth5", "log_ask_depth5", "book_skew", "spread_velocity", # 4
    "funding_rate", "oi_change", "liq_intensity", "basis",             # 4
]
N_FEATURES = len(FEATURE_COLS)  # должно быть 17
assert N_FEATURES == 17, f"Ожидается 17 фичей, получено {N_FEATURES}"

EPS = 1e-8
WINDOW = 500   # скользящее окно для trade_imbalance и liq_intensity


def _compute_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """Слой 1: Trade Flow — 5 фичей."""
    out = pd.DataFrame(index=df.index)
    out["log_price"] = np.log(df["price"].clip(lower=EPS))
    out["log_amount"] = np.log(df["amount"].clip(lower=EPS))
    out["side"] = df["side_int"].astype(float)  # 0=buy, 1=sell

    # dt в секундах (из микросекунд)
    dt_sec = df["dt_us"] / 1_000_000.0
    out["log_dt"] = np.log(dt_sec.clip(lower=EPS))

    # Trade imbalance: скользящее окно (только trade-события)
    is_trade = df["event_type"] == 0   # bool Series — & работает корректно
    buy_col  = df["side_int"] == 0
    sell_col = df["side_int"] == 1
    buy_vol  = np.where(is_trade & buy_col,  df["amount"], 0.0)
    sell_vol = np.where(is_trade & sell_col, df["amount"], 0.0)

    roll_buy  = pd.Series(buy_vol,  index=df.index).rolling(WINDOW, min_periods=1).sum()
    roll_sell = pd.Series(sell_vol, index=df.index).rolling(WINDOW, min_periods=1).sum()
    total = roll_buy + roll_sell + EPS
    out["trade_imbalance"] = (roll_buy - roll_sell) / total

    return out


def _compute_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Слой 2: Book State — 8 фичей.
    Данные стакана forward-fill-ятся между book_snapshot обновлениями.
    """
    out = pd.DataFrame(index=df.index)

    # Book-события (event_type==1) содержат данные стакана
    # Для остальных событий forward-fill
    book_mask = df["event_type"] == 1

    def _book_col(col: str, fallback: float = np.nan) -> pd.Series:
        """Взять колонку только из book-строк, ffill остальное."""
        s = pd.Series(np.where(book_mask, df.get(col, fallback), np.nan), index=df.index)
        return s.ffill().fillna(fallback)

    bid0_p = _book_col("bid0_price", df["price"].median())
    ask0_p = _book_col("ask0_price", df["price"].median())
    bid0_v = _book_col("bid0_amount", EPS)
    ask0_v = _book_col("ask0_amount", EPS)
    bid_depth5 = _book_col("bid_depth5", EPS)
    ask_depth5 = _book_col("ask_depth5", EPS)
    imb5_val   = _book_col("imbalance_L5_raw", 0.0)

    spread    = (ask0_p - bid0_p).clip(lower=0.0)
    mid_price = (bid0_p + ask0_p) / 2.0
    total_L1  = bid0_v + ask0_v + EPS
    imb_L1    = (bid0_v - ask0_v) / total_L1

    out["spread"]          = spread
    out["mid_price"]       = mid_price
    out["imbalance_L1"]    = imb_L1
    out["imbalance_L5"]    = imb5_val
    out["log_bid_depth5"]  = np.log(bid_depth5.clip(lower=EPS))
    out["log_ask_depth5"]  = np.log(ask_depth5.clip(lower=EPS))
    out["book_skew"]       = 0.6 * imb_L1 + 0.4 * imb5_val
    out["spread_velocity"] = spread.diff().fillna(0.0)

    return out


def _compute_deriv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Слой 3: Derivatives Context — 4 фичи. Forward-fill между обновлениями."""
    out = pd.DataFrame(index=df.index)
    deriv_mask = df["event_type"] == 2

    def _deriv_col(col: str, fallback: float = 0.0) -> pd.Series:
        s = pd.Series(np.where(deriv_mask, df.get(col, fallback), np.nan), index=df.index)
        return s.ffill().fillna(fallback)

    funding_rate = _deriv_col("funding_rate", 0.0)
    open_interest = _deriv_col("open_interest", 1.0)
    mark_price   = _deriv_col("mark_price", df["price"].median())
    index_price  = _deriv_col("index_price", df["price"].median())

    oi_chg = open_interest.pct_change().fillna(0.0).clip(-0.1, 0.1)

    # Liquidation intensity — forward-fill rolling sum
    liq_mask = df["event_type"] == 3
    liq_amt  = pd.Series(np.where(liq_mask, df["amount"], 0.0), index=df.index)
    liq_roll = liq_amt.rolling(WINDOW, min_periods=1).sum().clip(lower=EPS)

    basis = (mark_price - index_price) / (index_price.clip(lower=EPS))

    out["funding_rate"]  = funding_rate
    out["oi_change"]     = oi_chg
    out["liq_intensity"] = np.log(liq_roll)
    out["basis"]         = basis

    return out


def build_features(
    events_df: pd.DataFrame,
    book_df: Optional[pd.DataFrame] = None,
    deriv_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Вычислить все 17 фичей из объединённого потока событий.

    Args:
        events_df: DataFrame из core/events.py (timestamp_us, event_type,
                   price, amount, side_int, dt_us + дополнительные колонки)
        book_df:   Исходный book_snapshot_5 DataFrame (опционально, для
                   точных 5-уровневых данных)
        deriv_df:  Исходный derivative_ticker DataFrame (опционально)

    Returns:
        DataFrame с 17 фичами + колонка dt_sec для передачи в CfC timespans
    """
    df = events_df.copy()

    # Если переданы исходные DataFrames — обогатим данными стакана и деривативов
    if book_df is not None:
        df = _enrich_with_book(df, book_df)
    if deriv_df is not None:
        df = _enrich_with_deriv(df, deriv_df)

    layer1 = _compute_trade_features(df)
    layer2 = _compute_book_features(df)
    layer3 = _compute_deriv_features(df)

    features = pd.concat([layer1, layer2, layer3], axis=1)

    # Сохраняем служебные колонки для обучения и бэктеста
    for col in ["timestamp_us", "event_type", "side_int", "price", "amount"]:
        if col in df.columns:
            features[col] = df[col].values

    # dt в секундах (для CfC timespans — отдельная колонка, НЕ входит в 17)
    features["dt_sec"] = (df["dt_us"] / 1_000_000.0).clip(lower=EPS)

    assert features[FEATURE_COLS].shape[1] == 17, "Нарушен контракт: не 17 фичей!"

    nan_count = features[FEATURE_COLS].isna().sum().sum()
    if nan_count > 0:
        logger.warning("NaN в фичах: %d. Применяем fillna(0).", nan_count)
        features[FEATURE_COLS] = features[FEATURE_COLS].fillna(0.0)

    logger.info("Фичи построены: %d строк × %d фичей", len(features), N_FEATURES)
    return features


def _enrich_with_book(df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    """Добавить точные 5-уровневые данные стакана к потоку событий через merge_asof."""
    book = book_df.copy()
    book["timestamp_us"] = book["timestamp"].astype(np.int64)

    # Агрегируем 5 уровней bid/ask
    def _sum_depth(prefix: str, n: int = 5) -> pd.Series:
        cols = [f"{prefix}[{i}].amount" for i in range(n) if f"{prefix}[{i}].amount" in book.columns]
        return book[cols].sum(axis=1)

    # L5 imbalance
    bid5 = _sum_depth("bids")
    ask5 = _sum_depth("asks")
    total5 = (bid5 + ask5).clip(lower=EPS)
    book["imbalance_L5_raw"] = (bid5 - ask5) / total5
    book["bid_depth5"] = bid5.clip(lower=EPS)
    book["ask_depth5"] = ask5.clip(lower=EPS)
    book["bid0_price"]  = book.get("bids[0].price", np.nan)
    book["ask0_price"]  = book.get("asks[0].price", np.nan)
    book["bid0_amount"] = book.get("bids[0].amount", EPS).clip(lower=EPS)
    book["ask0_amount"] = book.get("asks[0].amount", EPS).clip(lower=EPS)

    book_cols = book[["timestamp_us", "bid0_price", "ask0_price",
                       "bid0_amount", "ask0_amount", "bid_depth5",
                       "ask_depth5", "imbalance_L5_raw"]].sort_values("timestamp_us")

    df_sorted = df.sort_values("timestamp_us")
    merged = pd.merge_asof(df_sorted, book_cols, on="timestamp_us", direction="backward")
    return merged.sort_values("timestamp_us").reset_index(drop=True)


def _enrich_with_deriv(df: pd.DataFrame, deriv_df: pd.DataFrame) -> pd.DataFrame:
    """Добавить данные деривативов через merge_asof."""
    deriv = deriv_df.copy()
    deriv["timestamp_us"] = deriv["timestamp"].astype(np.int64)

    deriv_cols = deriv[["timestamp_us", "funding_rate", "open_interest",
                         "mark_price", "index_price"]].sort_values("timestamp_us")

    df_sorted = df.sort_values("timestamp_us")
    merged = pd.merge_asof(df_sorted, deriv_cols, on="timestamp_us", direction="backward")
    return merged.sort_values("timestamp_us").reset_index(drop=True)


# ─── Нормализация ────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """
    Z-score нормализатор.
    Обучается на train-данных, применяется на test — без look-ahead bias.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        """Вычислить mean и std на train-данных."""
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + EPS
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Применить нормализацию. Гарантирует возвращение float32."""
        if self.mean_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """Сохранить параметры нормализации в .npz файл."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, std=self.std_)
        logger.info(f"Параметры нормализации сохранены в {path}")

    def load(self, path: str) -> None:
        """Загрузить параметры нормализации из .npz файла."""
        data = np.load(path)
        self.mean_ = data["mean"].astype(np.float32)
        self.std_ = data["std"].astype(np.float32)
        logger.info(f"Параметры нормализации загружены из {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Тест с синтетическими данными (нет реальных CSV)
    N = 1000
    rng = np.random.default_rng(42)

    df_fake = pd.DataFrame({
        "timestamp_us": np.cumsum(rng.integers(100, 10000, N)),
        "event_type":   rng.integers(0, 4, N),
        "price":        65000 + rng.standard_normal(N) * 100,
        "amount":       rng.exponential(0.05, N),
        "side_int":     rng.integers(0, 2, N),
        "dt_us":        rng.integers(100, 10000, N),
    })

    features = build_features(df_fake)
    X = features[FEATURE_COLS].to_numpy()

    print(f"Фичей: {X.shape[1]}  (ожидается 17)")
    print(f"NaN:   {np.isnan(X).sum()}")
    print(f"dt_sec медиана: {features['dt_sec'].median():.6f} сек")

    norm = FeatureNormalizer()
    X_norm = norm.fit_transform(X)
    print(f"После нормализации mean≈0: {X_norm.mean(axis=0).max():.4f}")
    print("core/features.py OK")
```

### [core/model.py](core/model.py)
```python
"""
core/model.py
LiquidHawkesModel — CfC нейросеть для моделирования потока ордеров.

Архитектура:
  CfCCell(units=cfc_neurons) с proj_size через CfC + кастомный loop
  → 3 головы:
  - intensity_head  → λ_buy, λ_sell, λ_cancel (Softplus, всегда ≥ 0)
  - action_head     → позиция [-1, +1]         (Tanh)
  - confidence_head → уверенность [0, 1]       (Sigmoid)

Примечание по API ncps 1.0.1:
  Баг в CfC.forward(): ts = timespans[:, t].squeeze() → форма (B,).
  В CfCCell: t_a * ts → reshape error (B, hidden) * (B,).
  Workaround: передаём timespans как (B, L, 1), тогда squeeze() → (B,)...
  но нужен (B, 1) для broadcast. РЕШЕНИЕ: кастомный loop с ts.unsqueeze(-1).

Ключевой момент: timespans (dt между событиями) = сердце «жидкости».
"""
import logging
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ncps.torch import WiredCfCCell
from ncps.wirings import AutoNCP

from core.types import ModelConfig

logger = logging.getLogger(__name__)


class LiquidHawkesModel(nn.Module):
    """
    Liquid Neural Network для моделирования интенсивности потока ордеров.
    Использует CfC (Closed-form Continuous-time) с NCP (Neural Circuit Policy) архитектурой.
    Реализовано через WiredCfCCell + кастомный loop для стабильного broadcasting dt.

    Args:
        cfg: ModelConfig с гиперпараметрами (или None → defaults)
    """

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # NCP Wiring: sensory -> inter -> command -> motor
        self.wiring = AutoNCP(cfg.cfc_neurons, cfg.cfc_motor)
        
        # WiredCfCCell — специально для работы с NCP wiring
        self.rnn_cell = WiredCfCCell(
            input_size=cfg.input_size,
            wiring=self.wiring,
        )

        # 3 головы принимают на вход выходы моторных нейронов (8 штук)
        
        # λ_buy, λ_sell, λ_cancel
        self.intensity_head = nn.Sequential(
            nn.Linear(cfg.cfc_motor, 16), nn.SiLU(),
            nn.Linear(16, 3), nn.Softplus(),
        )
        # Позиция: -1 = full short, +1 = full long
        self.action_head = nn.Sequential(
            nn.Linear(cfg.cfc_motor, 16), nn.SiLU(),
            nn.Linear(16, 1), nn.Tanh(),
        )
        # Уверенность: 0..1
        self.confidence_head = nn.Sequential(
            nn.Linear(cfg.cfc_motor, 8), nn.SiLU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

        logger.info(
            "LiquidHawkesModel (AutoNCP Cell) создана: %d параметров | CfC %d нейронов, %d моторных",
            self.count_parameters(),
            cfg.cfc_neurons,
            cfg.cfc_motor,
        )

    def forward(
        self,
        x: torch.Tensor,
        timespans: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямой проход через кастомный loop.

        Args:
            x:         (batch, seq_len, input_size)
            timespans: (batch, seq_len) или (batch, seq_len, 1)
            hx:        (batch, cfc_neurons) или None

        Returns:
            intensities, actions, confidence, h_n (full state)
        """
        batch_size, seq_len, _ = x.shape

        if timespans.dim() == 3:
            timespans = timespans.squeeze(-1)

        if hx is None:
            # WiredCfCCell internally tracks neurons, but for AutoNCP
            # initial state should be size of total neurons (units)
            hx = torch.zeros(batch_size, self.cfg.cfc_neurons, device=x.device)

        outputs = []
        h = hx
        for t in range(seq_len):
            inp = x[:, t, :]
            ts = timespans[:, t].unsqueeze(-1) # (B, 1)
            # WiredCfCCell.forward returns (motor_outputs, new_hidden_state)
            # motor_outputs shape: (B, wiring.output_dim) where output_dim = cfc_motor
            # new_hidden_state shape: (B, wiring.units) where units = cfc_neurons
            motor_out, h = self.rnn_cell(inp, h, ts)
            outputs.append(motor_out)

        # Stack → (B, L, cfc_motor)
        motor_stack = torch.stack(outputs, dim=1)

        return (
            self.intensity_head(motor_stack),
            self.action_head(motor_stack),
            self.confidence_head(motor_stack),
            h,
        )

    def count_parameters(self) -> int:
        """Подсчитать число обучаемых параметров."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: Optional[ModelConfig] = None) -> LiquidHawkesModel:
    """Фабричная функция для создания модели из конфига."""
    return LiquidHawkesModel(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = ModelConfig()
    model = build_model(cfg)

    B, L = 2, cfg.seq_len
    x = torch.randn(B, L, cfg.input_size)
    dt = torch.randn(B, L, 1) # B, L, 1
    
    # hx must match total neurons in AutoNCP
    hx = torch.zeros(B, cfg.cfc_neurons)

    t0 = time.perf_counter()
    intensities, actions, confidence, h_n = model(x, dt, hx=hx)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"intensities : {intensities.shape}")
    print(f"actions     : {actions.shape}")
    print(f"confidence  : {confidence.shape}")
    print(f"h_n         : {h_n.shape}")
    print(f"Параметров  : {model.count_parameters():,}  (лимит: 50 000)")
    print(f"Inference   : {elapsed_ms:.1f} ms  (лимит: 50 ms)")
    print("core/model.py OK")

### [core/trainer.py](core/trainer.py)
```python
"""
core/trainer.py
Логика обучения LiquidHawkesModel.

Две стадии:
1. Supervised Learning (SL): Модель учится предсказывать интенсивность (λ) 
   событий в будущем окне. Это стабилизирует веса CfC.
2. Reinforcement Learning (RL): Тонкая настройка для максимизации PnL.

Использует timespans для CfC (секунды между событиями).
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List

from core.model import LiquidHawkesModel
from core.types import TrainConfig, ModelConfig
from core.features import FEATURE_COLS

logger = logging.getLogger(__name__)

# ─── Dataset ────────────────────────────────────────────────────────────────

class HawkesDataset(Dataset):
    """
    Dataset для последовательностей событий.
    Возвращает x (features), dt (timespans), y_int (intensities) и y_ret (returns).
    """
    def __init__(
        self, 
        features_df: pd.DataFrame, 
        seq_len: int = 512,
        prediction_window: int = 100,
        normalizer = None
    ):
        self.seq_len = seq_len
        self.predict_window = prediction_window
        
        # Данные
        X_raw = features_df[FEATURE_COLS].values.astype(np.float32)
        if normalizer is not None:
            self.X = normalizer.transform(X_raw).astype(np.float32)
        else:
            self.X = X_raw
            
        self.dt = features_df["dt_sec"].values.astype(np.float32)
        
        # Цели для Stage 1 Supervised Learning
        # 1. Интенсивности (3 головы)
        self.y_int = self._prepare_intensities(features_df)
        
        # 2. Будущие доходности (для action head supervision)
        # horizon 100 as per plan
        prices = features_df["price"].values
        self.y_ret = self._prepare_returns(prices)
        
        # Индексы для сэмплинга
        self.valid_indices = np.arange(seq_len, len(features_df) - prediction_window)

    def _prepare_intensities(self, df: pd.DataFrame) -> np.ndarray:
        """Вычислить целевые интенсивности (λ) для Stage 1."""
        et = df["event_type"].values
        side = df["side_int"].values
        is_buy_trade = (et == 0) & (side == 0)
        is_sell_trade = (et == 0) & (side == 1)
        is_liq = (et == 3)
        
        def roll_sum_future(arr, window):
            cs = np.cumsum(arr)
            res = np.zeros_like(arr, dtype=np.float32)
            res[:-window] = cs[window:] - cs[:-window]
            return res
            
        t_buy = roll_sum_future(is_buy_trade.astype(int), self.predict_window)
        t_sell = roll_sum_future(is_sell_trade.astype(int), self.predict_window)
        t_liq = roll_sum_future(is_liq.astype(int), self.predict_window)
        
        targets = np.stack([t_buy, t_sell, t_liq], axis=1)
        return np.log1p(targets).astype(np.float32)

    def _prepare_returns(self, prices: np.ndarray) -> np.ndarray:
        """Вычислить будущие доходности (log returns) через horizon."""
        log_p = np.log(prices + 1e-8)
        returns = np.zeros_like(log_p, dtype=np.float32)
        # return[i] = log(price[i + horizon] / price[i])
        returns[:-self.predict_window] = log_p[self.predict_window:] - log_p[:-self.predict_window]
        return returns

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx] - self.seq_len
        end_idx = self.valid_indices[idx]
        
        x = self.X[start_idx:end_idx]
        dt = self.dt[start_idx:end_idx]
        y_int = self.y_int[start_idx:end_idx]
        y_ret = self.y_ret[start_idx:end_idx]
        
        return (
            torch.from_numpy(x), 
            torch.from_numpy(dt), 
            torch.from_numpy(y_int), 
            torch.from_numpy(y_ret)
        )

# ─── Trainer ────────────────────────────────────────────────────────────────

class LiquidTrainer:
    def __init__(
        self, 
        model: LiquidHawkesModel, 
        train_cfg: TrainConfig,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.cfg = train_cfg
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=train_cfg.phase1_lr, weight_decay=1e-5)
        self.mse = nn.MSELoss()

    def prepare_phase2(self):
        """Вызвать ОДИН РАЗ перед первой RL-эпохой."""
        for name, param in self.model.named_parameters():
            if any(x in name for x in ["rnn_cell", "proj", "intensity_head"]):
                param.requires_grad = False
            else:
                param.requires_grad = True
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=self.cfg.phase2_lr, weight_decay=1e-5)

    def train_epoch_sl(self, dataloader: DataLoader, callback=None) -> float:
        """Одна эпоха Supervised обучения (Stage 1) — по Master Plan."""
        self.model.train()
        total_loss = 0
        n_batches = len(dataloader)
        
        for batch_idx, (x, dt, y_int, y_ret) in enumerate(dataloader):
            x, dt, y_int, y_ret = x.to(self.device), dt.to(self.device), y_int.to(self.device), y_ret.to(self.device)
            
            self.optimizer.zero_grad()
            dt = torch.clamp(dt, 1e-6, 10.0)
            
            # Предсказания: (B, L, 3), (B, L, 1), (B, L, 1)
            pred_int, pred_act, pred_conf, _ = self.model(x, dt)
            
            # 1. Intensity Loss: MSE(log(pred), target)
            # Модель уже выдает Softplus, берем log для стабильности
            loss_int = self.mse(torch.log1p(pred_int), y_int)
            
            # 2. Direction Loss: MSE(action, tanh(ret * 100))
            target_act = torch.tanh(y_ret.unsqueeze(-1) * 100.0)
            loss_dir = self.mse(pred_act, target_act)
            
            # 3. Confidence Weighted Loss: conf * (act - target)^2
            error_sq = (pred_act - target_act).pow(2)
            loss_conf = (pred_conf * error_sq.detach()).mean()
            
            # Total Loss = direction_loss + 0.3 * intensity_loss + 0.5 * confidence_weighted_loss
            loss = loss_dir + 0.3 * loss_int + 0.5 * loss_conf
            
            loss_val = loss.item()
            if np.isnan(loss_val) or np.isinf(loss_val):
                raise ValueError("NaN/Inf detected in loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Grad clip 1.0 for SL
            self.optimizer.step()
            total_loss += loss_val
            
            if callback:
                callback(batch_idx, n_batches, loss_val)
                
        return total_loss / n_batches

    def train_epoch_rl(self, dataloader: DataLoader, callback=None) -> float:
        """Одна эпоха Reinforcement Learning (Stage 2) — REINFORCE.
        Перед первым использованием вызвать prepare_phase2().
        """
        self.model.train()
        total_reward = 0
        n_batches = len(dataloader)
        
        for batch_idx, (x, dt, _, y_ret) in enumerate(dataloader):
            x, dt, y_ret = x.to(self.device), dt.to(self.device), y_ret.to(self.device)
            
            self.optimizer.zero_grad()
            dt = torch.clamp(dt, 1e-6, 10.0)
            
            _, actions, confidence, _ = self.model(x, dt)
            
            # PnL шага
            pnl = actions.squeeze(-1) * y_ret
            
            # 3. Штраф за изменение позиции (комиссия) — по плану
            # Считаем разницу между соседними тиками в батче
            action_diff = torch.abs(actions[:, 1:] - actions[:, :-1])
            # Применяем штраф к доходности соответствующего тика
            pnl[:, 1:] -= action_diff.squeeze(-1) * self.cfg.commission
            
            # REINFORCE style Loss: максимизируем PnL, взвешенный по уверенности
            loss = -(pnl * confidence.squeeze(-1)).mean()
            
            reward_val = pnl.mean().item()
            if np.isnan(loss.item()):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) # Grad clip 0.5 for RL
            self.optimizer.step()
            
            total_reward += reward_val
            if callback:
                callback(batch_idx, n_batches, reward_val)
            
        return total_reward / n_batches

    def save_checkpoint(self, path: str, epoch: int = 0, history: list = None, metadata: dict = None):
        """Сохранить веса модели, состояние оптимизатора, историю и метаданные."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'history': history or [],
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f"Чекпоинт сохранен: {path} (эпоха {epoch})")

    def load_checkpoint(self, path: str) -> dict:
        """Загрузить чекпоинт."""
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        except Exception as e:
            logger.warning(f"Ошибка загрузки весов (вероятно, смена архитектуры): {e}")
            
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass
        logger.info(f"Загружен чекпоинт: {path}")
        return checkpoint

    @staticmethod
    def get_checkpoint_info(path: str) -> dict:
        """Безопасно извлечь метаданные без загрузки в модель."""
        try:
            checkpoint = torch.load(path, map_location="cpu")
            info = {
                "epoch": checkpoint.get("epoch", 0),
                "history_len": len(checkpoint.get("history", [])),
                "metadata": checkpoint.get("metadata", {})
            }
            return info
        except Exception as e:
            return {"error": str(e)}
```

### [core/backtest.py](core/backtest.py)
```python
"""
core/backtest.py
Движок бэктестинга — симуляция торговой логики на истории.
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass
from core.types import Trade, BacktestResult, Signal
from core.features import FEATURE_COLS

class BacktestEngine:
    def __init__(self, commission=0.0004, slippage=0.0):
        self.commission = commission
        self.slippage = slippage

    def run(
        self, 
        model: torch.nn.Module, 
        features_df: pd.DataFrame,
        confidence_threshold: float = 0.5,
        batch_size: int = 4096,
        device: str = "cpu",
        callback = None
    ) -> BacktestResult:
        model.eval()
        model.to(device)
        
        X = features_df[FEATURE_COLS].values.astype(np.float32)
        dt = features_df["dt_sec"].values.astype(np.float32)
        prices = features_df["price"].values.astype(np.float32)
        timestamps = features_df["timestamp_us"].values
        
        n = len(X)
        equity = np.ones(n)
        trades = []
        
        position = 0.0 # -1 to 1
        balance = 1.0
        
        # CfC Hidden State
        hx = torch.zeros(1, model.cfg.cfc_neurons).to(device)
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                end_i = min(i + batch_size, n)
                
                # Inference (по одному шагу для честности скрытого состояния)
                # В оптимизированном виде можно батчами, если seq_len=1
                x_batch = torch.from_numpy(X[i:end_i]).unsqueeze(0).to(device) # (1, B, F)
                dt_batch = torch.from_numpy(dt[i:end_i]).unsqueeze(0).to(device) # (1, B)
                
                _, actions, confidence, hx = model(x_batch, dt_batch, hx=hx)
                
                # (B,)
                actions = actions.squeeze().cpu().numpy()
                confidence = confidence.squeeze().cpu().numpy()
                
                if actions.ndim == 0: # single step case
                    actions = np.array([actions])
                    confidence = np.array([confidence])

                for j in range(len(actions)):
                    idx = i + j
                    if idx == 0: continue
                    
                    price = prices[idx]
                    prev_price = prices[idx-1]
                    
                    # 1. Update Equity based on current position
                    ret = (price / prev_price) - 1.0
                    pnl = position * ret
                    balance *= (1.0 + pnl)
                    equity[idx] = balance
                    
                    # 2. Decision logic
                    target_pos = actions[j]
                    conf = confidence[j]
                    
                    if conf >= confidence_threshold:
                        # Если уверенность выше порога — меняем позицию
                        new_pos = target_pos
                    else:
                        # Иначе — закрываемся (или сидим в кэше)
                        new_pos = 0.0
                        
                    # 3. Handle Trades & Commissions
                    if abs(new_pos - position) > 1e-5:
                        change = abs(new_pos - position)
                        comm = change * self.commission
                        balance -= comm # вычитаем комиссию из баланса
                        equity[idx] = balance
                        
                        trades.append(Trade(
                            step=idx,
                            timestamp_us=timestamps[idx],
                            side="long" if new_pos > position else "short",
                            size=change,
                            price=price,
                            commission=comm,
                            pnl=0.0, # pnl считается накопленным
                            confidence=float(conf)
                        ))
                        position = new_pos
                
                if callback:
                    callback(end_i, n)

        return self._calculate_metrics(equity, trades)

    def _calculate_metrics(self, equity: np.ndarray, trades: List[Trade]) -> BacktestResult:
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] / equity[0]) - 1.0
        total_trades = len(trades)
        
        if len(returns) < 2:
            return BacktestResult(equity, trades, 0, 0, 0, 0, 0, 0, total_return, total_trades)
            
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365 * 24 * 60) # Упрощенно к году
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)
        
        # Остальные метрики
        win_rate = 0.0 # Нужно считать по закрытым сделкам
        profit_factor = 1.0
        
        return BacktestResult(
            equity=equity,
            trades=trades,
            sharpe=float(sharpe),
            sortino=0.0,
            calmar=0.0,
            max_drawdown=float(max_dd),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=float(total_return),
            total_trades=total_trades
        )

### [core/walk_forward.py](core/walk_forward.py)
```python
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
```

### [ui/app.py](ui/app.py)
```python
"""
ui/app.py
Точка входа Streamlit — навигация и статус проекта.
Запуск: streamlit run ui/app.py

Это заглушка Фазы 1. UI будет наполняться по мере реализации Фаз 2-6.
"""
import streamlit as st
from pathlib import Path

# ─── Конфигурация страницы ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Liquid Hawkes",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Статус проекта ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent


def check_status() -> dict:
    """Проверить, что готово в проекте."""
    return {
        "config":  (PROJECT_ROOT / "config.toml").exists(),
        "model":   (PROJECT_ROOT / "core" / "model.py").exists(),
        "data":    any((PROJECT_ROOT / "data" / "cache").glob("*.csv.gz")),
        "trained": any((PROJECT_ROOT / "models").glob("*.pt")),
        "results": any((PROJECT_ROOT / "results").glob("*.json")),
    }


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=64)
    st.title("Liquid Hawkes")
    st.caption("BTC Perpetual · Binance Futures")
    st.divider()

    status = check_status()
    st.subheader("Статус проекта")
    st.write("✅ Конфиг" if status["config"] else "❌ Конфиг")
    st.write("✅ Модель" if status["model"] else "❌ Модель")
    st.write("✅ Данные" if status["data"] else "⬜ Данные (Фаза 2)")
    st.write("✅ Обучена" if status["trained"] else "⬜ Обучена (Фаза 3)")
    st.write("✅ Backtest" if status["results"] else "⬜ Backtest (Фаза 4)")

# ─── Главная страница ─────────────────────────────────────────────────────────
st.title("💧 Liquid Hawkes Platform")
st.subheader("CfC Neural Network · Order Flow Intensity Modeling")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Модель**: CfC + AutoNCP\n\n17 фичей → λ_buy, λ_sell, λ_cancel")
with col2:
    st.info("**Данные**: Tardis.dev\n\nTrades · Book · Derivatives · Liquidations")
with col3:
    st.info("**Стратегия**: Intensity modeling\n\nHorizon: 100 событий · Sharpe target: 2.0")

st.divider()
st.subheader("Roadmap")

phases = [
    ("✅", "Фаза 1 — MVP", "Модель + конфиг + тесты"),
    ("⬜", "Фаза 2 — Данные", "Tardis.dev + парсинг + 17 фичей"),
    ("⬜", "Фаза 3 — Обучение", "Supervised + RL Fine-Tuning"),
    ("⬜", "Фаза 4 — Backtest", "Walk-Forward валидация"),
    ("⬜", "Фаза 5 — Нейроны", "Визуализация CfC состояний"),
    ("⬜", "Фаза 6 — Live", "Binance WebSocket · Paper Trading"),
]

for icon, name, desc in phases:
    with st.expander(f"{icon} {name}"):
        st.write(desc)

st.divider()
st.caption("Используй меню слева для навигации по страницам (появятся по мере реализации фаз).")

### [ui/__init__.py](ui/__init__.py)
```python
"""Пустой файл — ui-пакет."""
```

### [ui/pages/2_📥_Data.py](ui/pages/2_📥_Data.py)
```python
"""
ui/pages/2_📥_Data.py
Страница управления данными: загрузка, парсинг, инспекция признаков.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import datetime
import logging

from core.config import load_config
from data.download import download, list_downloaded
from core.events import build_event_stream, save_event_stream, load_event_stream
from core.features import build_features, FEATURE_COLS

st.set_page_config(page_title="Data Management", page_icon="📥", layout="wide")

st.title("📥 Управление данными")

DATA_DIR = Path("./data/cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Sidebar: Загрузка ───────────────────────────────────────────────────────
st.sidebar.header("Загрузить новые данные")
default_date = datetime.date(2026, 3, 1)
target_date = st.sidebar.date_input("Дата (1-е число для бесплатного доступа)", default_date)
api_key = st.sidebar.text_input("Tardis API Key (опционально)", type="password")

if st.sidebar.button("Скачать данные"):
    with st.spinner(f"Загрузка BTCUSDT за {target_date}..."):
        try:
            from_date = str(target_date)
            to_date = str(target_date + datetime.timedelta(days=1))
            files = download(from_date, to_date, str(DATA_DIR), api_key=api_key or None)
            st.sidebar.success(f"Скачано {len(files)} файлов")
        except Exception as e:
            st.sidebar.error(f"Ошибка: {e}")

# ─── Main: Список файлов ─────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Локальный кэш (CSV.gz)")
    csv_files = list_downloaded(str(DATA_DIR))
    if csv_files:
        df_csv = pd.DataFrame([
            {"Файл": f.name, "Размер (MB)": round(f.stat().st_size / (1024*1024), 2)}
            for f in csv_files if f.name.endswith(".csv.gz")
        ])
        st.table(df_csv)
    else:
        st.info("Нет скачанных CSV файлов")

with col2:
    st.subheader("💎 Обработанные данные (Parquet)")
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if parquet_files:
        df_pq = pd.DataFrame([
            {"Файл": f.name, "Размер (MB)": round(f.stat().st_size / (1024*1024), 2)}
            for f in parquet_files
        ])
        st.table(df_pq)
    else:
        st.info("Нет обработанных Parquet файлов")

# ─── Инспекция признаков ─────────────────────────────────────────────────────
st.divider()
st.subheader("🔬 Инспектор признаков")

available_features = [f.name for f in DATA_DIR.glob("features_*.parquet")]
if available_features:
    selected_file = st.selectbox("Выберите файл признаков", available_features, key="selected_feature_file")
    
    if st.button("📥 Загрузить данные"):
        st.session_state.data_loaded = False  # Сброс перед новой загрузкой
        with st.spinner("Загрузка данных..."):
            file_path = DATA_DIR / selected_file
            st.session_state.df_features = pd.read_parquet(file_path)
            st.session_state.data_loaded = True
            st.success(f"Загружено {len(st.session_state.df_features):,} строк")

    if st.session_state.get("data_loaded"):
        df = st.session_state.df_features
        
        # Статистика
        st.write("Статистика признаков:")
        st.dataframe(df[FEATURE_COLS].describe())
        
        # Визуализация
        st.divider()
        st.subheader("📊 Графики")
        
        # Используем session_state для default выбора, если он еще не задан
        if "feat_to_plot" not in st.session_state:
            st.session_state.feat_to_plot = ["log_price", "trade_imbalance"]

        feat_to_plot = st.multiselect(
            "Выберите признаки для отображения", 
            FEATURE_COLS, 
            default=st.session_state.feat_to_plot,
            key="feat_selector"
        )
        
        if feat_to_plot:
            # Берем сэмпл для скорости
            sample_n = st.slider("Количество событий для отображения", 1000, min(50000, len(df)), 10000)
            plot_df = df.iloc[:sample_n].copy()
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Разделяем на цену и остальные индикаторы
            has_price = "log_price" in feat_to_plot
            other_feats = [f for f in feat_to_plot if f != "log_price"]
            
            if has_price and other_feats:
                # Создаем график с двумя осями Y
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Основная ось: Индикаторы
                for f in other_feats:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[f], name=f), secondary_y=False)
                
                # Вторичная ось: Цена
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["log_price"], name="log_price (Price)", line=dict(color='gold', width=3)), secondary_y=True)
                
                fig.update_layout(
                    title=f"Динамика: Индикаторы vs log_price (первые {sample_n} точек)",
                    hovermode="x unified",
                    height=600
                )
                fig.update_yaxes(title_text="Индикаторы", secondary_y=False)
                fig.update_yaxes(title_text="log_price", secondary_y=True)
                
            else:
                # Обычный график если выбрано что-то одно (или только цена)
                fig = px.line(plot_df, y=feat_to_plot, title=f"Динамика признаков (первые {sample_n} событий)")
                fig.update_layout(hovermode="x unified", height=500)

            st.plotly_chart(fig, use_container_width=True)
            
            # Корреляция
            if len(feat_to_plot) > 1:
                st.subheader("🔗 Матрица корреляции")
                corr = df[feat_to_plot].corr()
                fig_corr = px.imshow(
                    corr, 
                    text_auto=".2f", 
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    range_color=[-1, 1]
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Выберите хотя бы один признак для отображения графика.")

else:
    st.warning("Сначала обработайте данные (создайте features_*.parquet)")

# Кнопка запуска обработки (если есть CSV но нет Parquet)
if csv_files and st.button("⚙️ Обработать свежие данные за " + str(target_date)):
    with st.status("Обработка данных...") as status:
        st.write("1. Создание потока событий...")
        ev_df = build_event_stream(DATA_DIR, str(target_date))
        save_event_stream(ev_df, DATA_DIR / f"events_{target_date}.parquet")
        
        st.write("2. Генерация признаков...")
        # Упрощенная генерация без полной enrichment для примера в UI
        # (в проде лучше использовать полный скрипт)
        feat_df = build_features(ev_df)
        feat_df.to_parquet(DATA_DIR / f"features_{target_date}.parquet", index=False)
        
        status.update(label="Обработка завершена!", state="complete")
        st.rerun()
```

### [ui/pages/3_��_Model.py](ui/pages/3_🧠_Model.py)
```python
"""
ui/pages/3_🧠_Model.py
Интерфейс обучения модели LiquidHawkesModel.
"""
import streamlit as st
import pandas as pd
import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader

from core.model import LiquidHawkesModel, build_model
from core.trainer import HawkesDataset, LiquidTrainer
from core.types import ModelConfig, TrainConfig
from core.config import load_config
from core.features import FeatureNormalizer, FEATURE_COLS

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Обучение CfC-модели")

DATA_DIR = Path("./data/cache")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Константы путей (Рефакторинг: 3 файла)
MODEL_S1 = MODEL_DIR / "stage1.pth"
MODEL_S2 = MODEL_DIR / "stage2.pth"
NORM_PATH = MODEL_DIR / "normalizer.npz"

# ─── Выбор данных ────────────────────────────────────────────────────────────
available_features = [f.name for f in DATA_DIR.glob("features_*.parquet")]
if not available_features:
    st.warning("Нет доступных признаков. Сначала обработайте данные на странице 📥 Data.")
    st.stop()

selected_file = st.selectbox("Выберите файл признаков для обучения", available_features)

# Получение реального размера файла для слайдера
def get_parquet_row_count(path):
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except:
        return 1_000_000 # fallback

total_rows = get_parquet_row_count(DATA_DIR / selected_file)

# ─── Sidebar: Настройки ──────────────────────────────────────────────────────
st.sidebar.header("Параметры обучения")
m_cfg, t_cfg, _ = load_config()

cfc_neurons = st.sidebar.number_input("CfC Neurons", 8, 1024, m_cfg.cfc_neurons)
cfc_motor = st.sidebar.number_input("CfC Motor", 4, 256, m_cfg.cfc_motor)
batch_size = st.sidebar.number_input("Batch Size", 1, 1024, t_cfg.batch_size)
epochs = st.sidebar.number_input("Stage 1 Epochs (SL)", 1, 100, t_cfg.phase1_epochs)
lr = st.sidebar.number_input("SL Learning Rate", 1e-5, 1e-1, t_cfg.phase1_lr, format="%.5f")
st.sidebar.divider()
st.sidebar.subheader("Stage 2 Settings (RL)")
rl_epochs = st.sidebar.number_input("Stage 2 Epochs (RL)", 1, 100, t_cfg.phase2_epochs)
rl_lr = st.sidebar.number_input("RL Learning Rate", 1e-6, 1e-2, t_cfg.phase2_lr, format="%.6f")

st.sidebar.divider()
resume = st.sidebar.checkbox("Дообучить (Resume)", value=True, help="Загрузить последний чекпоинт перед началом (SL или RL)")

# Динамический слайдер лимита данных c магнитом 500к
default_limit = min(1_000_000, total_rows)
# Ставим min_value=0, чтобы шаги были ровными: 0, 500k, 1m...
data_limit = st.sidebar.slider("Data Limit (rows)", 0, total_rows, default_limit, step=500_000)
st.sidebar.caption(f"Максимум в файле: **{total_rows:,}** строк")

st.sidebar.divider()
st.sidebar.subheader("Hardware")
available_devices = ["cpu"]
if torch.backends.mps.is_available():
    available_devices.append("mps")
if torch.cuda.is_available():
    available_devices.append("cuda")
device = st.sidebar.radio("Устройство для вычислений", available_devices, index=len(available_devices)-1)

# ─── Тренировочный цикл ────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Статус Обучения")
    st.info(f"💡 Текущее устройство: **{device.upper()}**")
    
    # ─── Stage 1: Card ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🚀 Stage 1: Supervised Learning")
        st.caption("Обучение базового понимания рынка (предсказание интенсивности).")
        btn_sl = st.button("Начать Stage 1", use_container_width=True)
        
        # Локальные плейсхолдеры для Stage 1
        s1_metrics = st.columns(2)
        s1_speed = s1_metrics[0].empty()
        s1_loss = s1_metrics[1].empty()
        s1_progress = st.progress(0)
        s1_status = st.empty()
        s1_chart = st.empty()

        if btn_sl:
            with st.status("Обучение Stage 1...", expanded=True) as status:
                st.write("🔧 Подготовка и нормализация данных...")
                df = pd.read_parquet(DATA_DIR / selected_file)
                
                if len(df) > data_limit:
                    df = df.iloc[:data_limit]
                
                norm = FeatureNormalizer()
                norm.fit(df[FEATURE_COLS].values)
                st.write("✅ Z-score нормализация готова.")
                
                st.write(f"📊 Вычисление таргетов для {len(df):,} событий...")
                dataset = HawkesDataset(df, seq_len=m_cfg.seq_len, normalizer=norm)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                st.write(f"🏗️ Сборка модели LNN ({device})...")
                model = build_model(ModelConfig(cfc_neurons=cfc_neurons, cfc_motor=cfc_motor))
                trainer = LiquidTrainer(model, TrainConfig(phase1_lr=lr, batch_size=batch_size), device=device)
                
                losses = []
                start_epoch = 0
                if resume and MODEL_S1.exists():
                    try:
                        checkpoint = trainer.load_checkpoint(str(MODEL_S1))
                        start_epoch = checkpoint.get('epoch', 0) + 1
                        losses = checkpoint.get('history', [])
                        if NORM_PATH.exists():
                            norm.load(str(NORM_PATH))
                        st.write(f"✅ Чекпоинт загружен: **{len(losses)}** эпох истории. Продолжаем с эпохи **{start_epoch+1}**.")
                    except Exception as e:
                        st.error(f"Не удалось загрузить чекпоинт: {e}. Попробуйте начать без Resume.")
                        st.stop()
                elif resume:
                    st.warning("⚠️ Чекпоинт не найден, начинаем с нуля.")
                
                # Инициализация сессии и пре-заполнение графика
                st.session_state.losses = losses
                if losses:
                    s1_chart.line_chart(pd.DataFrame({"SL Loss": losses}))
                
                def sl_callback(batch, total, loss_val):
                    pct = (batch + 1) / total
                    s1_progress.progress(pct)
                    elapsed = time.time() - start_epoch_time
                    batch_time = elapsed / (batch + 1)
                    eta = batch_time * (total - (batch + 1))
                    events_per_sec = batch_size / batch_time if batch_time > 0 else 0
                    
                    s1_speed.metric("Speed", f"{events_per_sec:.0f} ev/s")
                    s1_loss.metric("Loss", f"{loss_val:.4f}")
                    s1_status.text(f"Эпоха {epoch+1}/{epochs} | Батч {batch+1}/{total} | ETA: {eta:.0f}s")

                st.write("🔥 Запуск тренировочного цикла SL...")
                start_train_time = time.time()
                for epoch in range(start_epoch, epochs):
                    status.update(label=f"Обучение SL (Эпоха {epoch+1}/{epochs})...")
                    start_epoch_time = time.time()
                    avg_loss = trainer.train_epoch_sl(dataloader, callback=sl_callback)
                    losses.append(avg_loss)
                    st.session_state.losses = losses
                    s1_chart.line_chart(pd.DataFrame({"SL Loss": losses}))
                    
                    trainer.save_checkpoint(str(MODEL_S1), epoch=epoch, history=losses,
                        metadata={
                            "trained_samples": data_limit if len(df) > data_limit else len(df),
                            "cfc_neurons": cfc_neurons,
                            "cfc_motor": cfc_motor,
                            "batch_size": batch_size,
                            "epochs_s1": epochs,
                            "lr_s1": lr,
                            "source_file": selected_file,
                            "dataset_start_us": int(df["timestamp_us"].iloc[0]),
                            "dataset_end_us": int(df["timestamp_us"].iloc[-1])
                        })
                    norm.save(str(NORM_PATH))
                
                duration = time.time() - start_train_time
                st.success(f"Stage 1 завершена за {duration:.1f} сек")
                status.update(label="Stage 1 завершена!", state="complete")

    st.write("") # Отступ между карточками

    # ─── Stage 2: Card ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 📈 Stage 2: RL Optimization")
        st.caption("Тонкая настройка для максимизации прибыли (PnL).")
        btn_rl = st.button("Начать Stage 2", use_container_width=True)

        # Локальные плейсхолдеры для Stage 2
        s2_metrics = st.columns(2)
        s2_speed = s2_metrics[0].empty()
        s2_reward = s2_metrics[1].empty()
        s2_progress = st.progress(0)
        s2_status = st.empty()
        s2_chart = st.empty()

        if btn_rl:
            with st.status("Оптимизация PnL...", expanded=True) as status:
                st.write("Загрузка данных...")
                df = pd.read_parquet(DATA_DIR / selected_file)
                if len(df) > data_limit:
                    df = df.iloc[:data_limit]
                
                st.write(f"🔧 Настройка RL-трейнера ({device})...")
                model = build_model(ModelConfig(cfc_neurons=cfc_neurons, cfc_motor=cfc_motor))
                trainer = LiquidTrainer(model, TrainConfig(phase2_lr=rl_lr, batch_size=batch_size), device=device)
                
                rewards = []
                start_epoch_rl = 0
                
                if resume and MODEL_S2.exists():
                    try:
                        checkpoint = trainer.load_checkpoint(str(MODEL_S2))
                        start_epoch_rl = checkpoint.get('epoch', 0) + 1
                        rewards = checkpoint.get('history', [])
                        st.write(f"✅ RL-чекпоинт загружен: **{len(rewards)}** эпох истории. Продолжаем с эпохи **{start_epoch_rl+1}**.")
                        if start_epoch_rl >= rl_epochs:
                            st.info(f"✅ Модель уже дообучена до эпохи {start_epoch_rl}. (Цель: {rl_epochs})")
                    except Exception as e:
                        st.warning(f"Не удалось загрузить RL-чекпоинт: {e}. Пробуем базу SL.")
                
                if not rewards:
                    try:
                        trainer.load_checkpoint(str(MODEL_S1))
                        st.write("✅ Базовая SL модель загружена для старта RL.")
                    except Exception as e:
                        st.error(f"Ошибка: Базовая модель SL не найдена! ({e})")
                        st.stop()
                
                norm = FeatureNormalizer()
                if NORM_PATH.exists():
                    norm.load(str(NORM_PATH))
                else:
                    st.warning("⚠️ Файл нормализации не найден. RL может быть крайне нестабильным.")

                st.write(f"📊 Подготовка данных ({len(df):,} событий)...")
                dataset = HawkesDataset(df, seq_len=m_cfg.seq_len, normalizer=norm)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Настройка слоев и оптимизатора перед RL
                if not rewards:
                    # Первый запуск RL — полная подготовка (заморозка + новый optimizer)
                    trainer.prepare_phase2()
                    st.write(f"🔒 Backbone заморожен. Новый optimizer. RL LR: **{rl_lr}**")
                else:
                    # Resume — только заморозка, optimizer уже загружен из чекпоинта
                    for name, param in trainer.model.named_parameters():
                        if any(x in name for x in ["rnn_cell", "proj", "intensity_head"]):
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    st.write(f"🔒 Backbone заморожен. Optimizer восстановлен из чекпоинта.")

                st.session_state.rewards = rewards
                if rewards:
                    s2_chart.line_chart(pd.DataFrame({"RL Reward (PnL)": rewards}))
                
                def rl_callback(batch, total, reward_val):
                    pct = (batch + 1) / total
                    s2_progress.progress(pct)
                    elapsed = time.time() - start_epoch_time
                    batch_time = elapsed / (batch + 1)
                    eta = batch_time * (total - (batch + 1))
                    events_per_sec = batch_size / batch_time if batch_time > 0 else 0
                    s2_speed.metric("Speed", f"{events_per_sec:.0f} ev/s")
                    s2_reward.metric("Avg Reward", f"{reward_val:.6f}")
                    s2_status.text(f"Эпоха {epoch+1}/{rl_epochs} | RL Батч {batch+1}/{total} | ETA: {eta:.0f}s")

                st.write("🚀 Запуск цикла оптимизации PnL...")
                start_train_time_rl = time.time()
                for epoch in range(start_epoch_rl, rl_epochs):
                    status.update(label=f"Обучение RL (Эпоха {epoch+1}/{rl_epochs})...")
                    start_epoch_time = time.time()
                    avg_reward = trainer.train_epoch_rl(dataloader, callback=rl_callback)
                    rewards.append(avg_reward)
                    st.session_state.rewards = rewards
                    s2_chart.line_chart(pd.DataFrame({"RL Reward (PnL)": rewards}))
                    trainer.save_checkpoint(str(MODEL_S2), epoch=epoch, history=rewards,
                    metadata={
                        "trained_samples": data_limit if len(df) > data_limit else len(df),
                        "cfc_neurons": cfc_neurons,
                        "cfc_motor": cfc_motor,
                        "batch_size": batch_size,
                        "epochs_s2": rl_epochs,
                        "lr_s2": rl_lr,
                        "source_file": selected_file,
                        "dataset_start_us": int(df["timestamp_us"].iloc[0]),
                        "dataset_end_us": int(df["timestamp_us"].iloc[-1])
                    })
                
                duration_rl = time.time() - start_train_time_rl
                status.update(label="Stage 2 завершена!", state="complete")
                st.success(f"📈 RL-дообучение завершено! Прошло {duration_rl:.1f} сек.")


with col2:
    st.subheader("Модель и Производительность")
    st.info("Stage 1 учит модель предсказывать интенсивность событий. Это базовый слой 'понимания' рынка.")
    
    st.markdown("""
    ### Архитектура:
    - **Backbone**: MLP для извлечения паттернов из 17 фичей.
    - **Liquid Core**: CfC ячейки, учитывающие время между событиями (`dt`).
    - **Output Heads**:
        1. `Intensity`: Предсказание будущей активности.
        2. `Action`: Позиция [-1, +1].
        3. `Confidence`: Уверенность (для управления риском).
    """)
    
    if "losses" in st.session_state and st.session_state.losses:
        st.write("История Stage 1 (SL Loss):")
        st.line_chart(st.session_state.losses)
    
    if "rewards" in st.session_state and st.session_state.rewards:
        st.write("История Stage 2 (RL Reward):")
        st.line_chart(st.session_state.rewards)

# ─── Список моделей ──────────────────────────────────────────────────────────
st.divider()
st.subheader("💾 Сохранённые модели")
models = sorted(list(MODEL_DIR.glob("*.pth")))
if models:
    model_data = []
    for m in models:
        info = LiquidTrainer.get_checkpoint_info(str(m))
        meta = info.get("metadata", {})
        model_data.append({
            "Имя": m.name,
            "Эпох": f"{info.get('epoch', 0) + 1}",
            "Neurons": meta.get("cfc_neurons", "N/A"),
            "Motor": meta.get("cfc_motor", "N/A"),
            "Batch": meta.get("batch_size", "N/A"),
            "Источник": meta.get("source_file", "N/A"),
            "History": f"{info.get('history_len', 0)} pts",
            "Дата": time.strftime("%Y-%m-%d %H:%M", time.localtime(m.stat().st_mtime))
        })
    st.table(pd.DataFrame(model_data))
else:
    st.info("Нет сохранённых моделей")

### [ui/pages/4_📈_Backtest.py](ui/pages/4_📈_Backtest.py)
```python
"""
ui/pages/4_📈_Backtest.py
Страница бэктестинга: запуск симуляции и анализ результатов.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import torch
import time
import toml
import numpy as np

from core.model import build_model
from core.backtest import BacktestEngine
from core.types import ModelConfig
from core.features import FeatureNormalizer

def display_model_passport(meta):
    if not meta:
        return
    
    import datetime
    def fmt_ts(ts_us):
        if not ts_us: return "N/A"
        return datetime.datetime.fromtimestamp(ts_us / 1_000_000).strftime('%Y-%m-%d %H:%M:%S')

    stage = meta.get('stage', 0)
    stage_name = f"Stage {stage} ({'SL' if stage==1 else 'RL'})" if stage else "Универсальный чекпоинт"
    
    with st.expander(f"📄 Паспорт модели: {stage_name}", expanded=True):
        # Выделяем количество точек обучения крупнее
        pts = meta.get('trained_samples', 0)
        st.metric("Количество точек обучения (Samples)", f"{pts:,}")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"📁 **Файл данных:** `{meta.get('source_file', 'N/A')}`")
            st.write(f"🧠 **Neurons/Motor:** `{meta.get('cfc_neurons', 0)} / {meta.get('cfc_motor', 0)}`")
            st.write(f"🚀 **Batch Size:** `{meta.get('batch_size', 0)}`")
        with col2:
            st.write(f"⏱️ **Начало:** `{fmt_ts(meta.get('dataset_start_us'))}`")
            st.write(f"⏱️ **Конец:** `{fmt_ts(meta.get('dataset_end_us'))}`")
            
            # Показываем именно тот LR и Эпохи, которые соответствуют стадии
            if stage == 1:
                lr = meta.get('lr_s1', 0)
                epochs = meta.get('epochs_s1', 0)
                st.write(f"📈 **SL LR:** `{lr:.6f}`")
                st.write(f"🔄 **SL Epochs:** `{epochs}`")
            elif stage == 2:
                lr = meta.get('lr_s2', 0)
                epochs = meta.get('epochs_s2', 0)
                st.write(f"📈 **RL LR:** `{lr:.6f}`")
                st.write(f"🔄 **RL Epochs:** `{epochs}`")
            else:
                # Fallback для старых/общих метаданных
                lr = meta.get('lr_s2', meta.get('lr_s1', 1e-4))
                st.write(f"📈 **Learning Rate:** `{lr:.6f}`")

st.set_page_config(page_title="Backtest Results", page_icon="📈", layout="wide")

st.title("📈 Результаты бэктестинга")

DATA_DIR = Path("./data/cache")
MODEL_DIR = Path("./models")
NORM_PATH = MODEL_DIR / "normalizer.npz"
CONFIG_PATH = Path("./config.toml")

def load_config():
    return toml.load(CONFIG_PATH)

cfg = load_config()

# ─── Sidebar: Настройки ──────────────────────────────────────────────────────
st.sidebar.header("Конфигурация теста")

available_models = [f.name for f in MODEL_DIR.glob("*.pth")]
if not available_models:
    st.warning("Нет сохранённых моделей. Сначала обучите модель на странице 🧠 Model.")
    st.stop()
selected_model = st.sidebar.selectbox("Выберите модель", available_models)

available_data = [f.name for f in DATA_DIR.glob("features_*.parquet")]
if not available_data:
    st.warning("Нет данных. Сначала обработайте данные на странице 📥 Data.")
    st.stop()
selected_data = st.sidebar.selectbox("Выберите данные", available_data)

# Получение реального размера файла для слайдеров
def get_parquet_row_count(path):
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except:
        return 1_000_000

total_rows_data = get_parquet_row_count(DATA_DIR / selected_data)

# 🚀 Проактивная загрузка метаданных модели
checkpoint_path = MODEL_DIR / selected_model
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model_meta = checkpoint.get('metadata', {})

display_model_passport(model_meta)

# Smart Sync Логика для Split Point
if model_meta:
    # Пытаемся быстро проверить начало файла без полной загрузки
    first_row = pd.read_parquet(DATA_DIR / selected_data, columns=["timestamp_us"]).iloc[0]
    current_start = int(first_row["timestamp_us"])
    
    if model_meta.get('dataset_start_us') == current_start:
        auto_split = model_meta.get('trained_samples', 0)
        # Инициализируем или обновляем состояние ползунка (теперь следим и за файлом данных)
        sync_key = f"{selected_model}_{selected_data}"
        if "split_point_slider" not in st.session_state or st.session_state.get('last_model_sync') != sync_key:
            st.session_state.split_point_slider = auto_split
            st.session_state.last_model_sync = sync_key
        st.info(f"🔗 **Smart Sync:** Модель узнала данные. Обучение закончилось на `{auto_split:,}`.")

st.sidebar.divider()
st.sidebar.subheader("Данные и Разделение")
# Магнит 500к для лимита, по умолчанию - весь файл
data_limit = st.sidebar.number_input("Лимит данных (строк)", 0, total_rows_data, total_rows_data, step=500_000)

# Защита от сброса Split Point при изменении лимита:
if "split_point_slider" in st.session_state:
    st.session_state.split_point_slider = min(st.session_state.split_point_slider, data_limit)

# Используем key для возможности программного изменения + магнит 500к
st.sidebar.slider(
    "Точка разделения (IS/OOS)", 
    0, data_limit, 
    key="split_point_slider",
    step=500_000
)
# Актуальное значение всегда берем из session_state
cur_split = st.session_state.split_point_slider

# Устройство заблокировано на CPU для максимальной скорости (последовательный инференс на Mac быстрее на CPU)
device = "cpu"

st.sidebar.subheader("Параметры рынка")
commission = st.sidebar.number_input("Комиссия (%)", 0.0, 1.0, cfg['trading']['commission']*100, step=0.01) / 100.0
slippage = st.sidebar.number_input("Проскальзывание (bps)", 0, 100, int(cfg['trading']['slippage']*10000))
initial_balance = st.sidebar.number_input("Начальный депозит (USD)", 100, 1000000, 10000)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, cfg['trading']['confidence_threshold'])

# ─── Запуск бэктеста ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("### 🔍 Запуск симуляции")
    st.caption("Проверка обученной модели на исторических данных с учетом комиссий и проскальзывания.")
    
    btn_run = st.button("🚀 Запустить бэктест", use_container_width=True)
    
    if btn_run:
        with st.status("Выполнение бэктеста...", expanded=True) as status:
            st.write("📂 Загрузка и нарезка данных...")
            df_full = pd.read_parquet(DATA_DIR / selected_data)
            df = df_full.head(data_limit).copy()
            # Берем актуальное значение из слайдера
            cur_split = st.session_state.split_point_slider
            st.write(f"📊 Всего строк: {len(df)} (Split: {cur_split:,})")
            
            st.info(f"💡 Вычисления: **{device.upper()}**")
            
            # --- Progress UI ---
            st.divider()
            prog_col1, prog_col2 = st.columns([3, 1])
            bt_progress = prog_col1.progress(0)
            bt_speed = prog_col2.empty()
            bt_status = st.empty()
            
            start_bt_time = time.time()
            
            def backtest_callback(step, total, phase):
                pct = step / total
                bt_progress.progress(pct)
                elapsed = time.time() - start_bt_time
                speed = step / elapsed if elapsed > 0 else 0
                eta = (total - step) / speed if speed > 0 else 0
                
                bt_speed.metric("Speed", f"{speed:.0f} ev/s")
                bt_status.text(f"⚡ Фаза: {phase} | {step:,} / {total:,} | ETA: {eta:.0f}s")

            st.write(f"🏗️ Инициализация модели LNN...")
            m_cfg = ModelConfig(
                cfc_neurons=cfg['model']['cfc_neurons'],
                cfc_motor=cfg['model']['cfc_motor']
            ) 
            model = build_model(m_cfg).to(device)
            
            # Используем уже загруженный чекпоинт
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                st.write(f"✅ Успешно загружен чекпоинт (эпоха {checkpoint.get('epoch', 'N/A')}).")
            else:
                model.load_state_dict(checkpoint) # для старых файлов весов
                st.write("✅ Модель загружена (старый формат весов).")
            
            model.eval()
            
            # Загрузка нормализатора (единый файл)
            norm = None
            if NORM_PATH.exists():
                norm = FeatureNormalizer()
                norm.load(str(NORM_PATH))
                st.write("✅ Единый нормализатор подключен.")
            else:
                st.warning("⚠️ Файл `normalizer.npz` не найден. Работа на сырых данных!")

            engine = BacktestEngine(
                commission=commission,
                slippage_bps=slippage,
                initial_balance=initial_balance
            )
            
            st.write("📉 Симуляция торговых циклов...")
            t_start = time.time()
            result = engine.run(
                model, 
                df, 
                device=device, 
                normalizer=norm,
                seq_len=cfg['model']['seq_len'],
                conf_threshold=conf_threshold,
                callback=backtest_callback
            )
            elapsed = time.time() - t_start
            
            status.update(label=f"Бэктест завершён! ({elapsed:.1f}с)", state="complete")
            
            # ─── Вывод метрик (IS vs OOS) ──────────────────────────────────────────
            st.divider()
            
            equity = result.equity
            is_part = equity[:cur_split]
            oos_part = equity[cur_split:]
            
            def calc_simple_ret(arr):
                if len(arr) < 2: return 0
                return (arr[-1] / arr[0] - 1)
            
            is_ret = calc_simple_ret(is_part)
            oos_ret = calc_simple_ret(oos_part)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🟢 IN-SAMPLE (Trained)")
                st.metric("Return (IS)", f"{is_ret*100:.2f}%")
            with c2:
                st.markdown("##### 🔵 OUT-OF-SAMPLE (Unseen)")
                color = "normal" if oos_ret > 0 else "inverse"
                st.metric("Return (OOS)", f"{oos_ret*100:.2f}%", delta=f"{(oos_ret - is_ret)*100:.1f}% vs IS", delta_color=color)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{result.total_return*100:.2f}%")
            m2.metric("Sharpe Ratio", f"{result.sharpe:.2f}")
            m3.metric("Max Drawdown", f"{result.max_drawdown*100:.2f}%", delta_color="inverse")
            m4.metric("Total Trades", result.total_trades)
            
            # ─── График Equity ────────────────────────────────────────────────────
            st.subheader("Визуализация: Обучение vs Реальность")
            
            sample_size = 20000
            step_val = max(1, len(result.equity) // sample_size)
            idx = range(0, len(result.equity), step_val)
            
            equity_sampled = result.equity[idx]
            price_sampled = df["price"].values[idx]
            steps_sampled = np.array(list(range(len(result.equity))))[idx]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps_sampled, y=equity_sampled, name="Equity (USD)", line=dict(color="#00FFAA", width=2), yaxis="y1"))
            fig.add_trace(go.Scatter(x=steps_sampled, y=price_sampled, name="BTC Price", line=dict(color="rgba(255,255,255,0.2)", width=1), yaxis="y2"))
            
            fig.add_vline(x=cur_split, line_width=2, line_dash="dash", line_color="orange")
            fig.add_annotation(x=cur_split/2 if cur_split > 0 else 0, y=1.05, yref="paper", text="IN-SAMPLE", showarrow=False, font=dict(color="#00FFAA"))
            fig.add_annotation(x=cur_split + (len(equity)-cur_split)/2 if len(equity)>cur_split else cur_split, y=1.05, yref="paper", text="OUT-OF-SAMPLE", showarrow=False, font=dict(color="#00BBFF"))
            
            fig.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                yaxis=dict(title="Equity (USD)", side="left"),
                yaxis2=dict(title="BTC Price", overlaying="y", side="right", showgrid=False),
                margin=dict(l=0, r=0, t=50, b=0),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
```
