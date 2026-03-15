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
