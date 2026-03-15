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
