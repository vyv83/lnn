"""
tests/test_events.py
Тесты для core/events.py.

Запуск: pytest tests/test_events.py -v

Тесты используют синтетические CSV-файлы в tmpdir — не требуют скачивания.
"""
import gzip
import io
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from core.events import (
    _parse_trades,
    _parse_book,
    _parse_deriv,
    _parse_liquidations,
    stream_to_events,
    ET_TRADE, ET_BOOK, ET_DERIV, ET_LIQ,
)
from core.types import Event


# ─── Фикстуры: синтетические CSV-файлы ──────────────────────────────────────

def _write_gz(content: str, path: Path) -> None:
    """Записать строку в .csv.gz файл."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture
def trades_csv(tmp_path) -> Path:
    p = tmp_path / "trades.csv.gz"
    content = (
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,1000000,1000100,1,buy,85000.0,0.01\n"
        "binance-futures,BTCUSDT,1001000,1001100,2,sell,84990.0,0.02\n"
        "binance-futures,BTCUSDT,1002000,1002100,3,buy,85010.0,0.015\n"
    )
    _write_gz(content, p)
    return p


@pytest.fixture
def book_csv(tmp_path) -> Path:
    p = tmp_path / "book.csv.gz"
    content = (
        "exchange,symbol,timestamp,local_timestamp,"
        "asks[0].price,asks[0].amount,asks[1].price,asks[1].amount,"
        "asks[2].price,asks[2].amount,asks[3].price,asks[3].amount,"
        "asks[4].price,asks[4].amount,"
        "bids[0].price,bids[0].amount,bids[1].price,bids[1].amount,"
        "bids[2].price,bids[2].amount,bids[3].price,bids[3].amount,"
        "bids[4].price,bids[4].amount\n"
        "binance-futures,BTCUSDT,1000500,1000600,"
        "85005.0,1.0,85010.0,0.5,85015.0,0.3,85020.0,0.2,85025.0,0.1,"
        "84995.0,1.0,84990.0,0.5,84985.0,0.3,84980.0,0.2,84975.0,0.1\n"
    )
    _write_gz(content, p)
    return p


@pytest.fixture
def deriv_csv(tmp_path) -> Path:
    p = tmp_path / "deriv.csv.gz"
    content = (
        "exchange,symbol,timestamp,local_timestamp,"
        "funding_timestamp,funding_rate,predicted_funding_rate,"
        "open_interest,last_price,index_price,mark_price\n"
        "binance-futures,BTCUSDT,1000200,1000300,"
        "1001000,0.0001,0.0001,50000.0,85000.0,84990.0,84995.0\n"
    )
    _write_gz(content, p)
    return p


@pytest.fixture
def liq_csv(tmp_path) -> Path:
    p = tmp_path / "liq.csv.gz"
    content = (
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,1001500,1001600,99,sell,84500.0,0.5\n"
    )
    _write_gz(content, p)
    return p


# ─── Тесты парсеров ─────────────────────────────────────────────────────────

def test_parse_trades_schema(trades_csv):
    """Парсер trades возвращает правильные колонки и event_type."""
    df = _parse_trades(trades_csv)
    assert "timestamp_us" in df.columns
    assert "event_type" in df.columns
    assert (df["event_type"] == ET_TRADE).all()
    assert len(df) == 3


def test_parse_trades_side(trades_csv):
    """side_int: buy→0, sell→1."""
    df = _parse_trades(trades_csv)
    assert df.iloc[0]["side_int"] == 0  # buy
    assert df.iloc[1]["side_int"] == 1  # sell


def test_parse_book_schema(book_csv):
    """Парсер book возвращает price как mid-price."""
    df = _parse_book(book_csv)
    assert df.iloc[0]["event_type"] == ET_BOOK
    # mid price = (85005 + 84995) / 2 = 85000
    assert abs(df.iloc[0]["price"] - 85000.0) < 1.0


def test_parse_deriv_schema(deriv_csv):
    """Парсер derivative_ticker возвращает mark_price."""
    df = _parse_deriv(deriv_csv)
    assert df.iloc[0]["event_type"] == ET_DERIV
    assert df.iloc[0]["price"] == pytest.approx(84995.0)


def test_parse_liquidations_schema(liq_csv):
    """Парсер liquidations возвращает event_type=LIQ."""
    df = _parse_liquidations(liq_csv)
    assert df.iloc[0]["event_type"] == ET_LIQ
    assert df.iloc[0]["side_int"] == 1  # sell


def test_parse_liquidations_empty(tmp_path):
    """Пустой файл ликвидаций не должен вызывать ошибку."""
    p = tmp_path / "liq_empty.csv.gz"
    _write_gz(
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n",
        p,
    )
    df = _parse_liquidations(p)
    assert len(df) == 0


# ─── Тест потока событий ─────────────────────────────────────────────────────

def test_stream_to_events(trades_csv):
    """Перевод DataFrame → поток объектов Event."""
    df = _parse_trades(trades_csv)
    # Добавим dt_us
    df = df.copy()
    df["dt_us"] = [0, 1000, 1000]

    events = list(stream_to_events(df))
    assert len(events) == 3
    assert all(isinstance(e, Event) for e in events)
    assert events[0].event_type == ET_TRADE
    assert events[0].side == 0     # buy
    assert events[1].side == 1     # sell
