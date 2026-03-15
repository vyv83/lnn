"""
tests/test_features.py
Тесты для core/features.py.

Запуск: pytest tests/test_features.py -v
"""
import pytest
import numpy as np
import pandas as pd

from core.features import (
    build_features,
    FEATURE_COLS,
    N_FEATURES,
    FeatureNormalizer,
)


# ─── Фикстуры ───────────────────────────────────────────────────────────────

@pytest.fixture
def events_df() -> pd.DataFrame:
    """Синтетический поток событий для тестов."""
    rng = np.random.default_rng(42)
    N = 2000
    return pd.DataFrame({
        "timestamp_us": np.cumsum(rng.integers(100, 5000, N)),
        "event_type":   rng.integers(0, 4, N),
        "price":        85000 + rng.standard_normal(N) * 200,
        "amount":       rng.exponential(0.05, N).clip(1e-8),
        "side_int":     rng.integers(0, 2, N),
        "dt_us":        rng.integers(100, 5000, N),
    })


@pytest.fixture
def features_df(events_df) -> pd.DataFrame:
    return build_features(events_df)


@pytest.fixture
def X(features_df) -> np.ndarray:
    return features_df[FEATURE_COLS].to_numpy()


# ─── Тесты ──────────────────────────────────────────────────────────────────

def test_feature_count(features_df):
    """Должно быть ровно 17 признаков."""
    assert len(FEATURE_COLS) == 17, f"Ожидается 17, получено {len(FEATURE_COLS)}"
    assert N_FEATURES == 17
    present = [c for c in FEATURE_COLS if c in features_df.columns]
    assert len(present) == 17, f"В DataFrame только {len(present)} из 17 колонок"


def test_no_nan(X):
    """После build_features не должно быть NaN."""
    nan_count = np.isnan(X).sum()
    assert nan_count == 0, f"Найдено {nan_count} NaN в матрице фичей"


def test_no_inf(X):
    """Не должно быть бесконечных значений."""
    inf_count = np.isinf(X).sum()
    assert inf_count == 0, f"Найдено {inf_count} inf в матрице фичей"


def test_dt_sec_positive(features_df):
    """dt_sec должен быть строго положительным (нужен для CfC timespans)."""
    assert (features_df["dt_sec"] > 0).all(), "dt_sec должен быть > 0"


def test_side_binary(features_df):
    """Признак side ∈ {0, 1}."""
    sides = features_df["side"].unique()
    assert set(sides).issubset({0.0, 1.0}), f"side содержит посторонние значения: {sides}"


def test_trade_imbalance_range(features_df):
    """trade_imbalance ∈ [-1, 1]."""
    col = features_df["trade_imbalance"]
    assert col.min() >= -1.0 - 1e-6
    assert col.max() <= 1.0 + 1e-6


def test_imbalance_L1_range(features_df):
    """imbalance_L1 ∈ [-1, 1]."""
    col = features_df["imbalance_L1"]
    assert col.min() >= -1.0 - 1e-6
    assert col.max() <= 1.0 + 1e-6


def test_output_length(events_df, features_df):
    """Длина фичей = длине входного DataFrame."""
    assert len(features_df) == len(events_df)


def test_normalizer_fit_transform(X):
    """После нормализации mean ≈ 0 и std ≈ 1 на train-данных (для нёконстантных признаков)."""
    norm = FeatureNormalizer()
    X_norm = norm.fit_transform(X)

    col_means = X_norm.mean(axis=0)
    col_stds  = X_norm.std(axis=0)

    # Проверяем только ненулевые признаки (синтетика без стакана = константные book-колонки)
    nonconst = norm.std_ > 1e-6

    assert np.abs(col_means[nonconst]).max() < 0.01, \
        f"mean не ≈ 0: {np.abs(col_means[nonconst]).max():.4f}"
    assert np.abs(col_stds[nonconst] - 1.0).max() < 0.01, \
        f"std не ≈ 1: {np.abs(col_stds[nonconst]-1).max():.4f}"


def test_normalizer_no_fit_raises(X):
    """transform() без fit() должен бросать RuntimeError."""
    norm = FeatureNormalizer()
    with pytest.raises(RuntimeError, match="fit"):
        norm.transform(X)


def test_normalizer_train_test_split(X):
    """Нормализатор обучается только на train, применяется на test — без look-ahead."""
    split = len(X) * 4 // 5
    X_train, X_test = X[:split], X[split:]

    norm = FeatureNormalizer()
    norm.fit(X_train)

    X_train_norm = norm.transform(X_train)
    X_test_norm  = norm.transform(X_test)

    # Оба не должны содержать NaN или inf
    assert not np.isnan(X_train_norm).any()
    assert not np.isnan(X_test_norm).any()
