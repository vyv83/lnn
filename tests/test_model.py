"""
tests/test_model.py
Критические тесты LiquidHawkesModel.

Запуск: pytest tests/test_model.py -v

Тесты проверяют:
  - Правильность форм выходных тензоров
  - Физические ограничения (Softplus ≥ 0, Tanh ∈ [-1,1], Sigmoid ∈ [0,1])
  - Что разные dt дают разный выход (проверка «жидкости»)
  - Перенос скрытого состояния между вызовами
  - Компактность: < 50 000 параметров
  - Скорость inference: < 50ms
"""
import time
import pytest
import torch

from core.types import ModelConfig
from core.model import build_model, LiquidHawkesModel

# ─── Фикстуры ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> LiquidHawkesModel:
    """Единственный экземпляр модели на все тесты в модуле."""
    return build_model(ModelConfig())


@pytest.fixture(scope="module")
def sample_batch(model: LiquidHawkesModel):
    """Случайный батч совместимый с конфигом модели."""
    cfg = model.cfg
    B, L = 4, cfg.seq_len
    x = torch.randn(B, L, cfg.input_size)
    dt = torch.rand(B, L, 1) * 0.5 + 0.001
    return x, dt, B, L


# ─── Тесты ──────────────────────────────────────────────────────────────────

def test_output_shapes(model: LiquidHawkesModel, sample_batch):
    """intensities=(B,L,3), actions=(B,L,1), confidence=(B,L,1)."""
    x, dt, B, L = sample_batch
    intensities, actions, confidence, h_n = model(x, dt)

    assert intensities.shape == (B, L, 3), f"intensities: {intensities.shape}"
    assert actions.shape == (B, L, 1),     f"actions: {actions.shape}"
    assert confidence.shape == (B, L, 1),  f"confidence: {confidence.shape}"


def test_intensity_positive(model: LiquidHawkesModel, sample_batch):
    """Softplus гарантирует λ ≥ 0 (физический смысл: интенсивность)."""
    x, dt, *_ = sample_batch
    intensities, _, _, _ = model(x, dt)
    assert (intensities >= 0).all(), "Интенсивности должны быть ≥ 0"


def test_action_bounded(model: LiquidHawkesModel, sample_batch):
    """Tanh гарантирует action ∈ [-1, +1]."""
    x, dt, *_ = sample_batch
    _, actions, _, _ = model(x, dt)
    assert actions.min() >= -1.0 - 1e-6, f"action.min = {actions.min()}"
    assert actions.max() <= +1.0 + 1e-6, f"action.max = {actions.max()}"


def test_confidence_bounded(model: LiquidHawkesModel, sample_batch):
    """Sigmoid гарантирует confidence ∈ [0, 1]."""
    x, dt, *_ = sample_batch
    _, _, confidence, _ = model(x, dt)
    assert confidence.min() >= 0.0 - 1e-6, f"conf.min = {confidence.min()}"
    assert confidence.max() <= 1.0 + 1e-6, f"conf.max = {confidence.max()}"


def test_different_dt_gives_different_output(model: LiquidHawkesModel):
    """
    КРИТИЧЕСКИЙ ТЕСТ «ЖИДКОСТИ».
    Если разные dt дают одинаковый выход — timespans не работают.
    Это означает что модель = обычная RNN, а не Liquid Network.
    """
    cfg = model.cfg
    B, L = 2, cfg.seq_len
    x = torch.randn(B, L, cfg.input_size)

    dt_fast = torch.full((B, L, 1), 0.001)  # быстрый поток: 1ms между событиями
    dt_slow = torch.full((B, L, 1), 10.0)   # медленный: 10 секунд между событиями

    _, actions_fast, _, _ = model(x, dt_fast)
    _, actions_slow, _, _ = model(x, dt_slow)

    # Выходы должны существенно отличаться
    diff = (actions_fast - actions_slow).abs().mean().item()
    assert diff > 1e-4, (
        f"Разные timespans дали почти одинаковый выход (diff={diff:.2e}). "
        "«Жидкость» не работает!"
    )


def test_hidden_state_persistence(model: LiquidHawkesModel):
    """hx можно передавать между вызовами для сохранения контекста."""
    cfg = model.cfg
    B, L = 2, cfg.seq_len
    x = torch.randn(B, L, cfg.input_size)
    dt = torch.rand(B, L, 1) * 0.5 + 0.001

    # Первый вызов — без hx
    _, _, _, h_n1 = model(x, dt)
    assert h_n1 is not None, "h_n должен быть возвращён"

    # Второй вызов — с переданным hx
    _, _, _, h_n2 = model(x, dt, hx=h_n1)
    assert h_n2 is not None, "h_n2 должен быть возвращён"
    assert h_n2.shape == h_n1.shape, "Форма h_n должна совпадать"


def test_parameter_count(model: LiquidHawkesModel):
    """Компактность: < 50 000 параметров (ключевое свойство LNN)."""
    params = model.count_parameters()
    assert params < 50_000, (
        f"Слишком много параметров: {params:,}. "
        "LNN должна быть компактной (< 50k)."
    )
    print(f"\n  Параметров: {params:,}")


def test_inference_speed(model: LiquidHawkesModel):
    """Inference < 50ms на batch=4, seq_len=512 (CPU)."""
    cfg = model.cfg
    B, L = 4, cfg.seq_len
    x = torch.randn(B, L, cfg.input_size)
    dt = torch.rand(B, L, 1) * 0.5 + 0.001

    # Прогрев
    with torch.no_grad():
        model(x, dt)

    # Замер
    t0 = time.perf_counter()
    with torch.no_grad():
        model(x, dt)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 50, (
        f"Inference слишком медленный: {elapsed_ms:.1f}ms (лимит: 50ms)"
    )
    print(f"\n  Inference: {elapsed_ms:.1f}ms")
