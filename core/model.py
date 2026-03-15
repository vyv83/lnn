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
from ncps.torch import CfC, CfCCell

from core.types import ModelConfig

logger = logging.getLogger(__name__)


class LiquidHawkesModel(nn.Module):
    """
    Liquid Neural Network для моделирования интенсивности потока ордеров.

    Args:
        cfg: ModelConfig с гиперпараметрами (или None → defaults)
    """

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # Используем CfCCell напрямую — полный контроль над timespans
        self.rnn_cell = CfCCell(
            input_size=cfg.input_size,
            hidden_size=cfg.cfc_neurons,
            backbone_units=cfg.backbone_units,
            backbone_layers=cfg.backbone_layers,
        )

        # Проекция из скрытого состояния cfc_neurons → cfc_motor
        self.proj = nn.Linear(cfg.cfc_neurons, cfg.cfc_motor)

        # λ_buy, λ_sell, λ_cancel — интенсивность потоков (всегда > 0)
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
            "LiquidHawkesModel создана: %d параметров | CfC %d→%d нейронов",
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
        Прямой проход.

        Args:
            x:         (batch, seq_len, input_size) — нормализованные фичи
            timespans: (batch, seq_len) или (batch, seq_len, 1) — dt в секундах
            hx:        (batch, cfc_neurons) или None

        Returns:
            intensities: (batch, seq_len, 3) — λ_buy, λ_sell, λ_cancel
            actions:     (batch, seq_len, 1) — позиция [-1, +1]
            confidence:  (batch, seq_len, 1) — уверенность [0, 1]
            h_n:         (batch, cfc_neurons) — финальное скрытое состояние
        """
        batch_size, seq_len, _ = x.shape

        # Нормализуем timespans → (B, L)
        if timespans.dim() == 3:
            timespans = timespans.squeeze(-1)

        # Инициализация скрытого состояния
        if hx is None:
            hx = torch.zeros(batch_size, self.cfg.cfc_neurons, device=x.device)

        outputs = []
        h = hx
        for t in range(seq_len):
            inp = x[:, t, :]          # (B, input_size)
            ts = timespans[:, t].unsqueeze(-1)  # (B,) → (B, 1) ← ключ!
            h, _ = self.rnn_cell(inp, h, ts)
            outputs.append(self.proj(h))  # (B, cfc_motor)

        # Stack → (B, L, cfc_motor)
        cfc_out = torch.stack(outputs, dim=1)

        return (
            self.intensity_head(cfc_out),
            self.action_head(cfc_out),
            self.confidence_head(cfc_out),
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
    dt = torch.rand(B, L) * 0.5 + 0.001

    t0 = time.perf_counter()
    intensities, actions, confidence, h_n = model(x, dt)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"intensities : {intensities.shape}")
    print(f"actions     : {actions.shape}")
    print(f"confidence  : {confidence.shape}")
    print(f"h_n         : {h_n.shape}")
    print(f"Параметров  : {model.count_parameters():,}  (лимит: 50 000)")
    print(f"Inference   : {elapsed_ms:.1f} ms  (лимит: 50 ms)")
    print("core/model.py OK")
