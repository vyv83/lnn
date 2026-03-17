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
