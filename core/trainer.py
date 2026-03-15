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
    Возвращает x (features), dt (timespans) и y (targets).
    """
    def __init__(
        self, 
        features_df: pd.DataFrame, 
        seq_len: int = 512,
        prediction_window: int = 100, # сколько событий вперед смотрим для интенсивности
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
        
        # Цели для Supervised Learning (интенсивность)
        # Упрощенно: считаем количество событий типов 0 (trade buy), 0 (trade sell), 3 (liq)
        # в окне prediction_window.
        self.targets = self._prepare_targets(features_df)
        
        # Индексы для сэмплинга (чтобы не выходить за границы)
        self.valid_indices = np.arange(seq_len, len(features_df) - prediction_window)

    def _prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Вычислить целевые интенсивности (λ) для Supervised обучения."""
        logger.info("Вычисление целевых интенсивностей (Stage 1 targets)...")
        # event_type: 0=trade, 3=liq
        # side_int: 0=buy, 1=sell
        et = df["event_type"].values
        side = df["side_int"].values
        
        is_buy_trade = (et == 0) & (side == 0)
        is_sell_trade = (et == 0) & (side == 1)
        is_liq = (et == 3)
        
        # Скользящее окно вперед (суммируем события в будущем)
        # Используем трюк с кумулятивной суммой для скорости
        def roll_sum_future(arr, window):
            cs = np.cumsum(arr)
            res = np.zeros_like(arr, dtype=np.float32)
            res[:-window] = cs[window:] - cs[:-window]
            return res
            
        t_buy = roll_sum_future(is_buy_trade.astype(int), self.predict_window)
        t_sell = roll_sum_future(is_sell_trade.astype(int), self.predict_window)
        t_liq = roll_sum_future(is_liq.astype(int), self.predict_window)
        
        # Нормализуем цели (log(1 + count))
        targets = np.stack([t_buy, t_sell, t_liq], axis=1)
        return np.log1p(targets).astype(np.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx] - self.seq_len
        end_idx = self.valid_indices[idx]
        
        x = self.X[start_idx:end_idx]
        dt = self.dt[start_idx:end_idx]
        y = self.targets[start_idx:end_idx] # предсказываем интенсивность для всей последовательности (RNN mode)
        
        return torch.from_numpy(x), torch.from_numpy(dt), torch.from_numpy(y)

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
        self.criterion = nn.MSELoss()

    def train_epoch_sl(self, dataloader: DataLoader, callback=None) -> float:
        """Одна эпоха Supervised обучения (Stage 1)."""
        self.model.train()
        total_loss = 0
        n_batches = len(dataloader)
        
        for batch_idx, (x, dt, y) in enumerate(dataloader):
            x, dt, y = x.to(self.device), dt.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # forward: intensities, actions, confidence, h_n
            # Клиппинг dt для стабильности ODE (epsilon=1e-6)
            dt = torch.clamp(dt, 1e-6, 10.0)
            
            intensities, _, _, _ = self.model(x, dt)
            
            loss = self.criterion(intensities, y)
            
            # ЗАЩИТА: проверка на NaN
            loss_val = loss.item()
            if np.isnan(loss_val) or np.isinf(loss_val):
                logger.error(f"Взрыв градиентов обнаружен на батче {batch_idx}! Остановка.")
                raise ValueError("NaN/Inf detected in loss")

            loss.backward()
            
            # Clip gradients для CfC (важно!)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            total_loss += loss_val
            
            if callback:
                callback(batch_idx, n_batches, loss_val)
                
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}/{n_batches}, Loss: {loss_val:.6f}")
                
        return total_loss / n_batches

    def save_checkpoint(self, path: str, epoch: int = 0, history: list = None):
        """Сохранить веса модели, состояние оптимизатора и историю метрик."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'history': history or [],
        }
        torch.save(checkpoint, path)
        logger.info(f"Чекпоинт сохранен: {path} (эпоха {epoch})")

    def load_checkpoint(self, path: str) -> dict:
        """Загрузить чекпоинт в формате dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # ЗАЩИТА: проверка весов на NaN после загрузки
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error(f"Параметр {name} содержит NaN/Inf в чекпоинте! Файл поврежден.")
                raise ValueError(f"Checkpoint {path} is corrupted with NaN/Inf")
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                logger.warning(f"Не удалось загрузить состояние оптимизатора: {e}")
        
        logger.info(f"Загружен чекпоинт: {path} (эпоха {checkpoint.get('epoch', 'N/A')})")
        return checkpoint

    def train_epoch_rl(self, dataloader: DataLoader, callback=None) -> float:
        """
        Одна эпоха Reinforcement Learning (Stage 2).
        Цель: максимизация PnL.
        """
        self.model.train()
        total_reward = 0
        n_batches = len(dataloader)
        
        for batch_idx, (x, dt, _) in enumerate(dataloader):
            x, dt = x.to(self.device), dt.to(self.device)
            
            self.optimizer.zero_grad()
            
            # forward
            _, actions, confidence, _ = self.model(x, dt)
            
            # Упрощенная логика PnL для RL:
            # Reward = sign(price_change_future) * action
            # Для этого нам нужны цены из x. 
            # Допустим log_price — первая фича (index 0).
            log_prices = x[:, :, 0]
            # Ценовое изменение на следующем шаге (shift -1)
            price_change = torch.zeros_like(log_prices)
            price_change[:, :-1] = log_prices[:, 1:] - log_prices[:, :-1]
            
            # Награда = Позиция * Изменение цены (в лог-пунктах) - штраф за транзакции
            action_change = torch.abs(actions[:, 1:] - actions[:, :-1])
            transaction_costs = action_change * self.cfg.commission
            
            rewards = actions[:, :-1, 0] * price_change[:, :-1]
            # Вычитаем издержки (только там где был сдвиг)
            rewards[:, :transaction_costs.shape[1]] -= transaction_costs.squeeze(-1)
            
            # Loss = -Reward (минимизируем отрицательную награду)
            # Мы используем действия как вероятности выбора стороны (Policy Gradient)
            # Если action > 0 и цена выросла -> хорошо.
            reward_val = rewards.mean().item()
            loss = -(rewards * confidence[:, :-1, 0]).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_val
            
            if callback:
                callback(batch_idx, n_batches, reward_val)
            
        return total_reward / n_batches

# ─── Stage 2: RL (Placeholder) ──────────────────────────────────────────────

    def train_step_rl(self, x, dt, reward):
        """
        Упрощенный Policy Gradient (Stage 2).
        reward — изменение PnL после действия.
        """
        # TODO: Реализовать PPO или Simple Policy Gradient
        pass
