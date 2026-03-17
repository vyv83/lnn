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
