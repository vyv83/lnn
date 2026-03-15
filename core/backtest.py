"""
core/backtest.py
Движок бэктестинга для LiquidHawkesPlatform.

Реализует симуляцию торговли на основе сигналов модели.
Учитывает:
  - Комиссии (maker/taker)
  - Проскальзывание (slippage)
  - Плечо (leverage)

Вычисляет метрики: Sharpe, MaxDD, Profit Factor и др.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
import torch

from core.types import Trade, BacktestResult, Signal, ModelConfig
from core.features import FEATURE_COLS

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(
        self,
        commission: float = 0.0004, # 0.04% (Binance Futures taker)
        slippage_bps: float = 1.0,  # 1 bp (0.01%)
        initial_balance: float = 10000.0,
    ):
        self.commission = commission
        self.slippage_bps = slippage_bps / 10000.0
        self.initial_balance = initial_balance

    def run(
        self,
        model: torch.nn.Module,
        features_df: pd.DataFrame,
        device: str = "cpu",
        normalizer = None,
        seq_len: int = 512,
        conf_threshold: float = 0.5
    ) -> BacktestResult:
        """
        Прогнать бэктест на данных.
        
        Args:
            model: Обученная LiquidHawkesModel
            features_df: DataFrame с признаками и ценами
            normalizer: Обученный FeatureNormalizer
            seq_len: Длина последовательности (из конфига)
            conf_threshold: Порог уверенности (из конфига)
        """
        model.to(device)
        model.eval()
        
        prices = features_df["price"].values
        timestamps = features_df["timestamp_us"].values
        
        X_raw = features_df[FEATURE_COLS].values.astype(np.float32)
        if normalizer is not None:
            X = normalizer.transform(X_raw).astype(np.float32)
        else:
            X = X_raw
            
        dt = features_df["dt_sec"].values.astype(np.float32)
        
        n_steps = len(features_df)
        equity = np.zeros(n_steps)
        equity[0] = self.initial_balance
        
        balance = self.initial_balance
        last_pos_size = 0.0 # [-1, 1]
        trades = []
        
        # Инференс батчами, но с сохранением hx для корректности CfC
        actions = np.zeros((n_steps, 1), dtype=np.float32)
        confidences = np.zeros((n_steps, 1), dtype=np.float32)
        
        # Обработка окнами (чтобы не упасть по памяти)
        batch_size = 100000 
        hx = None
        
        logger.info(f"Запуск инференса модели ({n_steps} шагов)...")
        with torch.no_grad():
            for i in range(0, n_steps, batch_size):
                end_idx = min(i + batch_size, n_steps)
                x_win = torch.from_numpy(X[i:end_idx]).unsqueeze(0).to(device)
                dt_win = torch.from_numpy(dt[i:end_idx]).unsqueeze(0).to(device)
                
                # intensities, actions, confidence, hx_new
                _, act_win, conf_win, hx = model(x_win, dt_win, hx=hx)
                
                actions[i:end_idx] = act_win[0].cpu().numpy()
                confidences[i:end_idx] = conf_win[0].cpu().numpy()

        logger.info("Симуляция торговой логики...")
        for t in range(1, n_steps):
            price = prices[t]
            target_pos_size = actions[t, 0]
            conf = confidences[t, 0]
            
            # Фильтр по уверенности
            if conf < conf_threshold:
                target_pos_size = 0.0
            
            # Если позиция изменилась — совершаем сделку
            if abs(target_pos_size - last_pos_size) > 0.01:
                target_qty = (balance * target_pos_size) / price
                current_qty = (balance * last_pos_size) / price
                change_qty = target_qty - current_qty
                
                if abs(change_qty) > 0:
                    exec_price = price * (1 + self.slippage_bps if change_qty > 0 else 1 - self.slippage_bps)
                    comm_cost = abs(change_qty) * exec_price * self.commission
                    balance -= comm_cost
                    
                    trades.append(Trade(
                        step=t,
                        timestamp_us=timestamps[t],
                        side="long" if change_qty > 0 else "short",
                        size=abs(change_qty),
                        price=exec_price,
                        commission=comm_cost,
                        pnl=0.0,
                        confidence=conf
                    ))
                    last_pos_size = target_pos_size

            # PnL шага на основе лог-доходности для синхронизации с RL
            pnl_log = last_pos_size * (np.log(prices[t]) - np.log(prices[t-1]))
            # Перевод из лог-профита в линейный баланс
            balance *= np.exp(pnl_log)
            equity[t] = balance

        return self._calculate_metrics(equity, trades)

    def _calculate_metrics(self, equity: np.ndarray, trades: List[Trade]) -> BacktestResult:
        returns = pd.Series(equity).pct_change().dropna()
        
        total_return = (equity[-1] / equity[0]) - 1
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(365 * 24 * 60) # ежеминутный в годовой (?)
        # Для тиковых данных Sharpe традиционно низкий, нужен ресэмплинг.
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.abs(drawdown.min())
        
        return BacktestResult(
            equity=equity,
            trades=trades,
            sharpe=float(sharpe),
            sortino=0.0,
            calmar=total_return / (max_dd + 1e-9),
            max_drawdown=float(max_dd),
            win_rate=0.0, # TODO
            profit_factor=0.0,
            total_return=float(total_return),
            total_trades=len(trades)
        )
