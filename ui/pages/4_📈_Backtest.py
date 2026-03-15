"""
ui/pages/4_📈_Backtest.py
Страница бэктестинга: запуск симуляции и анализ результатов.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import torch

from core.model import build_model
from core.backtest import BacktestEngine
from core.types import ModelConfig
from core.features import FeatureNormalizer

st.set_page_config(page_title="Backtest Results", page_icon="📈", layout="wide")

st.title("📈 Результаты бэктестинга")

DATA_DIR = Path("./data/cache")
MODEL_DIR = Path("./models")
NORM_PATH = MODEL_DIR / "normalizer.npz"
CONFIG_PATH = Path("./config.toml")

import time
import toml
def load_config():
    return toml.load(CONFIG_PATH)

cfg = load_config()

# ─── Sidebar: Настройки ──────────────────────────────────────────────────────
st.sidebar.header("Конфигурация теста")

available_models = [f.name for f in MODEL_DIR.glob("*.pth")]
if not available_models:
    st.warning("Нет сохранённых моделей. Сначала обучите модель на странице 🧠 Model.")
    st.stop()
selected_model = st.sidebar.selectbox("Выберите модель", available_models)

available_data = [f.name for f in DATA_DIR.glob("features_*.parquet")]
if not available_data:
    st.warning("Нет данных. Сначала обработайте данные на странице 📥 Data.")
    st.stop()
selected_data = st.sidebar.selectbox("Выберите данные", available_data)

st.sidebar.divider()
st.sidebar.subheader("Данные и Разделение")
data_limit = st.sidebar.number_input("Лимит данных (строк)", 1000, 10_000_000, 1_000_000, step=100_000)
split_point = st.sidebar.slider("Точка разделения (IS/OOS)", 0, data_limit, int(data_limit * 0.8), step=10_000)
available_devices = ["cpu"]
if torch.backends.mps.is_available():
    available_devices.append("mps")
device = st.sidebar.radio("Устройство для вычислений", available_devices, index=len(available_devices)-1)

st.sidebar.subheader("Параметры рынка")
commission = st.sidebar.number_input("Комиссия (%)", 0.0, 1.0, cfg['trading']['commission']*100, step=0.01) / 100.0
slippage = st.sidebar.number_input("Проскальзывание (bps)", 0, 100, int(cfg['trading']['slippage']*10000))
initial_balance = st.sidebar.number_input("Начальный депозит (USD)", 100, 1000000, 10000)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, cfg['trading']['confidence_threshold'])

# ─── Запуск бэктеста ──────────────────────────────────────────────────────────
# ─── Запуск бэктеста ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("### 🔍 Запуск симуляции")
    st.caption("Проверка обученной модели на исторических данных с учетом комиссий и проскальзывания.")
    
    btn_run = st.button("🚀 Запустить бэктест", use_container_width=True)
    
    if btn_run:
        with st.status("Выполнение бэктеста...", expanded=True) as status:
            st.write("📂 Загрузка и нарезка данных...")
            df_full = pd.read_parquet(DATA_DIR / selected_data)
            df = df_full.head(data_limit).copy()
            st.write(f"📊 Всего строк: {len(df)} (Split: {split_point})")
            
            st.info(f"💡 Вычисления: **{device.upper()}**")
            
            st.write(f"🏗️ Инициализация модели LNN...")
            m_cfg = ModelConfig(
                cfc_neurons=cfg['model']['cfc_neurons'],
                cfc_motor=cfg['model']['cfc_motor']
            ) 
            model = build_model(m_cfg).to(device)
            
            # Универсальная загрузка чекпоинта (старый формат или словарь)
            checkpoint = torch.load(MODEL_DIR / selected_model, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                st.write(f"✅ Успешно загружен чекпоинт (эпоха {checkpoint.get('epoch', 'N/A')}).")
            else:
                model.load_state_dict(checkpoint) # для старых файлов весов
                st.write("✅ Модель загружена (старый формат весов).")
            
            model.eval()
            
            # Загрузка нормализатора (единый файл)
            norm = None
            if NORM_PATH.exists():
                norm = FeatureNormalizer()
                norm.load(str(NORM_PATH))
                st.write("✅ Единый нормализатор подключен.")
            else:
                st.warning("⚠️ Файл `normalizer.npz` не найден. Работа на сырых данных!")

            engine = BacktestEngine(
                commission=commission,
                slippage_bps=slippage,
                initial_balance=initial_balance
            )
            
            st.write("📉 Симуляция торговых циклов...")
            start_time = time.time()
            # Вызов обновленного engine.run с поддержкой hx и оконной обработки
            result = engine.run(
                model, 
                df, 
                device=device, 
                normalizer=norm,
                seq_len=cfg['model']['seq_len'],
                conf_threshold=conf_threshold
            )
            elapsed = time.time() - start_time
            
            status.update(label=f"Бэктест завершён! ({elapsed:.1f}с)", state="complete")
            
            # ─── Вывод метрик (IS vs OOS) ──────────────────────────────────────────
            st.divider()
            
            # Расчет раздельных метрик
            equity = result.equity
            is_part = equity[:split_point]
            oos_part = equity[split_point:]
            
            def calc_simple_ret(arr):
                return (arr[-1] / arr[0] - 1) if len(arr) > 1 else 0
            
            is_ret = calc_simple_ret(is_part)
            oos_ret = calc_simple_ret(oos_part) if len(oos_part) > 1 else 0
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🟢 IN-SAMPLE (Trained)")
                st.metric("Return (IS)", f"{is_ret*100:.2f}%")
            with c2:
                st.markdown("##### 🔵 OUT-OF-SAMPLE (Unseen)")
                color = "normal" if oos_ret > 0 else "inverse"
                st.metric("Return (OOS)", f"{oos_ret*100:.2f}%", delta=f"{(oos_ret - is_ret)*100:.1f}% vs IS", delta_color=color)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{result.total_return*100:.2f}%", help="Общая прибыль за весь период")
            m2.metric("Sharpe Ratio", f"{result.sharpe:.2f}", help="Коэффициент Шарпа")
            m3.metric("Max Drawdown", f"{result.max_drawdown*100:.2f}%", delta_color="inverse")
            m4.metric("Total Trades", result.total_trades)
            
            # ─── График Equity ────────────────────────────────────────────────────
            st.subheader("Визуализация: Обучение vs Реальность")
            
            # Сэмплирование для производительности UI
            sample_size = 20000
            step = max(1, len(result.equity) // sample_size)
            idx = range(0, len(result.equity), step)
            
            equity_sampled = result.equity[idx]
            price_sampled = df["price"].values[idx]
            steps_sampled = np.array(list(range(len(result.equity))))[idx]
            
            fig = go.Figure()
            # Основная ось: Equity
            fig.add_trace(go.Scatter(
                x=steps_sampled, 
                y=equity_sampled, 
                name="Equity (USD)", 
                line=dict(color="#00FFAA", width=2),
                yaxis="y1"
            ))
            # Вспомогательная ось: Цена актива
            fig.add_trace(go.Scatter(
                x=steps_sampled, 
                y=price_sampled, 
                name="BTC Price", 
                line=dict(color="rgba(255,255,255,0.2)", width=1),
                yaxis="y2"
            ))
            
            # Вертикальная линия разделения
            fig.add_vline(x=split_point, line_width=2, line_dash="dash", line_color="orange")
            fig.add_annotation(x=split_point/2, y=1.05, yref="paper", text="IN-SAMPLE", showarrow=False, font=dict(color="#00FFAA"))
            fig.add_annotation(x=split_point + (len(equity)-split_point)/2, y=1.05, yref="paper", text="OUT-OF-SAMPLE", showarrow=False, font=dict(color="#00BBFF"))

            fig.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                yaxis=dict(title="Equity (USD)", side="left"),
                yaxis2=dict(title="BTC Price", overlaying="y", side="right", showgrid=False),
                margin=dict(l=0, r=0, t=50, b=0),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ─── Таблица сделок ───────────────────────────────────────────────────
            if result.trades:
                with st.expander("📋 Посмотреть последние 100 сделок"):
                    trades_df = pd.DataFrame([
                        {
                            "Step": t.step,
                            "Side": t.side.upper(),
                            "Size (BTC)": f"{t.size:.4f}",
                            "Price": f"{t.price:.1f}",
                            "Comm ($)": f"{t.commission:.2f}",
                            "Conf": f"{t.confidence:.2f}"
                        } for t in result.trades
                    ])
                    st.table(trades_df.tail(100))
            else:
                st.info("💡 Модель не нашла точек входа с текущим порогом уверенности.")
    else:
        st.info("Настройте параметры в боковой панели и нажмите 'Запустить бэктест'")
