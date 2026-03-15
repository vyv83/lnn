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
import time
import toml
import numpy as np

from core.model import build_model
from core.backtest import BacktestEngine
from core.types import ModelConfig
from core.features import FeatureNormalizer

def display_model_passport(meta):
    if not meta:
        return
    
    import datetime
    def fmt_ts(ts_us):
        if not ts_us: return "N/A"
        return datetime.datetime.fromtimestamp(ts_us / 1_000_000).strftime('%Y-%m-%d %H:%M:%S')

    stage = meta.get('stage', 0)
    stage_name = f"Stage {stage} ({'SL' if stage==1 else 'RL'})" if stage else "Универсальный чекпоинт"
    
    with st.expander(f"📄 Паспорт модели: {stage_name}", expanded=True):
        # Выделяем количество точек обучения крупнее
        pts = meta.get('trained_samples', 0)
        st.metric("Количество точек обучения (Samples)", f"{pts:,}")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"📁 **Файл данных:** `{meta.get('source_file', 'N/A')}`")
            st.write(f"🧠 **Neurons/Motor:** `{meta.get('cfc_neurons', 0)} / {meta.get('cfc_motor', 0)}`")
            st.write(f"🚀 **Batch Size:** `{meta.get('batch_size', 0)}`")
        with col2:
            st.write(f"⏱️ **Начало:** `{fmt_ts(meta.get('dataset_start_us'))}`")
            st.write(f"⏱️ **Конец:** `{fmt_ts(meta.get('dataset_end_us'))}`")
            
            # Показываем именно тот LR и Эпохи, которые соответствуют стадии
            if stage == 1:
                lr = meta.get('lr_s1', 0)
                epochs = meta.get('epochs_s1', 0)
                st.write(f"📈 **SL LR:** `{lr:.6f}`")
                st.write(f"🔄 **SL Epochs:** `{epochs}`")
            elif stage == 2:
                lr = meta.get('lr_s2', 0)
                epochs = meta.get('epochs_s2', 0)
                st.write(f"📈 **RL LR:** `{lr:.6f}`")
                st.write(f"🔄 **RL Epochs:** `{epochs}`")
            else:
                # Fallback для старых/общих метаданных
                lr = meta.get('lr_s2', meta.get('lr_s1', 1e-4))
                st.write(f"📈 **Learning Rate:** `{lr:.6f}`")

st.set_page_config(page_title="Backtest Results", page_icon="📈", layout="wide")

st.title("📈 Результаты бэктестинга")

DATA_DIR = Path("./data/cache")
MODEL_DIR = Path("./models")
NORM_PATH = MODEL_DIR / "normalizer.npz"
CONFIG_PATH = Path("./config.toml")

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

# Получение реального размера файла для слайдеров
def get_parquet_row_count(path):
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except:
        return 1_000_000

total_rows_data = get_parquet_row_count(DATA_DIR / selected_data)

# 🚀 Проактивная загрузка метаданных модели
checkpoint_path = MODEL_DIR / selected_model
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model_meta = checkpoint.get('metadata', {})

display_model_passport(model_meta)

# Smart Sync Логика для Split Point
if model_meta:
    # Пытаемся быстро проверить начало файла без полной загрузки
    first_row = pd.read_parquet(DATA_DIR / selected_data, columns=["timestamp_us"]).iloc[0]
    current_start = int(first_row["timestamp_us"])
    
    if model_meta.get('dataset_start_us') == current_start:
        auto_split = model_meta.get('trained_samples', 0)
        # Инициализируем или обновляем состояние ползунка (теперь следим и за файлом данных)
        sync_key = f"{selected_model}_{selected_data}"
        if "split_point_slider" not in st.session_state or st.session_state.get('last_model_sync') != sync_key:
            st.session_state.split_point_slider = auto_split
            st.session_state.last_model_sync = sync_key
        st.info(f"🔗 **Smart Sync:** Модель узнала данные. Обучение закончилось на `{auto_split:,}`.")

st.sidebar.divider()
st.sidebar.subheader("Данные и Разделение")
# Магнит 500к для лимита, по умолчанию - весь файл
data_limit = st.sidebar.number_input("Лимит данных (строк)", 0, total_rows_data, total_rows_data, step=500_000)

# Защита от сброса Split Point при изменении лимита:
# Если текущее значение в session_state превышает лимит, ограничиваем его, но не сбрасываем в 0.
if "split_point_slider" in st.session_state:
    st.session_state.split_point_slider = min(st.session_state.split_point_slider, data_limit)

# Используем key для возможности программного изменения + магнит 500к (min_value=0 для ровного шага)
st.sidebar.slider(
    "Точка разделения (IS/OOS)", 
    0, data_limit, 
    key="split_point_slider",
    step=500_000
)
# Актуальное значение всегда берем из session_state
cur_split = st.session_state.split_point_slider

# Устройство заблокировано на CPU для максимальной скорости (последовательный инференс на Mac быстрее на CPU)
device = "cpu"

st.sidebar.subheader("Параметры рынка")
commission = st.sidebar.number_input("Комиссия (%)", 0.0, 1.0, cfg['trading']['commission']*100, step=0.01) / 100.0
slippage = st.sidebar.number_input("Проскальзывание (bps)", 0, 100, int(cfg['trading']['slippage']*10000))
initial_balance = st.sidebar.number_input("Начальный депозит (USD)", 100, 1000000, 10000)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, cfg['trading']['confidence_threshold'])

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
            # Берем актуальное значение из слайдера
            cur_split = st.session_state.split_point_slider
            st.write(f"📊 Всего строк: {len(df)} (Split: {cur_split:,})")
            
            st.info(f"💡 Вычисления: **{device.upper()}**")
            
            # --- Progress UI ---
            st.divider()
            prog_col1, prog_col2 = st.columns([3, 1])
            bt_progress = prog_col1.progress(0)
            bt_speed = prog_col2.empty()
            bt_status = st.empty()
            
            start_bt_time = time.time()
            
            def backtest_callback(step, total, phase):
                pct = step / total
                bt_progress.progress(pct)
                elapsed = time.time() - start_bt_time
                speed = step / elapsed if elapsed > 0 else 0
                eta = (total - step) / speed if speed > 0 else 0
                
                bt_speed.metric("Speed", f"{speed:.0f} ev/s")
                bt_status.text(f"⚡ Фаза: {phase} | {step:,} / {total:,} | ETA: {eta:.0f}s")

            st.write(f"🏗️ Инициализация модели LNN...")
            m_cfg = ModelConfig(
                cfc_neurons=cfg['model']['cfc_neurons'],
                cfc_motor=cfg['model']['cfc_motor']
            ) 
            model = build_model(m_cfg).to(device)
            
            # Используем уже загруженный чекпоинт
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
            t_start = time.time()
            result = engine.run(
                model, 
                df, 
                device=device, 
                normalizer=norm,
                seq_len=cfg['model']['seq_len'],
                conf_threshold=conf_threshold,
                callback=backtest_callback
            )
            elapsed = time.time() - t_start
            
            status.update(label=f"Бэктест завершён! ({elapsed:.1f}с)", state="complete")
            
            # ─── Вывод метрик (IS vs OOS) ──────────────────────────────────────────
            st.divider()
            
            equity = result.equity
            is_part = equity[:cur_split]
            oos_part = equity[cur_split:]
            
            def calc_simple_ret(arr):
                if len(arr) < 2: return 0
                return (arr[-1] / arr[0] - 1)
            
            is_ret = calc_simple_ret(is_part)
            oos_ret = calc_simple_ret(oos_part)
            
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
            m1.metric("Total Return", f"{result.total_return*100:.2f}%")
            m2.metric("Sharpe Ratio", f"{result.sharpe:.2f}")
            m3.metric("Max Drawdown", f"{result.max_drawdown*100:.2f}%", delta_color="inverse")
            m4.metric("Total Trades", result.total_trades)
            
            # ─── График Equity ────────────────────────────────────────────────────
            st.subheader("Визуализация: Обучение vs Реальность")
            
            sample_size = 20000
            step_val = max(1, len(result.equity) // sample_size)
            idx = range(0, len(result.equity), step_val)
            
            equity_sampled = result.equity[idx]
            price_sampled = df["price"].values[idx]
            steps_sampled = np.array(list(range(len(result.equity))))[idx]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps_sampled, y=equity_sampled, name="Equity (USD)", line=dict(color="#00FFAA", width=2), yaxis="y1"))
            fig.add_trace(go.Scatter(x=steps_sampled, y=price_sampled, name="BTC Price", line=dict(color="rgba(255,255,255,0.2)", width=1), yaxis="y2"))
            
            fig.add_vline(x=cur_split, line_width=2, line_dash="dash", line_color="orange")
            fig.add_annotation(x=cur_split/2 if cur_split > 0 else 0, y=1.05, yref="paper", text="IN-SAMPLE", showarrow=False, font=dict(color="#00FFAA"))
            fig.add_annotation(x=cur_split + (len(equity)-cur_split)/2 if len(equity)>cur_split else cur_split, y=1.05, yref="paper", text="OUT-OF-SAMPLE", showarrow=False, font=dict(color="#00BBFF"))
            
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
            
            if result.trades:
                with st.expander("📋 Посмотреть последние 100 сделок"):
                    trades_df = pd.DataFrame([{
                        "Step": t.step,
                        "Side": t.side.upper(),
                        "Size (BTC)": f"{t.size:.4f}",
                        "Price": f"{t.price:.1f}",
                        "Comm ($)": f"{t.commission:.2f}",
                        "Conf": f"{t.confidence:.2f}"
                    } for t in result.trades])
                    st.table(trades_df.tail(100))
    else:
        st.info("Настройте параметры в боковой панели и нажмите 'Запустить бэктест'")
