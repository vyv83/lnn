"""
ui/pages/3_🧠_Model.py
Интерфейс обучения модели LiquidHawkesModel.
"""
import streamlit as st
import pandas as pd
import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader

from core.model import LiquidHawkesModel, build_model
from core.trainer import HawkesDataset, LiquidTrainer
from core.types import ModelConfig, TrainConfig
from core.config import load_config
from core.features import FeatureNormalizer, FEATURE_COLS

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Обучение CfC-модели")

DATA_DIR = Path("./data/cache")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Константы путей (Рефакторинг: 3 файла)
MODEL_S1 = MODEL_DIR / "stage1.pth"
MODEL_S2 = MODEL_DIR / "stage2.pth"
NORM_PATH = MODEL_DIR / "normalizer.npz"

# ─── Выбор данных ────────────────────────────────────────────────────────────
available_features = [f.name for f in DATA_DIR.glob("features_*.parquet")]
if not available_features:
    st.warning("Нет доступных признаков. Сначала обработайте данные на странице 📥 Data.")
    st.stop()

selected_file = st.selectbox("Выберите файл признаков для обучения", available_features)

# Получение реального размера файла для слайдера
def get_parquet_row_count(path):
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except:
        return 1_000_000 # fallback

total_rows = get_parquet_row_count(DATA_DIR / selected_file)

# ─── Sidebar: Настройки ──────────────────────────────────────────────────────
st.sidebar.header("Параметры обучения")
m_cfg, t_cfg, _ = load_config()

cfc_neurons = st.sidebar.number_input("CfC Neurons", 8, 1024, m_cfg.cfc_neurons)
cfc_motor = st.sidebar.number_input("CfC Motor", 4, 256, m_cfg.cfc_motor)
batch_size = st.sidebar.number_input("Batch Size", 1, 1024, t_cfg.batch_size)
epochs = st.sidebar.number_input("Stage 1 Epochs (SL)", 1, 100, t_cfg.phase1_epochs)
lr = st.sidebar.number_input("SL Learning Rate", 1e-5, 1e-1, t_cfg.phase1_lr, format="%.5f")
st.sidebar.divider()
st.sidebar.subheader("Stage 2 Settings (RL)")
rl_epochs = st.sidebar.number_input("Stage 2 Epochs (RL)", 1, 100, t_cfg.phase2_epochs)
rl_lr = st.sidebar.number_input("RL Learning Rate", 1e-6, 1e-2, t_cfg.phase2_lr, format="%.6f")

st.sidebar.divider()
resume = st.sidebar.checkbox("Дообучить (Resume)", value=True, help="Загрузить последний чекпоинт перед началом (SL или RL)")

# Динамический слайдер лимита данных c магнитом 500к
default_limit = min(1_000_000, total_rows)
# Ставим min_value=0, чтобы шаги были ровными: 0, 500k, 1m...
data_limit = st.sidebar.slider("Data Limit (rows)", 0, total_rows, default_limit, step=500_000)
st.sidebar.caption(f"Максимум в файле: **{total_rows:,}** строк")

st.sidebar.divider()
st.sidebar.subheader("Hardware")
available_devices = ["cpu"]
if torch.backends.mps.is_available():
    available_devices.append("mps")
if torch.cuda.is_available():
    available_devices.append("cuda")
device = st.sidebar.radio("Устройство для вычислений", available_devices, index=len(available_devices)-1)

# ─── Тренировочный цикл ────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Статус Обучения")
    st.info(f"💡 Текущее устройство: **{device.upper()}**")
    
    # ─── Stage 1: Card ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🚀 Stage 1: Supervised Learning")
        st.caption("Обучение базового понимания рынка (предсказание интенсивности).")
        btn_sl = st.button("Начать Stage 1", use_container_width=True)
        
        # Локальные плейсхолдеры для Stage 1
        s1_metrics = st.columns(2)
        s1_speed = s1_metrics[0].empty()
        s1_loss = s1_metrics[1].empty()
        s1_progress = st.progress(0)
        s1_status = st.empty()
        s1_chart = st.empty()

        if btn_sl:
            with st.status("Обучение Stage 1...", expanded=True) as status:
                st.write("🔧 Подготовка и нормализация данных...")
                df = pd.read_parquet(DATA_DIR / selected_file)
                
                if len(df) > data_limit:
                    df = df.iloc[:data_limit]
                
                norm = FeatureNormalizer()
                norm.fit(df[FEATURE_COLS].values)
                st.write("✅ Z-score нормализация готова.")
                
                st.write(f"📊 Вычисление таргетов для {len(df):,} событий...")
                dataset = HawkesDataset(df, seq_len=m_cfg.seq_len, normalizer=norm)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                st.write(f"🏗️ Сборка модели LNN ({device})...")
                model = build_model(ModelConfig(cfc_neurons=cfc_neurons, cfc_motor=cfc_motor))
                trainer = LiquidTrainer(model, TrainConfig(phase1_lr=lr, batch_size=batch_size), device=device)
                
                losses = []
                start_epoch = 0
                if resume and MODEL_S1.exists():
                    try:
                        checkpoint = trainer.load_checkpoint(str(MODEL_S1))
                        start_epoch = checkpoint.get('epoch', 0) + 1
                        losses = checkpoint.get('history', [])
                        if NORM_PATH.exists():
                            norm.load(str(NORM_PATH))
                        st.write(f"✅ Чекпоинт загружен: **{len(losses)}** эпох истории. Продолжаем с эпохи **{start_epoch+1}**.")
                    except Exception as e:
                        st.error(f"Не удалось загрузить чекпоинт: {e}. Попробуйте начать без Resume.")
                        st.stop()
                elif resume:
                    st.warning("⚠️ Чекпоинт не найден, начинаем с нуля.")
                
                # Инициализация сессии и пре-заполнение графика
                st.session_state.losses = losses
                if losses:
                    s1_chart.line_chart(pd.DataFrame({"SL Loss": losses}))
                
                def sl_callback(batch, total, loss_val):
                    pct = (batch + 1) / total
                    s1_progress.progress(pct)
                    elapsed = time.time() - start_epoch_time
                    batch_time = elapsed / (batch + 1)
                    eta = batch_time * (total - (batch + 1))
                    events_per_sec = batch_size / batch_time if batch_time > 0 else 0
                    
                    s1_speed.metric("Speed", f"{events_per_sec:.0f} ev/s")
                    s1_loss.metric("Loss", f"{loss_val:.4f}")
                    s1_status.text(f"Эпоха {epoch+1}/{epochs} | Батч {batch+1}/{total} | ETA: {eta:.0f}s")

                st.write("🔥 Запуск тренировочного цикла SL...")
                start_train_time = time.time()
                for epoch in range(start_epoch, epochs):
                    status.update(label=f"Обучение SL (Эпоха {epoch+1}/{epochs})...")
                    start_epoch_time = time.time()
                    avg_loss = trainer.train_epoch_sl(dataloader, callback=sl_callback)
                    losses.append(avg_loss)
                    st.session_state.losses = losses
                    s1_chart.line_chart(pd.DataFrame({"SL Loss": losses}))
                    
                    trainer.save_checkpoint(str(MODEL_S1), epoch=epoch, history=losses,
                        metadata={
                            "trained_samples": data_limit if len(df) > data_limit else len(df),
                            "cfc_neurons": cfc_neurons,
                            "cfc_motor": cfc_motor,
                            "batch_size": batch_size,
                            "epochs_s1": epochs,
                            "lr_s1": lr,
                            "source_file": selected_file,
                            "dataset_start_us": int(df["timestamp_us"].iloc[0]),
                            "dataset_end_us": int(df["timestamp_us"].iloc[-1])
                        })
                    norm.save(str(NORM_PATH))
                
                duration = time.time() - start_train_time
                st.success(f"Stage 1 завершена за {duration:.1f} сек")
                status.update(label="Stage 1 завершена!", state="complete")

    st.write("") # Отступ между карточками

    # ─── Stage 2: Card ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 📈 Stage 2: RL Optimization")
        st.caption("Тонкая настройка для максимизации прибыли (PnL).")
        btn_rl = st.button("Начать Stage 2", use_container_width=True)

        # Локальные плейсхолдеры для Stage 2
        s2_metrics = st.columns(2)
        s2_speed = s2_metrics[0].empty()
        s2_reward = s2_metrics[1].empty()
        s2_progress = st.progress(0)
        s2_status = st.empty()
        s2_chart = st.empty()

        if btn_rl:
            with st.status("Оптимизация PnL...", expanded=True) as status:
                st.write("Загрузка данных...")
                df = pd.read_parquet(DATA_DIR / selected_file)
                if len(df) > data_limit:
                    df = df.iloc[:data_limit]
                
                st.write(f"🔧 Настройка RL-трейнера ({device})...")
                model = build_model(ModelConfig(cfc_neurons=cfc_neurons, cfc_motor=cfc_motor))
                trainer = LiquidTrainer(model, TrainConfig(phase2_lr=rl_lr, batch_size=batch_size), device=device)
                
                rewards = []
                start_epoch_rl = 0
                
                if resume and MODEL_S2.exists():
                    try:
                        checkpoint = trainer.load_checkpoint(str(MODEL_S2))
                        start_epoch_rl = checkpoint.get('epoch', 0) + 1
                        rewards = checkpoint.get('history', [])
                        st.write(f"✅ RL-чекпоинт загружен: **{len(rewards)}** эпох истории. Продолжаем с эпохи **{start_epoch_rl+1}**.")
                        if start_epoch_rl >= rl_epochs:
                            st.info(f"✅ Модель уже дообучена до эпохи {start_epoch_rl}. (Цель: {rl_epochs})")
                    except Exception as e:
                        st.warning(f"Не удалось загрузить RL-чекпоинт: {e}. Пробуем базу SL.")
                
                if not rewards:
                    try:
                        trainer.load_checkpoint(str(MODEL_S1))
                        st.write("✅ Базовая SL модель загружена для старта RL.")
                    except Exception as e:
                        st.error(f"Ошибка: Базовая модель SL не найдена! ({e})")
                        st.stop()
                
                norm = FeatureNormalizer()
                if NORM_PATH.exists():
                    norm.load(str(NORM_PATH))
                else:
                    st.warning("⚠️ Файл нормализации не найден. RL может быть крайне нестабильным.")

                st.write(f"📊 Подготовка данных ({len(df):,} событий)...")
                dataset = HawkesDataset(df, seq_len=m_cfg.seq_len, normalizer=norm)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Настройка слоев и оптимизатора перед RL
                if not rewards:
                    # Первый запуск RL — полная подготовка (заморозка + новый optimizer)
                    trainer.prepare_phase2()
                    st.write(f"🔒 Backbone заморожен. Новый optimizer. RL LR: **{rl_lr}**")
                else:
                    # Resume — только заморозка, optimizer уже загружен из чекпоинта
                    for name, param in trainer.model.named_parameters():
                        if any(x in name for x in ["rnn_cell", "proj", "intensity_head"]):
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    st.write(f"🔒 Backbone заморожен. Optimizer восстановлен из чекпоинта.")

                st.session_state.rewards = rewards
                if rewards:
                    s2_chart.line_chart(pd.DataFrame({"RL Reward (PnL)": rewards}))
                
                def rl_callback(batch, total, reward_val):
                    pct = (batch + 1) / total
                    s2_progress.progress(pct)
                    elapsed = time.time() - start_epoch_time
                    batch_time = elapsed / (batch + 1)
                    eta = batch_time * (total - (batch + 1))
                    events_per_sec = batch_size / batch_time if batch_time > 0 else 0
                    s2_speed.metric("Speed", f"{events_per_sec:.0f} ev/s")
                    s2_reward.metric("Avg Reward", f"{reward_val:.6f}")
                    s2_status.text(f"Эпоха {epoch+1}/{rl_epochs} | RL Батч {batch+1}/{total} | ETA: {eta:.0f}s")

                st.write("🚀 Запуск цикла оптимизации PnL...")
                start_train_time_rl = time.time()
                if start_epoch_rl >= rl_epochs:
                    st.warning("⚠️ Целевое количество эпох уже достигнуто.")
                
                for epoch in range(start_epoch_rl, rl_epochs):
                    status.update(label=f"Обучение RL (Эпоха {epoch+1}/{rl_epochs})...")
                    start_epoch_time = time.time()
                    avg_reward = trainer.train_epoch_rl(dataloader, callback=rl_callback)
                    rewards.append(avg_reward)
                    st.session_state.rewards = rewards
                    s2_chart.line_chart(pd.DataFrame({"RL Reward (PnL)": rewards}))
                    trainer.save_checkpoint(str(MODEL_S2), epoch=epoch, history=rewards,
                    metadata={
                        "trained_samples": data_limit if len(df) > data_limit else len(df),
                        "cfc_neurons": cfc_neurons,
                        "cfc_motor": cfc_motor,
                        "batch_size": batch_size,
                        "epochs_s2": rl_epochs,
                        "lr_s2": rl_lr,
                        "source_file": selected_file,
                        "dataset_start_us": int(df["timestamp_us"].iloc[0]),
                        "dataset_end_us": int(df["timestamp_us"].iloc[-1])
                    })
                
                duration_rl = time.time() - start_train_time_rl
                status.update(label="Stage 2 завершена!", state="complete")
                st.success(f"📈 RL-дообучение завершено! Прошло {duration_rl:.1f} сек.")


with col2:
    st.subheader("Модель и Производительность")
    st.info("Stage 1 учит модель предсказывать интенсивность событий. Это базовый слой 'понимания' рынка.")
    
    st.markdown("""
    ### Архитектура:
    - **Backbone**: MLP для извлечения паттернов из 17 фичей.
    - **Liquid Core**: CfC ячейки, учитывающие время между событиями (`dt`).
    - **Output Heads**:
        1. `Intensity`: Предсказание будущей активности.
        2. `Action`: Позиция [-1, +1].
        3. `Confidence`: Уверенность (для управления риском).
    """)
    
    if "losses" in st.session_state and st.session_state.losses:
        st.write("История Stage 1 (SL Loss):")
        st.line_chart(st.session_state.losses)
    
    if "rewards" in st.session_state and st.session_state.rewards:
        st.write("История Stage 2 (RL Reward):")
        st.line_chart(st.session_state.rewards)

# ─── Список моделей ──────────────────────────────────────────────────────────
st.divider()
st.subheader("💾 Сохранённые модели")
models = sorted(list(MODEL_DIR.glob("*.pth")))
if models:
    model_data = []
    for m in models:
        info = LiquidTrainer.get_checkpoint_info(str(m))
        meta = info.get("metadata", {})
        model_data.append({
            "Имя": m.name,
            "Эпох": f"{info.get('epoch', 0) + 1}",
            "Neurons": meta.get("cfc_neurons", "N/A"),
            "Motor": meta.get("cfc_motor", "N/A"),
            "Batch": meta.get("batch_size", "N/A"),
            "Источник": meta.get("source_file", "N/A"),
            "History": f"{info.get('history_len', 0)} pts",
            "Дата": time.strftime("%Y-%m-%d %H:%M", time.localtime(m.stat().st_mtime))
        })
    st.table(pd.DataFrame(model_data))
else:
    st.info("Нет сохранённых моделей")
