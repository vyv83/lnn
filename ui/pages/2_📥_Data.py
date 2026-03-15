"""
ui/pages/2_📥_Data.py
Страница управления данными: загрузка, парсинг, инспекция признаков.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import datetime
import logging

from core.config import load_config
from data.download import download, list_downloaded
from core.events import build_event_stream, save_event_stream, load_event_stream
from core.features import build_features, FEATURE_COLS

st.set_page_config(page_title="Data Management", page_icon="📥", layout="wide")

st.title("📥 Управление данными")

DATA_DIR = Path("./data/cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Sidebar: Загрузка ───────────────────────────────────────────────────────
st.sidebar.header("Загрузить новые данные")
default_date = datetime.date(2026, 3, 1)
target_date = st.sidebar.date_input("Дата (1-е число для бесплатного доступа)", default_date)
api_key = st.sidebar.text_input("Tardis API Key (опционально)", type="password")

if st.sidebar.button("Скачать данные"):
    with st.spinner(f"Загрузка BTCUSDT за {target_date}..."):
        try:
            from_date = str(target_date)
            to_date = str(target_date + datetime.timedelta(days=1))
            files = download(from_date, to_date, str(DATA_DIR), api_key=api_key or None)
            st.sidebar.success(f"Скачано {len(files)} файлов")
        except Exception as e:
            st.sidebar.error(f"Ошибка: {e}")

# ─── Main: Список файлов ─────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Локальный кэш (CSV.gz)")
    csv_files = list_downloaded(str(DATA_DIR))
    if csv_files:
        df_csv = pd.DataFrame([
            {"Файл": f.name, "Размер (MB)": round(f.stat().st_size / (1024*1024), 2)}
            for f in csv_files if f.name.endswith(".csv.gz")
        ])
        st.table(df_csv)
    else:
        st.info("Нет скачанных CSV файлов")

with col2:
    st.subheader("💎 Обработанные данные (Parquet)")
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if parquet_files:
        df_pq = pd.DataFrame([
            {"Файл": f.name, "Размер (MB)": round(f.stat().st_size / (1024*1024), 2)}
            for f in parquet_files
        ])
        st.table(df_pq)
    else:
        st.info("Нет обработанных Parquet файлов")

# ─── Инспекция признаков ─────────────────────────────────────────────────────
st.divider()
st.subheader("🔬 Инспектор признаков")

available_features = [f.name for f in DATA_DIR.glob("features_*.parquet")]
if available_features:
    selected_file = st.selectbox("Выберите файл признаков", available_features, key="selected_feature_file")
    
    if st.button("📥 Загрузить данные"):
        st.session_state.data_loaded = False  # Сброс перед новой загрузкой
        with st.spinner("Загрузка данных..."):
            file_path = DATA_DIR / selected_file
            st.session_state.df_features = pd.read_parquet(file_path)
            st.session_state.data_loaded = True
            st.success(f"Загружено {len(st.session_state.df_features):,} строк")

    if st.session_state.get("data_loaded"):
        df = st.session_state.df_features
        
        # Статистика
        st.write("Статистика признаков:")
        st.dataframe(df[FEATURE_COLS].describe())
        
        # Визуализация
        st.divider()
        st.subheader("📊 Графики")
        
        # Используем session_state для default выбора, если он еще не задан
        if "feat_to_plot" not in st.session_state:
            st.session_state.feat_to_plot = ["log_price", "trade_imbalance"]

        feat_to_plot = st.multiselect(
            "Выберите признаки для отображения", 
            FEATURE_COLS, 
            default=st.session_state.feat_to_plot,
            key="feat_selector"
        )
        
        if feat_to_plot:
            # Берем сэмпл для скорости
            sample_n = st.slider("Количество событий для отображения", 1000, min(50000, len(df)), 10000)
            plot_df = df.iloc[:sample_n].copy()
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Разделяем на цену и остальные индикаторы
            has_price = "log_price" in feat_to_plot
            other_feats = [f for f in feat_to_plot if f != "log_price"]
            
            if has_price and other_feats:
                # Создаем график с двумя осями Y
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Основная ось: Индикаторы
                for f in other_feats:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[f], name=f), secondary_y=False)
                
                # Вторичная ось: Цена
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["log_price"], name="log_price (Price)", line=dict(color='gold', width=3)), secondary_y=True)
                
                fig.update_layout(
                    title=f"Динамика: Индикаторы vs log_price (первые {sample_n} точек)",
                    hovermode="x unified",
                    height=600
                )
                fig.update_yaxes(title_text="Индикаторы", secondary_y=False)
                fig.update_yaxes(title_text="log_price", secondary_y=True)
                
            else:
                # Обычный график если выбрано что-то одно (или только цена)
                fig = px.line(plot_df, y=feat_to_plot, title=f"Динамика признаков (первые {sample_n} событий)")
                fig.update_layout(hovermode="x unified", height=500)

            st.plotly_chart(fig, use_container_width=True)
            
            # Корреляция
            if len(feat_to_plot) > 1:
                st.subheader("🔗 Матрица корреляции")
                corr = df[feat_to_plot].corr()
                fig_corr = px.imshow(
                    corr, 
                    text_auto=".2f", 
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    range_color=[-1, 1]
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Выберите хотя бы один признак для отображения графика.")

else:
    st.warning("Сначала обработайте данные (создайте features_*.parquet)")

# Кнопка запуска обработки (если есть CSV но нет Parquet)
if csv_files and st.button("⚙️ Обработать свежие данные за " + str(target_date)):
    with st.status("Обработка данных...") as status:
        st.write("1. Создание потока событий...")
        ev_df = build_event_stream(DATA_DIR, str(target_date))
        save_event_stream(ev_df, DATA_DIR / f"events_{target_date}.parquet")
        
        st.write("2. Генерация признаков...")
        # Упрощенная генерация без полной enrichment для примера в UI
        # (в проде лучше использовать полный скрипт)
        feat_df = build_features(ev_df)
        feat_df.to_parquet(DATA_DIR / f"features_{target_date}.parquet", index=False)
        
        status.update(label="Обработка завершена!", state="complete")
        st.rerun()
