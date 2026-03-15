"""
ui/app.py
Точка входа Streamlit — навигация и статус проекта.
Запуск: streamlit run ui/app.py

Это заглушка Фазы 1. UI будет наполняться по мере реализации Фаз 2-6.
"""
import streamlit as st
from pathlib import Path

# ─── Конфигурация страницы ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Liquid Hawkes",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Статус проекта ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent


def check_status() -> dict:
    """Проверить, что готово в проекте."""
    return {
        "config":  (PROJECT_ROOT / "config.toml").exists(),
        "model":   (PROJECT_ROOT / "core" / "model.py").exists(),
        "data":    any((PROJECT_ROOT / "data" / "cache").glob("*.csv.gz")),
        "trained": any((PROJECT_ROOT / "models").glob("*.pt")),
        "results": any((PROJECT_ROOT / "results").glob("*.json")),
    }


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=64)
    st.title("Liquid Hawkes")
    st.caption("BTC Perpetual · Binance Futures")
    st.divider()

    status = check_status()
    st.subheader("Статус проекта")
    st.write("✅ Конфиг" if status["config"] else "❌ Конфиг")
    st.write("✅ Модель" if status["model"] else "❌ Модель")
    st.write("✅ Данные" if status["data"] else "⬜ Данные (Фаза 2)")
    st.write("✅ Обучена" if status["trained"] else "⬜ Обучена (Фаза 3)")
    st.write("✅ Backtest" if status["results"] else "⬜ Backtest (Фаза 4)")

# ─── Главная страница ─────────────────────────────────────────────────────────
st.title("💧 Liquid Hawkes Platform")
st.subheader("CfC Neural Network · Order Flow Intensity Modeling")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Модель**: CfC + AutoNCP\n\n17 фичей → λ_buy, λ_sell, λ_cancel")
with col2:
    st.info("**Данные**: Tardis.dev\n\nTrades · Book · Derivatives · Liquidations")
with col3:
    st.info("**Стратегия**: Intensity modeling\n\nHorizon: 100 событий · Sharpe target: 2.0")

st.divider()
st.subheader("Roadmap")

phases = [
    ("✅", "Фаза 1 — MVP", "Модель + конфиг + тесты"),
    ("⬜", "Фаза 2 — Данные", "Tardis.dev + парсинг + 17 фичей"),
    ("⬜", "Фаза 3 — Обучение", "Supervised + RL Fine-Tuning"),
    ("⬜", "Фаза 4 — Backtest", "Walk-Forward валидация"),
    ("⬜", "Фаза 5 — Нейроны", "Визуализация CfC состояний"),
    ("⬜", "Фаза 6 — Live", "Binance WebSocket · Paper Trading"),
]

for icon, name, desc in phases:
    with st.expander(f"{icon} {name}"):
        st.write(desc)

st.divider()
st.caption("Используй меню слева для навигации по страницам (появятся по мере реализации фаз).")
