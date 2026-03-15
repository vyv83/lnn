"""
core/config.py
Загружает config.toml и возвращает типизированные объекты конфигурации.
Использует стандартный tomllib (Python 3.11+) — никаких доп. зависимостей.
"""
import tomllib
import logging
from pathlib import Path
from typing import Optional, Tuple

from core.types import ModelConfig, TrainConfig, DataConfig

logger = logging.getLogger(__name__)

# Путь к конфигу по умолчанию — ищем снизу вверх от текущей директории
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def load_raw(config_path: Optional[Path] = None) -> dict:
    """Загрузить сырой словарь из config.toml."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    logger.info("Конфиг загружен из %s", path)
    return data


def load_model_config(config_path: Optional[Path] = None) -> ModelConfig:
    """Загрузить секцию [model] из config.toml."""
    raw = load_raw(config_path)
    section = raw.get("model", {})
    return ModelConfig(
        input_size=section.get("input_size", 17),
        cfc_neurons=section.get("cfc_neurons", 32),
        cfc_motor=section.get("cfc_motor", 8),
        backbone_units=section.get("backbone_units", 64),
        backbone_layers=section.get("backbone_layers", 1),
        seq_len=section.get("seq_len", 512),
        horizon=section.get("horizon", 100),
    )


def load_train_config(config_path: Optional[Path] = None) -> TrainConfig:
    """Загрузить секции [training] + [trading] из config.toml."""
    raw = load_raw(config_path)
    tr = raw.get("training", {})
    td = raw.get("trading", {})
    return TrainConfig(
        phase1_epochs=tr.get("phase1_epochs", 20),
        phase1_lr=tr.get("phase1_lr", 1e-3),
        phase2_epochs=tr.get("phase2_epochs", 10),
        phase2_lr=tr.get("phase2_lr", 1e-4),
        batch_size=tr.get("batch_size", 32),
        commission=td.get("commission", 0.0004),
        confidence_threshold=td.get("confidence_threshold", 0.5),
    )


def load_data_config(config_path: Optional[Path] = None) -> DataConfig:
    """Загрузить секцию [data] из config.toml."""
    raw = load_raw(config_path)
    section = raw.get("data", {})
    return DataConfig(
        exchange=section.get("exchange", "binance-futures"),
        symbol=section.get("symbol", "BTCUSDT"),
        data_dir=section.get("data_dir", "./data/cache"),
        feature_window=section.get("feature_window", 500),
        train_ratio=section.get("train_ratio", 0.8),
    )


def load_config(config_path: Optional[Path] = None) -> Tuple[ModelConfig, TrainConfig, DataConfig]:
    """Загрузить все конфигурации разом."""
    return (
        load_model_config(config_path),
        load_train_config(config_path),
        load_data_config(config_path)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_cfg = load_model_config()
    train_cfg = load_train_config()
    data_cfg = load_data_config()
    print(f"ModelConfig: {model_cfg}")
    print(f"TrainConfig: {train_cfg}")
    print(f"DataConfig:  {data_cfg}")
    print("core/config.py OK")
