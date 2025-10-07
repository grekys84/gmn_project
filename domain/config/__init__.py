import yaml
from pathlib import Path
from .settings import AppSettings


def load_settings(config_path: Path) -> AppSettings:
    """
    Загружает настройки из YAML файла.

    Args:
        config_path: Путь к файлу конфигурации YAML.

    Returns:
        Экземпляр AppSettings с загруженной конфигурацией.

    """
    with open(config_path, "r", encoding="utf-8") as f:
        # Загружаем данные как есть, сохраняя иерархическую структуру
        data = yaml.safe_load(f)

    return AppSettings(**data)
