from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import (
    BaseModel,
    model_validator,
)
from typing import List
import torch
import yaml


class ModelConfig(BaseModel):
    """Конфигурация модели GMN."""

    name: str = "GMN"
    version: str = "1.0.0"
    encoder: str = "GINConv"
    readout: str = "global_add_pool"
    in_channels: int = 5
    hidden_dim: int = 64
    num_layers: int = 3


class TrainingConfig(BaseModel):
    """Конфигурация обучения модели."""

    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.0005
    weight_decay: float = 0.0001
    scheduler: str = "one_cycle"
    early_stopping_patience: int = 5
    validation_split: float = 0.1
    seed: int = 42
    shuffle: bool = True
    margin: float = 1.0
    loss: str = "contrastive"


class AugmentationConfig(BaseModel):
    """Конфигурация аугментаций для графов."""

    noise_scale: float = 0.1  # смещение координат в мм (гауссовский шум)
    rotation_max_deg: float = 10.0  # поворот в градусах
    scale_jitter: float = 0.05  # относительное изменение масштаба
    flip_prob: float = 0.0  # вероятность отражения
    edge_dropout: float = 0.1  # вероятность удаления ребра
    node_dropout: float = 0.05  # вероятность удаления узла


class AugmentationFlagsConfig(BaseModel):
    """Флаги включения различных типов аугментаций."""

    apply_noise: bool = True
    apply_rotation: bool = True
    apply_scale: bool = True
    apply_flip: bool = True
    apply_edge_dropout: bool = True
    apply_node_dropout: bool = True


class AugmentationGapConfig(BaseModel):
    """Конфигурация аугментации типа 'щели'."""

    gap_range: List[float] = [0.01, 0.05]  # диапазон размера щелей в мм
    min_gap_multiplier: int = (
        3  # минимальная разница между щелью и длиной грани
    )
    random_position_range: List[float] = [
        0.1,
        0.9,
    ]  # рандомное расположение щели на грани
    contours_per_file: int = 1  # количество контуров для модификации


class AugmentationHoleUnclosedConfig(BaseModel):
    """Конфигурация аугментации типа 'щели'."""

    gap_range: List[float] = [0.01, 0.05]  # диапазон размера щелей в мм
    min_gap_multiplier: int = 2  # множитель для минимальной длины ребра
    contours_per_file: int = 3  # сколько контуров разрывать в файле
    max_hole_edge_length: int = 2  # мм (макс длина ребра отверстия)
    random_position_range: List[float] = [
        0.3,
        0.7,
    ]  # относительная позиция разрыва (30-70%)


class AugmentationLadderConfig(BaseModel):
    """Конфигурация аугментации типа 'лесенка'."""

    step_length: float = 5  # длина шага лесенки (мм)
    offset: float = 2.0  # смещение для H/V (мм)
    diag_offset: float = 0.0  # смещение для диагоналей (мм)
    sides: int = 0  # число преобразуемых сторон (мм)


class AugmentationTabsConfig(BaseModel):
    """Конфигурация аугментации типа 'перемычка'."""

    width_range: List[float] = [1.0, 2.0]  # диапазон ширины перемычек (мм)
    length_range: List[float] = [2.0, 6.0]  # диапазон длин перемычек (мм)
    tabs_per_contour: int = 2  # количество перемычек на контур
    min_edge_length: float = 3.0  # минимальная длина ребра


class AugmentationShearConfig(BaseModel):
    """Конфигурация аугментации типа 'срезы'."""

    shearing_width_range: List[float] = [0.01, 0.05]  # диапазон ширины линии
    shearing_length_range: List[float] = [1, 5]  # диапазон длины линии
    shearing_per_file: int = 3  # количество линий
    min_edge_length: float = 5.0  # минимальная длина линии (мм)


class AugmentationPunchConfig(BaseModel):
    """Конфигурация аугментации типа 'пуансоны'."""

    artifact_types: list[str] = ["rectangle", "circle", "oblong"]
    artifacts_per_file: int = 3
    rectangle_size: tuple[float, float] = (50.0, 5.0)
    circle_diameter: float = 10.0
    oblong_size: tuple[float, float] = (20.0, 8.0)


class DataConfig(BaseModel):
    """Конфигурация путей к данным."""

    pair_file: str
    new_pair_file: str
    master_graph: str
    new_graph: str
    master_svg: str
    new_svg: str
    overlays: str
    augmentation_gap: str
    augmentation_hole_unclosed: str
    augmentation_ladder: str
    augmentation_tabs: str
    augmentation_shear: str
    augmentation_punch: str

    @model_validator(mode="after")
    def convert_to_absolute_paths(self) -> "DataConfig":
        """
        Преобразует все пути в абсолютные.

        Returns:
                DataConfig: Объект конфигурации с абсолютными путями
            Raises:
                ValueError: Если путь не является абсолютным

        """
        for field_name, value in self.model_dump().items():
            if isinstance(value, str):
                p = Path(value)
                if not p.is_absolute():
                    raise ValueError(
                        f"Путь должен быть абсолютным: {field_name}={value}"
                    )
                # Сохраняем как Path
                setattr(self, field_name, p)
        return self


class VisualizationConfig(BaseModel):
    """Конфигурация путей для визуализации."""

    prepare_pairs: str
    prepare_graphs: str

    @model_validator(mode="after")
    def convert_and_create_dirs(self) -> "VisualizationConfig":
        """
        Преобразует пути в абсолютные и создает директории.

        Returns:
                VisualizationConfig: Объект конфигурации с
                                     абсолютными путями и
                                     созданными директориями
            Raises:
                ValueError: Если путь не является абсолютным

        """
        for field_name, value in self.model_dump().items():
            if isinstance(value, str):
                p = Path(value)
                if not p.is_absolute():
                    raise ValueError(
                        f"Путь должен быть абсолютным: {field_name}={value}"
                    )
                p.mkdir(parents=True, exist_ok=True)
                # Сохраняем как Path
                setattr(self, field_name, p)
        return self


class ModelsConfig(BaseModel):
    """Конфигурация путей к моделям и метаданным."""

    current_model: str
    new_model: str
    val_errors: str


class ServiceConfig(BaseModel):
    """Конфигурация сервисных параметров."""

    embedding_db_path: str
    device: str = "auto"
    top_k: int = 3
    similarity_threshold: float = 0.9

    @model_validator(mode="after")
    def resolve_device(self) -> "ServiceConfig":
        """
        Определяет устройство для вычислений (CPU или GPU).

        Returns:
                ServiceConfig: Объект конфигурации с определенным устройством

        """
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self

    @model_validator(mode="after")
    def validate_embedding_path(self) -> "ServiceConfig":
        """
        Проверяет и создает путь к базе данных эмбеддингов.

        Returns:
                ServiceConfig: Объект конфигурации с валидированным путем
            Raises:
                ValueError: Если путь не является абсолютным

        """
        p = Path(self.embedding_db_path)
        if not p.is_absolute():
            raise ValueError(
                f"Путь должен быть абсолютным: embedding_db_path={p}"
            )
        p.mkdir(parents=True, exist_ok=True)
        self.embedding_db_path = p
        return self


class NormalizerConfig(BaseModel):
    """Конфигурация нормализации данных."""

    scale: float = 1 / 2857.0
    precision: float = 0.01


class GeometryConfig(BaseModel):
    """Конфигурация геометрических параметров сравнения."""

    slow_threshold: float = 50.0
    tol: float = 0.0001
    angle_tol: float = 0.001
    use_angles: bool = True
    normalize_position: bool = True
    normalize_scale: bool = True


class EvaluationConfig(BaseModel):
    """Конфигурация параметров оценки модели."""

    threshold: float = 0.5
    min_target_accuracy: float = 0.90
    test_mode: bool = False


class LoggingConfig(BaseModel):
    """Конфигурация путей для логирования."""

    log_dir: str

    @model_validator(mode="after")
    def create_log_dir(self) -> "LoggingConfig":
        """
        Создает директорию для логов.

        Returns:
                LoggingConfig: Объект конфигурации с
                               созданной директорией логов
            Raises:
                ValueError: Если путь не является абсолютным

        """
        p = Path(self.log_dir)
        if not p.is_absolute():
            raise ValueError(f"Путь должен быть абсолютным: log_dir={p}")
        p.mkdir(parents=True, exist_ok=True)
        self.log_dir = p
        return self


class AppSettings(BaseSettings):
    """Основная конфигурация приложения."""

    # Объявляем все секции конфигурации
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    augmentation_flags: AugmentationFlagsConfig
    augmentation_gap: AugmentationGapConfig
    augmentation_hole_unclosed: AugmentationHoleUnclosedConfig
    augmentation_ladder: AugmentationLadderConfig
    augmentation_tabs: AugmentationTabsConfig
    augmentation_shear: AugmentationShearConfig
    augmentation_punch: AugmentationPunchConfig
    data: DataConfig
    visualization: VisualizationConfig
    models: ModelsConfig
    service: ServiceConfig
    normalizer: NormalizerConfig
    geometry: GeometryConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

    model_config = SettingsConfigDict()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """
        Настраивает источники конфигурации.

        Args:
                settings_cls: Класс настроек
                init_settings: Инициализационные настройки
                env_settings: Настройки из переменных окружения
                dotenv_settings: Настройки из .env файла
                file_secret_settings: Секретные настройки из файлов
            Returns:
                tuple: Кортеж с источниками настроек

        """

        def yaml_settings_source() -> dict:
            """Загружает настройки из YAML файла."""
            config_path = Path("/app/gmn_project/config.yaml")
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            yaml_settings_source,
        )


# Глобальный экземпляр настроек
settings = AppSettings()
