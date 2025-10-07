import argparse
import json
import logging
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from infrastructure.scripts.graph_utils import (
    pyg_to_networkx,
    remove_frame_by_longest_edges_keep_nodes,
)
from domain.config.settings import settings


"""
Утилиты для создания пар графов в миллиметровых единицах измерения.

Ожидается, что все координаты узлов, полученные с помощью этого модуля,
будут указаны в миллиметрах.
"""


# --- Настраиваем пути ---
GRAPH_DIR = settings.data.master_graph
OUTPUT_FILE = settings.data.pair_file

# --- Параметры генерации пар ---
POSITIVE_PAIRS_PER_GRAPH = 2  # Сколько положительных пар на один граф
NEGATIVE_PAIRS_PER_GRAPH = 2  # Сколько отрицательных пар на один граф
SEED = 42

# --- Параметры аугментации ---
NOISE_SCALE = 0.5  # Масштаб шума (в мм)
ROTATION_MAX_DEG = 10.0  # Максимальный угол поворота (в градусах)

# --- Параметры удаления рамки ---
REMOVE_FRAME = True  # Удалять рамку при подготовке пар (ВКЛЮЧЕНО)

# --- Параметры визуализации ---
# Установите в True, чтобы автоматически сохранять
# изображение n-сгенерированных пар
RUN_VISUALIZATION_AFTER_GENERATION = True

# Количество пар для автоматической визуализации
NUM_PAIRS_TO_VISUALIZE = 10

# Директория для сохранения изображений визуализации
VISUALIZATION_OUTPUT_DIR = settings.visualization.prepare_pairs

# Путь к файлу лога
log_file = settings.logging.log_dir / "prepare_pairs.log"

# === Настройка логгера ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("PreparePairs")

# Установка seed для воспроизводимости
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def collect_graphs(graph_dir: Path):
    """Собирает список всех графов по мета-файлам."""
    meta_files = sorted(graph_dir.glob("g_*_meta.json"))
    if len(meta_files) == 0:
        raise FileNotFoundError("Не найдены мета-файлы графов!")
    graphs = []
    for meta_file in meta_files:
        with open(meta_file, "r") as f:
            meta = json.load(f)
        graph_id = meta["id"]
        pt_path = (graph_dir / f"{graph_id}.pt").resolve()
        graphs.append({"id": graph_id, "pt": str(pt_path), "meta": meta})
    logger.info(f"Найдено {len(graphs)} графов")
    return graphs


def load_graph(pt_path):
    """Загружает граф из .pt файла."""
    try:
        return torch.load(pt_path, weights_only=False)
    except Exception as e:
        logger.error(f"Ошибка загрузки графа {pt_path}: {e}")
        return None


def add_noise_to_nodes(data, scale=0.5):
    """Добавляет гауссов шум к координатам узлов."""
    if hasattr(data, "x") and data.x is not None and data.x.shape[1] >= 2:
        noise = torch.randn_like(data.x[:, :2]) * scale
        data = data.clone()
        data.x = data.x.clone()
        data.x[:, :2] += noise
    return data


def rotate_nodes(data, angle_deg):
    """Поворачивает узлы вокруг центра на угол в градусах."""
    if hasattr(data, "x") and data.x is not None and data.x.shape[1] >= 2:
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        r = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float)

        data = data.clone()
        data.x = data.x.clone()

        # Центрирование
        center = data.x[:, :2].mean(dim=0)
        centered = data.x[:, :2] - center

        # Поворот
        rotated = torch.matmul(centered, r.T)

        # Возврат в исходное положение
        data.x[:, :2] = rotated + center
    return data


def augment_graph(data):
    """Применяет случайную аугментацию к графу."""
    if data is None:
        return None

    aug_data = data.clone()

    # Добавляем шум
    if random.random() > 0.5:
        noise = torch.randn_like(aug_data.x[:, :2]) * NOISE_SCALE
        aug_data.x[:, :2] += noise

    # Поворот
    if random.random() > 0.5:
        angle_deg = random.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG)
        # ... поворот с data = data.clone() внутри
        aug_data = rotate_nodes(aug_data, angle_deg)

    return aug_data


def preprocess_graph_for_training(data):
    """
    Предобработка графа для обучения.

    Удаление N самых длинных рёбер (рамки), но без удаления узлов.
    """
    if data is None:
        return None

    if REMOVE_FRAME:
        data = remove_frame_by_longest_edges_keep_nodes(
            data, num_edges_to_remove=4
        )

    return data


def make_pairs(graphs):
    """Создаёт список положительных и отрицательных пар."""
    pairs = []
    logger.info("Генерация положительных пар (с аугментацией в памяти)...")
    for g in tqdm(graphs, desc="Положительные пары"):
        graph1_path = g["pt"]
        # Положительная пара: оригинал + аугментированная версия (в памяти)
        for _ in range(POSITIVE_PAIRS_PER_GRAPH):
            pairs.append(
                {
                    "graph1": graph1_path,
                    "graph2": graph1_path,
                    "label": 1,
                    "augment": True,  # Флаг: второй граф нужно аугментировать
                    "augment_both": False,
                    "remove_frame": REMOVE_FRAME,  # Флаг удаления рамки
                }
            )

    logger.info("Генерация отрицательных пар...")
    for g in tqdm(graphs, desc="Отрицательные пары"):
        graph1_path = g["pt"]
        for _ in range(NEGATIVE_PAIRS_PER_GRAPH):
            g2 = random.choice(graphs)
            while g2["id"] == g["id"]:
                g2 = random.choice(graphs)
            pairs.append(
                {
                    "graph1": graph1_path,
                    "graph2": g2["pt"],
                    "label": 0,
                    "augment": False,  # Не аугментировать
                    "remove_frame": REMOVE_FRAME,  # Флаг удаления рамки
                }
            )

    random.shuffle(pairs)
    logger.info(
        f"Сгенерировано {len(pairs)} пар: "
        f"{sum(1 for p in pairs if p['label'] == 1)} положительных, "
        f"{sum(1 for p in pairs if p['label'] == 0)} отрицательных"
    )
    return pairs


def validate_sample_pairs(pairs, sample_size=10):
    """Валидация нескольких пар для проверки корректности подготовки."""
    logger.info("Проверка корректности подготовки пар...")

    sample_pairs = random.sample(pairs, min(sample_size, len(pairs)))

    for i, pair in enumerate(sample_pairs):
        try:
            # Загружаем графы
            graph1 = load_graph(pair["graph1"])
            graph2 = load_graph(pair["graph2"])

            if graph1 is None or graph2 is None:
                logger.warning(f"Не удалось загрузить графы для пары {i}")
                continue

            # Применяем предобработку (удаление рамки)
            if pair.get("remove_frame", False):
                graph1 = preprocess_graph_for_training(graph1)
                graph2 = preprocess_graph_for_training(graph2)

            # Применяем аугментацию если нужно
            if pair.get("augment", False):
                if pair.get("augment_both", False):
                    graph1 = augment_graph(graph1)
                    graph2 = augment_graph(graph2)
                else:
                    graph2 = augment_graph(graph2)

            logger.info(
                f"Пара {i}: Граф1({graph1.x.shape[0]} узлов) vs "
                f"Граф2({graph2.x.shape[0]} узлов), "
                f"Метка: {pair['label']}, "
                f"Аугментация: {pair.get('augment', False)}"
            )

        except Exception as e:
            logger.error(f"Ошибка при проверке пары {i}: {e}")


def visualize_multiple_pairs(
    pairs_file, num_pairs=10, output_dir=None, start_index=0
):
    """Функция визуализации пар."""
    pairs_file_path = Path(pairs_file)
    if not pairs_file_path.exists():
        logger.error(f"Файл с парами не найден: {pairs_file_path}")
        return

    with open(pairs_file_path, "r") as f:
        pairs = json.load(f)
    logger.info(f"Загружено {len(pairs)} пар для визуализации")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    end_index = min(start_index + num_pairs, len(pairs))
    pairs_to_visualize = pairs[start_index:end_index]
    logger.info(f"Визуализация пар с {start_index} по {end_index - 1}")

    for i, pair in enumerate(pairs_to_visualize):
        global_index = start_index + i
        save_path = None
        if output_dir:
            save_path = (
                Path(output_dir)
                / f"pair_{global_index:04d}_label_{pair['label']}.png"
            )

        try:
            _visualize_single_pair_builtin(
                pair, global_index, save_path, apply_remove_frame=False
            )
        except Exception as e:
            logger.error(f"Ошибка при визуализации пары {global_index}: {e}")


def _visualize_single_pair_builtin(
    pair, pair_index, save_path=None, apply_remove_frame=True
):
    """
    Визуализирует одну пару графов (встроенная версия).

    Args:
        pair: Пара графов и метаданные.
        pair_index: Индекс пары в общем списке.
        save_path: Путь для сохранения изображения, если требуется.
        apply_remove_frame: Применять ли ``preprocess_graph_for_training`` для
            удаления рамки. По умолчанию ``True`` для сохранения прежнего
            поведения, но может быть отключено при визуализации,
            чтобы сохранить все рёбра.

    """
    import matplotlib.pyplot as plt

    def _load_graph(pt_path_str):
        pt_path = Path(pt_path_str)
        if not pt_path.exists():
            logger.error(f"Файл графа не найден: {pt_path}")
            return None
        return load_graph(str(pt_path))

    graph1 = _load_graph(pair["graph1"])
    graph2 = _load_graph(pair["graph2"])

    if graph1 is None or graph2 is None:
        logger.error(f"Не удалось загрузить графы для пары {pair_index}")
        return

    # Применяем предобработку (удаление рамки)
    if apply_remove_frame and pair.get("remove_frame", False):
        graph1 = preprocess_graph_for_training(graph1)
        graph2 = preprocess_graph_for_training(graph2)

    # Применяем аугментацию если нужно
    if pair.get("augment", False):
        if pair.get("augment_both", False):
            graph1 = augment_graph(graph1)
            graph2 = augment_graph(graph2)
        else:
            graph2 = augment_graph(graph2)

    # Конвертируем в NetworkX для визуализации
    g1 = pyg_to_networkx(graph1)
    g2 = pyg_to_networkx(graph2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Визуализация первого графа ---
    if g1.number_of_nodes() > 0:
        pos1 = {n: (g1.nodes[n]["x"], -g1.nodes[n]["y"]) for n in g1.nodes()}
        nx.draw(
            g1,
            pos1,
            ax=ax1,
            node_size=20,
            node_color="red",
            edge_color="black",
            linewidths=0.5,
            alpha=0.7,
            with_labels=False,
        )
    ax1.set_title(
        f"Граф 1 (оригинал)\nУзлов: {g1.number_of_nodes()}, "
        f"Рёбер: {g1.number_of_edges()}"
    )
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Визуализация второго графа ---
    title_suffix = (
        " (аугментированный)" if pair.get("augment", False) else " (оригинал)"
    )
    if g2.number_of_nodes() > 0:
        pos2 = {n: (g2.nodes[n]["x"], -g2.nodes[n]["y"]) for n in g2.nodes()}
        nx.draw(
            g2,
            pos2,
            ax=ax2,
            node_size=20,
            node_color="blue",
            edge_color="black",
            linewidths=0.5,
            alpha=0.7,
            with_labels=False,
        )
    ax2.set_title(
        f"Граф 2{title_suffix}\nУзлов: {g2.number_of_nodes()}, "
        f"Рёбер: {g2.number_of_edges()}"
    )
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- Общий заголовок ---
    pair_type = "ПОЛОЖИТЕЛЬНАЯ" if pair["label"] == 1 else "ОТРИЦАТЕЛЬНАЯ"
    fig.suptitle(
        f"Пара {pair_index} - {pair_type}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=150, bbox_inches="tight")
        logger.info(f"Пара {pair_index} сохранена: {save_path_obj}")
        plt.close(fig)
    else:
        plt.show()


def main(visualize_flag=False):
    """
    Основная функция генерации пар.

    Args:
        visualize_flag (bool): Если True,
        запускает визуализацию после генерации.

    """
    logger.info("Генерация пар графов...")
    logger.info(f"Удаление рамки: {REMOVE_FRAME}")  # Будет выводить True

    graphs = collect_graphs(GRAPH_DIR)
    pairs = make_pairs(graphs)

    # Валидация нескольких пар
    validate_sample_pairs(pairs)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(pairs, f, indent=2)

    logger.info(f"Файл пар сохранён: {OUTPUT_FILE}")
    logger.info(
        "Готово! Аугментация и удаление рамки будут "
        "применяться при загрузке в DataLoader."
    )

    # --- Визуализация ---
    # Визуализация запускается, если:
    # 1. Указан флаг --visualize в командной строке (visualize_flag=True)
    # 2. ИЛИ установлена переменная RUN_VISUALIZATION_AFTER_GENERATION=True
    if visualize_flag or RUN_VISUALIZATION_AFTER_GENERATION:
        logger.info("Запуск визуализации пар...")
        try:
            visualize_multiple_pairs(
                pairs_file=OUTPUT_FILE,
                num_pairs=NUM_PAIRS_TO_VISUALIZE,
                output_dir=VISUALIZATION_OUTPUT_DIR
                if VISUALIZATION_OUTPUT_DIR
                else None,
                start_index=0,
            )
            if VISUALIZATION_OUTPUT_DIR:
                logger.info(
                    f" Визуализации сохранены в: {VISUALIZATION_OUTPUT_DIR}"
                )
        except Exception as e:
            logger.error(f"Ошибка при визуализации: {e}")


if __name__ == "__main__":
    # --- Обработка аргументов командной строки ---
    parser = argparse.ArgumentParser(
        description="Генерация пар графов для обучения Siamese GNN"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Визуализировать пары после генерации",
    )
    # Можно добавить другие аргументы для переопределения настроек
    # parser.add_argument("--output_dir", type=str,
    # help="Директория для сохранения пар")
    # parser.add_argument("--num_positive", type=int,
    # default=POSITIVE_PAIRS_PER_GRAPH,
    # help="Кол-во положительных пар на граф")

    args = parser.parse_args()

    # Запуск основной логики с флагом визуализации
    main(visualize_flag=args.visualize)
