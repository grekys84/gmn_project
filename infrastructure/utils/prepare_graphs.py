from pathlib import Path
import logging
from datetime import datetime
import json

from tqdm import tqdm
import networkx as nx
import torch

from infrastructure.scripts.svg_parser import svg_to_graph
from torch_geometric.data import Data
from domain.config.settings import settings


# Пути из конфига
INPUT_DIR = settings.data.master_svg
SAVE_DIR = settings.data.master_graph
DRAW_GRAPH = True  # Включить True для отладки визуализации
VISUALIZATION_OUTPUT_DIR = settings.visualization.prepare_graphs

# Ограничения на размер графа, чтобы предотвратить создание огромных файлов
MAX_NODES = 100_000
MAX_EDGES = 300_000


# Путь к файлу лога
log_file = settings.logging.log_dir / "prepare_graphs.log"

# === Настройка логгера ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("PrepareGraphs")


# === Сохранение графа ===
def save_graph(
    graph: nx.Graph, out_dir: Path, graph_id: str, source_file: Path
):
    # --- PyG формат (.pt)
    edge_index = (
        torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    )
    x = torch.tensor(
        [[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in graph.nodes],
        dtype=torch.float,
    )
    data = Data(x=x, edge_index=edge_index)
    torch.save(data, out_dir / f"{graph_id}.pt")

    # --- Метаинформация (.json)
    meta = {
        "id": graph_id,
        "source": source_file.stem,
        "created": datetime.now().isoformat(timespec="seconds"),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
    }
    with open(out_dir / f"{graph_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# === Визуализация ===
def draw_graph(g, title="Graph", save_path=VISUALIZATION_OUTPUT_DIR):
    """
    Визуализирует граф.

    Args:
        g: networkx.Graph
        title: заголовок
        save_path: путь для сохранения (опционально)

    """
    import matplotlib.pyplot as plt

    # Позиции узлов: (x, y), Y инвертирован — как в SVG
    pos = {n: (g.nodes[n]["x"], -g.nodes[n]["y"]) for n in g.nodes}

    plt.figure(figsize=(10, 8))
    nx.draw(
        g,
        pos,
        with_labels=False,
        node_size=30,
        node_color="red",
        edge_color="black",
        linewidths=1.5,
        alpha=0.9,
    )
    plt.title(title, fontsize=14)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Граф визуализирован: {save_path}")

    else:
        plt.show()
    plt.close()


# === Основной запуск ===
def main():
    input_dir = INPUT_DIR
    graph_save_dir = SAVE_DIR
    viz_save_dir = VISUALIZATION_OUTPUT_DIR

    # Поиск SVG-файлов (с учётом регистра)
    svg_files = sorted(
        set(input_dir.glob("*.svg")) | set(input_dir.glob("*.SVG"))
    )

    # Сортируем для консистентности
    svg_files.sort()

    logger.info(f"Найдено {len(svg_files)} уникальных SVG-файлов...")
    logger.info(f"Список файлов: {[f.name for f in svg_files]}")

    for idx, svg_path in enumerate(tqdm(svg_files, desc="Конвертация")):
        try:
            graph = svg_to_graph(svg_path)
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()

            if node_count > MAX_NODES or edge_count > MAX_EDGES:
                logger.warning(
                    f"{svg_path.name}: пропуск — {node_count} узлов, "
                    f"{edge_count} рёбер (лимиты: {MAX_NODES}/{MAX_EDGES})"
                )
                continue
            graph_id = f"g_{idx:04d}"

            # --- Сохраняем граф ---
            save_graph(graph, graph_save_dir, graph_id, svg_path)

            # --- Визуализируем (в visualization/prepare_graphs) ---
            if DRAW_GRAPH:
                title = (
                    f"{svg_path.name} | Nodes: {node_count}, "
                    f"Edges: {edge_count}"
                )
                save_img_path = (
                    viz_save_dir / f"{graph_id}_graph.png"
                )  # сохранять как PNG
                draw_graph(graph, title=title, save_path=save_img_path)

            logger.info(
                f"{svg_path.name}: {node_count} узлов, {edge_count} рёбер"
            )

        except Exception as e:
            logger.error(f"Ошибка при обработке {svg_path.name}: {e}")

    logger.info(f"[OK] Все графы сохранены в {SAVE_DIR}")
    logger.info(
        f"Обработано {len(svg_files)} SVG-файлов, "
        f"создано {len(list(graph_save_dir.glob('g_*.pt')))} графов"
    )


if __name__ == "__main__":
    main()
