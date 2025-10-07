import sys
import json
import networkx as nx
from pathlib import Path
from typing import Callable, Iterable, Optional
from io import BytesIO

import torch
import torch.nn.functional as F  # noqa
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch

from application.warmup import warmup_model
from infrastructure.scripts.geometry_compare import geometry_compare_slow

# from infrastructure.scripts.geometry_compare import geometry_compare_fast
from domain.models.siamese_gnn import SiameseGNN
from infrastructure.scripts.graph_utils import pyg_to_networkx
from domain.config.settings import settings

"""Модуль сопоставления графов с базой данных эмбеддингов."""

# --- Готовые пути ---
GRAPH_DB_DIR = settings.data.master_graph
DB_EMB_PATH = settings.service.embedding_db_path / "db_embeddings.pt"
DB_META_PATH = settings.service.embedding_db_path / "db_meta.json"
MODEL_PATH = Path(settings.models.current_model) / "GMN_v1.0.0.pt"
TOP_K = settings.service.top_k
CONFIDENCE_THRESHOLD = settings.service.similarity_threshold

# --- Папка для визуализаций ---
VISUALIZATION_OUTPUT_DIR = settings.data.overlays

# --- Устройство ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Глобальные переменные, инициализируемые при первом обращении
db_embeddings = None
db_meta = None
model = None


def load_resources() -> tuple[
    Optional[torch.nn.Module], Optional[torch.Tensor], Optional[list]
]:
    global db_embeddings, db_meta, model

    # Если уже загружено — просто вернуть
    if model is not None and db_embeddings is not None and db_meta is not None:
        return model, db_embeddings, db_meta
    if not DB_EMB_PATH.exists():
        raise FileNotFoundError(f"Файл {DB_EMB_PATH} не найден")
    if db_embeddings is None or db_meta is None:
        if DB_EMB_PATH.exists() and DB_META_PATH.exists():
            try:
                db_embeddings = torch.load(
                    DB_EMB_PATH, map_location=device, weights_only=False
                )
                with DB_META_PATH.open("r") as f:
                    db_meta = json.load(f)

                if len(db_meta) == 0 or db_embeddings.size(0) == 0:
                    raise RuntimeError("База эталонных графов пуста!")

                db_embeddings = F.normalize(db_embeddings, p=2, dim=1)
                print(
                    f"[INFO] Загружено {len(db_meta)} эталонных графов "
                    f"с эмбеддингами"
                )
            except (
                FileNotFoundError,
                RuntimeError,
                OSError,
            ) as e:  # pragma: no cover - информационные сообщения
                print(f"[WARN] Не удалось загрузить базу эмбеддингов: {e}")
                db_embeddings = None
                db_meta = None
        else:  # pragma: no cover
            print(
                "[WARN] Нет готовых эмбеддингов базы. "
                "Сначала сгенерируйте db_embeddings.pt и db_meta.json"
            )

    if model is None and SiameseGNN is not None and MODEL_PATH.exists():
        try:
            model = SiameseGNN(embed_dim=128).to(device)
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=device, weights_only=False)
            )
            model.eval()
            warmup_model(model)
        except (FileNotFoundError, RuntimeError, OSError) as e:
            print(f"[WARN] Не удалось загрузить модель: {e}")
            model = None
    elif model is None:  # pragma: no cover
        print(
            "[WARN] Модель не найдена или отсутствует зависимость SiameseGNN"
        )
    return model, db_embeddings, db_meta


# --- Вспомогательные функции ---
def draw_graph(
    data: Data, title: str = "Graph", save_path: Path | str | None = None
):
    """
    Визуализирует граф с помощью matplotlib и networkx.

    Args:
        data: PyG Data объект графа для визуализации
        title: заголовок графика
        save_path: путь для сохранения изображения.
        Если None — не сохраняет.

    """
    # Преобразуем PyG в NetworkX
    g = nx.Graph()
    pos = {}

    # Узлы
    for i in range(data.x.size(0)):
        x, y = data.x[i].tolist()
        g.add_node(i)
        pos[i] = (x, -y)  # инвертируем Y, как в SVG

    # Рёбра
    edge_list = data.edge_index.t().tolist()
    g.add_edges_from(edge_list)

    # Рисуем
    plt.figure(figsize=(8, 6))
    nx.draw(
        g,
        pos,
        with_labels=False,
        node_size=20,
        edge_color="black",
        node_color="red",
        linewidths=1.5,
        alpha=0.7,
    )
    plt.title(title)
    plt.axis("equal")
    try:
        plt.tight_layout()
    except Exception:
        pass

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Граф сохранён: {save_path}")

    plt.close()  # Обязательно закрываем, чтобы освободить память


def _default_graph_loader(id_str: str) -> Optional[Data]:
    try:
        # Извлекаем номер из id_str
        print(f"[DEBUG] graph_loader: id_str = {id_str}")
        # Если id_str = "g_0000", извлекаем 0
        if id_str.startswith("g_"):
            idx = int(id_str.split("_")[1])
        else:
            idx = int(id_str)
        path = GRAPH_DB_DIR / f"g_{idx:04d}.pt"
        if not path.exists():
            return None
        obj = torch.load(path, weights_only=False)
        if not isinstance(obj, Data):
            return None
        return obj
    except (ValueError, FileNotFoundError, RuntimeError):
        return None


def geometry_verified_id(
    g_pred: Data,
    candidate_ids: Iterable[str],
    graph_loader: Optional[Callable[[str], Optional[Data]]] = None,
    threshold: float = 90,
) -> Optional[str]:
    """
    Проверяет кандидатов по геометрии.

    Возвращает первый, соответствующий порогу.

    Args:
        g_pred: Предсказанный граф.
        candidate_ids: ID кандидатов.
        graph_loader: Функция загрузки графа.
        threshold: Порог сходства в процентах.

    Returns:
        ID подходящего графа или None.

    """
    threshold = float(threshold)
    graph_loader = graph_loader or _default_graph_loader
    pred_nx = pyg_to_networkx(g_pred)
    for cid in candidate_ids:
        cand = graph_loader(cid)
        if cand is None:
            continue
        cand_nx = pyg_to_networkx(cand)
        raw = geometry_compare_slow(cand_nx, pred_nx)
        try:
            percent = float(raw)
        except (TypeError, ValueError):
            continue
        if percent >= threshold:
            return cid
    return None


@torch.no_grad()
def get_embedding(graph: Data) -> torch.Tensor:
    """Строит L2-нормализованный эмбеддинг графа."""
    load_resources()
    if model is None:
        raise RuntimeError("Модель не загружена")
    if Batch is None:
        raise RuntimeError("Batch недоступен")
    batch = Batch.from_data_list([graph]).to(device)
    emb = model.forward_once(batch)
    return emb.cpu()


def _postprocess_match_result(
    graph: Data,
    top_k: int = TOP_K,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Общая логика: match + geometry verification + formatting."""
    load_resources()
    if db_embeddings is None or db_meta is None:
        raise RuntimeError("База эмбеддингов не загружена")

    matches = match_graph(graph, top_k=top_k)
    if not matches:
        return [
            {
                "reference_source": "unknown",
                "similarity": 0.0,
                "reference_nodes": None,
            }
        ]

    meta_by_id = {m["id"]: m for m in db_meta}
    candidate_ids = [m["id"] for m in matches]
    verified_id = geometry_verified_id(graph, candidate_ids)

    chosen = next((m for m in matches if m["id"] == verified_id), matches[0])
    score = chosen["similarity_percent"] / 100.0

    meta = meta_by_id.get(chosen["id"], {})
    if score >= threshold:
        return [
            {
                "reference_source": meta.get("source"),
                "similarity": round(score, 3),
                "reference_nodes": meta.get("num_nodes"),
            }
        ]
    else:
        return [
            {
                "reference_source": "unknown",
                "similarity": round(score, 3),
                "reference_nodes": None,
            }
        ]


@torch.no_grad()
def match_graph(
    g1: Data,
    master_db=None,
    top_k: int = TOP_K,
    threshold: float | None = None,
):
    """
    Находит top-k наиболее похожих графов для ``g1``.

    Args:
        g1 (Data): Граф, который необходимо сопоставить с базой.
        master_db(tuple or None): Необязательная пара (embeddings, meta).
                Если ``None`` используется глобальная база
                ``db_embeddings``/``db_meta``.
        top_k(int): Сколько наиболее похожих графов вернуть.
        threshold(float | None): Мин. значение косинусного сходства ``[0, 1]``.
                Если задано, результаты с меньшим сходством отбрасываются.

    Returns:
        list[dict]: Список словарей вида ``{"id": <глобальный ID>,
                                            "similarity_percent": <в %>}``

    """
    embeddings = db_embeddings
    meta = db_meta
    if master_db is not None:
        embeddings, meta = master_db
    else:
        load_resources()

    if embeddings is None or meta is None:
        raise RuntimeError("База эмбеддингов не загружена")

    test_embed = get_embedding(g1)

    similarities = F.cosine_similarity(
        test_embed.to(device), embeddings.to(device)
    )
    k = min(top_k, similarities.size(0))
    if k == 0:
        return []

    topk = torch.topk(similarities, k=k)
    results = []
    for idx, score in zip(
        topk.indices.tolist(), topk.values.tolist(), strict=False
    ):
        sim = float(score)
        if threshold is None or sim >= threshold:
            results.append(
                {
                    "id": meta[idx]["id"],
                    "source": meta[idx].get("source"),
                    "similarity_percent": round(sim * 100, 2),
                }
            )
    return results


def match_graph_from_path(
    graph_path: str | Path,
    top_k: int = TOP_K,
    threshold: float = CONFIDENCE_THRESHOLD,
):
    load_resources()
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Граф {graph_path} не найден")

    graph = torch.load(graph_path, weights_only=False)
    return _postprocess_match_result(graph, top_k=top_k, threshold=threshold)


def match_graph_from_bytes(
    data_bytes: bytes,
    top_k: int = TOP_K,
    threshold: float = CONFIDENCE_THRESHOLD,
):
    load_resources()
    buffer = BytesIO(data_bytes)
    graph = torch.load(buffer, weights_only=False, map_location="gpu")
    return _postprocess_match_result(graph, top_k=top_k, threshold=threshold)


# --- CLI с визуализацией ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python match_graph.py <path_to_graph.pt> [threshold]")
        sys.exit(1)

    graph_path = Path(sys.argv[1])
    if not graph_path.exists():
        raise FileNotFoundError(f"Файл не найден: {graph_path}")

    if len(sys.argv) >= 3:
        threshold = float(sys.argv[2])
    else:
        threshold = float(CONFIDENCE_THRESHOLD)

    # Используем правильную функцию для пути
    top_results = match_graph_from_path(
        graph_path, top_k=TOP_K, threshold=threshold
    )
    print(json.dumps(top_results, indent=2, ensure_ascii=False))

    # Визуализация топ-1 совпадения
    if top_results and top_results[0]["reference_source"] != "unknown":
        ref_source = top_results[0]["reference_source"]
        ref_id = None
        for meta in db_meta:
            if meta["source"] == ref_source:
                ref_id = meta["id"]
                break

        if ref_id:
            ref_path = GRAPH_DB_DIR / f"{ref_id}.pt"
            if ref_path.exists():
                ref_graph = torch.load(ref_path, weights_only=False)
                draw_graph(
                    ref_graph,
                    title=f"Лучшее совпадение: {ref_source}",
                    save_path=VISUALIZATION_OUTPUT_DIR / f"match_{ref_id}.png",
                )
            else:
                print(f"[WARN] Эталонный граф не найден: {ref_path}")
