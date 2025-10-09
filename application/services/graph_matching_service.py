import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import time

import torch
import torch.nn.functional as F  # noqa
import networkx as nx
from torch_geometric.data import Data

from infrastructure.scripts.svg_parser import svg_to_graph
from infrastructure.utils.visualization import svg_overlay
from domain.models.siamese_gnn import SiameseGNN
from infrastructure.utils.match_graph import match_graph
from infrastructure.scripts.geometry_compare import geometry_compare_slow
from domain.config.settings import settings


"""Сервис для сопоставления SVG-графиков со справочной базой данных."""

logger = logging.getLogger("GraphMatchingService")


# --- Ошибки ---
class MissingModelError(RuntimeError):
    """Возникает, когда отсутствует файл модели."""


class MissingDatabaseError(RuntimeError):
    """Возникает, когда отсутствуют файлы эмбеддинга."""


class GraphMatchingService:
    """Сервис, организующий сопоставление графов со справочной базой данных."""

    def __init__(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        model_path: Optional[Path] = None,
        embedding_db_path: Optional[Path] = None,
        overlay_dir: Optional[Path] = None,
    ) -> None:
        """
        Инициализирует сервис.

        Загружает модель, базу эмбеддингов и настройки.
        """
        # --- Устройство ---
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --- Пути ---
        self.model_path = (
            model_path or Path(settings.models.current_model) / "GMN_v1.0.0.pt"
        )
        self.embedding_db_path = embedding_db_path or Path(
            settings.service.embedding_db_path
        )
        self.graph_db_dir = Path(settings.data.master_graph)
        self.overlay_dir = overlay_dir or Path(settings.data.overlays)

        self.db_emb_path = self.embedding_db_path / "db_embeddings.pt"
        self.db_meta_path = self.embedding_db_path / "db_meta.json"

        # --- Конфигурация ---
        self.top_k = top_k if top_k is not None else settings.service.top_k
        self.threshold = (
            threshold
            if threshold is not None
            else settings.service.similarity_threshold
        )
        self.slow_geometry_threshold = settings.geometry.slow_threshold

        # --- Загрузка модели ---
        if not self.model_path.exists():
            raise MissingModelError(f"Model file not found: {self.model_path}")
        if self.model_path.is_dir():
            raise MissingModelError(
                f"Model path is a directory: {self.model_path}"
            )

        self.model = SiameseGNN(embed_dim=128).to(self.device)
        state_dict = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # --- Загрузка базы ---
        if not self.db_emb_path.exists():
            raise MissingDatabaseError(
                f"db_embeddings.pt not found: {self.db_emb_path}"
            )
        if not self.db_meta_path.exists():
            raise MissingDatabaseError(
                f"db_meta.json not found: {self.db_meta_path}"
            )

        self.db_embeddings = torch.load(
            self.db_emb_path, map_location=self.device, weights_only=False
        )
        with self.db_meta_path.open("r", encoding="utf-8") as f:
            self.db_meta = json.load(f)

        if len(self.db_meta) == 0 or self.db_embeddings.size(0) == 0:
            raise MissingDatabaseError("База данных эмбеддингов отсутствует.")

        self.db_embeddings = F.normalize(self.db_embeddings, p=2, dim=1)

        # --- Директории и отчёт ---
        self.log_dir = settings.logging.log_dir

        self.report_path = self.log_dir / "endpoint.csv"

        # Создаём CSV с заголовком, если файла нет
        if not self.report_path.exists():
            with self.report_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",  # дата и время обработки
                        "input_filename",  # имя входящего файла
                        "matched_id",  # ID из базы (например, g_0001)
                        "matched_source",  # source (например, имя эталона)
                        "similarity_percent",  # процент сходства
                        "valid",  # признано валидным или нет
                    ]
                )

        # --- Метаданные ---
        self.id_to_source = {m["id"]: m.get("source") for m in self.db_meta}

    # ------------------------------------------------------------------
    @staticmethod
    def _data_to_networkx(data: Data) -> nx.Graph:
        """Конвертирует PyG ``Data`` граф в NetworkX граф с координатами."""
        g = nx.Graph()
        for i in range(data.x.size(0)):
            x, y = data.x[i].tolist()
            g.add_node(
                int(i), x=float(x), y=float(y), pos=(float(x), float(y))
            )
        edges = data.edge_index.t().tolist()
        g.add_edges_from(edges)
        return g

    # ------------------------------------------------------------------
    def predict_path(
        self, svg_path: Path, filename: Optional[str] = None
    ) -> dict:
        """
        Возвращает лучшее совпадение для SVG-файла по пути.

        Args:
              svg_path (Path): Путь к SVG-файлу.
              filename (str, optional): Имя файла (используется в логах).
              По умолчанию — имя файла из пути.

        Returns:
               dict: Результат с ключами: 'id', 'similarity_percent',
               'overlay_path', 'valid'

        """
        filename = filename or svg_path.name
        logger.info(f"Начало обработки SVG-файла: {svg_path}")
        start_total = time.perf_counter()

        # 1. Конвертируем SVG → Data
        logger.info("Парсинг SVG...")
        start_step = time.perf_counter()
        try:
            nx_graph = svg_to_graph(svg_path)
            elapsed_time = time.perf_counter() - start_step

            # --- ЛОГИРОВАНИЕ ИНФОРМАЦИИ ИЗ SVG_TO_GRAPH ---
            # Теперь логируем информацию о графе здесь, используя наш логгер
            num_nodes = nx_graph.number_of_nodes()
            num_edges = nx_graph.number_of_edges()
            width_mm = nx_graph.graph.get("width_mm", 0)
            height_mm = nx_graph.graph.get("height_mm", 0)
            area_mm2 = nx_graph.graph.get("area_mm2", 0)

            logger.info(
                f"SVG успешно преобразован: {num_nodes} узлов, "
                f"{num_edges} рёбер "
                f"(за {elapsed_time:.2f} с)"
            )
            logger.info(
                f"Габариты детали: {width_mm:.3f} × {height_mm:.3f} мм "
                f"(площадь: {area_mm2:.3f} мм²)"
            )

        except Exception as e:
            logger.error(f"Ошибка парсинга SVG {svg_path}: {e}")
            raise ValueError(f"Ошибка парсинга SVG: {e}")

        x = torch.tensor(
            [
                [nx_graph.nodes[n]["x"], nx_graph.nodes[n]["y"]]
                for n in nx_graph.nodes
            ],
            dtype=torch.float,
        )
        edge_index = (
            torch.tensor(list(nx_graph.edges), dtype=torch.long)
            .t()
            .contiguous()
        )
        data = Data(x=x, edge_index=edge_index)
        pred_nx = self._data_to_networkx(data)

        # 2. Используем готовую функцию поиска по эмбеддингам
        logger.info("Поиск по базе эмбеддингов...")
        start_step = time.perf_counter()
        matches = match_graph(
            data, master_db=(self.db_embeddings, self.db_meta)
        )
        logger.info(
            f"Найдено {len(matches)} кандидатов (за "
            f"{time.perf_counter() - start_step:.2f} с)"
        )

        # 3. Геометрическая проверка
        logger.info("Геометрическая проверка совпадений...")
        start_geo = time.perf_counter()
        best_id: Optional[str] = None
        best_percent: float = 0.0
        best_cand_nx: Optional[nx.Graph] = None
        valid = False

        def _load_candidate_graph(candidate_id: str) -> Optional[nx.Graph]:
            """Загружает граф кандидата из базы."""

            cand_path = self.graph_db_dir / f"{candidate_id}.pt"
            if not cand_path.exists():
                logger.warning(f"Файл кандидата не найден: {cand_path}")
                return None

            try:
                cand_data = torch.load(cand_path, weights_only=False)
            except Exception as e:  # pragma: no cover - логирование ошибок
                logger.warning(
                    f"Ошибка загрузки кандидата {candidate_id}: {e}"
                )
                return None

            try:
                return self._data_to_networkx(cand_data)
            except Exception as e:  # pragma: no cover - логирование ошибок
                logger.warning(
                    f"Ошибка преобразования кандидата {candidate_id}: {e}"
                )
                return None

        below_threshold: list[tuple[dict, str]] = []
        threshold_percent = (
            self.threshold * 100 if self.threshold <= 1 else self.threshold
        )

        for item in matches:
            cand_id = item["id"]
            sim_percent = item["similarity_percent"] / 100.0  # в [0,1]

            # 3.1. Проверка порога по GNN
            if sim_percent < self.threshold:
                logger.debug(
                    f"Пропуск {cand_id}: сходство {sim_percent:.3f} < "
                    f"порога {self.threshold}"
                )
                cand_meta = next(
                    (m for m in self.db_meta if m["id"] == cand_id), {}
                )
                cand_source = cand_meta.get("source", "unknown_source")
                below_threshold.append((item, cand_source))
                continue
            # Найти source для cand_id из метаданных
            cand_meta = next(
                (m for m in self.db_meta if m["id"] == cand_id), {}
            )
            cand_source = cand_meta.get("source", "unknown_source")

            logger.info(
                f"Проверка кандидата: "
                f"{cand_id} ({cand_source}) (GNN сходство: "
                f"{sim_percent:.3f})"
            )

            # 3.2. Загрузка эталонного графа
            cand_nx = _load_candidate_graph(cand_id)
            if cand_nx is None:
                continue

            # 3.3. Геометрическое сравнение
            try:
                robust_score = geometry_compare_slow(
                    pred_nx,
                    cand_nx,
                    tol=settings.geometry.tol,
                    angle_tol=settings.geometry.angle_tol,
                    use_angles=settings.geometry.use_angles,
                    normalize_position=settings.geometry.normalize_position,
                    normalize_scale=settings.geometry.normalize_scale,
                )
                logger.info(f"Точное сравнение (slow): {robust_score:.2f}%")

                if robust_score >= self.slow_geometry_threshold:
                    logger.info(
                        f"Совпадение подтверждено: "
                        f"{cand_id} ({cand_source}), точность: "
                        f"{robust_score:.2f}%"
                    )
                    best_id = cand_id
                    best_percent = robust_score
                    best_cand_nx = cand_nx
                    valid = True
                    break
                else:
                    logger.debug(
                        f"Не прошёл точное сравнение: {robust_score:.2f}% < "
                        f"{self.slow_geometry_threshold}%"
                    )
            except Exception as e:
                logger.warning(
                    f"Ошибка в geometry_compare_slow "
                    f"для {cand_id} ({cand_source}): {e}"
                )
        logger.info(
            f"Геометрическая проверка завершена за "
            f"{time.perf_counter() - start_geo:.2f} с"
        )

        if not valid and below_threshold:
            fallback_item, fallback_source = max(
                below_threshold,
                key=lambda x: x[0].get("similarity_percent", 0.0),
            )
            fallback_id = fallback_item["id"]
            fallback_similarity = fallback_item.get("similarity_percent", 0.0)
            logger.info(
                "Ни один кандидат не подтверждён геометрической проверкой. "
                "Запускаем проверку лучшего кандидата ниже порога GNN %s (%s): "
                "%s%% < %s%%.",
                fallback_id,
                fallback_source,
                f"{fallback_similarity:.2f}",
                f"{threshold_percent:.2f}",
            )

            cand_nx = _load_candidate_graph(fallback_id)
            if cand_nx is not None:
                try:
                    robust_score = geometry_compare_slow(
                        pred_nx,
                        cand_nx,
                        tol=settings.geometry.tol,
                        angle_tol=settings.geometry.angle_tol,
                        use_angles=settings.geometry.use_angles,
                        normalize_position=settings.geometry.normalize_position,
                        normalize_scale=settings.geometry.normalize_scale,
                    )
                    logger.info(
                        "Точное сравнение (slow) для fallback-кандидата %s: %.2f%%",
                        fallback_id,
                        robust_score,
                    )
                    if robust_score >= self.slow_geometry_threshold:
                        logger.info(
                            "Совпадение подтверждено по fallback-кандидату: %s (%s), точность: %.2f%%",
                            fallback_id,
                            fallback_source,
                            robust_score,
                        )
                        best_id = fallback_id
                        best_percent = robust_score
                        best_cand_nx = cand_nx
                        valid = True
                except Exception as e:  # pragma: no cover - логирование ошибок
                    logger.warning(
                        "Ошибка в geometry_compare_slow для fallback-кандидата %s (%s): %s",
                        fallback_id,
                        fallback_source,
                        e,
                    )

        # 4. Оверлей
        logger.info("Создание оверлея...")
        start_step = time.perf_counter()
        overlay_path = None
        if best_id is not None:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = self.overlay_dir / f"{best_id}_overlay.svg"
            svg_overlay(pred_nx, best_cand_nx, overlay_path)
            logger.info(
                f"Оверлей сохранён: {overlay_path} "
                f"(за {time.perf_counter() - start_step:.2f} с)"
            )

        # Берём в поле id именно значение "source" из мета-информации
        meta = next((m for m in self.db_meta if m["id"] == best_id), {})
        source = meta.get("source")

        # Лог в CSV
        logger.info("Запись в отчёт...")
        with self.report_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    filename,
                    best_id or "",
                    source or "",
                    round(best_percent, 2),
                    valid,
                ]
            )
            logger.info("Запись в CSV завершена")

            total_time = time.perf_counter() - start_total
            logger.info(f"Обработка завершена за {total_time:.2f} секунд")

        return {
            "id": source,
            "similarity_percent": round(best_percent, 2),
            "overlay_path": str(overlay_path) if overlay_path else None,
            "valid": valid,
        }

    # ------------------------------------------------------------------
    def predict_bytes(
        self, file_bytes: bytes, filename: str = "uploaded.svg"
    ) -> dict:
        """
        Возвращает лучшее совпадение для SVG-файла в виде байтов.

        Args:
            file_bytes (bytes): Содержимое SVG-файла.
            filename (str): Имя файла (для логов).

        Returns:
            dict: Результат, как в `predict_path`.

        """
        logger.info(
            f"Начало обработки файла: {filename}, "
            f"размер: {len(file_bytes)} байт"
        )

        from tempfile import NamedTemporaryFile
        import os

        with NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self.predict_path(tmp_path, filename=filename)
        except Exception as e:
            logger.error(
                f"Ошибка в predict_bytes для {filename}: {e}", exc_info=True
            )
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, FileNotFoundError):
                pass
