import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from infrastructure.scripts.svg_parser import svg_to_graph
from infrastructure.utils.visualization import svg_overlay
from domain.models.siamese_gnn import SiameseGNN
from infrastructure.utils.match_graph import match_graph
from infrastructure.scripts.geometry_compare import geometry_compare_slow
from domain.config.settings import settings


logger = logging.getLogger("GraphMatchingService")


# ============================================================
#                   ИСКЛЮЧЕНИЯ
# ============================================================

class MissingModelError(RuntimeError):
    """Возникает, когда отсутствует файл модели."""


class MissingDatabaseError(RuntimeError):
    """Возникает, когда отсутствует база эмбеддингов."""


# ============================================================
#                   РЕЗУЛЬТАТ СРАВНЕНИЯ
# ============================================================

@dataclass
class MatchResult:
    """Результат сопоставления одного кандидата."""

    id: Optional[str] = None
    source: Optional[str] = None
    percent: float = 0.0
    valid: bool = False
    reason: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"<MatchResult id={self.id} valid={self.valid} "
            f"percent={self.percent:.2f} reason={self.reason}>"
        )

    def to_dict(self) -> dict:
        """Преобразует объект в словарь."""
        return asdict(self)


# ============================================================
#                   ОСНОВНОЙ СЕРВИС
# ============================================================

class GraphMatchingService:
    """Сервис для сопоставления SVG-графов со справочной базой данных."""

    def __init__(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        model_path: Optional[Path] = None,
        embedding_db_path: Optional[Path] = None,
        overlay_dir: Optional[Path] = None,
    ) -> None:
        """Инициализация и загрузка необходимых компонентов."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Пути
        self.model_path = model_path or Path(settings.models.current_model) / "GMN_v1.0.0.pt"
        self.embedding_db_path = embedding_db_path or Path(settings.service.embedding_db_path)
        self.graph_db_dir = Path(settings.data.master_graph)
        self.overlay_dir = overlay_dir or Path(settings.data.overlays)

        self.db_emb_path = self.embedding_db_path / "db_embeddings.pt"
        self.db_meta_path = self.embedding_db_path / "db_meta.json"

        # Конфигурация
        self.top_k = top_k or settings.service.top_k
        self.threshold = threshold or settings.service.similarity_threshold
        self.slow_geometry_threshold = settings.geometry.slow_threshold
        self.max_node_mismatch_percent = settings.geometry.max_node_mismatch_percent
        self.max_edge_mismatch_percent = settings.geometry.max_edge_mismatch_percent
        self.max_bbox_mismatch_percent = settings.geometry.max_bbox_mismatch_percent

        # Загрузка модели
        if not self.model_path.exists():
            raise MissingModelError(f"Model file not found: {self.model_path}")
        if self.model_path.is_dir():
            raise MissingModelError(f"Model path is a directory: {self.model_path}")

        self.model = SiameseGNN(embed_dim=128).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Загрузка базы
        if not self.db_emb_path.exists() or not self.db_meta_path.exists():
            raise MissingDatabaseError("Отсутствует база эмбеддингов.")

        self.db_embeddings = torch.load(self.db_emb_path, map_location=self.device, weights_only=False)
        with self.db_meta_path.open("r", encoding="utf-8") as f:
            self.db_meta = json.load(f)

        if len(self.db_meta) == 0 or self.db_embeddings.size(0) == 0:
            raise MissingDatabaseError("База данных эмбеддингов отсутствует.")

        self.db_embeddings = F.normalize(self.db_embeddings, p=2, dim=1)

        # Отчёт
        self.log_dir = settings.logging.log_dir
        self.report_path = self.log_dir / "endpoint.csv"
        if not self.report_path.exists():
            with self.report_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "input_filename", "matched_id", "matched_source", "similarity_percent", "valid", "candidate_nodes", "candidate_edges", "candidate_width_mm", "candidate_height_mm", "candidate_area_mm2"])

    # ============================================================
    #                   ОСНОВНОЙ МЕТОД
    # ============================================================

    def predict_path(self, svg_path: Path, filename: Optional[str] = None) -> dict:
        """Находит наиболее похожий граф из базы по SVG-файлу."""
        filename = filename or svg_path.name
        logger.info(f"Начало обработки SVG-файла: {svg_path}")
        start_total = time.perf_counter()

        nx_graph = self._parse_svg(svg_path)
        data, pred_nx = self._graph_from_nx(nx_graph)
        matches = self._find_embedding_matches(data)
        best_result = self._select_best_match(pred_nx, matches)
        overlay_path = self._generate_overlay(pred_nx, best_result)
        self._write_report(filename, best_result, overlay_path, start_total)

        logger.info(
            f"Финальный результат: {best_result.source or '—'}, "
            f"сходство {best_result.percent:.2f}%, valid={best_result.valid}"
        )

        return {
            "id": best_result.source,
            "similarity_percent": round(best_result.percent, 2),
            "overlay_path": str(overlay_path) if overlay_path else None,
            "valid": best_result.valid,
        }

    # ============================================================
    #                   ОБЁРТКА ДЛЯ БАЙТОВ
    # ============================================================

    def predict_bytes(self, file_bytes: bytes, filename: str = "uploaded.svg") -> dict:
        """Принимает SVG-файл в виде байтов и выполняет предсказание."""
        logger.info(f"Начало обработки файла: {filename}, размер: {len(file_bytes)} байт")

        from tempfile import NamedTemporaryFile
        import os

        with NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self.predict_path(tmp_path, filename=filename)
        except Exception as e:
            logger.error(f"Ошибка в predict_bytes для {filename}: {e}", exc_info=True)
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, FileNotFoundError):
                pass

    # ============================================================
    #                   ПАРСИНГ И ГРАФ
    # ============================================================

    def _parse_svg(self, svg_path: Path) -> nx.Graph:
        """Преобразует SVG-файл в граф NetworkX и логирует базовые параметры."""
        try:
            start = time.perf_counter()
            nx_graph = svg_to_graph(svg_path)
            elapsed = time.perf_counter() - start

            num_nodes = nx_graph.number_of_nodes()
            num_edges = nx_graph.number_of_edges()
            width_mm = nx_graph.graph.get("width_mm", 0)
            height_mm = nx_graph.graph.get("height_mm", 0)
            area_mm2 = nx_graph.graph.get("area_mm2", 0)

            logger.info(f"SVG преобразован: {num_nodes} узлов, {num_edges} рёбер (за {elapsed:.2f} с)")
            logger.info(f"Габариты детали: {width_mm:.3f} × {height_mm:.3f} мм (площадь: {area_mm2:.3f} мм²)")
            return nx_graph
        except Exception as e:
            logger.error(f"Ошибка парсинга SVG {svg_path}: {e}")
            raise ValueError(f"Ошибка парсинга SVG: {e}")

    def _graph_from_nx(self, nx_graph: nx.Graph) -> tuple[Data, nx.Graph]:
        """Преобразует граф NetworkX в PyTorch Geometric Data и обратно."""
        x = torch.tensor([[nx_graph.nodes[n]["x"], nx_graph.nodes[n]["y"]] for n in nx_graph.nodes], dtype=torch.float)
        edge_index = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        return data, self._data_to_networkx(data)

    def _find_embedding_matches(self, data: Data) -> list[dict]:
        """Находит ближайшие эмбеддинги графов из базы."""
        start = time.perf_counter()
        matches = match_graph(data, master_db=(self.db_embeddings, self.db_meta), top_k=self.top_k, threshold=None)
        logger.info(f"Найдено {len(matches)} кандидатов (за {time.perf_counter() - start:.2f} с)")
        return matches

    # ============================================================
    #                   ПРОВЕРКА КАНДИДАТОВ
    # ============================================================

    def _select_best_match(self, pred_nx: nx.Graph, matches: list[dict]) -> MatchResult:
        """Проверяет всех кандидатов и выбирает лучший."""
        pred_stats = self._graph_measurements(pred_nx)
        best = MatchResult()
        for item in matches:
            result = self._evaluate_candidate(item, pred_nx, pred_stats)
            if result.valid:
                return result
        logger.info("Все кандидаты отклонены, совпадений не найдено.")
        return best

    def _generate_overlay(self, pred_nx: nx.Graph, best_result: MatchResult) -> Optional[Path]:
        """Создаёт SVG-оверлей между исходным графом и найденным кандидатом."""
        if not best_result.valid or not best_result.id:
            return None
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = self.overlay_dir / f"{best_result.id}_overlay.svg"
        cand_nx = self._load_candidate_graph(best_result.id)
        if cand_nx:
            svg_overlay(pred_nx, cand_nx, overlay_path)
            logger.info(f"Оверлей сохранён: {overlay_path}")
            return overlay_path
        return None

    def _evaluate_candidate(self, item: dict, pred_nx: nx.Graph, pred_stats: dict) -> MatchResult:
        """Проверяет одного кандидата (GNN + геометрия) с подробным логированием."""
        cand_id = item["id"]
        cand_meta = next((m for m in self.db_meta if m["id"] == cand_id), {})
        cand_source = cand_meta.get("source", "unknown_source")
        sim_percent = item["similarity_percent"]

        logger.info(f" ▶️ Проверка кандидата: {cand_id} ({cand_source}) ")
        logger.info(f"GNN сходство {sim_percent:.2f}%")

        if sim_percent < self.threshold:
            logger.info(f"Пропуск {cand_id} ({cand_source}) — ниже порога GNN ({self.threshold:.1f}%)")
            return MatchResult(id=cand_id, source=cand_source, reason="gnn_threshold")

        cand_nx = self._load_candidate_graph(cand_id)
        if cand_nx is None:
            return MatchResult(id=cand_id, source=cand_source, reason="load_failed")

        cand_stats = self._graph_measurements(cand_nx)

        # --- Лог сравнения размеров ---
        node_diff = self._percentage_diff(pred_stats["nodes"], cand_stats["nodes"])
        edge_diff = self._percentage_diff(pred_stats["edges"], cand_stats["edges"])
        width_diff = self._percentage_diff(pred_stats["width"], cand_stats["width"])
        height_diff = self._percentage_diff(pred_stats["height"], cand_stats["height"])
        area_diff = self._percentage_diff(pred_stats["area"], cand_stats["area"])
        logger.info(
            f"Сравнение размеров: узлы {pred_stats['nodes']}→{cand_stats['nodes']} (Δ {node_diff:.2f}%), "
            f"рёбра {pred_stats['edges']}→{cand_stats['edges']} (Δ {edge_diff:.2f}%), "
            f"ширина {pred_stats['width']:.1f}→{cand_stats['width']:.1f} (Δ {width_diff:.2f}%), "
            f"высота {pred_stats['height']:.1f}→{cand_stats['height']:.1f} (Δ {height_diff:.2f}%), "
            f"площадь {pred_stats['area']:.1f}→{cand_stats['area']:.1f} (Δ {area_diff:.2f}%)"
        )

        size_reason = self._validate_geometry_size(pred_stats, cand_stats)
        if size_reason:
            logger.info(f" 🔴 Отказ {cand_id} ({cand_source}) — несоответствие размеров: {size_reason}")
            return MatchResult(id=cand_id, source=cand_source, reason="geometry_size")

        try:
            result = geometry_compare_slow(
                pred_nx, cand_nx,
                tol=settings.geometry.tol,
                angle_tol=settings.geometry.angle_tol,
                use_angles=settings.geometry.use_angles,
                normalize_position=settings.geometry.normalize_position,
                normalize_scale=settings.geometry.normalize_scale,
                feature_weights=settings.geometry.feature_weights,
                feature_coverage_importance=(
                    settings.geometry.feature_coverage_importance
                ),
            )
            if isinstance(result, dict):
                score = result.get("score", 0.0)
                node_match = result.get("node_match", 0.0)
                edge_match = result.get("edge_match", 0.0)
                logger.info(f"Геометрия: сходство {score:.2f}%, узлы {node_match:.2f}%, рёбра {edge_match:.2f}%")
            else:
                score = float(result)
                logger.info(f"Геометрия: сходство {score:.2f}%")

            if score >= self.slow_geometry_threshold:
                logger.info(f"✅ Совпадение подтверждено: {cand_id} ({cand_source})")
                return MatchResult(id=cand_id, source=cand_source, percent=score, valid=True)
            return MatchResult(id=cand_id, source=cand_source, percent=score, reason="geometry_mismatch")

        except Exception as e:
            logger.warning(f"❌ Ошибка сравнения {cand_id}: {e}")
            return MatchResult(id=cand_id, source=cand_source, reason="geometry_exception")

    def _write_report(
            self,
            filename: str,
            best_result: MatchResult,
            overlay_path: Optional[Path],
            start_total: float) -> None:
        """Добавляет запись в CSV-отчёт с расширенной информацией о совпадении."""
        try:
            # --- логируем подробности ---
            if best_result and best_result.valid:
                cand_nx = self._load_candidate_graph(best_result.id)
                if cand_nx:
                    cand_stats = self._graph_measurements(cand_nx)
                    logger.info(
                        f"Финальный кандидат: {best_result.id} ({best_result.source}) | "
                        f"узлы={cand_stats['nodes']}, рёбра={cand_stats['edges']}, "
                        f"ширина={cand_stats['width']:.2f} мм, высота={cand_stats['height']:.2f} мм, "
                        f"площадь={cand_stats['area']:.1f} мм²"
                    )
                else:
                    cand_stats = {"nodes": 0, "edges": 0, "width": 0.0, "height": 0.0, "area": 0.0}
            else:
                cand_stats = {"nodes": 0, "edges": 0, "width": 0.0, "height": 0.0, "area": 0.0}

            # --- запись в CSV ---
            with self.report_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    filename,
                    best_result.id or "",
                    best_result.source or "",
                    round(best_result.percent, 2),
                    best_result.valid,
                    cand_stats["nodes"],
                    cand_stats["edges"],
                    round(cand_stats["width"], 2),
                    round(cand_stats["height"], 2),
                    round(cand_stats["area"], 2),
                ])

            logger.info(
                f"Запись в CSV завершена (время: {time.perf_counter() - start_total:.2f} с), "
                f"добавлены геометрические параметры кандидата."
            )

        except Exception as e:
            logger.error(f"Ошибка при записи отчёта: {e}", exc_info=True)

    # ============================================================
    #                   ВАЛИДАЦИЯ РАЗМЕРОВ
    # ============================================================

    def _validate_geometry_size(self, pred_stats: dict, cand_stats: dict) -> Optional[str]:
        """Проверяет базовые геометрические параметры (узлы, рёбра, bbox, площадь)."""
        node_diff = self._percentage_diff(pred_stats["nodes"], cand_stats["nodes"])
        if node_diff > self.max_node_mismatch_percent:
            return f"разница в количестве узлов {node_diff:.2f}%"

        edge_diff = self._percentage_diff(pred_stats["edges"], cand_stats["edges"])
        if edge_diff > self.max_edge_mismatch_percent:
            return f"разница в количестве рёбер {edge_diff:.2f}%"

        width_diff = self._percentage_diff(pred_stats["width"], cand_stats["width"])
        height_diff = self._percentage_diff(pred_stats["height"], cand_stats["height"])
        area_diff = self._percentage_diff(pred_stats["area"], cand_stats["area"])
        bbox_diff = max(width_diff, height_diff, area_diff)
        if bbox_diff > self.max_bbox_mismatch_percent:
            return f"отличие габаритов/площади {bbox_diff:.2f}%"

        abs_width_diff = abs(pred_stats["width"] - cand_stats["width"])
        abs_height_diff = abs(pred_stats["height"] - cand_stats["height"])
        abs_area_diff = abs(pred_stats["area"] - cand_stats["area"])
        max_abs_width = settings.geometry.max_absolute_bbox_diff_mm
        max_abs_area = settings.geometry.max_absolute_area_diff_mm2

        if abs_width_diff > max_abs_width or abs_height_diff > max_abs_width:
            return f"абсолютная разница габаритов {abs_width_diff:.2f}×{abs_height_diff:.2f} мм (допуск ±{max_abs_width:.1f} мм)"
        if abs_area_diff > max_abs_area:
            return f"абсолютная разница площади {abs_area_diff:.1f} мм² (допуск ±{max_abs_area:.0f} мм²)"

        ratio_pred = pred_stats["width"] / max(pred_stats["height"], 1e-9)
        ratio_cand = cand_stats["width"] / max(cand_stats["height"], 1e-9)
        ratio_diff = self._percentage_diff(ratio_pred, ratio_cand)
        if ratio_diff > 10.0:
            return f"несоответствие пропорций (ширина/высота) {ratio_diff:.1f}%"
        return None

    # ============================================================
    #                   ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
    # ============================================================

    @staticmethod
    def _data_to_networkx(data: Data) -> nx.Graph:
        """Конвертирует PyTorch Geometric Data в NetworkX-граф."""
        g = nx.Graph()
        for i in range(data.x.size(0)):
            x, y = data.x[i].tolist()
            g.add_node(int(i), x=float(x), y=float(y), pos=(float(x), float(y)))
        edges = data.edge_index.t().tolist()
        g.add_edges_from(edges)
        return g

    @staticmethod
    def _graph_measurements(graph: nx.Graph) -> dict[str, float]:
        """Вычисляет базовые геометрические характеристики графа."""
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        xs = [float(graph.nodes[n].get("x", 0.0)) for n in graph.nodes]
        ys = [float(graph.nodes[n].get("y", 0.0)) for n in graph.nodes]
        if xs and ys:
            width, height = max(xs) - min(xs), max(ys) - min(ys)
        else:
            width = height = 0.0
        area = width * height
        return {"nodes": nodes, "edges": edges, "width": width, "height": height, "area": area}

    @staticmethod
    def _percentage_diff(a: float, b: float) -> float:
        """Возвращает относительную разницу между a и b в процентах."""
        baseline = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / baseline * 100.0

    def _load_candidate_graph(self, candidate_id: str) -> Optional[nx.Graph]:
        """Загружает граф-кандидат из базы данных по ID."""
        cand_path = self.graph_db_dir / f"{candidate_id}.pt"
        if not cand_path.exists():
            logger.warning(f"Файл кандидата не найден: {cand_path}")
            return None
        try:
            cand_data = torch.load(cand_path, weights_only=False)
            return self._data_to_networkx(cand_data)
        except Exception as e:
            logger.warning(f"Ошибка загрузки кандидата {candidate_id}: {e}")
            return None
