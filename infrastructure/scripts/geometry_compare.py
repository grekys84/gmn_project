import math
import networkx as nx

import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations
from typing import Mapping, Optional, Union


def extract_and_normalize_coordinates(G, normalize_position, normalize_scale):  # noqa N803
    """
    Извлекает и нормализует координаты узлов графа.

    Args:
        G: Граф NetworkX
        normalize_position: Флаг нормализации положения
        normalize_scale: Флаг нормализации масштаба

    Returns:
        numpy.ndarray: Нормализованные координаты

    """
    # Извлекаем координаты
    coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G.nodes])

    # Нормализация положения (сдвиг)
    if normalize_position:
        coords = coords - coords.mean(axis=0)

    # Нормализация масштаба (опционально)
    if normalize_scale:
        scale = np.max(cdist(coords, coords)) if len(coords) > 1 else 1.0
        if scale > 1e-6:
            coords = coords / scale

    return coords


def compare_node_coordinates(coords_pred, coords_true, tol):
    """
    Сравнивает координаты узлов двух графов.

    Args:
        coords_pred: Координаты предсказанного графа
        coords_true: Координаты эталонного графа
        tol: Допуск для сравнения

    Returns:
        tuple: (количество совпадений, общее количество узлов)

    """
    if len(coords_pred) == 0 or len(coords_true) == 0:
        return 0, len(coords_true)

    dist_matrix = cdist(coords_pred, coords_true)
    min_dists = dist_matrix.min(axis=1)
    matched_coords = np.sum(min_dists <= tol)
    total_coords = len(coords_true)

    return matched_coords, total_coords


def get_edge_vectors(G, coords):  # noqa N803
    """
    Получает нормализованные векторы рёбер графа.

    Args:
        G: Граф NetworkX
        coords: Координаты узлов

    Returns:
        list: Список нормализованных векторов рёбер

    """
    vectors = []
    node_list = list(G.nodes)

    for u, v in G.edges():
        try:
            i = node_list.index(u)
            j = node_list.index(v)
            vec = coords[j] - coords[i]
            length = np.linalg.norm(vec)
            if length > 1e-6:  # Используем маленькую константу вместо tol
                vectors.append(vec / length)
        except Exception:
            continue

    return vectors


def compare_edge_directions(G_pred, G_true, coords_pred, coords_true, tol):  # noqa N803
    """
    Сравнивает направления рёбер двух графов.

    Args:
        G_pred: Предсказанный граф
        G_true: Эталонный граф
        coords_pred: Координаты предсказанного графа
        coords_true: Координаты эталонного графа
        tol: Допуск для сравнения

    Returns:
        tuple: (количество совпадений, общее количество рёбер)

    """
    edges_pred = get_edge_vectors(G_pred, coords_pred)
    edges_true = get_edge_vectors(G_true, coords_true)

    if edges_pred and edges_true:
        edge_dir_matrix = cdist(edges_pred, edges_true, metric="cosine")
        min_dir_dists = edge_dir_matrix.min(axis=1)
        matched_edges = np.sum(min_dir_dists <= tol)
        total_edges = len(edges_true)
        return matched_edges, total_edges
    else:
        return 0, 0


def get_node_angles(coords, G_nx, min_tol=1e-4):  # noqa N803
    """
    Вычисляет углы между рёбрами для узлов с более чем 2 соседями.

    Args:
        coords: Координаты узлов
        G_nx: Граф NetworkX
        min_tol: Минимальный допуск

    Returns:
        list: Список углов в радианах

    """
    angles = []
    node_list = list(G_nx.nodes)

    for node in G_nx.nodes():
        neighbors = list(G_nx.neighbors(node))
        if len(neighbors) < 2:
            continue

        try:
            i = node_list.index(node)
            if i >= len(coords):
                continue
            x0, y0 = coords[i]
        except Exception:
            continue

        vecs = []
        for nbr in neighbors:
            try:
                j = node_list.index(nbr)
                if j >= len(coords):
                    continue
                x1, y1 = coords[j]
                vec = np.array([x1 - x0, y1 - y0])
                norm = np.linalg.norm(vec)
                if norm > min_tol:
                    vecs.append(vec)
            except Exception:
                continue

        if len(vecs) >= 2:
            for a, b in combinations(vecs, 2):
                cos_theta = np.dot(a, b) / (
                    np.linalg.norm(a) * np.linalg.norm(b)
                )
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angles.append(math.acos(cos_theta))

    return angles


def compare_node_angles(
    G_pred,  # noqa N803
    G_true,  # noqa N803
    coords_pred,
    coords_true,
    angle_tol,
    tol,
):
    """
    Сравнивает углы между рёбрами двух графов.

    Args:
        G_pred: Предсказанный граф
        G_true: Эталонный граф
        coords_pred: Координаты предсказанного графа
        coords_true: Координаты эталонного графа
        angle_tol: Допуск для сравнения углов
        tol: Общий допуск

    Returns:
        tuple: (количество совпадений, общее количество углов)

    """
    if len(G_true.nodes) <= 2 or len(G_pred.nodes) <= 2:
        return 0, 0

    angles_true = get_node_angles(coords_true, G_true, min_tol=tol)
    angles_pred = get_node_angles(coords_pred, G_pred, min_tol=tol)

    if angles_true and angles_pred:
        angle_matrix = cdist(
            np.array(angles_pred).reshape(-1, 1),
            np.array(angles_true).reshape(-1, 1),
        )
        min_angle_dists = angle_matrix.min(axis=1)
        matched_angles = np.sum(min_angle_dists <= angle_tol)
        total_angles = len(angles_true)
        return matched_angles, total_angles
    else:
        return 0, 0


def calculate_similarity_percentage(
    matched_coords,
    total_coords,
    matched_edges,
    total_edges,
    matched_angles,
    total_angles,
    feature_weights: Optional[Mapping[str, float]] = None,
    feature_coverage_importance: Union[float, Mapping[str, float], None] = 1.0,
):
    """
    Вычисляет процент схожести графов.

    Args:
        matched_coords: Совпавшие узлы
        total_coords: Общее количество узлов
        matched_edges: Совпавшие рёбра
        total_edges: Общее количество рёбер
        matched_angles: Совпавшие углы
        total_angles: Общее количество углов

    feature_weights: Веса для компонентов сравнения (узлы, рёбра, углы)
    feature_coverage_importance: Степень влияния покрытия признаков.
        Может быть положительным числом (экспонента для итогового покрытия)
        или словарём с весами для отдельных компонентов покрытия
        ("coords"/"nodes", "edges", "angles").

    Returns:
        float: Процент схожести (0-100)

    """
    weights = {
        "nodes": 1.0,
        "edges": 1.0,
        "angles": 1.0,
    }
    if feature_weights:
        for key, value in feature_weights.items():
            if key in weights:
                weights[key] = float(value)

    weighted_totals = 0.0
    weighted_matches = 0.0

    components = (
        ("nodes", matched_coords, total_coords),
        ("edges", matched_edges, total_edges),
        ("angles", matched_angles, total_angles),
    )
    for name, matched, total in components:
        weight = max(weights.get(name, 1.0), 0.0)
        if weight == 0.0 or total <= 0:
            continue
        weighted_totals += weight * float(total)
        weighted_matches += weight * float(matched)

    if weighted_totals <= 0:
        return 0.0

    # Сохраняем индивидуальные покрытия для компонентов, если потребуется
    coverage_components: dict[str, Optional[float]] = {}
    for name, matched, total in components:
        if total > 0:
            coverage_components[name] = max(0.0, min(1.0, float(matched) / float(total)))
        else:
            coverage_components[name] = None

    coverage_ratio = max(0.0, min(1.0, weighted_matches / weighted_totals))

    if isinstance(feature_coverage_importance, Mapping):
        # Поддерживаем несколько алиасов для удобства настройки из YAML
        def _get_weight(component: str, *aliases: str) -> float:
            keys = (component,) + aliases
            for key in keys:
                if key in feature_coverage_importance:
                    try:
                        value = float(feature_coverage_importance[key])
                        return max(0.0, value)
                    except (TypeError, ValueError):
                        return 0.0
            return 0.0

        raw_weights = {
            "nodes": _get_weight("nodes", "coords", "coordinates", "points"),
            "edges": _get_weight("edges"),
            "angles": _get_weight("angles"),
        }
        weight_sum = sum(raw_weights.values())
        if weight_sum > 0:
            weighted_ratio = 0.0
            has_components = False
            for name, weight in raw_weights.items():
                if weight <= 0:
                    continue
                component_ratio = coverage_components.get(name)
                if component_ratio is None:
                    continue
                weighted_ratio += (weight / weight_sum) * component_ratio
                has_components = True
            if has_components:
                coverage_ratio = max(0.0, min(1.0, weighted_ratio))

    else:
        try:
            exponent = float(feature_coverage_importance)
        except (TypeError, ValueError):
            exponent = 1.0
        if exponent <= 0:
            exponent = 1.0
        coverage_ratio = coverage_ratio ** exponent

    similarity = coverage_ratio * 100.0
    return max(0.0, min(100.0, float(similarity)))


def geometry_compare_slow(
    G_pred: nx.Graph,  # noqa N803
    G_true: nx.Graph,  # noqa N803
    tol,
    angle_tol,
    use_angles,
    normalize_position,
    normalize_scale,
    feature_weights: Optional[Mapping[str, float]] = None,
    feature_coverage_importance: Union[float, Mapping[str, float], None] = 1.0,
) -> float:
    """
    Устойчивое сравнение двух графов по геометрии.

    Игнорирует ID узлов, чувствительность к сдвигу/масштабу.

    Args:
        G_pred: Предсказанный граф
        G_true: Эталонный граф
        tol: Допуск при котором графы считать совпадающими
        angle_tol: Допуск для сравнения углов (в радианах)
        use_angles: Учитывать ли углы между рёбрами при сравнении
        normalize_position: Центрировать графы (вычесть среднее)
        normalize_scale: Нормализовать масштаб графов
                        (по максимальному расстоянию)
        feature_weights: Веса для компонентов сравнения
        feature_coverage_importance: Степень влияния покрытия признаков

    Returns:
        float: Процент схожести графов (0-100)

    """
    if len(G_true.nodes) == 0 or len(G_pred.nodes) == 0:
        return 0.0

    # 1. Извлекаем и нормализуем координаты
    coords_true = extract_and_normalize_coordinates(
        G_true, normalize_position, normalize_scale
    )
    coords_pred = extract_and_normalize_coordinates(
        G_pred, normalize_position, normalize_scale
    )

    # 2. Сравнение координат узлов
    matched_coords, total_coords = compare_node_coordinates(
        coords_pred, coords_true, tol
    )

    # 3. Сравнение направлений рёбер
    matched_edges, total_edges = compare_edge_directions(
        G_pred, G_true, coords_pred, coords_true, tol
    )

    # 4. Сравнение углов (опционально)
    matched_angles = 0
    total_angles = 0
    if use_angles:
        matched_angles, total_angles = compare_node_angles(
            G_pred, G_true, coords_pred, coords_true, angle_tol, tol
        )

    # 5. Вычисляем взвешенное среднее
    similarity = calculate_similarity_percentage(
        matched_coords,
        total_coords,
        matched_edges,
        total_edges,
        matched_angles,
        total_angles,
        feature_weights=feature_weights,
        feature_coverage_importance=feature_coverage_importance,
    )

    return similarity
