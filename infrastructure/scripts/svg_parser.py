import xml.etree.ElementTree as ET
import re
import math
import networkx as nx
from pathlib import Path


def _parse_path_d(d: str) -> list[tuple[float, float]]:
    """Разбирает атрибут d path и возвращает список точек (x, y)."""
    coords = re.findall(r"[-+]?(?:\d*\.\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", d)
    coords = [float(x) for x in coords]
    points = []
    for i in range(0, len(coords) - 1, 2):
        x, y = coords[i], coords[i + 1]
        if (
            not points
            or math.hypot(x - points[-1][0], y - points[-1][1]) > 1e-5
        ):
            points.append((x, y))
    return points


def _get_or_create_node(
    x: float, y: float, g: nx.Graph, coord_to_id: dict
) -> int:
    """Получает ID узла по координатам или создаёт новый."""
    key = (round(x, 3), round(y, 3))  # Точность до 0.001 мм
    if key not in coord_to_id:
        node_id = len(coord_to_id)
        g.add_node(node_id, x=x, y=y)  # Уже в мм
        coord_to_id[key] = node_id
    return coord_to_id[key]


def svg_to_graph(svg_path: Path) -> nx.Graph:
    """
    Конвертирует SVG в граф NetworkX.

    В граф добавляются метаданные: размеры, площадь, количество узлов/рёбер.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Не удалось распарсить SVG-файл {svg_path}: {e}")

    # Обработка namespace
    ns = (
        {"svg": "http://www.w3.org/2000/svg"}
        if root.tag == "{http://www.w3.org/2000/svg}svg"
        else {}
    )

    # Инициализация графа
    g = nx.Graph()
    coord_to_id = {}
    all_coords = []  # Для вычисления габаритов

    # Обработка всех <path>
    for path in root.findall(".//svg:path", ns):
        d = path.get("d", "").strip()
        if not d:
            continue

        coords = _parse_path_d(d)
        all_coords.extend(coords)

        if len(coords) >= 2:
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                if math.hypot(x2 - x1, y2 - y1) > 1e-5:
                    u = _get_or_create_node(x1, y1, g, coord_to_id)
                    v = _get_or_create_node(x2, y2, g, coord_to_id)
                    if u != v:
                        g.add_edge(u, v)

    if g.number_of_nodes() == 0:
        raise ValueError(f"SVG не содержит подходящих линий: {svg_path}")

    # --- Вычисление габаритных размеров ---
    if all_coords:
        xs, ys = zip(*all_coords, strict=False)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width_mm = max_x - min_x
        height_mm = max_y - min_y
        area_mm2 = width_mm * height_mm
    else:
        min_x = min_y = max_x = max_y = 0.0
        width_mm = height_mm = area_mm2 = 0.0

    # --- Добавляем метаданные в граф ---
    g.graph["source"] = svg_path.name
    g.graph["min_x"] = round(min_x, 3)
    g.graph["min_y"] = round(min_y, 3)
    g.graph["max_x"] = round(max_x, 3)
    g.graph["max_y"] = round(max_y, 3)
    g.graph["width_mm"] = round(width_mm, 3)
    g.graph["height_mm"] = round(height_mm, 3)
    g.graph["area_mm2"] = round(area_mm2, 3)
    g.graph["num_nodes"] = g.number_of_nodes()
    g.graph["num_edges"] = g.number_of_edges()

    return g
