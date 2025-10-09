import argparse
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import networkx as nx

import torch

from infrastructure.scripts.graph_utils import pyg_to_networkx

logger = logging.getLogger(__name__)


def svg_overlay(
    G_pred: nx.Graph,
    G_true: nx.Graph,
    output_path,
    *,
    scale: float = 1.0,
    units: str = "mm",
    true_edge_width_mm: float = 1.0,
    pred_edge_width_mm: float = 0.3,
    node_radius_mm: float = 0.8,
) -> dict:
    """
    Overlay predicted graph on ground truth and save as SVG.
    Uses mm-based coordinate system with inverted Y-axis (as in SVG).

    Parameters
    ----------
    G_pred, G_true : nx.Graph
        Graphs with node attributes 'x', 'y' in millimeters.
    output_path : str or Path
        Output SVG path.
    scale : float
        Scale factor for final SVG size.
    units : str
        Unit for width/height (e.g., 'mm', 'px').
    true_edge_width_mm : float
        Stroke width for ground truth edges (in mm).
    pred_edge_width_mm : float
        Stroke width for predicted edges (in mm).
    node_radius_mm : float
        Radius of nodes (in mm).

    Returns
    -------
    dict
        Matching statistics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _coords(G):
        coords = {}
        for n in G.nodes:
            x = G.nodes[n].get('x', 0.0)
            y = G.nodes[n].get('y', 0.0)
            coords[n] = (float(x), float(y))
        return coords

    coords_true = _coords(G_true)
    coords_pred = _coords(G_pred)

    logger.info(f"svg_overlay: сохранение в {output_path}")

    # --- Статистика ---
    tol = 0.1  # мм
    common_nodes = set(coords_true) & set(coords_pred)
    matched_nodes = sum(
        1
        for n in common_nodes
        if abs(coords_true[n][0] - coords_pred[n][0]) < tol
        and abs(coords_true[n][1] - coords_pred[n][1]) < tol
    )
    node_percentage = (
        matched_nodes / len(coords_true) * 100.0 if coords_true else 0.0
    )

    edges_true = {tuple(sorted(e)) for e in G_true.edges()}
    edges_pred = {tuple(sorted(e)) for e in G_pred.edges()}
    matched_edges = len(edges_true & edges_pred)
    edge_percentage = (
        matched_edges / len(edges_true) * 100.0 if edges_true else 0.0
    )

    success = node_percentage >= 95.0 and edge_percentage >= 95.0
    logger.info(
        "SVG overlay: success=%s, node_match=%.2f%%, edge_match=%.2f%%",
        success,
        node_percentage,
        edge_percentage,
    )

    # --- Границы ---
    all_coords = list(coords_true.values()) + list(coords_pred.values())
    if all_coords:
        xs, ys = zip(*all_coords)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x = min_y = 0.0
        max_x = max_y = 100.0

    width_mm = (max_x - min_x) if max_x > min_x else 100.0
    height_mm = (max_y - min_y) if max_y > min_y else 100.0

    # --- SVG ---
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        width=f"{width_mm * scale}{units}",
        height=f"{height_mm * scale}{units}",
        viewBox=f"{min_x} {min_y} {width_mm} {height_mm}",
    )

    # Инвертируем Y: через transform
    # translate(0, max_y + min_y) + scale(1, -1) → y = max_y + min_y - y
    transform = f"scale(1, -1) translate(0, {-max_y - min_y})"
    g = ET.SubElement(svg, "g", transform=transform)

    # --- Отрисовка ---
    # Рёбра: истинные — чёрные, предсказанные — красные
    for u, v in G_true.edges():
        x1, y1 = coords_true[u]
        x2, y2 = coords_true[v]
        ET.SubElement(
            g,
            "line",
            x1=str(x1),
            y1=str(y1),
            x2=str(x2),
            y2=str(y2),
            stroke="black",
            **{"stroke-width": str(true_edge_width_mm)},
        )

    for u, v in G_pred.edges():
        x1, y1 = coords_pred[u]
        x2, y2 = coords_pred[v]
        ET.SubElement(
            g,
            "line",
            x1=str(x1),
            y1=str(y1),
            x2=str(x2),
            y2=str(y2),
            stroke="red",
            **{"stroke-width": str(pred_edge_width_mm)},
        )

    # Узлы: истинные — чёрные, предсказанные — красные
    for n, (x, y) in coords_true.items():
        ET.SubElement(
            g,
            "circle",
            cx=str(x),
            cy=str(y),
            r=str(node_radius_mm),
            fill="black",
        )

    for n, (x, y) in coords_pred.items():
        ET.SubElement(
            g,
            "circle",
            cx=str(x),
            cy=str(y),
            r=str(node_radius_mm),
            fill="red",
        )

    # Сохранение
    ET.ElementTree(svg).write(output_path, encoding="utf-8", xml_declaration=True)

    return {
        "success": success,
        "node_percentage": node_percentage,
        "edge_percentage": edge_percentage,
        "overlay_path": str(output_path),
    }

def main():
    parser = argparse.ArgumentParser(description="Overlay predicted graph on ground truth and save as SVG")
    parser.add_argument("--pred", required=True, help="Path to predicted graph (.pt)")
    parser.add_argument("--true", required=True, help="Path to ground truth graph (.pt)")
    parser.add_argument("--out", required=True, help="Path to output SVG file")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for SVG size")
    parser.add_argument("--units", default="mm", help="Unit suffix (e.g., mm, px)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Загрузка графов из .pt
    try:
        data_pred = torch.load(args.pred, weights_only=False)
        data_true = torch.load(args.true, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки .pt файла: {e}")

    # Конвертация в NetworkX
    G_pred = pyg_to_networkx(data_pred)
    G_true = pyg_to_networkx(data_true)

    # Наложение
    result = svg_overlay(
        G_pred, G_true,
        output_path=args.out,
        scale=args.scale,
        units=args.units
    )

    logger.info(f"Overlay saved: {args.out}")
    logger.info(f"Result: success={result['success']}, "
                f"nodes={result['node_percentage']:.1f}%, "
                f"edges={result['edge_percentage']:.1f}%")

    return result
