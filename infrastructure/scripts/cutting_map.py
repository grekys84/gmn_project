# # infrastructure/processing/cutting_map_processor.py
#
# from pathlib import Path
# import networkx as nx
# import torch
# import json
# from datetime import datetime
# from typing import Dict, List, Tuple, Optional
# import logging
#
# try:
#     from torch_geometric.data import Data
# except ModuleNotFoundError:
#     from torch_geometric_stub.data import Data
#
# # === Настройка ===
# # BASE_DIR = Path(__file__).parent.parent
# # INPUT_FOLDER = BASE_DIR / "data" / "raw_svg"
# # OUTPUT_FOLDER = BASE_DIR / "data" / "processed_graphs"
# INPUT_FOLDER = Path(r"D:\Finogeev\test")
# OUTPUT_FOLDER = Path(r"D:\Finogeev\processed_graphs")
# logger = logging.getLogger(__name__)
#
#
# class CuttingMapProcessor:
#     """
#     Модуль для обработки карт раскроя из SVG:
#     - Загрузка графа
#     - Разделение на связные компоненты (детали)
#     - Сохранение полного графа и отдельных деталей
#     - Визуализация
#     - Сбор статистики
#     """
#
#     def __init__(
#         self,
#         input_folder: Path,
#         output_folder: Path,
#         draw_graph: bool = False,
#         save_components: bool = True,
#         min_nodes_per_component: int = 1
#     ):
#         self.input_folder = Path(input_folder)
#         self.output_folder = Path(output_folder)
#         self.draw_graph = draw_graph
#         self.save_components = save_components
#         self.min_nodes_per_component = min_nodes_per_component
#
#         # Подкаталоги
#         self.components_dir = self.output_folder / "components"
#         self.visualizations_dir = self.output_folder / "visualizations"
#
#         # Создаём структуру
#         self.output_folder.mkdir(parents=True, exist_ok=True)
#         self.components_dir.mkdir(exist_ok=True)
#         self.visualizations_dir.mkdir(exist_ok=True)
#
#         self.total_summary: Dict[str, int] = {}
#
#     def _load_graph(self, svg_path: Path) -> nx.Graph:
#         """
#         Загружает граф из SVG с помощью svg_to_graph.
#         Предполагается, что svg_to_graph определён в infrastructure.utils.svg_parser.
#         """
#         from infrastructure.utils.svg_parser import svg_to_graph
#         return svg_to_graph(svg_path)
#
#     def _save_graph(
#         self,
#         graph: nx.Graph,
#         out_dir: Path,
#         graph_id: str,
#         source_file: Path,
#         is_component: bool = False
#     ):
#         """Сохраняет граф в формате PyG (.pt) и мета (.json)."""
#         try:
#             edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
#             x = torch.tensor(
#                 [[graph.nodes[n]['x'], graph.nodes[n]['y']] for n in graph.nodes],
#                 dtype=torch.float
#             )
#             data = Data(x=x, edge_index=edge_index)
#             torch.save(data, out_dir / f"{graph_id}.pt")
#
#             meta = {
#                 "id": graph_id,
#                 "source": source_file.name,
#                 "created": datetime.now().isoformat(timespec='seconds'),
#                 "num_nodes": graph.number_of_nodes(),
#                 "num_edges": graph.number_of_edges(),
#                 "is_component": is_component
#             }
#             with open(out_dir / f"{graph_id}_meta.json", "w", encoding="utf-8") as f:
#                 json.dump(meta, f, indent=2, ensure_ascii=False)
#
#         except Exception as e:
#             logger.error(f"Ошибка при сохранении графа {graph_id}: {e}")
#             raise
#
#     def _visualize_components(
#         self,
#         graph: nx.Graph,
#         components: List[set],
#         graph_id: str,
#         title_suffix: str = ""
#     ):
#         """Визуализирует граф с раскрашенными компонентами."""
#         try:
#             import matplotlib.pyplot as plt
#             import matplotlib.cm as cm
#
#             plt.figure(figsize=(12, 10))
#             pos = {n: (graph.nodes[n]['x'], -graph.nodes[n]['y']) for n in graph.nodes}
#
#             cmap = cm.tab20 if len(components) <= 20 else cm.tab20b
#             colors = [cmap(i % 20) for i in range(len(components))]
#
#             for idx, comp in enumerate(components):
#                 subgraph = graph.subgraph(comp)
#                 nx.draw_networkx_nodes(subgraph, pos, node_color=[colors[idx]], node_size=50)
#                 nx.draw_networkx_edges(subgraph, pos, edge_color=colors[idx], width=2.5)
#
#             plt.title(f"Карта раскроя: {title_suffix} | {len(components)} деталей", fontsize=14)
#             plt.axis('equal')
#             plt.axis('off')
#             plt.tight_layout()
#
#             save_path = self.visualizations_dir / f"{graph_id}_components.png"
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             plt.close()
#
#             logger.info(f"Визуализация сохранена: {save_path}")
#         except Exception as e:
#             logger.warning(f"Не удалось создать визуализацию: {e}")
#
#     def _infer_shape(self, num_nodes: int) -> str:
#         """Простая эвристика определения формы детали."""
#         if num_nodes == 4:
#             return "rectangle"
#         elif 5 <= num_nodes <= 8:
#             return "polygon"
#         elif num_nodes > 8:
#             return "circle_approx"
#         else:
#             return "unknown"
#
#     def _clean_graph(self, graph: nx.Graph) -> nx.Graph:
#         """
#         Очищает граф:
#         1. Удаляет изолированные узлы (без рёбер).
#         2. Удаляет связные компоненты, в которых меньше 4 узлов.
#         """
#         G = graph.copy()
#         logger.info(f"Очистка графа: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
#
#         # 1. Удаляем изолированные узлы
#         isolated_nodes = [n for n, degree in G.degree() if degree == 0]
#         logger.info(f"Найдено изолированных узлов: {len(isolated_nodes)}")
#         G.remove_nodes_from(isolated_nodes)
#
#         # 2. Анализ компонент до удаления
#         components = list(nx.connected_components(G))
#         logger.info(f"Связных компонент: {len(components)}")
#
#         components_to_remove = []
#         for i, comp in enumerate(components):
#             sub = G.subgraph(comp)
#             num_nodes = len(comp)
#             num_edges = sub.number_of_edges()
#             is_connected = nx.is_connected(sub)
#             degrees = [d for _, d in sub.degree()]
#             is_all_degree_2 = all(d == 2 for d in degrees) if num_nodes > 0 else False
#             is_cycle = num_edges == num_nodes and is_all_degree_2  # простой цикл
#
#             logger.debug(f"  Компонент {i}: {num_nodes} узлов, {num_edges} рёбер, "
#                          f"степени={degrees}, цикл={is_cycle}")
#
#             if num_nodes < self.min_nodes_per_component:
#                 components_to_remove.append(comp)
#         # Удаляем малые компоненты
#         for comp in components_to_remove:
#             G.remove_nodes_from(comp)
#
#         logger.info(f"Удалено компонент (<{self.min_nodes_per_component} узлов): {len(components_to_remove)}")
#         logger.info(f"После очистки: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
#
#         return G
#
#
#     def process(self) -> Dict[str, int]:
#         logger.info(f"Запуск обработки карт раскроя из {self.input_folder}")
#
#         svg_files = list(self.input_folder.glob("*.svg")) + list(self.input_folder.glob("*.SVG"))
#         svg_files = sorted(set(svg_files))
#
#         if not svg_files:
#             logger.warning("SVG-файлы не найдены.")
#             return {}
#
#         logger.info(f"Найдено {len(svg_files)} SVG-файлов для обработки.")
#
#         for idx, svg_path in enumerate(svg_files):
#             graph_id = f"g_{idx:04d}"
#             try:
#                 logger.info(f"[{idx + 1}/{len(svg_files)}] Обработка: {svg_path.name}")
#                 raw_graph = self._load_graph(svg_path)
#                 logger.info(
#                     f"  Исходный граф: {raw_graph.number_of_nodes()} узлов, {raw_graph.number_of_edges()} рёбер")
#
#                 # Очистка графа
#                 graph = self._clean_graph(raw_graph)
#                 if graph.number_of_nodes() == 0:
#                     logger.warning(f"Граф после очистки пуст: {svg_path.name}")
#                     continue
#
#                 # Сохраняем очищенный полный граф
#                 self._save_graph(graph, self.output_folder, graph_id, svg_path)
#
#                 # Разделение на детали
#                 components = list(nx.connected_components(graph))
#                 n_components = len(components)
#                 logger.info(f"Найдено деталей после фильтрации: {n_components}")
#
#                 # Визуализация
#                 if self.draw_graph:
#                     self._visualize_components(graph, components, graph_id, svg_path.stem)
#
#                 # Обработка каждой детали
#                 if self.save_components:
#                     comp_summary = {}
#                     for comp_idx, comp_nodes in enumerate(components):
#                         subgraph = graph.subgraph(comp_nodes).copy()
#                         comp_id = f"{graph_id}_part_{comp_idx:02d}"
#
#                         self._save_graph(subgraph, self.components_dir, comp_id, svg_path, is_component=True)
#
#                         shape = self._infer_shape(subgraph.number_of_nodes())
#                         comp_summary[shape] = comp_summary.get(shape, 0) + 1
#
#                     # Обновляем общую статистику
#                     for shape, count in comp_summary.items():
#                         self.total_summary[shape] = self.total_summary.get(shape, 0) + count
#
#                     logger.info(f"Сводка по файлу {svg_path.name}: {comp_summary}")
#
#             except Exception as e:
#                 logger.error(f"Ошибка при обработке {svg_path.name}: {e}")
#
#         logger.info("Обработка завершена.")
#         return self.total_summary
#
#     def get_summary(self) -> Dict[str, int]:
#         """Возвращает итоговую статистику по всем обработанным файлам."""
#         return self.total_summary.copy()
#
#
# # === Запуск ===
# if __name__ == "__main__":
#     processor = CuttingMapProcessor(
#         input_folder=INPUT_FOLDER,
#         output_folder=OUTPUT_FOLDER,
#         draw_graph=True,
#         save_components=True
#     )
#     summary = processor.process()
#
#     print("\n" + "="*50)
#     print("ОБЩАЯ СТАТИСТИКА ПО ДЕТАЛЯМ:")
#     for shape, count in summary.items():
#         print(f"  {shape}: {count}")
#     print(f"Всего деталей: {sum(summary.values())}")

import xml.etree.ElementTree as ET
from pathlib import Path
import re
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# --- Настройки ---
INPUT_SVG = r"D:\Finogeev\test\cs2_14m_r_8050_c_00001_1250x1470_ns201_alt1_mtn1.svg"
OUTPUT_SVG = r"D:\Finogeev\test\input_cleaned.svg"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_PART_AREA = 50_000  # 200x250 мм
MAX_PART_AREA = 500_000
TOLERANCE = 0.5


def parse_path_d(d: str):
    """Извлекает координаты из d."""
    coords = re.findall(r"[-+]?(?:\d*\.\d+|\d+(?:\.\d*)?)(?=\s*[MLZ]|\s*$)", d, re.IGNORECASE)
    coords = [float(x) for x in coords]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]


def path_to_polygon(d: str) -> Polygon | None:
    """Преобразует d в полигон."""
    coords = parse_path_d(d)
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        return Polygon(coords)
    except:
        return None


def extract_sheet_and_blue_areas(svg_path, ns):
    """Извлекает серый лист и синие заливки."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    sheet_polygon = None
    blue_polygons = []

    for path in root.findall('.//svg:path', ns):
        style = path.get('style', '')
        d = path.get('d', '').strip()

        # Ищем серый фон (лист)
        if 'fill:#808080' in style:
            poly = path_to_polygon(d)
            if poly:
                sheet_polygon = poly
                print(f"✅ Найден серый лист: площадь = {poly.area:.2f} мм²")
                continue  # не пропускаем!

        # Ищем синие заливки (сетка)
        if 'fill:#0000ff' in style:
            poly = path_to_polygon(d)
            if poly:
                blue_polygons.append(poly)

    return sheet_polygon, blue_polygons, tree, root


def find_parts(sheet: Polygon, blue_polygons: list[Polygon]):
    """Находит детали как "дыры" в синей сетке."""
    if not sheet or not blue_polygons:
        return []

    # Объединяем все синие области
    blue_union = unary_union([geom.buffer(TOLERANCE) for geom in blue_polygons])

    # Вычитаем из листа
    try:
        remaining = sheet.difference(blue_union)
    except:
        remaining = sheet.difference(blue_union.buffer(TOLERANCE))

    # Извлекаем оставшиеся части
    if remaining.geom_type == 'Polygon':
        parts = [remaining]
    elif remaining.geom_type == 'MultiPolygon':
        parts = list(remaining.geoms)
    else:
        return []

    holes = []
    for part in parts:
        if MIN_PART_AREA <= part.area <= MAX_PART_AREA:
            holes.append(part)

    return holes


def visualize_result(sheet, blue_polygons, holes, filename):
    fig, ax = plt.subplots(figsize=(16, 12))

    # Серый фон
    if sheet:
        xs, ys = box(*sheet.bounds).exterior.xy
        ax.fill(xs, ys, color='lightgray', alpha=0.2)

    # Синие области
    for poly in blue_polygons:
        if poly.geom_type == 'Polygon':
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color='blue', alpha=0.3)

    # Детали (дыры)
    for hole in holes:
        xs, ys = hole.exterior.xy
        ax.fill(xs, ys, color='none', edgecolor='red', linewidth=2, label="Деталь")

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Найденные детали (дыры в синей сетке)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def save_parts_as_svg(holes, output_path):
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "version": "1.1",
        "width": "1608.41mm",
        "height": "1350.29mm",
        "viewBox": "-15.77 -13.24 1608.41 1350.29"
    })
    for hole in holes:
        coords = list(hole.exterior.coords)
        d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in coords) + " Z"
        ET.SubElement(svg, "path", {
            "d": d,
            "fill": "none",
            "stroke": "#000000",
            "stroke-width": "1.045"
        })

    ET.ElementTree(svg).write(output_path, encoding="utf-8", xml_declaration=True)


# --- Основной код ---
if __name__ == "__main__":
    ns = {
        'svg': 'http://www.w3.org/2000/svg',
        'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
        'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
    }
    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)

    print("1. Извлечение серого листа и синих областей...")
    sheet, blue_polygons, tree, root = extract_sheet_and_blue_areas(INPUT_SVG, ns)

    if not sheet:
        print("❌ Ошибка: не найден серый фон (fill:#808080)")
        exit()

    print(f"✅ Найдено {len(blue_polygons)} синих областей")

    print("2. Поиск деталей (дыр в синей сетке)...")
    holes = find_parts(sheet, blue_polygons)
    print(f"✅ Найдено {len(holes)} деталей")

    # Визуализация
    visualize_result(sheet, blue_polygons, holes, OUTPUT_DIR / "visualization_holes.png")

    # Сохранение
    save_parts_as_svg(holes, OUTPUT_DIR / OUTPUT_SVG)

    print(f"\n🎉 Готово!")
    print(f"  → Очищенный SVG: {OUTPUT_DIR / OUTPUT_SVG}")
    print(f"  → Визуализация: {OUTPUT_DIR / 'visualization_holes.png'}")