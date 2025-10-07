import xml.etree.ElementTree as ET
import random
import argparse
import sys
from pathlib import Path
import math
from datetime import datetime

from domain.config.settings import settings


# --- Конфиг из settings ---
INPUT_DIR = settings.data.master_svg
DEFAULT_SAVE_DIR = settings.data.augmentation_tabs
tab_width_range = settings.augmentation_tabs.width_range
tab_length_range = settings.augmentation_tabs.length_range
tabs_per_file = settings.augmentation_tabs.tabs_per_contour
min_edge_length = settings.augmentation_tabs.min_edge_length

log_file = settings.logging.log_dir / "augmentation_tabs.log"


# --- Геометрия ---
def polygon_area(points):
    """Формула шнурования: >0 CCW, <0 CW."""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0


def point_in_polygon(point, polygon):
    """Ray casting: возвращает True, если точка внутри полигона."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
    return inside


def create_tab_on_edge(
    start_point,
    end_point,
    tab_width,
    tab_length,
    position_t=0.5,
    orientation_val=1.0,
    contour_points=None,
):
    """Создаёт прямоугольную перемычку, гарантированно наружу."""
    x1, y1 = start_point
    x2, y2 = end_point

    dx, dy = x2 - x1, y2 - y1
    edge_length = math.hypot(dx, dy)
    if edge_length < 1e-6:
        return None

    ux, uy = dx / edge_length, dy / edge_length

    # нормаль по ориентации
    if orientation_val > 0:  # CCW → наружу слева
        nx, ny = -uy, ux
    else:  # CW → наружу справа
        nx, ny = uy, -ux

    # центр таба
    center_x = x1 + ux * edge_length * position_t
    center_y = y1 + uy * edge_length * position_t

    half_width = tab_width / 2
    base1_x, base1_y = center_x - ux * half_width, center_y - uy * half_width
    base2_x, base2_y = center_x + ux * half_width, center_y + uy * half_width

    tip1_x, tip1_y = base1_x + nx * tab_length, base1_y + ny * tab_length
    tip2_x, tip2_y = base2_x + nx * tab_length, base2_y + ny * tab_length

    # проверяем направление — midpoint между tip1 и tip2
    if contour_points is not None:
        mid_tip = ((tip1_x + tip2_x) / 2, (tip1_y + tip2_y) / 2)
        if point_in_polygon(mid_tip, contour_points):
            # инвертируем нормаль
            nx, ny = -nx, -ny
            tip1_x, tip1_y = (
                base1_x + nx * tab_length,
                base1_y + ny * tab_length,
            )
            tip2_x, tip2_y = (
                base2_x + nx * tab_length,
                base2_y + ny * tab_length,
            )

    return [
        (base1_x, base1_y),
        (tip1_x, tip1_y),
        (tip2_x, tip2_y),
        (base2_x, base2_y),
        (base1_x, base1_y),
    ]


def parse_line_points(d_attr):
    """Парсит path с одной линией (M x,y L x,y)."""
    tokens = d_attr.replace(",", " ").split()
    points = []
    i = 0
    while i < len(tokens) - 1:
        t = tokens[i]
        if t.upper() in ("M", "L"):
            try:
                x, y = float(tokens[i + 1]), float(tokens[i + 2])
                points.append((x, y))
                i += 3
            except Exception:
                i += 1
        else:
            i += 1
    return points if len(points) == 2 else None


# --- Обработка SVG ---
def log_message(msg: str, log_file: Path = None):
    """Печатает и логирует сообщение."""
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] {msg}\n")


def save_tree(tree, output_path: Path, msg: str = None, log_file: Path = None):
    """Сохраняет SVG и пишет сообщение, если есть."""
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    if msg:
        log_message(msg, log_file)
    return True


def collect_edges(root):
    """Ищет рёбра (прямые линии) в SVG."""
    edges = []
    for elem in root.iter():
        if elem.tag.endswith("path"):
            points = parse_line_points(elem.attrib.get("d", ""))
            if points:
                x1, y1 = points[0]
                x2, y2 = points[1]
                length = math.hypot(x2 - x1, y2 - y1)
                if length >= min_edge_length:
                    edges.append(
                        {
                            "start_point": (x1, y1),
                            "end_point": (x2, y2),
                            "length": length,
                            "elem": elem,
                        }
                    )
    return edges


def update_viewbox(root):
    """Пересчитывает viewBox, размеры SVG и фон <rect>."""
    coords = []
    for elem in root.iter():
        if elem.tag.endswith("path"):
            d = elem.attrib.get("d", "")
            tokens = d.replace(",", " ").split()
            i = 0
            while i < len(tokens) - 1:
                if tokens[i].upper() in ("M", "L"):
                    try:
                        x, y = float(tokens[i + 1]), float(tokens[i + 2])
                        coords.append((x, y))
                        i += 3
                        continue
                    except Exception:
                        pass
                i += 1
    if not coords:
        return

    min_x = min(x for x, _ in coords)
    max_x = max(x for x, _ in coords)
    min_y = min(y for _, y in coords)
    max_y = max(y for _, y in coords)

    width = max_x - min_x
    height = max_y - min_y

    root.set("viewBox", f"{min_x:.3f} {min_y:.3f} {width:.3f} {height:.3f}")
    root.set("width", f"{width:.2f}mm")
    root.set("height", f"{height:.2f}mm")

    # обновляем (или создаём) фон <rect>
    rect = None
    for elem in root:
        if elem.tag.endswith("rect"):
            rect = elem
            break

    if rect is None:
        rect = ET.Element("rect")
        rect.set("fill", "#212830")
        rect.set("fill-opacity", "1.0")
        root.insert(0, rect)

    rect.set("x", f"{min_x:.3f}")
    rect.set("y", f"{min_y:.3f}")
    rect.set("width", f"{width:.3f}")
    rect.set("height", f"{height:.3f}")


def process_svg_file(input_path, output_path, log_file=None):
    """
    Обрабатывает рёбра.

    Направляет табы наружу по ориентации контура + проверка.
    """
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(input_path)
        root = tree.getroot()

        edges = collect_edges(root)
        log_message(f"{input_path.name}: всего рёбер={len(edges)}", log_file)

        if not edges:
            return save_tree(tree, output_path)

        # вычисляем ориентацию контура (по всем рёбрам)
        points = [e["start_point"] for e in edges]
        if edges:
            points.append(edges[-1]["end_point"])
        orientation_val = polygon_area(points)
        orientation = "CCW" if orientation_val > 0 else "CW"
        log_message(
            f"{input_path.name}: ориентация контура = {orientation}", log_file
        )

        total_tabs_added = 0
        num_tabs = min(tabs_per_file, len(edges))
        selected_edges = random.sample(edges, num_tabs)

        for edge in selected_edges:
            tab_width = random.uniform(*tab_width_range)
            tab_length = random.uniform(*tab_length_range)
            position_t = random.uniform(0.2, 0.8)

            tab_points = create_tab_on_edge(
                edge["start_point"],
                edge["end_point"],
                tab_width,
                tab_length,
                position_t,
                orientation_val=orientation_val
                if abs(orientation_val) > 1e-6
                else 1.0,
                contour_points=points,
            )
            if tab_points:
                path_str = (
                    "M "
                    + " ".join(
                        [
                            f"{'L ' if i else ''}{x:.6g},{y:.6g}"
                            for i, (x, y) in enumerate(tab_points)
                        ]
                    )
                    + " Z"
                )
                new_tab = ET.SubElement(root, "path")
                new_tab.set("d", path_str)
                cls = edge["elem"].get("class")
                if cls:
                    new_tab.set("class", cls)
                total_tabs_added += 1

        update_viewbox(root)

        return save_tree(
            tree,
            output_path,
            msg=f"{input_path.name}: добавлено {total_tabs_added} перемычек",
            log_file=log_file,
        )

    except Exception as e:
        log_message(f"Ошибка обработки файла {input_path.name}: {e}", log_file)
        return False


def get_tab_suffix():
    return "_tabs"


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Добавление перемычек на длинные рёбра SVG"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=INPUT_DIR,
        help=f"Входной SVG файл или директория (по умолчанию: {INPUT_DIR})",
    )
    parser.add_argument("-o", "--output", help="Выходная директория")
    parser.add_argument("-l", "--log", help="Файл для логирования")
    parser.add_argument("--seed", type=int, help="Фиксированный seed")

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else DEFAULT_SAVE_DIR
    log_file_path = Path(args.log) if args.log else log_file

    suffix = get_tab_suffix()

    if input_path.is_file():
        if not input_path.suffix.lower() == ".svg":
            print("Файл должен быть .svg")
            sys.exit(1)

        output_path = (
            output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
        )
        if process_svg_file(input_path, output_path, log_file_path):
            print(f"Файл сохранен: {output_path}")
        else:
            sys.exit(1)

    elif input_path.is_dir():
        svg_files = list(input_path.glob("*.svg"))
        if not svg_files:
            print("В директории нет SVG файлов")
            sys.exit(1)

        success_count = 0
        for svg_file in svg_files:
            output_path = (
                output_dir / f"{svg_file.stem}{suffix}{svg_file.suffix}"
            )
            if process_svg_file(svg_file, output_path, log_file_path):
                print(f"Обработан: {svg_file.name} -> {output_path.name}")
                success_count += 1
            else:
                print(f"Ошибка при обработке: {svg_file.name}")

        print(f"\nОбработано файлов: {success_count}/{len(svg_files)}")
    else:
        print("Указанный путь не существует")
        sys.exit(1)


if __name__ == "__main__":
    main()
