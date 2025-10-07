import xml.etree.ElementTree as ET
import random
import argparse
import sys
from pathlib import Path
import math
from datetime import datetime
import traceback

from domain.config.settings import settings

INPUT_DIR = settings.data.master_svg
DEFAULT_SAVE_DIR = settings.data.augmentation_gap
min_gap_multiplier = settings.augmentation_gap.min_gap_multiplier
contours_per_file = settings.augmentation_gap.contours_per_file
gap_range = settings.augmentation_gap.gap_range
random_position_range = settings.augmentation_gap.random_position_range


def parse_svg_commands(d_clean):
    """Парсит команды SVG пути."""
    commands = []
    current_cmd = ""
    current_args = []
    i = 0
    while i < len(d_clean):
        char = d_clean[i]
        if char.upper() in "MmLlHhVvCcSsQqTtAaZz":
            if current_cmd:
                commands.append((current_cmd, current_args))
            current_cmd = char
            current_args = []
        elif char in "0123456789.-":
            num_str = ""
            while i < len(d_clean) and d_clean[i] in "0123456789.-eE":
                num_str += d_clean[i]
                i += 1
            if num_str:
                try:
                    current_args.append(float(num_str))
                except ValueError:
                    pass
            continue
        i += 1
    if current_cmd:
        commands.append((current_cmd, current_args))
    return commands


def extract_points_from_commands(commands):
    """Извлекает точки из команд M/L."""
    points = []
    cmd_indices = []
    for i, (cmd, args) in enumerate(commands):
        if cmd.upper() in ("M", "L") and len(args) >= 2:
            points.append((args[0], args[1]))
            cmd_indices.append(i)
    return points, cmd_indices


def calculate_edges(points, cmd_indices):
    """Вычисляет грани контура."""
    edges = []
    for i in range(len(points)):
        start_idx = i
        end_idx = (i + 1) % len(points)
        start_x, start_y = points[start_idx]
        end_x, end_y = points[end_idx]
        dx, dy = end_x - start_x, end_y - start_y
        length = math.hypot(dx, dy)
        edges.append(
            {
                "start_point": (start_x, start_y),
                "end_point": (end_x, end_y),
                "length": length,
                "cmd_idx": cmd_indices[end_idx]
                if end_idx < len(cmd_indices)
                else None,
            }
        )
    return edges


def filter_suitable_edges(edges, gap_width):
    """Фильтрует подходящие грани для щели."""
    suitable_edges = [
        e for e in edges if e["length"] > gap_width * min_gap_multiplier
    ]
    return suitable_edges


def select_edge_for_gap(edges, gap_width, suitable_edges):
    """Выбирает грань для добавления щели."""
    if not suitable_edges:
        if edges:
            max_edge = max(edges, key=lambda x: x["length"])
            if max_edge["length"] > gap_width:
                suitable_edges = [max_edge]
            else:
                return None
        else:
            return None
    return random.choice(suitable_edges)


def calculate_gap_points(selected_edge, gap_width):
    """Вычисляет точки разрыва для щели."""
    start_x, start_y = selected_edge["start_point"]
    end_x, end_y = selected_edge["end_point"]

    # Случайная позиция на грани
    t = random.uniform(*random_position_range)
    center_x = start_x + (end_x - start_x) * t
    center_y = start_y + (end_y - start_y) * t

    # Точки разрыва
    dx, dy = end_x - start_x, end_y - start_y
    length = max(math.hypot(dx, dy), 1e-10)
    nx, ny = dx / length, dy / length
    half_gap = gap_width / 2
    break1_x, break1_y = center_x - nx * half_gap, center_y - ny * half_gap
    break2_x, break2_y = center_x + nx * half_gap, center_y + ny * half_gap

    return (break1_x, break1_y), (break2_x, break2_y)


def build_modified_commands(
    commands, selected_edge, break1_point, break2_point
):
    """Создает модифицированные команды пути."""
    new_commands = []
    break1_x, break1_y = break1_point
    break2_x, break2_y = break2_point

    for i, (cmd, args) in enumerate(commands):
        if i == selected_edge["cmd_idx"]:
            new_commands.append((cmd, [break1_x, break1_y]))
            new_commands.append(("M", [break2_x, break2_y]))
            new_commands.append(("L", args))
        else:
            new_commands.append((cmd, args))
    return new_commands


def build_svg_path_string(new_commands, has_z):
    """Собирает строку SVG пути."""
    new_d = ""
    for cmd, args in new_commands:
        new_d += cmd + " ".join(f"{a:.6g}" for a in args) + " "
    if has_z:
        new_d += "Z "
    return new_d.strip()


def add_gap_to_path(d_attr, gap_width):
    """Добавляет щель в замкнутый контур в случайном месте и грани."""
    d_clean = d_attr.strip()
    has_z = d_clean.endswith(("Z", "z"))
    if has_z:
        d_clean = d_clean[:-1].strip()

    # Парсим команды пути
    commands = parse_svg_commands(d_clean)

    # Извлекаем точки
    points, cmd_indices = extract_points_from_commands(commands)

    if len(points) < 2:
        return d_attr

    # Строим список граней
    edges = calculate_edges(points, cmd_indices)

    # Отбираем подходящие грани
    suitable_edges = filter_suitable_edges(edges, gap_width)

    # Выбираем грань
    selected_edge = select_edge_for_gap(edges, gap_width, suitable_edges)
    if not selected_edge:
        return d_attr

    # Вычисляем точки разрыва
    break1_point, break2_point = calculate_gap_points(selected_edge, gap_width)

    # Создаем модифицированные команды
    new_commands = build_modified_commands(
        commands, selected_edge, break1_point, break2_point
    )

    # Собираем строку пути
    new_d = build_svg_path_string(new_commands, has_z)

    start_x, start_y = selected_edge["start_point"]
    end_x, end_y = selected_edge["end_point"]
    return new_d, (start_x, start_y, end_x, end_y)


def find_closed_paths(root):
    """Находит замкнутые контуры в SVG."""
    closed_paths = []
    path_elements = []
    for elem in root.iter():
        if elem.tag.endswith("path"):
            d = elem.attrib.get("d", "")
            if "Z" in d.upper() or "z" in d:
                closed_paths.append(d)
                path_elements.append(elem)
    return closed_paths, path_elements


def select_contours_for_modification(closed_paths):
    """Выбирает контуры для модификации."""
    num_to_modify = min(contours_per_file, len(closed_paths))
    selected_indices = random.sample(range(len(closed_paths)), num_to_modify)
    return selected_indices


def modify_selected_contours(
    closed_paths,
    path_elements,
    selected_indices,
    gap_width,
    input_path,
    log_file,
):
    """Модифицирует выбранные контуры."""
    modified_count = 0
    for idx in selected_indices:
        d = closed_paths[idx]
        elem = path_elements[idx]
        result = add_gap_to_path(d, gap_width)
        if isinstance(result, tuple):
            new_d, (sx, sy, ex, ey) = result
            if new_d != d:
                elem.attrib["d"] = new_d
                modified_count += 1
                msg = (
                    f"Модифицирован контур {input_path.name}: "
                    f"({sx:.1f},{sy:.1f})->({ex:.1f},{ey:.1f})"
                )
                print(msg)
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now()}] {msg}\n")
    return modified_count


def is_svg_file(file_path):
    """Проверяет, является ли файл SVG (игнорируя регистр расширения)."""
    return file_path.suffix.lower() == ".svg"


def get_svg_files_from_directory(directory):
    """Получает все SVG файлы из директории (игнорируя регистр расширения)."""
    svg_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and is_svg_file(file_path):
            svg_files.append(file_path)
    return svg_files


def process_svg_file(
    input_path: Path,
    output_path: Path,
    gap_width: float,
    log_file: Path = None,
):
    """Обрабатывает один SVG файл - модифицирует случайные контуры."""
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(input_path)
        root = tree.getroot()

        # Находим замкнутые контуры
        closed_paths, path_elements = find_closed_paths(root)

        if not closed_paths:
            msg = f"Нет замкнутых контуров в {input_path.name}"
            print(msg)
            if log_file:
                log_file.write_text(
                    f"[{datetime.now()}] {msg}\n", encoding="utf-8"
                )
            tree.write(output_path, encoding="utf-8", xml_declaration=True)
            return True

        # Выбираем контуры для модификации
        selected_indices = select_contours_for_modification(closed_paths)

        # Модифицируем выбранные контуры
        modify_selected_contours(
            closed_paths,
            path_elements,
            selected_indices,
            gap_width,
            input_path,
            log_file,
        )

        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return True
    except Exception as e:
        msg = f"Ошибка {input_path.name}: {e}\n{traceback.format_exc()}"
        print(msg)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] {msg}\n")
        return False


def get_gap_suffix(gap_width: float) -> str:
    """Возвращает суффикс для файла в зависимости от ширины щели."""
    gap_centi = int(round(gap_width * 100))
    return f"_gap{gap_centi:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Добавление щелей в SVG файлы"
    )
    parser.add_argument("input", help="Входной SVG файл или директория")
    parser.add_argument(
        "-w",
        "--width",
        type=float,
        default=0.05,
        help="Ширина щели в мм (0.01–0.05)",
    )
    parser.add_argument("-o", "--output", help="Выходная директория")
    parser.add_argument("-l", "--log", help="Файл для логирования")

    args = parser.parse_args()
    input_path = Path(args.input)
    gap_width = args.width
    output_dir = Path(args.output) if args.output else None
    log_file = (
        Path(args.log)
        if args.log
        else settings.logging.log_dir / "augmentation_gap.log"
    )

    if not (0.01 <= gap_width <= 0.05):
        print("Ширина щели должна быть в диапазоне 0.01–0.05 мм")
        sys.exit(1)

    suffix = get_gap_suffix(gap_width)

    if input_path.is_file():
        if not is_svg_file(input_path):
            print("Файл должен быть .svg")
            sys.exit(1)
        output_dir = output_dir or input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
        )
        if process_svg_file(input_path, output_path, gap_width, log_file):
            print(f"Файл сохранён: {output_path}")
        else:
            sys.exit(1)

    elif input_path.is_dir():
        svg_files = get_svg_files_from_directory(input_path)
        if not svg_files:
            print("Нет SVG файлов в директории")
            sys.exit(1)

        output_dir = output_dir or input_path
        output_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        for svg in svg_files:
            output_path = output_dir / f"{svg.stem}{suffix}{svg.suffix}"
            if process_svg_file(svg, output_path, gap_width, log_file):
                success += 1
        print(f"Обработано {success}/{len(svg_files)} файлов")

    else:
        print("Указанный путь не существует")
        sys.exit(1)


if __name__ == "__main__":
    main()
