import xml.etree.ElementTree as ET
import random
import argparse
import sys
from pathlib import Path
import math
from datetime import datetime
import traceback

from domain.config.settings import settings

# --- Конфиг ---
INPUT_DIR = settings.data.master_svg
DEFAULT_SAVE_DIR = settings.data.augmentation_hole_unclosed
min_gap_multiplier = settings.augmentation_hole_unclosed.min_gap_multiplier
contours_per_file = settings.augmentation_hole_unclosed.contours_per_file
gap_range = settings.augmentation_hole_unclosed.gap_range
random_position_range = (
    settings.augmentation_hole_unclosed.random_position_range
)


# --- Парсинг ---
def parse_svg_commands(d_clean):
    """
    Парсит команды SVG пути.

    Args:
        d_clean (str): Строка атрибута d SVG path без Z в конце
    Returns:
        list: Список кортежей (команда, аргументы)

    """
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
    """
    Извлекает точки из команд M/L.

    Args:
        commands (list): Список команд SVG
    Returns:
        tuple: (список точек, список индексов команд)

    """
    points = []
    cmd_indices = []
    for i, (cmd, args) in enumerate(commands):
        if cmd.upper() in ("M", "L") and len(args) >= 2:
            points.append((args[0], args[1]))
            cmd_indices.append(i)
    return points, cmd_indices


# --- Обработка граней ---
def calculate_edges(
    points, cmd_indices, merge_small_segments=True, min_merge_ratio=0.2
):
    """
    Вычисляет грани контура.

    Объединяет очень короткие сегменты (для окружностей).

    Args:
        points (list): Список точек контура
        cmd_indices (list): Индексы команд, соответствующих точкам
        merge_small_segments (bool): Флаг объединения коротких сегментов
        min_merge_ratio (float): Минимальное отношение длины сегмента
                                 для объединения
    Returns:
        list: Список ребер контура с координатами, длинами и индексами команд

    """
    edges = []
    i = 0
    while i < len(points):
        start_idx = i
        end_idx = (i + 1) % len(points)
        start_x, start_y = points[start_idx]
        end_x, end_y = points[end_idx]
        dx, dy = end_x - start_x, end_y - start_y
        length = math.hypot(dx, dy)

        if (
            merge_small_segments
            and length < gap_range[0] * min_merge_ratio
            and len(points) > 2
        ):
            next_idx = (end_idx + 1) % len(points)
            next_x, next_y = points[next_idx]
            dx2, dy2 = next_x - start_x, next_y - start_y
            length = math.hypot(dx2, dy2)
            edges.append(
                {
                    "start_point": (start_x, start_y),
                    "end_point": (next_x, next_y),
                    "length": length,
                    "cmd_start": cmd_indices[start_idx]
                    if start_idx < len(cmd_indices)
                    else None,
                    "cmd_end": cmd_indices[end_idx]
                    if end_idx < len(cmd_indices)
                    else None,
                }
            )
            i += 2
        else:
            edges.append(
                {
                    "start_point": (start_x, start_y),
                    "end_point": (end_x, end_y),
                    "length": length,
                    "cmd_start": cmd_indices[start_idx]
                    if start_idx < len(cmd_indices)
                    else None,
                    "cmd_end": cmd_indices[end_idx]
                    if end_idx < len(cmd_indices)
                    else None,
                }
            )
            i += 1
    return edges


def filter_suitable_edges(edges, gap_width):
    """
    Фильтрует ребра, подходящие для создания разрыва.

    Отбирает ребра, длина которых больше значения
    `gap_width * min_gap_multiplier`.

    Args:
        edges (list): Список ребер контура
        gap_width (float): Ширина разрыва в мм
    Returns:
        list: Список подходящих ребер

    """
    return [e for e in edges if e["length"] > gap_width * min_gap_multiplier]


def select_edge_for_gap(edges, gap_width, suitable_edges):
    """
    Выбирает ребро для создания разрыва.

    Args:
        edges (list): Все ребра контура
        gap_width (float): Ширина разрыва в мм
        suitable_edges (list): Список подходящих ребер
    Returns:
        dict or None: Выбранное ребро или None, если подходящего ребра нет

    """
    if suitable_edges:
        return random.choice(suitable_edges)
    elif edges:
        max_edge = max(edges, key=lambda x: x["length"])
        if max_edge["length"] > gap_width:
            return max_edge
    return None


def calculate_gap_points(selected_edge, gap_width):
    """
    Вычисляет точки разрыва на ребре в пределах принятого значения.

    Args:
        selected_edge (dict): Выбранное ребро для разрыва
        gap_width (float): Ширина разрыва в мм
    Returns:
        tuple: Две точки разрыва (break1_point, break2_point)

    """
    start_x, start_y = selected_edge["start_point"]
    end_x, end_y = selected_edge["end_point"]

    # Случайная позиция на ребре
    t = random.uniform(*random_position_range)
    center_x = start_x + (end_x - start_x) * t
    center_y = start_y + (end_y - start_y) * t

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
    """
    Создает модифицированные SVG команды с разрывом.

    Args:
        commands (list): Исходные команды SVG
        selected_edge (dict): Выбранное ребро для разрыва
        break1_point (tuple): Первая точка разрыва
        break2_point (tuple): Вторая точка разрыва
    Returns:
        list: Новые команды SVG с разрывом

    """
    new_commands = []
    break1_x, break1_y = break1_point
    break2_x, break2_y = break2_point

    end_idx = selected_edge["cmd_end"]

    for i, (cmd, args) in enumerate(commands):
        if i == end_idx:
            # Вставляем разрыв прямо перед концом сегмента
            new_commands.append(("L", [break1_x, break1_y]))
            new_commands.append(("M", [break2_x, break2_y]))
            new_commands.append(("L", args))
        else:
            new_commands.append((cmd, args))
    return new_commands


def build_svg_path_string(new_commands, has_z):
    """
    Создает строку SVG пути из команд.

    Args:
        new_commands (list): Список новых команд SVG
        has_z (bool): Флаг наличия замыкания (Z)

    Returns:
        str: Строка атрибута d для SVG path

    """
    new_d = ""
    for cmd, args in new_commands:
        new_d += cmd + " ".join(f"{a:.6g}" for a in args) + " "
    if has_z:
        new_d += "Z "
    return new_d.strip()


def add_gap_to_path(d_attr, gap_width):
    """
    Добавляет разрыв в SVG путь.

    Args:
        d_attr (str): Атрибут d исходного SVG path
        gap_width (float): Ширина разрыва в мм
    Returns:
        tuple or str: Кортеж (новая_строка_d, координаты_ребра)
        или исходная строка d

    """
    d_clean = d_attr.strip()
    has_z = d_clean.endswith(("Z", "z"))
    if has_z:
        d_clean = d_clean[:-1].strip()

    commands = parse_svg_commands(d_clean)
    points, cmd_indices = extract_points_from_commands(commands)
    if len(points) < 2:
        return d_attr

    edges = calculate_edges(points, cmd_indices)
    suitable_edges = filter_suitable_edges(edges, gap_width)
    selected_edge = select_edge_for_gap(edges, gap_width, suitable_edges)
    if not selected_edge:
        return d_attr

    break1_point, break2_point = calculate_gap_points(selected_edge, gap_width)
    new_commands = build_modified_commands(
        commands, selected_edge, break1_point, break2_point
    )
    new_d = build_svg_path_string(new_commands, has_z)

    return (
        new_d,
        (
            selected_edge["start_point"][0],
            selected_edge["start_point"][1],
            selected_edge["end_point"][0],
            selected_edge["end_point"][1],
        ),
    )


# --- Основная обработка ---
def find_closed_paths(root):
    """
    Находит все замкнутые пути в SVG.

    Args:
        root (xml.etree.ElementTree.Element): Корневой элемент XML-дерева SVG
    Returns:
        tuple: (список строк d, список элементов path)

    """
    closed_paths = []
    path_elements = []
    for elem in root.iter():
        if elem.tag.endswith("path"):
            d = elem.attrib.get("d", "")
            if "Z" in d.upper():
                closed_paths.append(d)
                path_elements.append(elem)
    return closed_paths, path_elements


def select_contours_for_modification(closed_paths):
    """
    Реализовал функционал случайного выбора контуров для модификации.

    Args:
        closed_paths (list): Список замкнутых контуров
    Returns:
        list: Список индексов выбранных контуров

    """
    num_to_modify = min(contours_per_file, len(closed_paths))
    return random.sample(range(len(closed_paths)), num_to_modify)


def modify_selected_contours(
    closed_paths,
    path_elements,
    selected_indices,
    gap_width,
    input_path,
    log_file,
):
    """
    Модифицирует выбранные контуры.

    Args:
        closed_paths (list): Список замкнутых контуров
        path_elements (list): Список элементов path
        selected_indices (list): Индексы выбранных контуров
        gap_width (float): Ширина разрыва в мм
        input_path (Path): Путь к входному файлу
        log_file (Path): Путь к файлу лога
    Returns:
        int: Количество модифицированных контуров

    """
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
                    f"({sx:.2f},{sy:.2f})->({ex:.2f},{ey:.2f})"
                )
                print(msg)
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now()}] {msg}\n")
    return modified_count


# --- Утилиты ---
def is_svg_file(file_path):
    """
    Проверяет, является ли файл SVG.

    Args:
        file_path (Path): Путь к файлу
    Returns:
        bool: True, если файл имеет расширение .svg

    """
    return file_path.suffix.lower() == ".svg"


def get_svg_files_from_directory(directory):
    """
    Получает список SVG файлов из директории.

    Args:
        directory (Path): Путь к директории
    Returns:
        list: Список путей к SVG файлам

    """
    return [f for f in directory.iterdir() if f.is_file() and is_svg_file(f)]


def process_svg_file(
    input_path: Path,
    output_path: Path,
    gap_width: float,
    log_file: Path = None,
):
    """
    Обрабатывает один SVG файл.

    Args:
        input_path (Path): Путь к входному файлу
        output_path (Path): Путь к выходному файлу
        gap_width (float): Ширина разрыва в мм
        log_file (Path, optional): Путь к файлу лога
    Returns:
        bool: True при успешной обработке, False при ошибке

    """
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(input_path)
        root = tree.getroot()

        closed_paths, path_elements = find_closed_paths(root)
        if not closed_paths:
            msg = f"Нет замкнутых контуров в {input_path.name}"
            print(msg)
            return True

        selected_indices = select_contours_for_modification(closed_paths)
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
    """
    Создает суффикс для имени файла по ширине разрыва.

    Args:
        gap_width (float): Ширина разрыва в мм
    Returns:
        str: Суффикс в формате _hole_unclosedXXX

    """
    gap_centi = int(round(gap_width * 100))
    return f"_hole_unclosed{gap_centi:03d}"


# --- CLI ---
def main():
    """Главная функция командной строки."""
    parser = argparse.ArgumentParser(
        description="Добавление щелей (gap) в SVG файлы"
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
        else settings.logging.log_dir / "augmentation_hole_unclosed.log"
    )

    if not (0.01 <= gap_width <= 0.05):
        print("Ширина щели должна быть в диапазоне 0.01–0.05 мм")
        sys.exit(1)

    suffix = get_gap_suffix(gap_width)

    if input_path.is_file():
        if not is_svg_file(input_path):
            print("Файл должен быть .svg")
            sys.exit(1)
        output_dir = output_dir or DEFAULT_SAVE_DIR
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

        output_dir = output_dir or DEFAULT_SAVE_DIR
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
