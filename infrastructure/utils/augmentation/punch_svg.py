"""
Скрипт для аугментации SVG-файлов путем добавления артефактов пуансонов.

Этот скрипт имитирует технологические отверстия или следы от инструментов
(пуансонов), которые могут присутствовать на деталях. Он добавляет случайные
геометрические формы (прямоугольники, круги, овалы) в SVG-файл.

Типы артефактов:
1. Прямоугольник 50×5 мм (длинное узкое отверстие)
2. Круг D=10 мм (круглое отверстие диаметром 10 мм)
3. Облонг 20×8 мм (овальное/удлиненное отверстие)

Цель: добавлять такие случайные "артефакты" на SVG для имитации:
- Остаточных отверстий от инструментов
- Следов от крепежа
- Технологических отверстий
"""

import xml.etree.ElementTree as ET
import random
import argparse
import sys
from pathlib import Path
from datetime import datetime

from domain.config.settings import settings

# --- Конфигурация из settings ---
INPUT_DIR = settings.data.master_svg
DEFAULT_SAVE_DIR = settings.data.augmentation_punch
log_file = settings.logging.log_dir / "augmentation_punch.log"

# Параметры аугментации из конфига
ARTIFACT_TYPES = (
    settings.augmentation_punch.artifact_types
)  # ['rectangle', 'circle', 'oblong']
RECTANGLE_SIZE = settings.augmentation_punch.rectangle_size  # (50, 5)
CIRCLE_DIAMETER = settings.augmentation_punch.circle_diameter  # 10
OBLONG_SIZE = settings.augmentation_punch.oblong_size  # (20, 8)


# --- Геометрические функции ---
def bbox_intersects(bbox1, bbox2, margin=0.1):
    """
    Проверяет пересечение двух ограничивающих прямоугольников (bounding boxes).

    Args:
        bbox1 (tuple): Ограничивающий прямоугольник в формате
                       (min_x, min_y, max_x, max_y)
        bbox2 (tuple): Ограничивающий прямоугольник в формате
                       (min_x, min_y, max_x, max_y)
        margin (float): Дополнительный зазор для проверки пересечения

    Returns:
        bool: True, если прямоугольники пересекаются, иначе False

    """
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2
    return not (
        max_x1 + margin < min_x2 - margin
        or max_x2 + margin < min_x1 - margin
        or max_y1 + margin < min_y2 - margin
        or max_y2 + margin < min_y1 - margin
    )


def _extract_coords_from_path(d: str) -> list[tuple[float, float]]:
    """Извлекает координаты из SVG path строки."""
    tokens = d.replace(",", " ").split()
    coords = []
    i = 0
    while i < len(tokens) - 2:
        if tokens[i].upper() in ("M", "L"):
            try:
                x, y = float(tokens[i + 1]), float(tokens[i + 2])
                coords.append((x, y))
                i += 3
                continue
            except Exception:
                pass
        i += 1
    return coords


def _get_bbox_from_coords(
    coords: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Вычисляет bounding box по списку координат."""
    if not coords:
        return None
    xs, ys = zip(*coords, strict=False)
    return (min(xs), min(ys), max(xs), max(ys))


def get_svg_bounds_and_existing_bboxes(root):
    """
    Получает границы SVG и ограничивающие прямоугольники всех элементов.

    Args:
        root (xml.etree.ElementTree.Element): Корневой элемент XML-дерева SVG
    Returns:
        tuple: Кортеж из двух элементов:
            - svg_bbox (tuple): Общие границы SVG в формате
                                (min_x, min_y, max_x, max_y)
            - existing_bboxes (list): Список ограничивающих прямоугольников
                                      существующих элементов

    """
    coords = []
    existing_bboxes = []

    for elem in root.iter():
        if elem.tag.endswith("path"):
            d = elem.attrib.get("d", "")
            if not d:
                continue
            coords_in_path = _extract_coords_from_path(d)
            coords.extend(coords_in_path)
            bbox = _get_bbox_from_coords(coords_in_path)
            if bbox:
                existing_bboxes.append(bbox)

    if not coords:
        vb = root.get("viewBox")
        if vb:
            parts = vb.split()
            if len(parts) == 4:
                min_x, min_y, width, height = map(float, parts)
                svg_bbox = (min_x, min_y, min_x + width, min_y + height)
                return svg_bbox, [svg_bbox]
        return (0, 0, 100, 100), [(0, 0, 100, 100)]

    xs, ys = zip(*coords, strict=False)
    svg_bbox = (min(xs), min(ys), max(xs), max(ys))
    return svg_bbox, existing_bboxes


# --- Генерация артефактов ---
def generate_artifact_path_at(artifact_type, x, y):
    """
    Генерирует SVG-путь для артефакта заданного типа с центром в точке (x, y).

    Args:
        artifact_type (str): Тип артефакта ('rectangle', 'circle', 'oblong')
        x (float): X-координата центра артефакта
        y (float): Y-координата центра артефакта
    Returns:
        str: Строка SVG-пути (атрибут d)

    Raises:
        ValueError: Если указан неизвестный тип артефакта

    """
    if artifact_type == "rectangle":
        w, h = RECTANGLE_SIZE
        return (
            f"M {x - w / 2},{y - h / 2} L {x + w / 2},{y - h / 2} "
            f"L {x + w / 2},{y + h / 2} L {x - w / 2},{y + h / 2} Z"
        )

    elif artifact_type == "circle":
        d = CIRCLE_DIAMETER
        r = d / 2  # радиус
        c = (
            0.551915024494 * r
        )  # "магический" коэффициент, масштабированный под радиус
        return (
            f"M {x},{y - r} "
            # Рисуем 4 четверти круга,
            # каждая аппроксимируется кубической кривой Безье
            # Первая четверть (верхняя правая)
            f"C {x + c},{y - r} {x + r},{y - c} {x + r},{y} "
            # Вторая четверть (нижняя правая)
            f"C {x + r},{y + c} {x + c},{y + r} {x},{y + r} "
            # Третья четверть (нижняя левая)
            f"C {x - c},{y + r} {x - r},{y + c} {x - r},{y} "
            # Четвертая четверть (верхняя левая)
            f"C {x - r},{y - c} {x - c},{y - r} {x},{y - r} "
        )

    elif artifact_type == "oblong":
        w, h = OBLONG_SIZE
        r = h / 2
        left_x = x - w / 2 + r
        right_x = x + w / 2 - r
        top_y = y - r
        bottom_y = y + r
        return (
            f"M {left_x},{top_y} "
            f"A {r},{r} 0 0,0 {left_x},{bottom_y} "
            f"L {right_x},{bottom_y} "
            f"A {r},{r} 0 0,0 {right_x},{top_y} "
            f"L {left_x},{top_y} Z"
        )

    else:
        raise ValueError(f"Неизвестный тип: {artifact_type}")


def is_placement_valid(
    new_bbox, existing_bboxes, svg_bbox, padding_ratio=0.02
):
    """
    Проверяет, является ли размещение нового артефакта допустимым.

    Args:
        new_bbox (tuple): Ограничивающий прямоугольник нового артефакта
                        (min_x, min_y, max_x, max_y)
        existing_bboxes (list): Список ограничивающих прямоугольников
                                существующих элементов
        svg_bbox (tuple): Общие границы SVG (min_x, min_y, max_x, max_y)
        padding_ratio (float): Отношение отступа к размеру SVG для границ
    Returns:
        bool: True, если размещение допустимо, иначе False

    """
    min_x, min_y, max_x, max_y = svg_bbox
    pad_x = (max_x - min_x) * padding_ratio
    pad_y = (max_y - min_y) * padding_ratio

    if (
        new_bbox[0] < min_x + pad_x
        or new_bbox[1] < min_y + pad_y
        or new_bbox[2] > max_x - pad_x
        or new_bbox[3] > max_y - pad_y
    ):
        return False

    for b in existing_bboxes:
        if bbox_intersects(new_bbox, b):
            return False
    return True


def place_artifact(artifact_type, svg_bbox, existing_bboxes, max_attempts=100):
    """
    Пытается найти допустимое место для размещения артефакта.

    Args:
        artifact_type (str): Тип артефакта ('rectangle', 'circle', 'oblong')
        svg_bbox (tuple): Общие границы SVG (min_x, min_y, max_x, max_y)
        existing_bboxes (list): Список ограничивающих прямоугольников
                                существующих элементов
        max_attempts (int): Максимальное количество попыток размещения
    Returns:
        tuple или None: Кортеж (cx, cy, path_d, suffix) если успешно,
                        иначе None

    """
    min_x, min_y, max_x, max_y = svg_bbox
    if artifact_type == "rectangle":
        w, h = RECTANGLE_SIZE
    elif artifact_type == "circle":
        w = h = CIRCLE_DIAMETER
    elif artifact_type == "oblong":
        w, h = OBLONG_SIZE
    else:
        return None

    for _ in range(max_attempts):
        cx = random.uniform(min_x + w / 2, max_x - w / 2)
        cy = random.uniform(min_y + h / 2, max_y - h / 2)
        path_d = generate_artifact_path_at(artifact_type, cx, cy)
        new_bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if is_placement_valid(new_bbox, existing_bboxes, svg_bbox):
            return (
                cx,
                cy,
                path_d,
                f"{artifact_type}_{int(w)}x{int(h)}"
                if artifact_type != "circle"
                else f"round_D{int(w)}",
            )
    return None


# --- Основная функция ---
def add_punch_artifacts_to_svg(
    input_path: Path, output_dir: Path, log_file_path: Path = None
):
    """
    Добавляет артефакты пуансонов в SVG файл и сохраняет с новым именем.

    Для каждого типа артефакта из конфигурации создается отдельный SVG-файл
    с одним экземпляром этого артефакта.

    Args:
        input_path (Path): Путь к входному SVG файлу
        output_dir (Path): Директория для выходных файлов
        log_file_path (Path, optional): Путь к файлу лога
    Returns:
        list: Список путей к созданным файлам

    """
    created_files = []
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(input_path)
        root = tree.getroot()
        svg_bbox, existing_bboxes = get_svg_bounds_and_existing_bboxes(root)

        for artifact_type in ARTIFACT_TYPES:
            res = place_artifact(artifact_type, svg_bbox, existing_bboxes)
            if not res:
                continue
            cx, cy, path_d, suffix = res
            new_tree = ET.parse(input_path)
            new_root = new_tree.getroot()
            art = ET.SubElement(new_root, "path")
            art.set("d", path_d)
            art.set("class", f"punch-{artifact_type}")
            out_name = f"{input_path.stem}_{suffix}{input_path.suffix}"
            out_path = output_dir / out_name
            new_tree.write(out_path, encoding="utf-8", xml_declaration=True)
            created_files.append(out_path)

            msg = f"Создан {out_name} с {artifact_type} ({cx:.1f},{cy:.1f})"
            print(msg)
            if log_file_path:
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] {msg}\n")

        return created_files
    except Exception as e:
        print(f"Ошибка {input_path.name}: {e}")
        return []


def main():
    """
    Функция командной строки для запуска аугментации артефактов пуансонов.

    Поддерживает обработку как отдельных файлов, так и целых директорий.
    """
    parser = argparse.ArgumentParser(
        description="Добавление артефактов в SVG файлы"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(INPUT_DIR),
        help=f"Входной SVG файл или директория (по умолчанию: {INPUT_DIR})",
    )
    parser.add_argument("-o", "--output", help="Выходная директория")
    parser.add_argument("-l", "--log", help="Файл лога")
    parser.add_argument(
        "--seed",
        type=int,
        help="Фиксированный seed для генератора случайных чисел",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    in_path = Path(args.input)
    out_dir = Path(args.output) if args.output else DEFAULT_SAVE_DIR
    log_file_path = Path(args.log) if args.log else log_file
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        add_punch_artifacts_to_svg(in_path, out_dir, log_file_path)
    elif in_path.is_dir():
        for svg_file in in_path.glob("*.svg"):
            add_punch_artifacts_to_svg(svg_file, out_dir, log_file_path)
    else:
        print("Указанный путь не существует")
        sys.exit(1)


if __name__ == "__main__":
    main()
