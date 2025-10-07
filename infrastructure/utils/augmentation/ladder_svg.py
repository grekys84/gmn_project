#!/usr/bin/env python3
import argparse
import sys
import math
import traceback
import random
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

from domain.config.settings import settings

INPUT_DIR = settings.data.master_svg
DEFAULT_SAVE_DIR = settings.data.augmentation_ladder
# Используем значения из settings как дефолтные
DEFAULT_STEP = settings.augmentation_ladder.step_length
DEFAULT_OFFSET = settings.augmentation_ladder.offset
DEFAULT_DIAG_OFFSET = settings.augmentation_ladder.diag_offset

# Путь к файлу лога
log_file = settings.logging.log_dir / "augmentation_ladder.log"


def make_stair_path(
    x1, y1, x2, y2, step=5.0, offset=2.0, diag_offset=None, seed=None
):
    """Преобразует прямую линию (x1,y1 -> x2,y2) в «лесенку»."""
    points = [(x1, y1)]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return f"M {x1},{y1} L {x2},{y2}"

    steps = max(1, int(length // step))
    if steps == 0:  # Добавлена проверка
        return f"M {x1},{y1} L {x2},{y2}"

    vx, vy = dx / length, dy / length
    nx, ny = -vy, vx  # нормаль

    # Определение смещения
    used_offset = offset
    if diag_offset is not None and abs(dx) > 1e-6 and abs(dy) > 1e-6:
        used_offset = diag_offset

    # Инициализация генератора случайных чисел
    rng = random.Random(seed) if seed is not None else random.Random()

    for i in range(1, steps + 1):
        t = i / steps
        px = x1 + dx * t
        py = y1 + dy * t
        if i % 2 == 1:
            direction = rng.choice([-1, 1])
            px += nx * used_offset * direction
            py += ny * used_offset * direction
        points.append((px, py))
    points.append((x2, y2))

    d = f"M {points[0][0]:.3f},{points[0][1]:.3f} "
    for x, y in points[1:]:
        d += f"L {x:.3f},{y:.3f} "
    return d.strip()


def transform_path(d_attr, step, offset, diag_offset, seed):
    """Трансформирует path d, если это простая линия."""
    d_attr = d_attr.strip()
    if not d_attr:
        return d_attr

    try:
        parts = d_attr.replace(",", " ").split()
        if (
            len(parts) >= 6
            and parts[0].upper() == "M"
            and parts[3].upper() == "L"
        ):
            x1, y1 = float(parts[1]), float(parts[2])
            x2, y2 = float(parts[4]), float(parts[5])
            return make_stair_path(
                x1, y1, x2, y2, step, offset, diag_offset, seed
            )
    except (ValueError, IndexError):
        # Логируем ошибку парсинга, но возвращаем оригинальный путь
        pass
    except Exception:
        # Логируем другие ошибки
        pass
    return d_attr


def is_valid_path_element(d):
    """Проверяет, является ли элемент допустимым для трансформации."""
    # Проверяем на сложные кривые
    if any(c in d.upper() for c in ("C", "Q", "A", "S", "T")):
        return False

    parts = d.replace(",", " ").split()
    # Проверяем формат: M x y L x y
    if len(parts) >= 6 and parts[0].upper() == "M" and parts[3].upper() == "L":
        # Дополнительная проверка на числовые значения
        try:
            float(parts[1])
            float(parts[2])
            float(parts[4])
            float(parts[5])
            return True
        except ValueError:
            return False
    return False


def find_transformable_paths(root):
    """Находит все пути, подходящие для трансформации."""
    path_elems = []
    for elem in root.iter():
        if elem.tag.endswith("path"):
            d = elem.attrib.get("d", "")
            if is_valid_path_element(d):
                path_elems.append(elem)
    return path_elems


def select_paths_for_transformation(path_elems, max_sides, seed):
    """Выбирает пути для трансформации."""
    if max_sides > 0 and len(path_elems) > max_sides:
        rng = random.Random(seed) if seed is not None else random.Random()
        return rng.sample(path_elems, max_sides)
    else:
        return path_elems


def transform_selected_paths(path_elems, step, offset, diag_offset, seed):
    """Трансформирует выбранные пути."""
    changed = 0
    for elem in path_elems:
        d = elem.attrib.get("d", "")
        new_d = transform_path(d, step, offset, diag_offset, seed)
        if new_d != d:
            elem.attrib["d"] = new_d
            changed += 1
    return changed


def process_svg_file(
    input_path,
    output_path,
    step,
    offset,
    diag_offset,
    seed,
    max_sides,
    log_file=None,
):
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(input_path)
        root = tree.getroot()

        # Находим кандидатов для трансформации
        path_elems = find_transformable_paths(root)

        # Выбираем элементы для трансформации
        to_transform = select_paths_for_transformation(
            path_elems, max_sides, seed
        )
        # Трансформируем выбранные элементы
        changed = transform_selected_paths(
            to_transform, step, offset, diag_offset, seed
        )
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        message = (
            f"{input_path.name}: преобразовано "
            f"{changed}/{len(path_elems)} path-элементов"
        )
        print(message)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{message}\n"
                )
        return True
    except Exception as e:
        message = (
            f"Ошибка обработки {input_path.name}: "
            f"{e}\n{traceback.format_exc()}"
        )
        print(message, file=sys.stderr)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{message}\n"
                )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Преобразование прямых в SVG в 'лесенку'"
    )
    parser.add_argument("input", help="Входной SVG файл или директория")
    parser.add_argument(
        "-o",
        "--output",
        help="Выходная директория (по умолчанию рядом с входом)",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=float,
        default=DEFAULT_STEP,
        help="Длина шага лесенки (мм)",
    )
    parser.add_argument(
        "-f",
        "--offset",
        type=float,
        default=DEFAULT_OFFSET,
        help="Смещение для H/V (мм)",
    )
    parser.add_argument(
        "-d",
        "--diag-offset",
        type=float,
        default=DEFAULT_DIAG_OFFSET,
        help="Смещение для диагоналей (мм)",
    )
    parser.add_argument("--seed", type=int, help="Seed для воспроизводимости")
    parser.add_argument(
        "-n",
        "--sides",
        type=int,
        default=settings.augmentation_ladder.sides,
        help="Число преобразуемых сторон (0 = все)",
    )
    parser.add_argument("-l", "--log", help="Файл для логирования")

    args = parser.parse_args()
    input_path = Path(args.input)

    # Определение директории вывода
    output_dir = Path(args.output) if args.output else None
    log_file_path = (
        Path(args.log)
        if args.log
        else settings.logging.log_dir / "augmentation_ladder.log"
    )

    if input_path.is_file():
        if not input_path.suffix.lower() == ".svg":
            print("Файл должен быть .svg")
            sys.exit(1)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir / f"{input_path.stem}_ladder{input_path.suffix}"
            )
        else:
            output_path = (
                input_path.parent
                / f"{input_path.stem}_ladder{input_path.suffix}"
            )

        process_svg_file(
            input_path,
            output_path,
            args.step,
            args.offset,
            args.diag_offset,
            args.seed,
            args.sides,
            log_file_path,
        )

    elif input_path.is_dir():
        svg_files = list(input_path.glob("*.svg"))
        if not svg_files:
            print("В директории нет SVG файлов")
            sys.exit(1)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_path

        success = 0
        for svg in svg_files:
            output_path = output_dir / f"{svg.stem}_ladder{svg.suffix}"
            if process_svg_file(
                svg,
                output_path,
                args.step,
                args.offset,
                args.diag_offset,
                args.seed,
                args.sides,
                log_file_path,
            ):
                success += 1
        print(f"\nОбработано файлов: {success}/{len(svg_files)}")

    else:
        print("Указанный путь не существует")
        sys.exit(1)


if __name__ == "__main__":
    main()
