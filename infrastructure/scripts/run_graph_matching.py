#!/usr/bin/env python
"""CLI-интерфейс для GraphMatchingService."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.services.graph_matching_service import GraphMatchingService
from infrastructure.utils.match_graph import load_resources


def main() -> None:
    parser = argparse.ArgumentParser(description="Сопоставление SVG с эталонной базой данных")
    parser.add_argument("--svg", type=Path, required=True, help="Путь к SVG-файлу")
    args = parser.parse_args()

    load_resources()
    service = GraphMatchingService()
    result = service.predict_path(args.svg)

    # Вынесение вердикта либо по результату работы сервиса, либо по наличию ID
    valid = result.get("valid")
    if valid is None:
        valid = True if result.get("id") else False

    print(f"id: {result.get('id')}")
    print(f"similarity_percent: {result.get('similarity_percent')}")
    print(f"valid: {valid}")
    print(f"overlay_path: {result.get('overlay_path')}")


if __name__ == "__main__":
    main()
