#!/usr/bin/env python3
"""检查三模态数据、YOLO 标签和文件配对关系。

脚本刻意只依赖 NumPy 与 Pillow，便于在安装完整训练环境之前先审计原始数据。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = [
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage can",
    "uav",
    "tricycle",
]


def image_index(directory: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """按不带扩展名的 stem 建索引，因此 PNG/JPG 可以混用。"""
    index: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)
    if not directory.is_dir():
        return index, duplicates
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in index:
            duplicates[path.stem].append(str(index[path.stem]))
            duplicates[path.stem].append(str(path))
        else:
            index[path.stem] = path
    return index, duplicates


def label_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {p.stem: p for p in sorted(directory.glob("*.txt")) if p.is_file()}


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    points = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        name: round(float(value), 8)
        for name, value in zip(
            ["min", "q25", "median", "q75", "q90", "q95", "q99", "max"], points
        )
    }


def inspect_images(index: dict[str, Path], modality: str, max_images: int) -> dict[str, Any]:
    selected = list(index.values())
    if max_images > 0:
        selected = selected[:max_images]

    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    global_min: float | None = None
    global_max: float | None = None
    zero_pixels = 0
    total_pixels = 0
    uint16_images = 0
    failures: list[dict[str, str]] = []

    for path in selected:
        try:
            with Image.open(path) as image:
                modes[image.mode] += 1
                array = np.asarray(image)
            shapes[str(tuple(array.shape))] += 1
            dtypes[str(array.dtype)] += 1
            if array.dtype == np.uint16:
                uint16_images += 1
            if array.size:
                value_min = float(array.min())
                value_max = float(array.max())
                global_min = value_min if global_min is None else min(global_min, value_min)
                global_max = value_max if global_max is None else max(global_max, value_max)
                if modality == "depth":
                    zero_pixels += int(np.count_nonzero(array == 0))
                    total_pixels += int(array.size)
        except Exception as exc:  # 记录坏图，但继续完成全量审计
            failures.append({"file": str(path), "error": str(exc)})

    return {
        "inspected": len(selected),
        "extensions": dict(Counter(path.suffix.lower() for path in index.values())),
        "modes": dict(modes),
        "shapes": dict(shapes),
        "dtypes": dict(dtypes),
        "min": global_min,
        "max": global_max,
        "uint16_images": uint16_images,
        "zero_ratio": round(zero_pixels / total_pixels, 8) if total_pixels else None,
        "read_failures": failures,
    }


def inspect_labels(index: dict[str, Path], num_classes: int) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_areas: list[float] = []
    empty_files = 0
    malformed: list[dict[str, Any]] = []
    invalid_class: list[dict[str, Any]] = []
    invalid_box: list[dict[str, Any]] = []

    for path in index.values():
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            empty_files += 1
            continue
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                malformed.append({"file": str(path), "line": line_number, "value": line})
                continue
            try:
                class_value, cx, cy, width, height = map(float, fields)
            except ValueError:
                malformed.append({"file": str(path), "line": line_number, "value": line})
                continue
            class_id = int(class_value)
            if class_value != class_id or not 0 <= class_id < num_classes:
                invalid_class.append({"file": str(path), "line": line_number, "class_id": class_value})
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                invalid_box.append({"file": str(path), "line": line_number, "box": [cx, cy, width, height]})
                continue
            class_counts[class_id] += 1
            bbox_widths.append(width)
            bbox_heights.append(height)
            bbox_areas.append(width * height)

    return {
        "class_count": len(class_counts),
        "class_distribution": {str(i): class_counts[i] for i in range(num_classes)},
        "target_count": sum(class_counts.values()),
        "empty_label_files": empty_files,
        "bbox_width": quantiles(bbox_widths),
        "bbox_height": quantiles(bbox_heights),
        "bbox_area": quantiles(bbox_areas),
        "malformed_lines": malformed,
        "invalid_classes": invalid_class,
        "invalid_boxes": invalid_box,
    }


def draw_distribution(counts: dict[str, int], output: Path) -> None:
    """不用 Matplotlib 也能生成稳定的类别分布图。"""
    width, height = 1400, 760
    left, top, right, bottom = 90, 80, 40, 150
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    values = [int(counts.get(str(i), 0)) for i in range(len(CLASS_NAMES))]
    maximum = max(values, default=1) or 1
    plot_w = width - left - right
    plot_h = height - top - bottom
    gap = 14
    bar_w = (plot_w - gap * (len(values) - 1)) / max(len(values), 1)

    draw.text((left, 24), "Class distribution (YOLO labels)", fill="black", font=font)
    draw.line((left, top, left, top + plot_h), fill="#222222", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#222222", width=2)
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * tick / 5
        draw.line((left - 5, y, left + plot_w, y), fill="#e5e7eb", width=1)
        draw.text((8, y - 7), f"{value:.0f}", fill="#374151", font=font)

    for i, (name, value) in enumerate(zip(CLASS_NAMES, values)):
        x0 = left + i * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = top + plot_h - (value / maximum) * plot_h
        draw.rectangle((x0, y0, x1, top + plot_h), fill="#2563eb")
        label = f"{i}:{name}"
        draw.text((x0, top + plot_h + 12), label, fill="#111827", font=font)
        draw.text((x0, max(top, y0 - 17)), str(value), fill="#111827", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def audit_split(root: Path, split: str, max_images: int) -> dict[str, Any]:
    split_dir = root / split
    visible, visible_duplicates = image_index(split_dir / "visible")
    infrared, infrared_duplicates = image_index(split_dir / "infrared")
    depth, depth_duplicates = image_index(split_dir / "depth")
    labels = label_index(split_dir / "labels") if split == "train" else {}

    indexes: dict[str, dict[str, Path]] = {"visible": visible, "infrared": infrared, "depth": depth}
    if split == "train":
        indexes["labels"] = labels
    union = set().union(*(set(index) for index in indexes.values()))
    complete = set.intersection(*(set(index) for index in indexes.values())) if indexes else set()
    missing = {
        name: sorted(union - set(index))
        for name, index in indexes.items()
        if union - set(index)
    }

    result: dict[str, Any] = {
        "directory": str(split_dir),
        "file_counts": {name: len(index) for name, index in indexes.items()},
        "matched_groups": len(complete),
        "union_groups": len(union),
        "missing": missing,
        "duplicates": {
            "visible": visible_duplicates,
            "infrared": infrared_duplicates,
            "depth": depth_duplicates,
        },
        "visible_info": inspect_images(visible, "visible", max_images),
        "infrared_info": inspect_images(infrared, "infrared", max_images),
        "depth_info": inspect_images(depth, "depth", max_images),
    }
    if split == "train":
        result["label_info"] = inspect_labels(labels, len(CLASS_NAMES))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 RGB/Infrared/Depth 三模态数据集")
    parser.add_argument("--data", type=Path, required=True, help="包含 train/ 和 test/ 的数据根目录")
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="每模态最多检查的图数；0 表示全量检查",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    report = {
        "data_root": str(root),
        "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "train": audit_split(root, "train", args.max_images),
        "test": audit_split(root, "test", args.max_images),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_distribution(report["train"]["label_info"]["class_distribution"], args.output / "class_distribution.png")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告: {report_path}")
    print(f"类别分布图: {args.output / 'class_distribution.png'}")


if __name__ == "__main__":
    main()
