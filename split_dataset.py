#!/usr/bin/env python3
"""固定 seed=42 将官方训练集划分为 85%/15%，绝不读取 test。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在: {directory}")
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def create_split(data_root: Path, output: Path, train_ratio: float, seed: int) -> tuple[int, int]:
    train_root = data_root / "train"
    indexes = {
        "visible": image_stems(train_root / "visible"),
        "infrared": image_stems(train_root / "infrared"),
        "depth": image_stems(train_root / "depth"),
        "labels": {path.stem for path in (train_root / "labels").glob("*.txt")},
    }
    union = set().union(*indexes.values())
    complete = set.intersection(*indexes.values())
    if union != complete:
        missing = {name: len(union - stems) for name, stems in indexes.items()}
        raise ValueError(f"训练数据配对不完整: {missing}")
    stems = sorted(complete)
    random.Random(seed).shuffle(stems)
    boundary = round(len(stems) * train_ratio)
    train_stems, val_stems = stems[:boundary], stems[boundary:]
    if set(train_stems) & set(val_stems):
        raise RuntimeError("训练集与验证集发生重叠")
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    (output / "val.txt").write_text("\n".join(val_stems) + "\n", encoding="utf-8")
    return len(train_stems), len(val_stems)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="划分官方训练集")
    parser.add_argument("--data", type=Path, default=Path("/root/autodl-fs/datasets"))
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_count, val_count = create_split(args.data, args.output, args.train_ratio, args.seed)
    print(f"划分完成: train={train_count}, val={val_count}, seed={args.seed}")


if __name__ == "__main__":
    main()
