#!/usr/bin/env python3
"""提交前检查文件数、六列格式、范围、类别和每图最多 100 框。"""

from __future__ import annotations

import argparse
from pathlib import Path

from check_dataset import image_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查提交目录")
    parser.add_argument("--submission", type=Path, default=Path("submission"))
    parser.add_argument("--test-visible", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, _ = image_index(args.test_visible)
    predictions = {path.stem: path for path in args.submission.glob("*.txt")}
    missing = sorted(set(images) - set(predictions))
    extra = sorted(set(predictions) - set(images))
    errors: list[str] = []
    if missing:
        errors.append(f"缺少 {len(missing)} 个 TXT，例如 {missing[:5]}")
    if extra:
        errors.append(f"多出 {len(extra)} 个 TXT，例如 {extra[:5]}")
    for stem in sorted(set(images) & set(predictions)):
        lines = [line.strip() for line in predictions[stem].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(lines) > 100:
            errors.append(f"{stem}.txt 超过 100 框: {len(lines)}")
        for number, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 6:
                errors.append(f"{stem}.txt:{number} 不是 6 列")
                continue
            try:
                class_value, cx, cy, width, height, confidence = map(float, fields)
            except ValueError:
                errors.append(f"{stem}.txt:{number} 含非数字字段")
                continue
            if class_value != int(class_value) or not 0 <= int(class_value) < 12:
                errors.append(f"{stem}.txt:{number} 类别非法: {class_value}")
            if not all(0 <= value <= 1 for value in (cx, cy, width, height, confidence)):
                errors.append(f"{stem}.txt:{number} 坐标或置信度超界")
            if width <= 0 or height <= 0:
                errors.append(f"{stem}.txt:{number} 框宽高非正")
    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"提交检查失败，共 {len(errors)} 个问题")
    print(f"提交检查通过: {len(images)} 张图，所有 TXT 格式合法")


if __name__ == "__main__":
    main()
