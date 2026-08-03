"""三模态数据集、同步增强、深度归一化和多尺度批处理。"""

from __future__ import annotations

import random
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SamplePaths:
    stem: str
    visible: Path
    infrared: Path
    depth: Path
    label: Path | None = None


def _index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"同一 stem 存在多个图像: {result[path.stem]} 和 {path}")
            result[path.stem] = path
    return result


def build_sample_index(data_root: str | Path, split: str, require_labels: bool) -> list[SamplePaths]:
    split_dir = Path(data_root) / split
    visible = _index_images(split_dir / "visible")
    infrared = _index_images(split_dir / "infrared")
    depth = _index_images(split_dir / "depth")
    labels = (
        {path.stem: path for path in sorted((split_dir / "labels").glob("*.txt"))}
        if require_labels
        else {}
    )
    indexes = {"visible": visible, "infrared": infrared, "depth": depth}
    if require_labels:
        indexes["labels"] = labels
    all_stems = set().union(*(set(value) for value in indexes.values()))
    complete = set.intersection(*(set(value) for value in indexes.values()))
    if complete != all_stems:
        summary = {name: len(all_stems - set(value)) for name, value in indexes.items()}
        raise ValueError(f"{split} 三模态或标签配对不完整，缺失数量: {summary}")
    return [
        SamplePaths(
            stem=stem,
            visible=visible[stem],
            infrared=infrared[stem],
            depth=depth[stem],
            label=labels.get(stem),
        )
        for stem in sorted(complete)
    ]


def read_split_file(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"划分文件不存在: {path}；请先运行 split_dataset.py")
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _read_unchanged(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"无法读取图像: {path}")
    return image


def read_modalities(sample: SamplePaths, strict_alignment: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible = _read_unchanged(sample.visible)
    infrared = _read_unchanged(sample.infrared)
    depth = _read_unchanged(sample.depth)

    if visible.ndim == 2:
        visible = cv2.cvtColor(visible, cv2.COLOR_GRAY2RGB)
    elif visible.shape[2] == 4:
        visible = cv2.cvtColor(visible, cv2.COLOR_BGRA2RGB)
    else:
        visible = cv2.cvtColor(visible, cv2.COLOR_BGR2RGB)
    if infrared.ndim == 3:
        infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY)
    if depth.ndim == 3:
        # 数据中的 JPG 深度是三通道 8-bit 灰度编码，不按毫米除以 20000。
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    shapes = {visible.shape[:2], infrared.shape[:2], depth.shape[:2]}
    if len(shapes) != 1:
        if strict_alignment:
            raise ValueError(
                f"空间对齐尺寸不一致 {sample.stem}: RGB={visible.shape}, IR={infrared.shape}, Depth={depth.shape}"
            )
        height, width = visible.shape[:2]
        infrared = cv2.resize(infrared, (width, height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return visible, infrared, depth


def read_labels(path: Path | None, width: int, height: int) -> np.ndarray:
    if path is None:
        return np.zeros((0, 5), dtype=np.float32)
    boxes: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"标签必须为 5 列: {path}:{line_number}")
        class_value, cx, cy, box_width, box_height = map(float, fields)
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < 12:
            raise ValueError(f"非法类别: {path}:{line_number} -> {class_value}")
        # 官方数据有 4 个坐标轻微超界，统一裁剪到图像边界。
        x1 = np.clip((cx - box_width / 2) * width, 0, width)
        y1 = np.clip((cy - box_height / 2) * height, 0, height)
        x2 = np.clip((cx + box_width / 2) * width, 0, width)
        y2 = np.clip((cy + box_height / 2) * height, 0, height)
        if x2 > x1 and y2 > y1:
            boxes.append([class_id, x1, y1, x2, y2])
        else:
            warnings.warn(f"忽略退化框: {path}:{line_number}", stacklevel=2)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 5)


def normalize_depth(depth: np.ndarray, minimum_mm: float, maximum_mm: float) -> np.ndarray:
    """16-bit PNG 按毫米裁剪；8-bit JPG 按其实际编码范围归一化。"""
    if depth.dtype == np.uint16 or float(depth.max(initial=0)) > 255:
        values = depth.astype(np.float32)
        valid = values >= minimum_mm
        values = np.clip(values, minimum_mm, maximum_mm) / maximum_mm
        values[~valid] = 0.0
        return values
    return depth.astype(np.float32) / 255.0


def random_crop_triplet(
    visible: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    boxes: np.ndarray,
    scale_range: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = visible.shape[:2]
    scale = random.uniform(float(scale_range[0]), float(scale_range[1]))
    crop_width, crop_height = max(2, round(width * scale)), max(2, round(height * scale))
    x0 = random.randint(0, max(0, width - crop_width))
    y0 = random.randint(0, max(0, height - crop_height))
    candidate = boxes.copy()
    if candidate.shape[0]:
        candidate[:, [1, 3]] = np.clip(candidate[:, [1, 3]] - x0, 0, crop_width)
        candidate[:, [2, 4]] = np.clip(candidate[:, [2, 4]] - y0, 0, crop_height)
        valid = (candidate[:, 3] - candidate[:, 1] >= 2) & (candidate[:, 4] - candidate[:, 2] >= 2)
        candidate = candidate[valid]
        if candidate.shape[0] == 0:
            return visible, infrared, depth, boxes
    slices = np.s_[y0 : y0 + crop_height, x0 : x0 + crop_width]
    return visible[slices], infrared[slices], depth[slices], candidate


def horizontal_flip_triplet(
    visible: np.ndarray, infrared: np.ndarray, depth: np.ndarray, boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = visible.shape[1]
    boxes = boxes.copy()
    if boxes.shape[0]:
        old_x1 = boxes[:, 1].copy()
        boxes[:, 1] = width - boxes[:, 3]
        boxes[:, 3] = width - old_x1
    return np.ascontiguousarray(visible[:, ::-1]), np.ascontiguousarray(infrared[:, ::-1]), np.ascontiguousarray(depth[:, ::-1]), boxes


def augment_rgb(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    values = image.astype(np.float32)
    brightness = 1.0 + random.uniform(-float(config["rgb_brightness"]), float(config["rgb_brightness"]))
    contrast = 1.0 + random.uniform(-float(config["rgb_contrast"]), float(config["rgb_contrast"]))
    mean = values.mean(axis=(0, 1), keepdims=True)
    values = (values - mean) * contrast + mean
    values *= brightness
    values = np.clip(values, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(values, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-config["hsv_h"], config["hsv_h"]) * 180) % 180
    hsv[..., 1] *= 1.0 + random.uniform(-config["hsv_s"], config["hsv_s"])
    hsv[..., 2] *= 1.0 + random.uniform(-config["hsv_v"], config["hsv_v"])
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_infrared(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    values = image.astype(np.float32) / 255.0
    if random.random() < float(config["ir_gamma_probability"]):
        gamma = random.uniform(*map(float, config["ir_gamma"]))
        values = np.power(np.clip(values, 0, 1), gamma)
    if random.random() < float(config["ir_noise_probability"]):
        values += np.random.normal(0.0, float(config["ir_noise_std"]), values.shape).astype(np.float32)
    return np.clip(values, 0, 1)


def letterbox_triplet(
    visible: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    boxes: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, tuple[float, float]]]:
    height, width = visible.shape[:2]
    ratio = min(size / width, size / height)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    pad_x = (size - resized_width) / 2
    pad_y = (size - resized_height) / 2
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)

    visible = cv2.resize(visible, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    infrared = cv2.resize(infrared, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    visible = cv2.copyMakeBorder(
        visible,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114 / 255, 114 / 255, 114 / 255),
    )
    infrared = cv2.copyMakeBorder(infrared, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.5)
    depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0)

    boxes = boxes.copy()
    if boxes.shape[0]:
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + left
        boxes[:, [2, 4]] = boxes[:, [2, 4]] * ratio + top
    return visible, infrared, depth, boxes, (ratio, (float(left), float(top)))


class MultimodalDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        image_size: int,
        data_config: dict[str, Any],
        augmentation_config: dict[str, Any] | None = None,
        split_file: str | Path | None = None,
        training: bool = False,
    ) -> None:
        self.samples = build_sample_index(data_root, split, require_labels=split == "train")
        if split_file is not None:
            selected = set(read_split_file(split_file))
            sample_by_stem = {sample.stem: sample for sample in self.samples}
            missing = selected - set(sample_by_stem)
            if missing:
                raise ValueError(f"划分文件含 {len(missing)} 个不存在的 stem，例如: {sorted(missing)[:5]}")
            self.samples = [sample_by_stem[stem] for stem in sorted(selected)]
        self.image_size = int(image_size)
        self.data_config = data_config
        self.augmentation = augmentation_config or {}
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self, num_classes: int) -> list[int]:
        """仅扫描当前划分的标签，供类别平衡损失计算权重。"""
        counts = [0 for _ in range(num_classes)]
        for sample in self.samples:
            if sample.label is None:
                continue
            for line_number, line in enumerate(
                sample.label.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(f"标签必须为 5 列: {sample.label}:{line_number}")
                class_value = float(fields[0])
                class_id = int(class_value)
                if class_value != class_id or not 0 <= class_id < num_classes:
                    raise ValueError(
                        f"非法类别: {sample.label}:{line_number} -> {class_value}"
                    )
                counts[class_id] += 1
        if any(value == 0 for value in counts):
            missing = [index for index, value in enumerate(counts) if value == 0]
            raise ValueError(f"训练划分中以下类别没有实例: {missing}")
        return counts

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        visible, infrared, depth = read_modalities(
            sample, strict_alignment=bool(self.data_config.get("strict_alignment", True))
        )
        original_shape = visible.shape[:2]
        boxes = read_labels(sample.label, original_shape[1], original_shape[0])

        if self.training and random.random() < float(self.augmentation.get("random_crop", 0.0)):
            visible, infrared, depth, boxes = random_crop_triplet(
                visible, infrared, depth, boxes, self.augmentation.get("crop_scale", [0.82, 1.0])
            )
        if self.training and random.random() < float(self.augmentation.get("horizontal_flip", 0.0)):
            visible, infrared, depth, boxes = horizontal_flip_triplet(visible, infrared, depth, boxes)
        if self.training:
            visible = augment_rgb(visible, self.augmentation)
            infrared = augment_infrared(infrared, self.augmentation)
        else:
            infrared = infrared.astype(np.float32) / 255.0
        depth = normalize_depth(
            depth,
            float(self.data_config.get("depth_min_mm", 300)),
            float(self.data_config.get("depth_max_mm", 20000)),
        )
        visible = visible.astype(np.float32) / 255.0

        visible, infrared, depth, boxes, ratio_pad = letterbox_triplet(
            visible, infrared, depth, boxes, self.image_size
        )
        labels = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        if boxes.shape[0]:
            labels[:, 0] = boxes[:, 0]
            labels[:, 1] = (boxes[:, 1] + boxes[:, 3]) / (2 * self.image_size)
            labels[:, 2] = (boxes[:, 2] + boxes[:, 4]) / (2 * self.image_size)
            labels[:, 3] = (boxes[:, 3] - boxes[:, 1]) / self.image_size
            labels[:, 4] = (boxes[:, 4] - boxes[:, 2]) / self.image_size
            labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)

        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(visible.transpose(2, 0, 1))),
            "infrared": torch.from_numpy(np.ascontiguousarray(infrared[None])),
            "depth": torch.from_numpy(np.ascontiguousarray(depth[None])),
            "labels": torch.from_numpy(labels),
            "stem": sample.stem,
            "original_shape": original_shape,
            "ratio_pad": ratio_pad,
            "visible_path": str(sample.visible),
        }


class MultimodalCollate:
    def __init__(self, training: bool, multi_scale: Sequence[int] | None = None) -> None:
        self.training = training
        self.multi_scale = [int(value) for value in (multi_scale or [])]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        rgb = torch.stack([item["rgb"] for item in batch])
        infrared = torch.stack([item["infrared"] for item in batch])
        depth = torch.stack([item["depth"] for item in batch])
        if self.training and self.multi_scale:
            size = random.choice(self.multi_scale)
            if rgb.shape[-1] != size:
                rgb = F.interpolate(rgb, size=(size, size), mode="bilinear", align_corners=False)
                infrared = F.interpolate(infrared, size=(size, size), mode="bilinear", align_corners=False)
                depth = F.interpolate(depth, size=(size, size), mode="nearest")

        targets: list[torch.Tensor] = []
        for batch_index, item in enumerate(batch):
            labels = item["labels"]
            if labels.shape[0]:
                batch_column = torch.full((labels.shape[0], 1), batch_index, dtype=labels.dtype)
                targets.append(torch.cat((batch_column, labels), dim=1))
        combined = torch.cat(targets, dim=0) if targets else torch.zeros((0, 6), dtype=torch.float32)
        return {
            "rgb": rgb,
            "infrared": infrared,
            "depth": depth,
            "targets": combined,
            "stems": [item["stem"] for item in batch],
            "original_shapes": [item["original_shape"] for item in batch],
            "ratio_pads": [item["ratio_pad"] for item in batch],
            "visible_paths": [item["visible_path"] for item in batch],
        }


def create_dataloader(
    dataset: MultimodalDataset,
    batch_size: int,
    data_config: dict[str, Any],
    shuffle: bool,
    multi_scale: Sequence[int] | None = None,
) -> DataLoader[dict[str, Any]]:
    available_cpus = max(1, (os.cpu_count() or 1) - 2)
    workers = min(int(data_config.get("num_workers", 8)), available_cpus)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=bool(data_config.get("persistent_workers", True)) and workers > 0,
        collate_fn=MultimodalCollate(training=dataset.training, multi_scale=multi_scale),
        drop_last=shuffle,
    )
