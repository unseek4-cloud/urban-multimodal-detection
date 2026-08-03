#!/usr/bin/env python3
"""在固定验证划分上计算 Precision、Recall、mAP50 和 mAP50-95。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset import MultimodalDataset, create_dataloader
from model import build_model
from utils.common import checkpoint_model_state, load_checkpoint, load_config, select_device
from utils.metrics import DetectionMetrics
from utils.nms import class_aware_nms, xywh_to_xyxy


def batch_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "rgb": batch["rgb"].to(device, non_blocking=True),
        "infrared": batch["infrared"].to(device, non_blocking=True),
        "depth": batch["depth"].to(device, non_blocking=True),
    }


def labels_for_image(
    targets: torch.Tensor, batch_index: int, image_width: int, image_height: int
) -> torch.Tensor:
    selected = targets[:, 0].long() == batch_index
    labels = targets[selected, 1:].clone()
    if labels.shape[0]:
        labels[:, 1:5] = xywh_to_xyxy(
            labels[:, 1:5] * labels.new_tensor([image_width, image_height, image_width, image_height])
        )
    return labels


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    confidence: float,
    nms_iou: float,
    max_detections: int,
    show_progress: bool = True,
) -> dict[str, float | list[float]]:
    model.eval()
    metrics = DetectionMetrics(num_classes)
    iterator = tqdm(loader, desc="验证", dynamic_ncols=True, disable=not show_progress)
    for batch in iterator:
        inputs = batch_inputs(batch, device)
        predictions = model(inputs)
        detections = class_aware_nms(predictions, confidence, nms_iou, max_detections)
        image_height, image_width = inputs["rgb"].shape[-2:]
        targets = batch["targets"].to(device)
        for batch_index, output in enumerate(detections):
            labels = labels_for_image(targets, batch_index, image_width, image_height)
            metrics.update(output, labels)
    return metrics.compute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证多模态检测模型")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=None, help="覆盖配置中的数据根目录")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-ema", action="store_true", help="不优先加载 EMA 权重")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data:
        config["data"]["root"] = args.data
    device = select_device(args.device or config.get("device", "auto"))
    checkpoint = load_checkpoint(args.weights, device)
    model_config = checkpoint.get("model_config", config["model"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint_model_state(checkpoint, prefer_ema=not args.no_ema))

    dataset = MultimodalDataset(
        config["data"]["root"],
        "train",
        config["training"]["image_size"],
        config["data"],
        split_file=config["data"]["val_split"],
        training=False,
    )
    loader = create_dataloader(dataset, config["training"]["batch_size"], config["data"], shuffle=False)
    validation = config["validation"]
    result = evaluate(
        model,
        loader,
        device,
        int(model_config["num_classes"]),
        float(validation["confidence"]),
        float(validation["nms_iou"]),
        int(validation["max_detections"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
