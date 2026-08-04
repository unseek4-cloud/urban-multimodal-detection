#!/usr/bin/env python3
"""缓存一次验证推理，网格搜索 confidence 与类别感知 NMS IoU。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from dataset import MultimodalDataset, create_dataloader
from model import build_model
from predict import flipped_predictions
from utils.common import checkpoint_model_state, load_checkpoint, load_config, select_device
from utils.metrics import DetectionMetrics
from utils.nms import class_aware_nms
from val import batch_inputs, labels_for_image


DEFAULT_CONFIDENCE_VALUES = [0.03, 0.05, 0.07, 0.10]
DEFAULT_NMS_IOU_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="搜索验证集后处理阈值")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="outputs/ema_best.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/threshold_search")
    parser.add_argument("--max-images", type=int, default=0, help="0 为完整验证集")
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用/禁用与 predict.py 相同的水平翻转 TTA；默认读取配置 prediction.tta",
    )
    parser.add_argument(
        "--confidence-values",
        type=float,
        nargs="+",
        default=DEFAULT_CONFIDENCE_VALUES,
        metavar="CONF",
    )
    parser.add_argument(
        "--nms-iou-values",
        type=float,
        nargs="+",
        default=DEFAULT_NMS_IOU_VALUES,
        metavar="IOU",
    )
    return parser.parse_args()


def normalized_grid(values: list[float], name: str) -> list[float]:
    grid = sorted(set(float(value) for value in values))
    if not grid or any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError(f"{name} 必须包含至少一个 0 到 1 之间的值")
    return grid


@torch.inference_mode()
def cache_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_images: int,
    min_confidence: float,
    use_tta: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    predictions_cache: list[torch.Tensor] = []
    labels_cache: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="缓存验证推理", dynamic_ncols=True):
        inputs = batch_inputs(batch, device)
        decoded = model(inputs)
        if use_tta:
            decoded = torch.cat((decoded, flipped_predictions(model, inputs)), dim=1)
        decoded = decoded.cpu()
        height, width = inputs["rgb"].shape[-2:]
        targets = batch["targets"]
        for index, prediction in enumerate(decoded):
            # 低于最小搜索阈值的候选不会参与任何组合。
            keep = prediction[:, 4:].amax(dim=1) >= min_confidence
            predictions_cache.append(prediction[keep])
            labels_cache.append(labels_for_image(targets, index, width, height).cpu())
            if max_images > 0 and len(predictions_cache) >= max_images:
                return predictions_cache, labels_cache
    return predictions_cache, labels_cache


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data:
        config["data"]["root"] = args.data
    device = select_device(args.device or config.get("device", "auto"))
    checkpoint = load_checkpoint(args.weights, device)
    model_config = checkpoint.get("model_config", config["model"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint_model_state(checkpoint, prefer_ema=True))
    dataset = MultimodalDataset(
        config["data"]["root"],
        "train",
        int(config["training"]["image_size"]),
        config["data"],
        split_file=config["data"]["val_split"],
        training=False,
    )
    loader = create_dataloader(
        dataset, int(config["training"]["batch_size"]), config["data"], shuffle=False
    )
    confidence_values = normalized_grid(args.confidence_values, "confidence-values")
    nms_iou_values = normalized_grid(args.nms_iou_values, "nms-iou-values")
    prediction_config = config.get("prediction", {})
    use_tta = bool(prediction_config.get("tta", False) if args.tta is None else args.tta)
    max_detections = int(
        prediction_config.get("max_detections", config["validation"]["max_detections"])
    )
    print(
        f"阈值搜索推理模式: TTA={use_tta}, confidence={confidence_values}, "
        f"nms_iou={nms_iou_values}, max_detections={max_detections}"
    )
    cached_predictions, cached_labels = cache_predictions(
        model,
        loader,
        device,
        args.max_images,
        min(confidence_values),
        use_tta,
    )
    rows: list[dict[str, float | bool]] = []
    for confidence in confidence_values:
        for nms_iou in nms_iou_values:
            metrics = DetectionMetrics(int(model_config["num_classes"]))
            for prediction, labels in zip(cached_predictions, cached_labels):
                detections = class_aware_nms(
                    prediction.unsqueeze(0), confidence, nms_iou,
                    max_detections,
                )[0]
                metrics.update(detections, labels)
            result = metrics.compute()
            rows.append(
                {
                    "tta": use_tta,
                    "confidence": confidence,
                    "nms_iou": nms_iou,
                    "precision": float(result["precision"]),
                    "recall": float(result["recall"]),
                    "map50": float(result["map50"]),
                    "map50_95": float(result["map50_95"]),
                }
            )
            print(
                f"conf={confidence:.2f} nms={nms_iou:.2f} "
                f"mAP50-95={result['map50_95']:.6f}"
            )
    best = max(rows, key=lambda row: row["map50_95"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "best_thresholds.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"最佳阈值: {json.dumps(best, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
