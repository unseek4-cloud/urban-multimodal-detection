#!/usr/bin/env python3
"""用随机张量执行模型前向、损失、反向、解码和 NMS 烟测。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from losses import DetectionLoss
from model import build_model
from utils.nms import class_aware_nms


def run_mode(mode: str) -> None:
    config = {
        "mode": mode,
        "num_classes": 12,
        "channels": [16, 32, 64, 128],
        "neck_channels": 32,
        "reg_max": 8,
        "attention_heads": 4,
        "attention_pool": 4,
        "modality_dropout": 0.05,
    }
    model = build_model(config)
    inputs = {
        "rgb": torch.rand(2, 3, 128, 128),
        "infrared": torch.rand(2, 1, 128, 128),
        "depth": torch.rand(2, 1, 128, 128),
    }
    targets = torch.tensor(
        [[0, 0, 0.50, 0.50, 0.25, 0.35], [1, 6, 0.30, 0.40, 0.15, 0.20]],
        dtype=torch.float32,
    )
    model.train()
    raw = model(inputs)
    assert [tuple(item.shape[-2:]) for item in raw] == [(32, 32), (16, 16), (8, 8), (4, 4)]
    criterion = DetectionLoss(model, {"box": 7.5, "cls": 0.5, "dfl": 2.0, "center_radius": 2.5})
    loss, details = criterion(raw, targets, (128, 128))
    assert torch.isfinite(loss), details
    loss.backward()
    model.eval()
    with torch.inference_mode():
        decoded = model(inputs)
        assert decoded.shape == (2, 1360, 16)
        detections = class_aware_nms(decoded, confidence=0.99, iou_threshold=0.7, max_detections=100)
        assert len(detections) == 2
    print(f"{mode}: OK, loss={float(loss):.4f}, decoded={tuple(decoded.shape)}")


def main() -> None:
    run_mode("feature_fusion")
    run_mode("early_fusion")


if __name__ == "__main__":
    main()
