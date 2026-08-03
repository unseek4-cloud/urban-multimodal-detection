#!/usr/bin/env python3
"""第二版模型的前向、TAL 损失、反向和解码烟测。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from losses import DetectionLoss
from model import build_model


def main() -> None:
    model_config = {
        "architecture": "yolov8m_p2_5ch",
        "mode": "yolov8m_p2_5ch",
        "num_classes": 12,
        "modality_dropout": 0.0,
        "verbose": False,
    }
    model = build_model(model_config)
    inputs = {
        "rgb": torch.rand(1, 3, 128, 128),
        "infrared": torch.rand(1, 1, 128, 128),
        "depth": torch.rand(1, 1, 128, 128),
    }
    targets = torch.tensor(
        [[0, 0, 0.50, 0.50, 0.25, 0.35], [0, 11, 0.25, 0.30, 0.10, 0.12]],
        dtype=torch.float32,
    )
    loss_config = {
        "assigner": "tal",
        "tal_topk": 10,
        "tal_alpha": 0.5,
        "tal_beta": 6.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "class_counts": [100, 20, 80, 40, 50, 45, 90, 10, 70, 30, 15, 5],
        "class_balance_beta": 0.999,
        "class_weight_min": 0.35,
        "class_weight_max": 3.0,
        "focal_gamma": 1.5,
        "negative_focal_alpha": 0.75,
    }
    criterion = DetectionLoss(model, loss_config, label_smoothing=0.0)

    model.train()
    raw = model(inputs)
    assert [tuple(item.shape[-2:]) for item in raw] == [
        (32, 32), (16, 16), (8, 8), (4, 4)
    ]
    assert all(item.shape[1] == 76 for item in raw)
    loss, details = criterion(raw, targets, (128, 128))
    assert torch.isfinite(loss), details
    assert details["positive_anchors"] > 0, details
    loss.backward()

    model.eval()
    with torch.inference_mode():
        decoded = model(inputs)
    assert decoded.shape == (1, 1360, 16), decoded.shape
    print(
        "v2 smoke test: OK, "
        f"loss={float(loss):.4f}, positives={details['positive_anchors']:.0f}, "
        f"decoded={tuple(decoded.shape)}"
    )


if __name__ == "__main__":
    main()
