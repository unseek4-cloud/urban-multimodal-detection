"""训练样本与检测结果可视化。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def draw_detections(
    image_rgb: np.ndarray,
    detections: torch.Tensor,
    class_names: list[str],
    output: str | Path,
) -> None:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for x1, y1, x2, y2, confidence, class_id in detections.detach().cpu().tolist():
        color = (37, 99, 235)
        cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        name = class_names[int(class_id)] if int(class_id) < len(class_names) else str(int(class_id))
        cv2.putText(
            canvas,
            f"{name} {confidence:.2f}",
            (round(x1), max(15, round(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
