"""与赛题 101 点插值定义一致的 mAP50-95 指标。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from utils.nms import box_iou


IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


def match_predictions(
    detections: torch.Tensor,
    labels: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> torch.Tensor:
    """按置信度顺序为每个 IoU 阈值独立做一对一匹配。"""
    correct = torch.zeros((detections.shape[0], iou_thresholds.numel()), dtype=torch.bool, device=detections.device)
    if detections.numel() == 0 or labels.numel() == 0:
        return correct
    ious = box_iou(detections[:, :4], labels[:, 1:5])
    same_class = detections[:, 5:6].long() == labels[:, 0].long().unsqueeze(0)
    ious = ious * same_class
    order = detections[:, 4].argsort(descending=True)
    for threshold_index, threshold in enumerate(iou_thresholds.to(detections.device)):
        used: set[int] = set()
        for prediction_index in order.tolist():
            values = ious[prediction_index].clone()
            if used:
                values[list(used)] = -1
            best_iou, target_index = values.max(dim=0)
            if best_iou >= threshold:
                correct[prediction_index, threshold_index] = True
                used.add(int(target_index))
    return correct


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    samples = np.linspace(0.0, 1.0, 101)
    values = np.zeros_like(samples)
    for i, sample in enumerate(samples):
        candidates = precision[recall >= sample]
        values[i] = candidates.max() if candidates.size else 0.0
    return float(values.mean())


@dataclass
class DetectionMetrics:
    num_classes: int
    iou_thresholds: torch.Tensor = field(default_factory=lambda: IOU_THRESHOLDS.clone())
    correct: list[torch.Tensor] = field(default_factory=list)
    confidence: list[torch.Tensor] = field(default_factory=list)
    predicted_class: list[torch.Tensor] = field(default_factory=list)
    target_class: list[torch.Tensor] = field(default_factory=list)

    def update(self, detections: torch.Tensor, labels: torch.Tensor) -> None:
        if detections.shape[0]:
            self.correct.append(match_predictions(detections, labels, self.iou_thresholds).cpu())
            self.confidence.append(detections[:, 4].detach().cpu())
            self.predicted_class.append(detections[:, 5].detach().cpu())
        self.target_class.append(labels[:, 0].detach().cpu() if labels.shape[0] else torch.empty(0))

    def compute(self) -> dict[str, float | list[float]]:
        target_cls = torch.cat(self.target_class).numpy().astype(np.int64) if self.target_class else np.empty(0, np.int64)
        if self.correct:
            correct = torch.cat(self.correct).numpy().astype(bool)
            confidence = torch.cat(self.confidence).numpy()
            predicted_cls = torch.cat(self.predicted_class).numpy().astype(np.int64)
            order = np.argsort(-confidence)
            correct, confidence, predicted_cls = correct[order], confidence[order], predicted_cls[order]
        else:
            correct = np.zeros((0, len(self.iou_thresholds)), dtype=bool)
            confidence = np.empty(0)
            predicted_cls = np.empty(0, np.int64)

        ap = np.zeros((self.num_classes, len(self.iou_thresholds)), dtype=np.float64)
        present = np.zeros(self.num_classes, dtype=bool)
        for class_id in range(self.num_classes):
            prediction_mask = predicted_cls == class_id
            targets = int((target_cls == class_id).sum())
            if targets == 0:
                continue
            present[class_id] = True
            tp = correct[prediction_mask].astype(np.float64)
            fp = 1.0 - tp
            if tp.shape[0] == 0:
                continue
            tp_cumulative = np.cumsum(tp, axis=0)
            fp_cumulative = np.cumsum(fp, axis=0)
            recall = tp_cumulative / max(targets, 1)
            precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
            for threshold_index in range(len(self.iou_thresholds)):
                ap[class_id, threshold_index] = interpolated_ap(
                    recall[:, threshold_index], precision[:, threshold_index]
                )

        valid_ap = ap[present]
        map50 = float(valid_ap[:, 0].mean()) if valid_ap.size else 0.0
        map50_95 = float(valid_ap.mean()) if valid_ap.size else 0.0

        # Precision/Recall 使用 IoU=0.5 下使 micro-F1 最大的置信度截点。
        total_targets = max(len(target_cls), 1)
        if correct.shape[0]:
            tp_curve = np.cumsum(correct[:, 0])
            fp_curve = np.cumsum(~correct[:, 0])
            recall_curve = tp_curve / total_targets
            precision_curve = tp_curve / np.maximum(tp_curve + fp_curve, 1)
            f1 = 2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-12)
            best = int(np.argmax(f1))
            precision_value = float(precision_curve[best])
            recall_value = float(recall_curve[best])
            best_confidence = float(confidence[best])
        else:
            precision_value = recall_value = best_confidence = 0.0

        return {
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision_value,
            "recall": recall_value,
            "best_f1_confidence": best_confidence,
            "ap_per_class": ap.mean(axis=1).tolist(),
            "targets": int(len(target_cls)),
            "predictions": int(len(predicted_cls)),
        }
