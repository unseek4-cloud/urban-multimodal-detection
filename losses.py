"""Anchor-free YOLO 损失：Legacy/TAL 分配、CIoU、类别平衡 Focal BCE 与 DFL。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from utils.nms import bbox_ciou, box_iou, xywh_to_xyxy


def make_anchor_points(
    raw_outputs: list[torch.Tensor], strides: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    points: list[torch.Tensor] = []
    stride_values: list[torch.Tensor] = []
    for output, stride in zip(raw_outputs, strides):
        height, width = output.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(height, device=output.device),
            torch.arange(width, device=output.device),
            indexing="ij",
        )
        grid = (torch.stack((x, y), dim=-1).reshape(-1, 2).to(output.dtype) + 0.5) * stride
        points.append(grid)
        stride_values.append(torch.full((grid.shape[0],), stride, device=output.device, dtype=output.dtype))
    return torch.cat(points), torch.cat(stride_values)


def flatten_outputs(
    raw_outputs: list[torch.Tensor], num_classes: int, reg_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    regression_channels = 4 * (reg_max + 1)
    regressions: list[torch.Tensor] = []
    classes: list[torch.Tensor] = []
    for output in raw_outputs:
        batch = output.shape[0]
        regression = output[:, :regression_channels].permute(0, 2, 3, 1)
        regressions.append(regression.reshape(batch, -1, 4, reg_max + 1))
        classification = output[:, regression_channels:].permute(0, 2, 3, 1)
        classes.append(classification.reshape(batch, -1, num_classes))
    return torch.cat(regressions, dim=1), torch.cat(classes, dim=1)


def decode_distributions(
    regression: torch.Tensor,
    points: torch.Tensor,
    strides: torch.Tensor,
    reg_max: int,
) -> torch.Tensor:
    projection = torch.arange(reg_max + 1, device=regression.device, dtype=regression.dtype)
    distances = regression.softmax(dim=-1).matmul(projection) * strides[None, :, None]
    xy_min = points[None] - distances[..., :2]
    xy_max = points[None] + distances[..., 2:]
    return torch.cat((xy_min, xy_max), dim=-1)


def assign_targets(
    points: torch.Tensor,
    strides: torch.Tensor,
    ground_truth_boxes: torch.Tensor,
    center_radius: float,
    reg_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """中心采样后按最小 GT 面积分配，优先保护小目标。"""
    anchors = points.shape[0]
    if ground_truth_boxes.shape[0] == 0:
        return torch.zeros(anchors, dtype=torch.bool, device=points.device), torch.full(
            (anchors,), -1, dtype=torch.long, device=points.device
        )
    left_top = points[:, None] - ground_truth_boxes[None, :, :2]
    right_bottom = ground_truth_boxes[None, :, 2:] - points[:, None]
    distances = torch.cat((left_top, right_bottom), dim=-1)
    inside = distances.amin(dim=-1) > 0
    # 只把目标分给 DFL 可表达的尺度，避免大框被错误分到 P2。
    representable = (distances / strides[:, None, None]).amax(dim=-1) < reg_max - 0.01
    centers = (ground_truth_boxes[:, :2] + ground_truth_boxes[:, 2:]) / 2
    center_distance = (points[:, None] - centers[None]).abs()
    center = center_distance.amax(dim=-1) < center_radius * strides[:, None]
    candidates = inside & center & representable
    # 极小框若没有中心采样点，退回任意内部点。
    missing_gt = ~candidates.any(dim=0)
    candidates[:, missing_gt] = (inside & representable)[:, missing_gt]
    still_missing = ~candidates.any(dim=0)
    candidates[:, still_missing] = inside[:, still_missing]
    areas = (
        (ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0])
        * (ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1])
    )
    costs = areas[None].expand(anchors, -1).clone()
    costs[~candidates] = torch.inf
    minimum, assigned = costs.min(dim=1)
    foreground = torch.isfinite(minimum)
    assigned[~foreground] = -1
    return foreground, assigned


def distribution_focal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reg_max: int,
    reduction: str = "mean",
) -> torch.Tensor:
    target = target.clamp(0, reg_max - 0.01)
    left = target.floor().long()
    right = left + 1
    right_weight = target - left.to(target.dtype)
    left_weight = 1.0 - right_weight
    flat_prediction = prediction.reshape(-1, reg_max + 1)
    loss_left = F.cross_entropy(flat_prediction, left.reshape(-1), reduction="none")
    loss_right = F.cross_entropy(flat_prediction, right.reshape(-1), reduction="none")
    loss = loss_left * left_weight.reshape(-1) + loss_right * right_weight.reshape(-1)
    loss = loss.reshape(target.shape)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError(f"未知 DFL reduction: {reduction}")
    return loss.mean()


class TaskAlignedAssigner:
    """YOLOv8 风格 Task-Aligned Assigner。

    分类置信度和预测框 IoU 共同决定每个 GT 的 top-k 正样本，并使用归一化
    alignment metric 作为软分类质量目标。
    """

    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0) -> None:
        if topk < 1:
            raise ValueError("TAL topk 必须大于 0")
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @torch.no_grad()
    def __call__(
        self,
        class_probabilities: torch.Tensor,
        predicted_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        ground_truth_classes: torch.Tensor,
        ground_truth_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_probabilities = class_probabilities.float()
        predicted_boxes = predicted_boxes.float()
        anchor_points = anchor_points.float()
        ground_truth_boxes = ground_truth_boxes.float()
        anchors = anchor_points.shape[0]
        foreground = torch.zeros(anchors, dtype=torch.bool, device=anchor_points.device)
        assigned = torch.full((anchors,), -1, dtype=torch.long, device=anchor_points.device)
        quality = anchor_points.new_zeros((anchors,))
        if ground_truth_boxes.numel() == 0:
            return foreground, assigned, quality

        left_top = anchor_points[None] - ground_truth_boxes[:, None, :2]
        right_bottom = ground_truth_boxes[:, None, 2:] - anchor_points[None]
        inside = torch.cat((left_top, right_bottom), dim=-1).amin(dim=-1) > 0
        # stride-4 下仍可能存在窄于一个网格的极小框；为这类 GT 保留最近锚点。
        missing_inside = ~inside.any(dim=1)
        if missing_inside.any():
            centers = (ground_truth_boxes[:, :2] + ground_truth_boxes[:, 2:]) / 2
            closest = (centers[:, None] - anchor_points[None]).pow(2).sum(dim=-1).argmin(dim=1)
            rows = missing_inside.nonzero(as_tuple=False).squeeze(1)
            inside[rows, closest[rows]] = True
        overlaps = box_iou(ground_truth_boxes, predicted_boxes).clamp_(min=0)
        class_scores = class_probabilities[:, ground_truth_classes].transpose(0, 1)
        alignment = class_scores.pow(self.alpha) * overlaps.pow(self.beta)

        topk = min(self.topk, anchors)
        ranked = alignment.masked_fill(~inside, -1.0)
        _, topk_indices = ranked.topk(topk, dim=1, largest=True)
        topk_valid = inside.gather(1, topk_indices)
        candidates = torch.zeros_like(inside)
        candidates.scatter_(1, topk_indices, topk_valid)

        candidate_count = candidates.sum(dim=0)
        foreground = candidate_count > 0
        if not foreground.any():
            return foreground, assigned, quality
        best_overlap_gt = overlaps.masked_fill(~candidates, -1.0).argmax(dim=0)
        assigned[foreground] = best_overlap_gt[foreground]

        selected_alignment = alignment * candidates
        selected_overlap = overlaps * candidates
        maximum_alignment = selected_alignment.amax(dim=1, keepdim=True)
        maximum_overlap = selected_overlap.amax(dim=1, keepdim=True)
        normalized = selected_alignment * maximum_overlap / (maximum_alignment + 1e-9)
        foreground_indices = foreground.nonzero(as_tuple=False).squeeze(1)
        quality[foreground] = normalized[
            assigned[foreground], foreground_indices
        ].clamp_(0, 1)
        return foreground, assigned, quality


def effective_number_class_weights(
    counts: list[int] | tuple[int, ...],
    beta: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.numel() == 0 or (values <= 0).any():
        raise ValueError("class_counts 必须为每类大于 0 的样本数")
    if not 0.0 <= beta < 1.0:
        raise ValueError("class_balance_beta 必须位于 [0, 1)")
    weights = (1.0 - beta) / (
        1.0 - torch.pow(torch.tensor(beta, dtype=values.dtype), values)
    )
    weights = weights / weights.mean()
    return weights.clamp(min=minimum, max=maximum).to(torch.float32)


class DetectionLoss(nn.Module):
    def __init__(self, model: nn.Module, config: dict[str, Any], label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.num_classes = int(model.num_classes)
        self.reg_max = int(model.reg_max)
        self.strides = tuple(int(value) for value in model.strides)
        self.box_weight = float(config.get("box", 7.5))
        self.cls_weight = float(config.get("cls", 0.5))
        self.dfl_weight = float(config.get("dfl", 2.0))
        self.center_radius = float(config.get("center_radius", 2.5))
        self.assigner_name = str(config.get("assigner", "legacy")).lower()
        self.tal = TaskAlignedAssigner(
            topk=int(config.get("tal_topk", 10)),
            alpha=float(config.get("tal_alpha", 0.5)),
            beta=float(config.get("tal_beta", 6.0)),
        )
        counts = config.get("class_counts")
        if counts is None or counts == "auto":
            class_weights = torch.ones(self.num_classes, dtype=torch.float32)
        else:
            if len(counts) != self.num_classes:
                raise ValueError(
                    f"class_counts 长度 {len(counts)} 与类别数 {self.num_classes} 不一致"
                )
            class_weights = effective_number_class_weights(
                [int(value) for value in counts],
                beta=float(config.get("class_balance_beta", 0.999)),
                minimum=float(config.get("class_weight_min", 0.25)),
                maximum=float(config.get("class_weight_max", 4.0)),
            )
        self.register_buffer("class_weights", class_weights)
        self.focal_gamma = float(config.get("focal_gamma", 0.0))
        self.negative_focal_alpha = float(config.get("negative_focal_alpha", 1.0))
        self.positive_label = 1.0 - label_smoothing / 2
        # Anchor-free 检测中背景锚点数量巨大；只平滑正标签，背景仍为 0。
        self.negative_label = 0.0

    def forward(
        self,
        raw_outputs: list[torch.Tensor],
        targets: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.assigner_name == "tal":
            return self._forward_tal(raw_outputs, targets, image_size)
        if self.assigner_name != "legacy":
            raise ValueError(f"未知正样本分配器: {self.assigner_name}")
        return self._forward_legacy(raw_outputs, targets, image_size)

    def _forward_legacy(
        self,
        raw_outputs: list[torch.Tensor],
        targets: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        regression, classification = flatten_outputs(raw_outputs, self.num_classes, self.reg_max)
        points, strides = make_anchor_points(raw_outputs, self.strides)
        predicted_boxes = decode_distributions(regression, points, strides, self.reg_max)
        batch_size, anchors = classification.shape[:2]
        class_targets = torch.full_like(classification, self.negative_label)
        box_losses: list[torch.Tensor] = []
        dfl_losses: list[torch.Tensor] = []
        positive_count = 0
        image_height, image_width = image_size

        for batch_index in range(batch_size):
            selected = targets[:, 0].long() == batch_index
            batch_targets = targets[selected]
            if batch_targets.shape[0] == 0:
                continue
            ground_truth_classes = batch_targets[:, 1].long()
            normalized = batch_targets[:, 2:6]
            scale = normalized.new_tensor([image_width, image_height, image_width, image_height])
            ground_truth_boxes = xywh_to_xyxy(normalized * scale)
            foreground, assigned = assign_targets(
                points, strides, ground_truth_boxes, self.center_radius, self.reg_max
            )
            if not foreground.any():
                continue
            anchor_indices = foreground.nonzero(as_tuple=False).squeeze(1)
            assigned_indices = assigned[foreground]
            assigned_boxes = ground_truth_boxes[assigned_indices]
            assigned_classes = ground_truth_classes[assigned_indices]
            class_targets[batch_index, anchor_indices, assigned_classes] = self.positive_label
            positive_count += int(anchor_indices.numel())

            prediction_boxes = predicted_boxes[batch_index, anchor_indices]
            box_losses.append((1.0 - bbox_ciou(prediction_boxes, assigned_boxes)).mean())
            anchor_points = points[anchor_indices]
            target_distances = torch.stack(
                (
                    anchor_points[:, 0] - assigned_boxes[:, 0],
                    anchor_points[:, 1] - assigned_boxes[:, 1],
                    assigned_boxes[:, 2] - anchor_points[:, 0],
                    assigned_boxes[:, 3] - anchor_points[:, 1],
                ),
                dim=1,
            ) / strides[anchor_indices, None]
            dfl_losses.append(
                distribution_focal_loss(
                    regression[batch_index, anchor_indices], target_distances, self.reg_max
                )
            )

        classification_loss = F.binary_cross_entropy_with_logits(
            classification, class_targets, reduction="sum"
        ) / max(positive_count, batch_size)
        box_loss = torch.stack(box_losses).mean() if box_losses else classification.sum() * 0
        dfl_loss = torch.stack(dfl_losses).mean() if dfl_losses else classification.sum() * 0
        total = (
            self.box_weight * box_loss
            + self.cls_weight * classification_loss
            + self.dfl_weight * dfl_loss
        )
        details = {
            "loss": float(total.detach()),
            "box_loss": float(box_loss.detach()),
            "cls_loss": float(classification_loss.detach()),
            "dfl_loss": float(dfl_loss.detach()),
            "positive_anchors": float(positive_count),
        }
        return total, details

    def _balanced_focal_bce(
        self, logits: torch.Tensor, targets: torch.Tensor, normalizer: torch.Tensor
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        positive = targets > 0
        probabilities = logits.sigmoid()
        probability_of_target = torch.where(positive, probabilities, 1.0 - probabilities)
        if self.focal_gamma > 0:
            loss = loss * (1.0 - probability_of_target).pow(self.focal_gamma)
        if self.negative_focal_alpha != 1.0:
            alpha = torch.where(
                positive,
                torch.ones_like(loss),
                torch.full_like(loss, self.negative_focal_alpha),
            )
            loss = loss * alpha
        balance = torch.where(
            positive,
            self.class_weights.view(1, 1, -1).to(dtype=loss.dtype),
            torch.ones_like(loss),
        )
        return (loss * balance).sum() / normalizer

    def _forward_tal(
        self,
        raw_outputs: list[torch.Tensor],
        targets: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        regression, classification = flatten_outputs(raw_outputs, self.num_classes, self.reg_max)
        points, strides = make_anchor_points(raw_outputs, self.strides)
        points, strides = points.float(), strides.float()
        predicted_boxes = decode_distributions(
            regression.float(), points, strides, self.reg_max
        )
        batch_size = classification.shape[0]
        class_targets = torch.zeros_like(classification, dtype=torch.float32)
        assigned_boxes = torch.zeros_like(predicted_boxes)
        foreground_masks = torch.zeros(
            classification.shape[:2], dtype=torch.bool, device=classification.device
        )
        image_height, image_width = image_size

        for batch_index in range(batch_size):
            selected = targets[:, 0].long() == batch_index
            batch_targets = targets[selected]
            if batch_targets.shape[0] == 0:
                continue
            ground_truth_classes = batch_targets[:, 1].long()
            scale = batch_targets.new_tensor(
                [image_width, image_height, image_width, image_height]
            )
            ground_truth_boxes = xywh_to_xyxy(batch_targets[:, 2:6] * scale)
            foreground, assigned, quality = self.tal(
                classification[batch_index].detach().sigmoid(),
                predicted_boxes[batch_index].detach(),
                points,
                ground_truth_classes,
                ground_truth_boxes,
            )
            if not foreground.any():
                continue
            anchor_indices = foreground.nonzero(as_tuple=False).squeeze(1)
            assigned_indices = assigned[foreground]
            classes = ground_truth_classes[assigned_indices]
            scores = quality[foreground] * self.positive_label
            foreground_masks[batch_index, anchor_indices] = True
            assigned_boxes[batch_index, anchor_indices] = ground_truth_boxes[assigned_indices]
            class_targets[batch_index, anchor_indices, classes] = scores

        target_score_sum = class_targets.sum().clamp(min=1.0)
        classification_loss = self._balanced_focal_bce(
            classification, class_targets, target_score_sum
        )
        if foreground_masks.any():
            foreground_scores = class_targets.sum(dim=-1)[foreground_masks].detach()
            prediction_boxes = predicted_boxes[foreground_masks]
            target_boxes = assigned_boxes[foreground_masks]
            box_loss = (
                (1.0 - bbox_ciou(prediction_boxes, target_boxes)) * foreground_scores
            ).sum() / target_score_sum

            foreground_batch, foreground_anchor = foreground_masks.nonzero(as_tuple=True)
            anchor_points = points[foreground_anchor]
            target_distances = torch.stack(
                (
                    anchor_points[:, 0] - target_boxes[:, 0],
                    anchor_points[:, 1] - target_boxes[:, 1],
                    target_boxes[:, 2] - anchor_points[:, 0],
                    target_boxes[:, 3] - anchor_points[:, 1],
                ),
                dim=1,
            ) / strides[foreground_anchor, None]
            dfl_per_side = distribution_focal_loss(
                regression[foreground_batch, foreground_anchor],
                target_distances,
                self.reg_max,
                reduction="none",
            )
            dfl_loss = (dfl_per_side.mean(dim=1) * foreground_scores).sum() / target_score_sum
            positive_count = int(foreground_masks.sum())
        else:
            zero = classification.sum() * 0
            box_loss = zero
            dfl_loss = zero
            positive_count = 0

        total = (
            self.box_weight * box_loss
            + self.cls_weight * classification_loss
            + self.dfl_weight * dfl_loss
        )
        details = {
            "loss": float(total.detach()),
            "box_loss": float(box_loss.detach()),
            "cls_loss": float(classification_loss.detach()),
            "dfl_loss": float(dfl_loss.detach()),
            "positive_anchors": float(positive_count),
        }
        return total, details
