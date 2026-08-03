"""无 torchvision 依赖的类别感知 NMS 与坐标转换。"""

from __future__ import annotations

import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    output = boxes.clone()
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return output


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    output = boxes.clone()
    output[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
    output[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
    output[..., 2] = boxes[..., 2] - boxes[..., 0]
    output[..., 3] = boxes[..., 3] - boxes[..., 1]
    return output


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    area1 = (box1[:, 2] - box1[:, 0]).clamp_(min=0) * (box1[:, 3] - box1[:, 1]).clamp_(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp_(min=0) * (box2[:, 3] - box2[:, 1]).clamp_(min=0)
    intersection = (
        torch.minimum(box1[:, None, 2:], box2[:, 2:])
        - torch.maximum(box1[:, None, :2], box2[:, :2])
    ).clamp_(min=0).prod(2)
    return intersection / (area1[:, None] + area2 - intersection + 1e-7)


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """逐元素 Complete IoU。"""
    inter_wh = (torch.minimum(box1[:, 2:], box2[:, 2:]) - torch.maximum(box1[:, :2], box2[:, :2])).clamp(min=0)
    intersection = inter_wh[:, 0] * inter_wh[:, 1]
    wh1 = (box1[:, 2:] - box1[:, :2]).clamp(min=1e-7)
    wh2 = (box2[:, 2:] - box2[:, :2]).clamp(min=1e-7)
    union = wh1.prod(1) + wh2.prod(1) - intersection + 1e-7
    iou = intersection / union
    center1 = (box1[:, :2] + box1[:, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    center_distance = (center1 - center2).pow(2).sum(1)
    enclosing = torch.maximum(box1[:, 2:], box2[:, 2:]) - torch.minimum(box1[:, :2], box2[:, :2])
    diagonal = enclosing.pow(2).sum(1) + 1e-7
    v = (4 / torch.pi**2) * (torch.atan(wh2[:, 0] / wh2[:, 1]) - torch.atan(wh1[:, 0] / wh1[:, 1])).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    return iou - center_distance / diagonal - alpha * v


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def class_aware_nms(
    predictions: torch.Tensor,
    confidence: float = 0.05,
    iou_threshold: float = 0.7,
    max_detections: int = 100,
) -> list[torch.Tensor]:
    """将 [B, N, 4+nc] 解码输出转换为每图 [x1,y1,x2,y2,conf,class]。"""
    outputs: list[torch.Tensor] = []
    for prediction in predictions:
        class_scores, class_ids = prediction[:, 4:].max(dim=1)
        valid = class_scores >= confidence
        boxes = prediction[valid, :4]
        scores = class_scores[valid]
        classes = class_ids[valid]
        detections: list[torch.Tensor] = []
        for class_id in classes.unique():
            selected = classes == class_id
            keep = nms(boxes[selected], scores[selected], iou_threshold)
            class_boxes = boxes[selected][keep]
            class_scores_kept = scores[selected][keep, None]
            class_column = torch.full_like(class_scores_kept, float(class_id.item()))
            detections.append(torch.cat((class_boxes, class_scores_kept, class_column), dim=1))
        if detections:
            output = torch.cat(detections, dim=0)
            output = output[output[:, 4].argsort(descending=True)[:max_detections]]
        else:
            output = prediction.new_zeros((0, 6))
        outputs.append(output)
    return outputs


def scale_boxes_to_original(
    boxes: torch.Tensor,
    original_shape: tuple[int, int],
    ratio_pad: tuple[float, tuple[float, float]],
) -> torch.Tensor:
    ratio, (pad_x, pad_y) = ratio_pad
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    height, width = original_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    return boxes
