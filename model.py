"""第一版三分支模型及第二版 YOLOv8m-P2 五通道模型构建入口。"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from backbone import ConvBNAct, ModalBackbone, PANFPN
from fusion import MultiScaleFusion


class DetectionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, reg_max: int, levels: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.regression = nn.ModuleList()
        self.classification = nn.ModuleList()
        self.regression_output = nn.ModuleList()
        self.classification_output = nn.ModuleList()
        for _ in range(levels):
            self.regression.append(
                nn.Sequential(ConvBNAct(channels, channels, 3), ConvBNAct(channels, channels, 3))
            )
            self.classification.append(
                nn.Sequential(ConvBNAct(channels, channels, 3), ConvBNAct(channels, channels, 3))
            )
            self.regression_output.append(nn.Conv2d(channels, 4 * (reg_max + 1), 1))
            self.classification_output.append(nn.Conv2d(channels, num_classes, 1))
        self._initialize_biases()

    def _initialize_biases(self) -> None:
        # 每幅图先验约 5 个目标，按各检测层网格数初始化，避免 P2 海量负样本主导首轮梯度。
        for layer, stride in zip(self.classification_output, (4, 8, 16, 32)):
            classification_bias = math.log(5.0 / self.num_classes / (640.0 / stride) ** 2)
            nn.init.constant_(layer.bias, classification_bias)
        for layer in self.regression_output:
            nn.init.constant_(layer.bias, 1.0)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for index, feature in enumerate(features):
            regression = self.regression_output[index](self.regression[index](feature))
            classification = self.classification_output[index](self.classification[index](feature))
            outputs.append(torch.cat((regression, classification), dim=1))
        return outputs


class MultimodalDetector(nn.Module):
    strides = (4, 8, 16, 32)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.mode = str(config.get("mode", "feature_fusion"))
        self.num_classes = int(config["num_classes"])
        self.reg_max = int(config.get("reg_max", 16))
        channels = [int(value) for value in config.get("channels", [64, 128, 256, 512])]
        neck_channels = int(config.get("neck_channels", 192))
        self.modality_dropout = float(config.get("modality_dropout", 0.0))

        if self.mode == "rgb_only":
            self.rgb_backbone = ModalBackbone(3, channels)
        elif self.mode == "feature_fusion":
            self.rgb_backbone = ModalBackbone(3, channels)
            self.infrared_backbone = ModalBackbone(1, channels)
            self.depth_backbone = ModalBackbone(1, channels)
            self.fusion = MultiScaleFusion(
                channels,
                heads=int(config.get("attention_heads", 4)),
                pool_size=int(config.get("attention_pool", 8)),
            )
        elif self.mode == "early_fusion":
            self.early_backbone = ModalBackbone(5, channels)
        else:
            raise ValueError(f"未知模型模式: {self.mode}")

        self.neck = PANFPN(channels, neck_channels)
        self.head = DetectionHead(neck_channels, self.num_classes, self.reg_max, levels=4)
        self.register_buffer("projection", torch.arange(self.reg_max + 1, dtype=torch.float32), persistent=False)

    def _drop_modalities(
        self, infrared: torch.Tensor, depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return infrared, depth
        batch = infrared.shape[0]
        ir_keep = (torch.rand(batch, 1, 1, 1, device=infrared.device) >= self.modality_dropout).to(infrared.dtype)
        depth_keep = (torch.rand(batch, 1, 1, 1, device=depth.device) >= self.modality_dropout).to(depth.dtype)
        # 两个辅助模态同时丢失时恢复红外，保持每个样本至少两个模态可用。
        both_missing = (ir_keep + depth_keep) == 0
        ir_keep = torch.where(both_missing, torch.ones_like(ir_keep), ir_keep)
        return infrared * ir_keep, depth * depth_keep

    def extract_features(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        rgb = inputs["rgb"]
        if self.mode == "rgb_only":
            return self.neck(self.rgb_backbone(rgb))
        infrared, depth = self._drop_modalities(inputs["infrared"], inputs["depth"])
        if self.mode == "early_fusion":
            features = self.early_backbone(torch.cat((rgb, infrared, depth), dim=1))
        else:
            features = self.fusion(
                self.rgb_backbone(rgb),
                self.infrared_backbone(infrared),
                self.depth_backbone(depth),
            )
        return self.neck(features)

    def forward(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor] | torch.Tensor:
        raw = self.head(self.extract_features(inputs))
        return raw if self.training else self.decode(raw)

    def decode(self, raw_outputs: list[torch.Tensor]) -> torch.Tensor:
        decoded: list[torch.Tensor] = []
        regression_channels = 4 * (self.reg_max + 1)
        for output, stride in zip(raw_outputs, self.strides):
            batch, _, height, width = output.shape
            regression = output[:, :regression_channels]
            regression = regression.reshape(batch, 4, self.reg_max + 1, height, width)
            regression = regression.permute(0, 3, 4, 1, 2).contiguous()
            distances = regression.softmax(dim=-1).matmul(self.projection.to(regression.dtype)) * stride
            classes = output[:, regression_channels:].sigmoid().permute(0, 2, 3, 1)
            y, x = torch.meshgrid(
                torch.arange(height, device=output.device),
                torch.arange(width, device=output.device),
                indexing="ij",
            )
            centers = torch.stack((x, y), dim=-1).to(output.dtype)
            centers = (centers + 0.5) * stride
            xy_min = centers[None] - distances[..., :2]
            xy_max = centers[None] + distances[..., 2:]
            boxes = torch.cat((xy_min, xy_max), dim=-1)
            decoded.append(torch.cat((boxes, classes), dim=-1).reshape(batch, -1, 4 + self.num_classes))
        return torch.cat(decoded, dim=1)


def build_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("architecture", "multimodal_detector"))
    if architecture == "yolov8m_p2_5ch":
        from yolov8_p2 import YOLOv8mP2FiveChannel

        return YOLOv8mP2FiveChannel(config)
    if architecture != "multimodal_detector":
        raise ValueError(f"未知模型架构: {architecture}")
    return MultimodalDetector(config)


def input_modalities(config: dict[str, Any]) -> tuple[str, ...]:
    """返回当前模型真正需要从磁盘加载的输入模态。"""
    if str(config.get("mode", "feature_fusion")) == "rgb_only":
        return ("rgb",)
    return ("rgb", "infrared", "depth")
