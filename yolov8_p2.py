"""YOLOv8m-P2 五通道检测器及 COCO 预训练权重迁移。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def yolov8m_p2_definition(num_classes: int) -> dict[str, Any]:
    """返回官方 YOLOv8-P2 拓扑，并固定使用 m 规模系数。"""
    return {
        "nc": num_classes,
        "scale": "m",
        "scales": {
            "n": [0.33, 0.25, 1024],
            "s": [0.33, 0.50, 1024],
            "m": [0.67, 0.75, 768],
            "l": [1.00, 1.00, 512],
            "x": [1.00, 1.25, 512],
        },
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 3, "C2f", [128, True]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 6, "C2f", [256, True]],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 6, "C2f", [512, True]],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 3, "C2f", [1024, True]],
            [-1, 1, "SPPF", [1024, 5]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 3, "C2f", [512]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 3, "C2f", [256]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 2], 1, "Concat", [1]],
            [-1, 3, "C2f", [128]],
            [-1, 1, "Conv", [128, 3, 2]],
            [[-1, 15], 1, "Concat", [1]],
            [-1, 3, "C2f", [256]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 12], 1, "Concat", [1]],
            [-1, 3, "C2f", [512]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 9], 1, "Concat", [1]],
            [-1, 3, "C2f", [1024]],
            [[18, 21, 24, 27], 1, "Detect", [num_classes]],
        ],
    }


class YOLOv8mP2FiveChannel(nn.Module):
    """把 RGB、红外和深度拼成五通道后送入 YOLOv8m-P2。"""

    strides = (4, 8, 16, 32)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        try:
            from ultralytics.nn.tasks import DetectionModel
        except ImportError as error:
            raise ImportError(
                "第二版模型需要 ultralytics；请执行 python tools/bootstrap.py 安装依赖"
            ) from error

        self.config = config
        self.mode = "yolov8m_p2_5ch"
        self.num_classes = int(config["num_classes"])
        definition = yolov8m_p2_definition(self.num_classes)
        self.network = DetectionModel(
            cfg=definition,
            ch=5,
            nc=self.num_classes,
            verbose=bool(config.get("verbose", False)),
        )
        detect = self.network.model[-1]
        # 本工程的 reg_max 表示 DFL 最大离散下标；Ultralytics 表示 bin 数量。
        self.reg_max = int(getattr(detect, "reg_max", 16)) - 1
        actual_strides = tuple(int(round(float(value))) for value in self.network.stride.tolist())
        if actual_strides != self.strides:
            raise RuntimeError(f"YOLOv8m-P2 stride 异常: {actual_strides}")
        self.modality_dropout = float(config.get("modality_dropout", 0.0))
        self.register_buffer(
            "projection", torch.arange(self.reg_max + 1, dtype=torch.float32), persistent=False
        )

    def _drop_modalities(
        self, infrared: torch.Tensor, depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return infrared, depth
        batch = infrared.shape[0]
        shape = (batch, 1, 1, 1)
        ir_keep = (torch.rand(shape, device=infrared.device) >= self.modality_dropout).to(infrared.dtype)
        depth_keep = (torch.rand(shape, device=depth.device) >= self.modality_dropout).to(depth.dtype)
        both_missing = (ir_keep + depth_keep) == 0
        ir_keep = torch.where(both_missing, torch.ones_like(ir_keep), ir_keep)
        return infrared * ir_keep, depth * depth_keep

    def forward(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor] | torch.Tensor:
        infrared, depth = self._drop_modalities(inputs["infrared"], inputs["depth"])
        five_channel = torch.cat((inputs["rgb"], infrared, depth), dim=1)
        output = self.network(five_channel)
        if self.training:
            return output
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError("Ultralytics Detect 未返回可解码的原始特征")
        return self.decode(output[1])

    def decode(self, raw_outputs: list[torch.Tensor]) -> torch.Tensor:
        decoded: list[torch.Tensor] = []
        regression_channels = 4 * (self.reg_max + 1)
        for output, stride in zip(raw_outputs, self.strides):
            batch, _, height, width = output.shape
            regression = output[:, :regression_channels]
            regression = regression.reshape(batch, 4, self.reg_max + 1, height, width)
            regression = regression.permute(0, 3, 4, 1, 2).contiguous()
            distances = regression.softmax(dim=-1).matmul(
                self.projection.to(dtype=regression.dtype)
            ) * stride
            classes = output[:, regression_channels:].sigmoid().permute(0, 2, 3, 1)
            y, x = torch.meshgrid(
                torch.arange(height, device=output.device),
                torch.arange(width, device=output.device),
                indexing="ij",
            )
            centers = (torch.stack((x, y), dim=-1).to(output.dtype) + 0.5) * stride
            boxes = torch.cat(
                (centers[None] - distances[..., :2], centers[None] + distances[..., 2:]),
                dim=-1,
            )
            decoded.append(
                torch.cat((boxes, classes), dim=-1).reshape(
                    batch, -1, 4 + self.num_classes
                )
            )
        return torch.cat(decoded, dim=1)

    def load_coco_pretrained(self, weights: str | Path) -> dict[str, Any]:
        """只从本地迁移 YOLOv8m 权重；此函数不会联网或自动下载。"""
        requested = Path(weights).expanduser()
        candidates = [requested]
        if not requested.is_absolute():
            candidates.extend(
                (
                    Path.cwd() / requested,
                    Path.cwd() / requested.name,
                    Path.cwd() / "weights" / requested.name,
                )
            )
        local_weights = next((path for path in candidates if path.is_file()), None)
        if local_weights is None:
            searched = "\n- ".join(str(path.resolve()) for path in dict.fromkeys(candidates))
            raise FileNotFoundError(
                "本地未找到 yolov8m.pt，程序不会自动下载。已检查:\n- " + searched
            )
        checkpoint = torch.load(local_weights, map_location="cpu")
        source_model: Any
        if isinstance(checkpoint, dict) and (checkpoint.get("ema") is not None):
            source_model = checkpoint["ema"]
        elif isinstance(checkpoint, dict) and (checkpoint.get("model") is not None):
            source_model = checkpoint["model"]
        else:
            source_model = checkpoint
        if hasattr(source_model, "float") and hasattr(source_model, "state_dict"):
            source_state = source_model.float().state_dict()
        elif isinstance(source_model, dict):
            source_state = source_model.get("state_dict", source_model)
        else:
            raise TypeError(f"无法识别预训练权重内容: {type(source_model)!r}")

        target_state = self.network.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        adapted_stem = 0
        for source_key, source_value in source_state.items():
            key = str(source_key)
            for prefix in ("module.", "network."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            if key not in target_state or not isinstance(source_value, torch.Tensor):
                continue
            target_value = target_state[key]
            value = source_value.detach().to(
                device=target_value.device, dtype=target_value.dtype
            )
            if value.shape == target_value.shape:
                transferred[key] = value
                continue
            if (
                key == "model.0.conv.weight"
                and value.ndim == 4
                and value.shape[1] == 3
                and target_value.shape[1] == 5
                and value.shape[0] == target_value.shape[0]
            ):
                expanded = target_value.clone()
                scale = 3.0 / 5.0
                expanded[:, :3] = value * scale
                expanded[:, 3:] = value.mean(dim=1, keepdim=True) * scale
                transferred[key] = expanded
                adapted_stem = 1

        incompatible = self.network.load_state_dict(transferred, strict=False)
        return {
            "source_path": str(local_weights.resolve()),
            "transferred_tensors": len(transferred),
            "target_tensors": len(target_state),
            "adapted_stem": adapted_stem,
            "missing_tensors": len(incompatible.missing_keys),
        }
