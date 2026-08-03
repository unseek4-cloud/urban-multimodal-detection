"""跨模态、通道与空间注意力模块。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CrossModalAttention(nn.Module):
    """在固定大小的特征网格上做三模态全局注意力，控制 P2 的显存开销。"""

    def __init__(self, channels: int, heads: int = 4, pool_size: int = 8) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} 必须能被 heads={heads} 整除")
        self.pool_size = pool_size
        self.modality_embedding = nn.Parameter(torch.zeros(3, 1, channels))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.ModuleList([nn.Conv2d(channels, channels, 1, bias=False) for _ in range(3)])

    def forward(
        self, rgb: torch.Tensor, infrared: torch.Tensor, depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = (rgb, infrared, depth)
        pooled = [F.adaptive_avg_pool2d(item, self.pool_size) for item in features]
        token_count = self.pool_size * self.pool_size
        tokens = []
        for modality, item in enumerate(pooled):
            sequence = item.flatten(2).transpose(1, 2)
            sequence = sequence + self.modality_embedding[modality]
            tokens.append(sequence)
        sequence = torch.cat(tokens, dim=1)
        normalized = self.norm(sequence)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        sequence = sequence + attended

        outputs: list[torch.Tensor] = []
        for modality, original in enumerate(features):
            start = modality * token_count
            item = sequence[:, start : start + token_count]
            item = item.transpose(1, 2).reshape(
                original.shape[0], original.shape[1], self.pool_size, self.pool_size
            )
            item = F.interpolate(item, size=original.shape[-2:], mode="bilinear", align_corners=False)
            outputs.append(original + self.output[modality](item))
        return outputs[0], outputs[1], outputs[2]


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = self.mlp(F.adaptive_avg_pool2d(inputs, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(inputs, 1))
        return inputs * torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = inputs.mean(dim=1, keepdim=True)
        maximum = inputs.amax(dim=1, keepdim=True)
        weights = torch.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))
        return inputs * weights
