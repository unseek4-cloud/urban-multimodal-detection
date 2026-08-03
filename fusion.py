"""三模态特征融合模块。"""

from __future__ import annotations

import torch
from torch import nn

from attention import ChannelAttention, CrossModalAttention, SpatialAttention
from backbone import ConvBNAct


class MultimodalFusionBlock(nn.Module):
    """先做跨模态注意力，再做通道和空间重标定，而不是简单拼接。"""

    def __init__(self, channels: int, heads: int = 4, pool_size: int = 8) -> None:
        super().__init__()
        self.cross_modal = CrossModalAttention(channels, heads=heads, pool_size=pool_size)
        self.merge = ConvBNAct(channels * 3, channels, 1)
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.residual = ConvBNAct(channels * 3, channels, 1, activation=False)

    def forward(
        self, rgb: torch.Tensor, infrared: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        attended = self.cross_modal(rgb, infrared, depth)
        concatenated = torch.cat(attended, dim=1)
        output = self.merge(concatenated)
        output = self.channel_attention(output)
        output = self.spatial_attention(output)
        return output + self.residual(torch.cat((rgb, infrared, depth), dim=1))


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: list[int], heads: int = 4, pool_size: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MultimodalFusionBlock(channel, heads=heads, pool_size=pool_size)
            for channel in channels
        )

    def forward(
        self,
        rgb_features: list[torch.Tensor],
        infrared_features: list[torch.Tensor],
        depth_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            block(rgb, infrared, depth)
            for block, rgb, infrared, depth in zip(
                self.blocks, rgb_features, infrared_features, depth_features
            )
        ]
