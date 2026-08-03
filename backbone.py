"""CSP 风格的模态骨干和包含 P2 的 PAN-FPN。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.activation = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        self.first = ConvBNAct(channels, channels, 3)
        self.second = ConvBNAct(channels, channels, 3)
        self.shortcut = shortcut

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.second(self.first(inputs))
        return inputs + output if self.shortcut else output


class C2f(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeats: int = 2) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.first = ConvBNAct(in_channels, hidden * 2, 1)
        self.blocks = nn.ModuleList(Bottleneck(hidden) for _ in range(repeats))
        self.final = ConvBNAct(hidden * (2 + repeats), out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        parts = list(self.first(inputs).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.final(torch.cat(parts, dim=1))


class ModalBackbone(nn.Module):
    """输出 stride 4/8/16/32 的 P2-P5 特征。"""

    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels 必须包含 P2、P3、P4、P5 四个通道数")
        stem_channels = max(24, channels[0] // 2)
        self.stem = ConvBNAct(in_channels, stem_channels, 3, 2)
        self.stages = nn.ModuleList()
        current = stem_channels
        for index, output in enumerate(channels):
            repeats = 1 if index == 0 else 2
            self.stages.append(
                nn.Sequential(ConvBNAct(current, output, 3, 2), C2f(output, output, repeats))
            )
            current = output

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        output = self.stem(inputs)
        features: list[torch.Tensor] = []
        for stage in self.stages:
            output = stage(output)
            features.append(output)
        return features


class PANFPN(nn.Module):
    """P5->P2 自顶向下，再 P2->P5 自底向上。"""

    def __init__(self, in_channels: list[int], out_channels: int = 192) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(ConvBNAct(c, out_channels, 1) for c in in_channels)
        self.top_down = nn.ModuleList(C2f(out_channels * 2, out_channels, 2) for _ in range(3))
        self.downsample = nn.ModuleList(
            ConvBNAct(out_channels, out_channels, 3, 2) for _ in range(3)
        )
        self.bottom_up = nn.ModuleList(C2f(out_channels * 2, out_channels, 2) for _ in range(3))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        p2, p3, p4, p5 = [layer(item) for layer, item in zip(self.lateral, features)]
        p4 = self.top_down[0](torch.cat((p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")), dim=1))
        p3 = self.top_down[1](torch.cat((p3, F.interpolate(p4, size=p3.shape[-2:], mode="nearest")), dim=1))
        p2 = self.top_down[2](torch.cat((p2, F.interpolate(p3, size=p2.shape[-2:], mode="nearest")), dim=1))
        n3 = self.bottom_up[0](torch.cat((p3, self.downsample[0](p2)), dim=1))
        n4 = self.bottom_up[1](torch.cat((p4, self.downsample[1](n3)), dim=1))
        n5 = self.bottom_up[2](torch.cat((p5, self.downsample[2](n4)), dim=1))
        return [p2, n3, n4, n5]
