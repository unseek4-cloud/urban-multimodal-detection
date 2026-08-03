#!/usr/bin/env python3
"""第二版独立训练入口：YOLOv8m-P2 五通道 + TAL + 类别平衡损失。"""

from train import main


if __name__ == "__main__":
    main("configs/yolov8m_p2_5ch.yaml")
