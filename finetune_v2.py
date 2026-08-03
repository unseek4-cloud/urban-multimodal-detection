#!/usr/bin/env python3
"""第二版独立 704 微调入口，只加载最佳模型权重并重置训练状态。"""

from train import main


if __name__ == "__main__":
    main("configs/yolov8m_p2_5ch_finetune.yaml")
