# 面向城市场景的视觉多模态目标检测

本工程面向 AIC“面向城市场景的视觉多模态目标检测”赛题，输入空间对齐的 RGB、Infrared 和 Depth，输出官方要求的六列预测：

```text
class_id norm_center_x norm_center_y norm_w norm_h confidence
```

这不是演示代码。工程包含真实数据审计、固定划分、三分支 Feature Fusion 主模型、五通道 Early Fusion 基线、训练、验证、断点恢复、EMA、TTA 推理、阈值搜索、提交压缩和提交格式检查。训练与推理均离线运行，不调用在线 API，也不读取官方数据之外的训练数据。

## 真实数据审计结论

已在本地 `G:\datasets` 执行审计：

- 训练集 2,000 组，RGB/Infrared/Depth/Label 全部按 stem 完整匹配。
- 测试集 1,000 组，三个模态全部完整匹配。
- 训练标签包含 12 类、14,413 个目标，无空标签文件。
- 图像分辨率包含 `640x360` 与 `1920x1080`。
- PNG 占训练集 1,851 组，JPG 占 149 组；测试集分别为 845 和 155 组。
- RGB 为三通道 `uint8`；Infrared 实际为三通道 `uint8` 灰度堆叠，加载时转单通道。
- Depth 不是单一格式：PNG 中存在单通道 `uint16` 毫米深度，最大值为 19999；JPG 是三通道 `uint8` 灰度编码。两类深度不能使用同一个除数。
- 抽检 200 张深度图的零值比例约为 48.94%。16 位深度按 `[300, 20000] mm` 有效范围裁剪并保留无效区为 0；8 位深度按 `[0,255]` 归一化。
- 官方标签有 4 个框轻微超出 `[0,1]`，数据加载器会裁剪到图像边界；格式错误、非法类别仍会立即报错。
- 目标类别长尾明显，最少类 `tricycle` 仅 25 个目标，最多类 `person` 有 5,496 个目标。

完整结果位于 `outputs/dataset_audit/dataset_report.json`，类别图位于 `outputs/dataset_audit/class_distribution.png`。

## 模型结构

主模型 `feature_fusion` 在 P2、P3、P4、P5 四个尺度均执行融合：

```text
RGB(3ch)   -> CSP Backbone --┐
Infrared   -> CSP Backbone --+-> Cross-Modal Attention
Depth      -> CSP Backbone --┘   -> Channel Attention
                                  -> Spatial Attention
                                  -> P2-P5 PAN-FPN
                                  -> Anchor-free YOLO Head
                                  -> 12 类 + DFL 边框分布
```

跨模态注意力先将高分辨率特征池化到固定网格，再通过多头注意力交换三模态信息，避免 P2 全分辨率注意力造成显存爆炸。融合后使用通道与空间注意力进行重标定，并保留残差路径。P2 检测层用于改善行人、标识、远距离车辆等小目标。

`early_fusion` 基线把 RGB 三通道、Infrared 单通道和 Depth 单通道拼成五通道，使用同一套 P2-P5 Neck 和检测头，便于做严格对照实验。

训练损失为 CIoU、分类 BCE 和 Distribution Focal Loss，默认权重分别为 `7.5 / 0.5 / 2.0`。后处理按类别执行 NMS，防止不同类别相互抑制，每图最终最多保留 100 个框。

## 目录与文件作用

```text
urban-multimodal-detection/
├── train.py                       # 完整训练、验证、保存、EMA、断点恢复和早停
├── val.py                         # 官方 101 点插值 mAP50-95 验证
├── predict.py                     # 原图+水平翻转 TTA，生成 TXT 和 ZIP
├── dataset.py                     # 三模态配对、同步增强、深度预处理、多尺度批处理
├── split_dataset.py               # seed=42 的 85%/15% 固定划分
├── model.py                       # 主模型、Early Fusion 基线、P2-P5 检测头与解码
├── backbone.py                    # CSP 风格模态骨干和 PAN-FPN
├── fusion.py                      # 四尺度三模态特征融合
├── attention.py                   # Cross-Modal、Channel、Spatial Attention
├── losses.py                      # CIoU、分类 BCE、DFL 与小目标优先分配
├── configs/
│   └── default.yaml               # RTX 4090 默认配置
├── utils/
│   ├── common.py                  # 配置、设备、EMA、调度器、早停、检查点和日志
│   ├── metrics.py                 # IoU 0.50:0.95、101 点插值 AP
│   ├── nms.py                     # 纯 PyTorch 类别感知 NMS 与坐标变换
│   └── visualization.py           # 检测结果可视化
├── tools/
│   ├── bootstrap.py               # 检查并自动安装缺失依赖
│   ├── check_dataset.py           # 全量配对/标签审计与图像格式抽检
│   ├── search_threshold.py        # 缓存推理并搜索 20 组 confidence/NMS 阈值
│   ├── validate_submission.py     # 检查提交文件数、六列格式和取值范围
│   └── smoke_test.py              # 两种模型的前向/损失/反向/NMS 烟测
├── outputs/                       # best.pt、ema_best.pt、last.pt、results.csv
├── experiments/                   # 每个实验的配置、结果、日志和最佳权重
├── weights/                       # 可选的本地权重存放目录
├── splits/
│   ├── train.txt                  # 已生成的 1,700 个训练 stem
│   └── val.txt                    # 已生成的 300 个验证 stem
├── submission/                    # 每张测试图同名的 TXT
├── requirements.txt
└── README.md
```

## 环境安装

当前依赖已按云端镜像固定为 Ubuntu 22.04、Python 3.10、PyTorch 2.1.2、Torchvision 0.16.2 和 CUDA 11.8。镜像已经预装 PyTorch，不需要另外创建虚拟环境或升级 PyTorch。

```bash
cd /root/autodl-fs/urban-multimodal-detection
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
python tools/bootstrap.py
```

项目终端当前使用 `/root/miniconda3/bin/python`；如果该解释器中尚未安装 PyTorch，普通的 `python tools/bootstrap.py` 会直接安装 2.1.2/cu118。只有同一解释器已经存在错误的 Torch/CUDA 版本时，脚本才会提示改用 `--repair-torch-stack`。始终使用 `python -m pip`，避免 `pip` 指向另一个 Python 环境。安装完成后确认 RTX 4090 可见：

```bash
python tools/bootstrap.py --check-only
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements.txt` 使用清华 PyPI 镜像下载普通依赖，PyTorch 仍使用官方 CUDA 11.8 wheel 索引；NumPy 固定在 1.26.4，避免 PyTorch 2.1.x 与 NumPy 2.x 的 ABI 兼容问题。安装命令启用 `--no-cache-dir`，减少 30 GB 系统盘占用。

## 数据检查和固定划分

上传代码后先运行完整审计。该命令只读官方数据，不修改原图或标签：

```bash
python tools/check_dataset.py \
  --data /root/autodl-fs/datasets \
  --output outputs/dataset_audit \
  --max-images 0
```

重新生成确定性划分时使用：

```bash
python split_dataset.py \
  --data /root/autodl-fs/datasets \
  --output splits \
  --train-ratio 0.85 \
  --seed 42
```

划分脚本只读取 `train/`，不会接触 `test/`。当前仓库已生成 `1700/300` 且无重叠的划分文件。

## 训练前烟测

先用随机张量确认两种模型的前向、损失、反向和 NMS 都能运行：

```bash
python tools/smoke_test.py
```

## 第一次训练

主模型默认启用 AMP、AdamW、5 epoch warmup、余弦退火、基于 mAP50-95 的 Plateau 衰减、EMA、梯度裁剪 `max_norm=10`、正标签平滑 `0.05`、P2-P5 检测和 `480/512/640/704` 多尺度训练。每 10 个 epoch 验证一次，200 epoch 上限，连续 30 epoch 无提升早停。

```bash
python train.py \
  --config configs/default.yaml \
  --data /root/autodl-fs/datasets \
  --mode feature_fusion \
  --name feature_fusion_main
```

RTX 4090 默认 `batch_size=12`、`num_workers=12`。若仍有显存余量，可先将 batch size 增至 16；若出现 OOM，降至 8，不要降低 P2 或输入尺度作为第一选择。

训练 Early Fusion 对照组：

```bash
python train.py \
  --config configs/default.yaml \
  --data /root/autodl-fs/datasets \
  --mode early_fusion \
  --name early_fusion_baseline
```

断点恢复会还原模型、EMA、优化器、学习率调度、AMP scaler、最佳指标和早停状态：

```bash
python train.py --config configs/default.yaml --resume outputs/last.pt
```

## 验证和阈值搜索

`best.pt` 内同时包含原始模型与 EMA，验证默认优先使用 EMA：

```bash
python val.py \
  --config configs/default.yaml \
  --weights outputs/best.pt \
  --data /root/autodl-fs/datasets
```

阈值搜索只做一次网络推理，然后在固定验证缓存上测试 5 个 confidence 和 4 个 NMS IoU：

```bash
python tools/search_threshold.py \
  --config configs/default.yaml \
  --weights outputs/ema_best.pt \
  --data /root/autodl-fs/datasets
```

结果写入 `outputs/threshold_search/results.csv` 与 `best_thresholds.json`。将最佳值更新到 `configs/default.yaml` 的 `prediction` 段后再生成提交。

## 第二版：YOLOv8m-P2 五通道 + TAL

第二版使用独立入口 `train_v2.py`，不会覆盖第一版 `outputs/best.pt`。它包含以下改动：

- 官方 YOLOv8m-P2 拓扑，检测层为 stride `4/8/16/32`；
- RGB 三通道、Infrared 单通道、Depth 单通道在输入端拼成五通道；
- 自动迁移 COCO `yolov8m.pt` 的所有形状兼容权重，首层卷积按 `3 -> 5` 通道做幅值保持初始化；
- 使用 Task-Aligned Assigner（`topk=10, alpha=0.5, beta=6.0`）；
- 根据当前训练划分自动统计 12 类频次，使用 Effective Number 权重与 Focal BCE；
- 初训权重写入 `outputs/v2`，704 微调权重写入 `outputs/v2_finetune`。

安装新增依赖并先做第二版烟测：

```bash
cd /root/autodl-fs/urban-multimodal-detection
python tools/bootstrap.py
python tools/smoke_test_v2.py
```

第一次运行会自动下载官方 `yolov8m.pt` 到 `weights/`，随后开始第二版初训：

```bash
export OMP_NUM_THREADS=8
nohup python train_v2.py \
  --name yolov8m_p2_5ch_main \
  > v2_train_console.log 2>&1 &
tail -f v2_train_console.log
```

如果 24 GB 显存发生 OOM，只把 `configs/yolov8m_p2_5ch.yaml` 的 `batch_size` 从 `8` 调到 `6` 或 `4`，不要删除 P2。断点续训使用完整状态：

```bash
python train_v2.py --resume outputs/v2/last.pt --name yolov8m_p2_5ch_main
```

初训结束后，独立 704 微调入口默认读取 `outputs/v2/ema_best.pt`，重新开始 30 epoch 的低学习率训练：

```bash
nohup python finetune_v2.py \
  --name yolov8m_p2_5ch_finetune704 \
  > v2_finetune_console.log 2>&1 &
tail -f v2_finetune_console.log
```

也可以显式指定任意第二版权重；该入口只接收结构完全一致的第二版检查点，不能加载第一版 `feature_fusion` 权重：

```bash
python finetune_v2.py \
  --finetune outputs/v2/ema_best.pt \
  --output-dir outputs/v2_finetune \
  --name yolov8m_p2_5ch_finetune704
```

微调后先搜索阈值，再验证和生成独立提交包：

```bash
python tools/search_threshold.py \
  --config configs/yolov8m_p2_5ch_finetune.yaml \
  --weights outputs/v2_finetune/ema_best.pt \
  --data /root/autodl-fs/datasets \
  --output outputs/v2_finetune/threshold_search

python val.py \
  --config configs/yolov8m_p2_5ch_finetune.yaml \
  --weights outputs/v2_finetune/ema_best.pt \
  --data /root/autodl-fs/datasets

python predict.py \
  --config configs/yolov8m_p2_5ch_finetune.yaml \
  --weights outputs/v2_finetune/ema_best.pt \
  --data /root/autodl-fs/datasets/test \
  --output submission_v2_finetune
```

## 预测和提交

命令兼容直接传入 `datasets/test`。默认使用 EMA 和原图+水平翻转 TTA：

```bash
python predict.py \
  --config configs/default.yaml \
  --weights outputs/best.pt \
  --data /root/autodl-fs/datasets/test
```

程序会为 1,000 张测试图逐一生成同名 TXT；无检测结果时生成空 TXT；每图按置信度最多保留 100 框；随后只把本次生成的文件打包为 `submission.zip`。

提交前执行最终格式检查：

```bash
python tools/validate_submission.py \
  --submission submission \
  --test-visible /root/autodl-fs/datasets/test/visible
```

## 后续提升 mAP50-95 的顺序

1. 先完成主模型与 Early Fusion 的同划分对照，确认提升来自 Feature Fusion，而非随机划分。
2. 使用阈值搜索结果，不重复历史上已经失败的强 HSV 实验；保留轻量 RGB 颜色扰动和红外 gamma/noise。
3. 重点查看 `tricycle`、`ball`、`boat`、`uav` 等少样本类别 AP。可在不引入外部数据的前提下使用类别均衡采样或分类损失重加权，但一次只改一个变量。
4. 对 `uint16 PNG` 与 `uint8 JPG` 深度分别统计训练/验证分布；若两域差异明显，可增加深度格式指示通道作为后续消融，但不要再次把 8 位深度按毫米处理。
5. 在 4090 显存允许时，固定 704 做最后 20-40 epoch 微调，优先验证小目标收益；不要简单延长到远超 200 epoch。
6. 分析模态质量差的样本。当前训练含辅助模态 dropout，可尝试质量感知门控，但需要与现有跨模态注意力做单变量对照。
7. 保留单模型规则。赛题禁止不同结构或阶段模型的简单投票/平均；本工程的 TTA 使用同一模型，符合该约束。
