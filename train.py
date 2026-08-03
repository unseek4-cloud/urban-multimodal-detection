#!/usr/bin/env python3
"""完整训练入口：AMP、AdamW、余弦/Plateau、EMA、断点恢复和早停。"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset import MultimodalDataset, create_dataloader
from losses import DetectionLoss
from model import build_model
from split_dataset import create_split
from utils.common import (
    CSVLogger,
    EarlyStopping,
    ModelEMA,
    WarmupCosinePlateauLR,
    checkpoint_model_state,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config,
    select_device,
    set_seed,
    setup_logger,
)
from val import evaluate


def parse_args(
    default_config: str = "configs/default.yaml", argv: list[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练城市三模态目标检测模型")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--data", default=None, help="覆盖数据根目录")
    parser.add_argument("--device", default=None)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume", nargs="?", const="outputs/last.pt", default=None,
        help="恢复完整训练状态（模型、优化器、调度器、EMA 和 epoch）",
    )
    checkpoint_group.add_argument(
        "--finetune", default=None,
        help="只加载模型/EMA 权重，重置优化器、调度器、早停和 epoch",
    )
    parser.add_argument(
        "--pretrained", default=None,
        help="首次训练时使用的 COCO YOLOv8m 权重；可为 yolov8m.pt",
    )
    parser.add_argument(
        "--mode",
        choices=["feature_fusion", "early_fusion", "yolov8m_p2_5ch"],
        default=None,
    )
    parser.add_argument("--name", default=None, help="experiments/ 下的实验名")
    parser.add_argument("--output-dir", default=None, help="权重和 results.csv 的独立输出目录")
    return parser.parse_args(argv)


def move_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "rgb": batch["rgb"].to(device, non_blocking=True),
        "infrared": batch["infrared"].to(device, non_blocking=True),
        "depth": batch["depth"].to(device, non_blocking=True),
    }


def checkpoint_state(
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosinePlateauLR,
    scaler: torch.cuda.amp.GradScaler,
    early_stopping: EarlyStopping,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "early_stopping": early_stopping.state_dict(),
        "best_metric": best_metric,
        "model_config": config["model"],
        "config": config,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: DetectionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    ema: ModelEMA,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip: float,
    epoch: int,
    log_interval: int,
    logger: Any,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"训练 {epoch + 1}", dynamic_ncols=True)
    for step, batch in enumerate(progress):
        inputs = move_inputs(batch, device)
        targets = batch["targets"].to(device, non_blocking=True)
        image_size = tuple(int(value) for value in inputs["rgb"].shape[-2:])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            raw_outputs = model(inputs)
            loss, details = criterion(raw_outputs, targets, image_size)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"出现非有限损失: {details}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

        for name, value in details.items():
            totals[name] += float(value)
        if step % log_interval == 0:
            progress.set_postfix(loss=f"{details['loss']:.3f}", size=image_size[0])
    count = max(1, len(loader))
    averages = {name: value / count for name, value in totals.items()}
    logger.info(
        "epoch=%d loss=%.4f box=%.4f cls=%.4f dfl=%.4f positives=%.1f",
        epoch + 1,
        averages.get("loss", 0.0),
        averages.get("box_loss", 0.0),
        averages.get("cls_loss", 0.0),
        averages.get("dfl_loss", 0.0),
        averages.get("positive_anchors", 0.0),
    )
    return averages


def ensure_splits(config: dict[str, Any]) -> None:
    train_path = Path(config["data"]["train_split"])
    val_path = Path(config["data"]["val_split"])
    if train_path.is_file() and val_path.is_file():
        return
    output = train_path.parent
    train_count, val_count = create_split(
        Path(config["data"]["root"]),
        output,
        float(config["data"].get("train_ratio", 0.85)),
        int(config["seed"]),
    )
    print(f"自动生成划分: train={train_count}, val={val_count}")


def main(
    default_config: str = "configs/default.yaml", argv: list[str] | None = None
) -> None:
    args = parse_args(default_config, argv)
    config = load_config(args.config)
    if args.data:
        config["data"]["root"] = args.data
    if args.device:
        config["device"] = args.device
    if args.mode:
        config["model"]["mode"] = args.mode
    configured_resume = str(config["training"].get("resume", "")).strip()
    configured_finetune = str(config["training"].get("finetune", "")).strip()
    if args.resume is not None:
        resume_path, finetune_path = args.resume, ""
    elif args.finetune is not None:
        resume_path, finetune_path = "", args.finetune
    else:
        resume_path, finetune_path = configured_resume, configured_finetune
    if resume_path and finetune_path:
        raise ValueError("--resume 与 --finetune 不能同时使用")
    pretrained_path = (
        args.pretrained
        if args.pretrained is not None
        else str(config["model"].get("pretrained", "")).strip()
    )
    mode = config["model"]["mode"]
    experiment_name = args.name or f"{mode}_seed{config['seed']}"
    output_dir = Path(
        args.output_dir or str(config["training"].get("output_dir", "outputs"))
    )
    experiment_dir = Path("experiments") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(experiment_dir / "train.log")
    set_seed(int(config["seed"]))
    device = select_device(str(config.get("device", "auto")))
    ensure_splits(config)

    checkpoint = None
    finetune_checkpoint = None
    if resume_path:
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"断点不存在: {resume_path}")
        checkpoint = load_checkpoint(resume_path, device)
        config["model"] = checkpoint.get("model_config", config["model"])
        logger.info("从断点恢复: %s", resume_path)
    elif finetune_path:
        if not Path(finetune_path).is_file():
            raise FileNotFoundError(f"微调权重不存在: {finetune_path}")
        finetune_checkpoint = load_checkpoint(finetune_path, device)
        logger.info("微调仅加载模型权重: %s", finetune_path)
    save_config(config, experiment_dir / "config.yaml")

    raw_model = build_model(config["model"]).to(device)
    if checkpoint is not None:
        raw_model.load_state_dict(checkpoint["model"])
    elif finetune_checkpoint is not None:
        raw_model.load_state_dict(
            checkpoint_model_state(finetune_checkpoint, prefer_ema=True), strict=True
        )
    elif pretrained_path:
        if not hasattr(raw_model, "load_coco_pretrained"):
            raise ValueError(
                f"当前模型 {type(raw_model).__name__} 不支持 --pretrained；"
                "该参数仅供 YOLOv8m-P2 五通道模型使用"
            )
        statistics = raw_model.load_coco_pretrained(pretrained_path)
        logger.info(
            "COCO 预训练迁移: %d/%d tensors, 五通道 stem=%s, missing=%d",
            statistics["transferred_tensors"],
            statistics["target_tensors"],
            bool(statistics["adapted_stem"]),
            statistics["missing_tensors"],
        )

    ema = ModelEMA(raw_model, decay=float(config["training"]["ema_decay"]))
    model: torch.nn.Module = raw_model
    if bool(config["training"].get("compile", False)):
        logger.warning("已启用 torch.compile；首次迭代会有编译开销")
        model = torch.compile(raw_model)

    train_dataset = MultimodalDataset(
        config["data"]["root"],
        "train",
        config["training"]["image_size"],
        config["data"],
        augmentation_config=config["augmentation"],
        split_file=config["data"]["train_split"],
        training=True,
    )
    val_dataset = MultimodalDataset(
        config["data"]["root"],
        "train",
        config["training"]["image_size"],
        config["data"],
        split_file=config["data"]["val_split"],
        training=False,
    )
    loss_config = dict(config["loss"])
    if loss_config.get("class_counts") == "auto":
        loss_config["class_counts"] = train_dataset.class_counts(
            int(config["model"]["num_classes"])
        )
        logger.info("训练划分类别实例数: %s", loss_config["class_counts"])
    criterion = DetectionLoss(
        raw_model,
        loss_config,
        label_smoothing=float(config["training"]["label_smoothing"]),
    ).to(device)
    if str(loss_config.get("assigner", "legacy")).lower() == "tal":
        logger.info(
            "类别平衡权重: %s",
            [round(float(value), 4) for value in criterion.class_weights.cpu()],
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler_config = config["training"]["scheduler"]
    scheduler = WarmupCosinePlateauLR(
        optimizer,
        total_epochs=int(config["training"]["epochs"]),
        warmup_epochs=int(config["training"]["warmup_epochs"]),
        min_lr=float(scheduler_config["min_lr"]),
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        mode=str(scheduler_config["mode"]),
    )
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    early_stopping = EarlyStopping(int(config["training"]["early_stopping_patience"]))
    start_epoch, best_metric = 0, -1.0
    if checkpoint is not None:
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        early_stopping.load_state_dict(checkpoint.get("early_stopping", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))
    elif finetune_checkpoint is not None:
        best_metric = float(finetune_checkpoint.get("best_metric", -1.0))
        if best_metric >= 0:
            early_stopping.best = best_metric
        baseline_state = checkpoint_state(
            raw_model,
            ema,
            optimizer,
            scheduler,
            scaler,
            early_stopping,
            -1,
            best_metric,
            config,
        )
        save_checkpoint(baseline_state, output_dir / "best.pt")
        ema_baseline = dict(baseline_state)
        ema_baseline["model"] = ema.ema.state_dict()
        ema_baseline.pop("ema", None)
        save_checkpoint(ema_baseline, output_dir / "ema_best.pt")
        save_checkpoint(ema_baseline, experiment_dir / "best.pt")
        logger.info(
            "微调状态已重置；基线 mAP50-95=%s，原始权重已保存到新输出目录",
            f"{best_metric:.6f}" if best_metric >= 0 else "未知",
        )

    train_loader = create_dataloader(
        train_dataset,
        int(config["training"]["batch_size"]),
        config["data"],
        shuffle=True,
        multi_scale=config["augmentation"]["multi_scale"],
    )
    val_loader = create_dataloader(
        val_dataset,
        int(config["training"]["batch_size"]),
        config["data"],
        shuffle=False,
    )
    logger.info(
        "样本: train=%d, val=%d, model=%s, assigner=%s, output=%s",
        len(train_dataset),
        len(val_dataset),
        mode,
        loss_config.get("assigner", "legacy"),
        output_dir,
    )

    fields = [
        "epoch", "lr", "loss", "box_loss", "cls_loss", "dfl_loss",
        "precision", "recall", "map50", "map50_95", "seconds",
    ]
    output_csv = CSVLogger(output_dir / "results.csv", fields)
    experiment_csv = CSVLogger(experiment_dir / "results.csv", fields)
    epochs = int(config["training"]["epochs"])
    val_interval = int(config["training"]["val_interval"])
    validation = config["validation"]

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        learning_rates = scheduler.step(epoch)
        losses = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            ema,
            device,
            amp_enabled,
            float(config["training"]["gradient_clip"]),
            epoch,
            int(config["training"]["log_interval"]),
            logger,
        )
        metrics: dict[str, Any] = {}
        should_validate = (epoch + 1) % val_interval == 0 or epoch + 1 == epochs
        if should_validate:
            metrics = evaluate(
                ema.ema,
                val_loader,
                device,
                int(config["model"]["num_classes"]),
                float(validation["confidence"]),
                float(validation["nms_iou"]),
                int(validation["max_detections"]),
            )
            current = float(metrics["map50_95"])
            scheduler.step(epoch, current)
            logger.info(
                "val epoch=%d P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f",
                epoch + 1,
                metrics["precision"], metrics["recall"], metrics["map50"], current,
            )
            if current > best_metric:
                best_metric = current
                state = checkpoint_state(
                    raw_model, ema, optimizer, scheduler, scaler, early_stopping,
                    epoch, best_metric, config,
                )
                save_checkpoint(state, output_dir / "best.pt")
                ema_state = dict(state)
                ema_state["model"] = ema.ema.state_dict()
                ema_state.pop("ema", None)
                save_checkpoint(ema_state, output_dir / "ema_best.pt")
                save_checkpoint(ema_state, experiment_dir / "best.pt")
            stop = early_stopping.update(current, val_interval)
        else:
            stop = False

        state = checkpoint_state(
            raw_model, ema, optimizer, scheduler, scaler, early_stopping,
            epoch, best_metric, config,
        )
        save_checkpoint(state, output_dir / "last.pt")
        row = {
            "epoch": epoch + 1,
            "lr": learning_rates[0],
            **losses,
            **metrics,
            "seconds": round(time.perf_counter() - epoch_start, 2),
        }
        output_csv.append(row)
        experiment_csv.append(row)
        if stop:
            logger.info("Early stopping: mAP50-95 连续 %d epoch 未提升", early_stopping.bad_epochs)
            break

    logger.info("训练结束，最佳 mAP50-95=%.6f", best_metric)


if __name__ == "__main__":
    main()
