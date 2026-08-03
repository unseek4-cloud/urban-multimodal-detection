"""配置、随机种子、EMA、日志和学习率调度等公共逻辑。"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是 YAML 映射: {path}")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 未检测到可用 GPU")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        name = torch.cuda.get_device_name(device)
        memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"使用 GPU: {name}, 显存 {memory:.1f} GiB")
    else:
        print("使用 CPU；完整训练建议使用 CUDA GPU")
    return device


def setup_logger(log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("urban_multimodal_detection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class ModelEMA:
    """指数滑动平均；验证和推理默认使用该权重。"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, updates: int = 0) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.updates = updates
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def _decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / 2000.0))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = self._decay()
        source_model = model.module if hasattr(model, "module") else model
        source_model = source_model._orig_mod if hasattr(source_model, "_orig_mod") else source_model
        model_state = source_model.state_dict()
        for key, value in self.ema.state_dict().items():
            source = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.ema.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema.load_state_dict(state["model"])
        self.updates = int(state.get("updates", 0))
        self.decay = float(state.get("decay", self.decay))


class WarmupCosinePlateauLR:
    """5 轮预热 + 余弦学习率，并叠加基于 mAP 的 Plateau 衰减。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        min_lr: float,
        factor: float = 0.5,
        patience: int = 10,
        mode: str = "max",
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.epoch = -1
        self.best: float | None = None
        self.bad_checks = 0
        self.plateau_scale = 1.0

    def _improved(self, metric: float) -> bool:
        if self.best is None:
            return True
        return metric > self.best if self.mode == "max" else metric < self.best

    def step(self, epoch: int, metric: float | None = None) -> list[float]:
        self.epoch = epoch
        if metric is not None:
            if self._improved(metric):
                self.best = metric
                self.bad_checks = 0
            else:
                self.bad_checks += 1
                if self.bad_checks >= self.patience:
                    self.plateau_scale *= self.factor
                    self.bad_checks = 0

        progress = max(0.0, epoch + 1 - self.warmup_epochs)
        span = max(1.0, self.total_epochs - self.warmup_epochs)
        if epoch < self.warmup_epochs:
            cosine_scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            cosine_scale = 0.5 * (1.0 + math.cos(math.pi * min(progress / span, 1.0)))

        lrs: list[float] = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            lr = max(self.min_lr, base_lr * cosine_scale * self.plateau_scale)
            group["lr"] = lr
            lrs.append(lr)
        return lrs

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "best": self.best,
            "bad_checks": self.bad_checks,
            "plateau_scale": self.plateau_scale,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", -1))
        self.best = state.get("best")
        self.bad_checks = int(state.get("bad_checks", 0))
        self.plateau_scale = float(state.get("plateau_scale", 1.0))
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))


class EarlyStopping:
    def __init__(self, patience_epochs: int) -> None:
        self.patience_epochs = patience_epochs
        self.best = -math.inf
        self.bad_epochs = 0

    def update(self, metric: float, elapsed_epochs: int) -> bool:
        if metric > self.best:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += elapsed_epochs
        return self.bad_epochs >= self.patience_epochs

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best = float(state.get("best", -math.inf))
        self.bad_epochs = int(state.get("bad_epochs", 0))


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fieldnames).writeheader()

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
            writer.writerow({name: row.get(name, "") for name in self.fieldnames})


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def checkpoint_model_state(checkpoint: dict[str, Any], prefer_ema: bool = True) -> dict[str, torch.Tensor]:
    if prefer_ema and "ema" in checkpoint:
        ema_state = checkpoint["ema"]
        return ema_state["model"] if isinstance(ema_state, dict) and "model" in ema_state else ema_state
    if "model" not in checkpoint:
        raise KeyError("检查点缺少 model 权重")
    return checkpoint["model"]


def dump_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
