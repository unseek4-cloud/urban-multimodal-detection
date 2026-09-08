#!/usr/bin/env python3
"""对官方 test 三模态数据推理，生成逐图 TXT 与 submission.zip。"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dataset import MultimodalDataset, create_dataloader
from model import build_model, input_modalities
from utils.common import checkpoint_model_state, load_checkpoint, load_config, select_device
from utils.nms import class_aware_nms, scale_boxes_to_original, xyxy_to_xywh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成赛题提交文件")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="outputs/ema_best.pt")
    parser.add_argument("--data", default=None, help="test 目录或包含 test/ 的数据根目录")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def resolve_data_root(value: str | None, configured_root: str) -> Path:
    root = Path(value or configured_root)
    if root.name == "test" and (root / "visible").is_dir():
        return root.parent
    if (root / "test" / "visible").is_dir():
        return root
    raise FileNotFoundError(f"找不到 test/visible: {root}")


def flipped_predictions(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    flipped_inputs = {name: torch.flip(value, dims=[-1]) for name, value in inputs.items()}
    predictions = model(flipped_inputs)
    width = inputs["rgb"].shape[-1]
    old_x1 = predictions[..., 0].clone()
    predictions[..., 0] = width - predictions[..., 2]
    predictions[..., 2] = width - old_x1
    return predictions


@torch.inference_mode()
def run_prediction(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output: Path,
    confidence: float,
    nms_iou: float,
    max_detections: int,
    use_tta: bool,
) -> Path:
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    filtered_invalid_boxes = 0
    for batch in tqdm(loader, desc="预测", dynamic_ncols=True):
        inputs = {
            name: batch[name].to(device, non_blocking=True)
            for name in ("rgb", "infrared", "depth")
            if name in batch
        }
        predictions = model(inputs)
        if use_tta:
            predictions = torch.cat((predictions, flipped_predictions(model, inputs)), dim=1)
        detections = class_aware_nms(predictions, confidence, nms_iou, max_detections)
        for index, (stem, detection) in enumerate(zip(batch["stems"], detections)):
            original_height, original_width = batch["original_shapes"][index]
            path = output / f"{stem}.txt"
            lines: list[str] = []
            if detection.shape[0]:
                boxes = scale_boxes_to_original(
                    detection[:, :4],
                    (original_height, original_width),
                    batch["ratio_pads"][index],
                )
                valid = (
                    torch.isfinite(boxes).all(dim=1)
                    & torch.isfinite(detection[:, 4:]).all(dim=1)
                    & (boxes[:, 2] > boxes[:, 0])
                    & (boxes[:, 3] > boxes[:, 1])
                )
                filtered_invalid_boxes += int((~valid).sum().item())
                boxes = boxes[valid]
                valid_detection = detection[valid]
                normalized = xyxy_to_xywh(boxes)
                normalized /= normalized.new_tensor(
                    [original_width, original_height, original_width, original_height]
                )
                normalized = normalized.clamp(0, 1)
                serializable = (normalized[:, 2] >= 1e-8) & (normalized[:, 3] >= 1e-8)
                filtered_invalid_boxes += int((~serializable).sum().item())
                normalized = normalized[serializable]
                valid_detection = valid_detection[serializable]
                for box, score, class_id in zip(
                    normalized, valid_detection[:, 4], valid_detection[:, 5]
                ):
                    cx, cy, width, height = box.tolist()
                    lines.append(
                        f"{int(class_id.item())} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f} {float(score):.8f}"
                    )
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written.append(path)

    archive = output.parent / f"{output.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(written):
            zip_file.write(path, arcname=path.name)
    if len(written) != len(loader.dataset):
        raise RuntimeError(f"提交文件数错误: {len(written)} != {len(loader.dataset)}")
    if filtered_invalid_boxes:
        print(f"已过滤裁剪后宽高非正或非有限的检测框: {filtered_invalid_boxes}")
    return archive


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(args.data, config["data"]["root"])
    device = select_device(args.device or config.get("device", "auto"))
    checkpoint = load_checkpoint(args.weights, device)
    model_config = checkpoint.get("model_config", config["model"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint_model_state(checkpoint, prefer_ema=not args.no_ema))
    dataset = MultimodalDataset(
        data_root,
        "test",
        int(config["training"]["image_size"]),
        config["data"],
        training=False,
        modalities=input_modalities(model_config),
    )
    loader = create_dataloader(
        dataset,
        int(config["training"]["batch_size"]),
        config["data"],
        shuffle=False,
    )
    prediction = config["prediction"]
    use_tta = bool(prediction["tta"] if args.tta is None else args.tta)
    archive = run_prediction(
        model,
        loader,
        device,
        Path(args.output or prediction["output"]),
        float(prediction["confidence"]),
        float(prediction["nms_iou"]),
        int(prediction["max_detections"]),
        use_tta,
    )
    print(f"提交文件已生成: {archive}")


if __name__ == "__main__":
    main()
