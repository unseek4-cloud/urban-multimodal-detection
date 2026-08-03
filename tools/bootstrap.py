#!/usr/bin/env python3
"""检查训练依赖；默认自动安装缺失项。"""

from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = {
    # module: (distribution, expected version, display name)
    "torch": ("torch", "2.1.2", "PyTorch"),
    "torchvision": ("torchvision", "0.16.2", "torchvision"),
    "numpy": ("numpy", "1.26.4", "NumPy"),
    "cv2": ("opencv-python-headless", "4.9.0.80", "OpenCV headless"),
    "PIL": ("Pillow", "10.3.0", "Pillow"),
    "yaml": ("PyYAML", "6.0.1", "PyYAML"),
    "tqdm": ("tqdm", "4.70.0", "tqdm"),
    "pandas": ("pandas", "2.3.3", "pandas"),
    "matplotlib": ("matplotlib", "3.9.0", "Matplotlib"),
    "ultralytics": ("ultralytics", "8.3.40", "Ultralytics YOLO"),
}


def dependency_problems() -> list[str]:
    problems: list[str] = []
    for module, (distribution, expected, display) in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module) is None:
            problems.append(f"{display} 未安装")
            continue
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            problems.append(f"{display} 不是要求的 {distribution} 发行包")
            continue
        # CUDA wheel 会带 +cu118，本体版本仍应严格匹配 2.1.2/0.16.2。
        if installed.split("+", 1)[0] != expected:
            problems.append(f"{display}={installed}，要求 {expected}")
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.version.cuda != "11.8":
            problems.append(f"PyTorch CUDA={torch.version.cuda}，要求 11.8")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查并安装项目依赖")
    parser.add_argument("--check-only", action="store_true", help="只检查，不自动安装")
    parser.add_argument(
        "--repair-torch-stack",
        action="store_true",
        help="移除冲突的 PyTorch/CUDA pip 包并恢复 torch 2.1.2 + CUDA 11.8",
    )
    return parser.parse_args()


def torch_stack_needs_repair() -> bool:
    try:
        torch_version = metadata.version("torch").split("+", 1)[0]
        vision_version = metadata.version("torchvision").split("+", 1)[0]
    except metadata.PackageNotFoundError:
        return False
    if torch_version != "2.1.2" or vision_version != "0.16.2":
        return True
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.version.cuda != "11.8"


def remove_conflicting_torch_stack() -> None:
    """清理 pip 安装的 CUDA 13 栈，防止 30 GB 系统盘残留两套运行库。"""
    cuda_prefixes = (
        "nvidia-cublas",
        "nvidia-cuda-",
        "nvidia-cudnn",
        "nvidia-cufft",
        "nvidia-cufile",
        "nvidia-curand",
        "nvidia-cusolver",
        "nvidia-cusparse",
        "nvidia-nccl",
        "nvidia-nvjitlink",
        "nvidia-nvshmem",
        "nvidia-nvtx",
    )
    installed_names = {
        (distribution.metadata.get("Name") or "").strip()
        for distribution in metadata.distributions()
    }
    removable = sorted(
        name
        for name in installed_names
        if name
        and (
            name.lower()
            in {
                "torch",
                "torchvision",
                "torchaudio",
                "triton",
                "cuda-bindings",
                "cuda-pathfinder",
                "cuda-toolkit",
            }
            or name.lower().startswith(cuda_prefixes)
        )
    )
    if not removable:
        return
    print("移除冲突的 PyTorch/CUDA pip 包:\n- " + "\n- ".join(removable))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", *removable]
    )


def main() -> None:
    args = parse_args()
    problems = dependency_problems()
    if not problems:
        import torch

        print(
            f"依赖版本完整；PyTorch={torch.__version__}, "
            f"CUDA runtime={torch.version.cuda}, GPU available={torch.cuda.is_available()}"
        )
        return
    print("依赖问题:\n- " + "\n- ".join(problems))
    if args.check_only:
        raise SystemExit(1)
    stack_mismatch = torch_stack_needs_repair()
    if stack_mismatch and not args.repair_torch_stack:
        raise SystemExit(
            "检测到冲突的 PyTorch/CUDA 栈。请显式运行: "
            "python tools/bootstrap.py --repair-torch-stack"
        )
    if stack_mismatch:
        remove_conflicting_torch_stack()
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    print(f"自动安装: {requirements}")
    # 云端系统盘仅 30 GB，不保留 pip wheel 缓存。
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)]
    )
    # 用新进程复核，避免当前进程仍缓存刚卸载的 torch 模块。
    subprocess.check_call([sys.executable, str(Path(__file__).resolve()), "--check-only"])
    print("依赖安装与版本复核完成")


if __name__ == "__main__":
    main()
