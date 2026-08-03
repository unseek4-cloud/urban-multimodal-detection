#!/usr/bin/env python3
"""检查点原子保存与训练输出目录锁的轻量回归测试。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils.common import (
    TrainingDirectoryLock,
    TrainingLockError,
    save_checkpoint,
)


def test_checkpoint_save(root: Path) -> None:
    target = root / "checkpoints" / "last.pt"
    save_checkpoint({"value": torch.tensor([1, 2, 3])}, target)
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["value"].tolist() == [1, 2, 3]
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))

    def fail_after_partial_write(state: object, path: str | Path) -> None:
        Path(path).write_bytes(b"partial checkpoint")
        raise RuntimeError("expected failure")

    with patch("utils.common.torch.save", side_effect=fail_after_partial_write):
        try:
            save_checkpoint({"value": 2}, target)
        except RuntimeError as error:
            assert str(error) == "expected failure"
        else:
            raise AssertionError("模拟保存失败时应抛出异常")
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_training_lock(root: Path) -> None:
    output_dir = root / "run"
    first = TrainingDirectoryLock(output_dir)
    first.acquire()
    assert (output_dir / ".training.lock").is_file()

    second = TrainingDirectoryLock(output_dir)
    try:
        second.acquire()
    except TrainingLockError as error:
        assert "请勿重复启动" in str(error)
    else:
        raise AssertionError("同一输出目录的第二个训练锁应被拒绝")

    other = TrainingDirectoryLock(root / "other-run")
    other.acquire()
    other.release()
    first.release()
    assert not (output_dir / ".training.lock").exists()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".training.lock").write_text(
        json.dumps({"pid": 999_999_999, "token": "stale"}),
        encoding="utf-8",
    )
    replacement = TrainingDirectoryLock(output_dir)
    replacement.acquire()
    replacement.release()
    assert not (output_dir / ".training.lock").exists()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="training-safety-") as directory:
        root = Path(directory)
        test_checkpoint_save(root)
        test_training_lock(root)
    print("training safety test: OK")


if __name__ == "__main__":
    main()
