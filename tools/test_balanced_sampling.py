#!/usr/bin/env python3
"""Regression tests for conservative repeat-factor sampling."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.sampling import repeat_factor_weights


def main() -> None:
    image_classes = [{0}, {0}, {1}, set()]
    weights, counts = repeat_factor_weights(
        image_classes, num_classes=2, repeat_threshold=0.50, max_repeat=3.0
    )

    assert counts == [2, 1], counts
    assert weights[0] == weights[1] == 1.0, weights
    assert abs(weights[2] - 2 ** 0.5) < 1e-9, weights
    assert weights[3] == 1.0, weights

    capped, _ = repeat_factor_weights(
        image_classes, num_classes=2, repeat_threshold=1.0, max_repeat=1.5
    )
    assert capped[2] == 1.5, capped

    print("balanced sampling test: OK")


if __name__ == "__main__":
    main()
