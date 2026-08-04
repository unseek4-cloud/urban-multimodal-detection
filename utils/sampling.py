"""Sampling utilities that do not depend on PyTorch or image libraries."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def repeat_factor_weights(
    image_classes: Sequence[Collection[int]],
    num_classes: int,
    repeat_threshold: float = 0.10,
    max_repeat: float = 3.0,
) -> tuple[list[float], list[int]]:
    """Calculate conservative LVIS-style repeat factors for training images."""
    if not 0.0 < repeat_threshold <= 1.0:
        raise ValueError("repeat_threshold must be in (0, 1]")
    if max_repeat < 1.0:
        raise ValueError("max_repeat must be at least 1")

    normalized: list[set[int]] = []
    image_counts = [0 for _ in range(num_classes)]
    for classes in image_classes:
        unique = set(classes)
        invalid = [class_id for class_id in unique if not 0 <= class_id < num_classes]
        if invalid:
            raise ValueError(f"class ids out of range: {sorted(invalid)}")
        normalized.append(unique)
        for class_id in unique:
            image_counts[class_id] += 1

    image_total = max(len(normalized), 1)
    class_factors = [1.0 for _ in range(num_classes)]
    for class_id, count in enumerate(image_counts):
        if count == 0:
            continue
        frequency = count / image_total
        class_factors[class_id] = min(
            max_repeat,
            max(1.0, (repeat_threshold / frequency) ** 0.5),
        )
    weights = [
        max((class_factors[class_id] for class_id in classes), default=1.0)
        for classes in normalized
    ]
    return weights, image_counts
