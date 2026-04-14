from __future__ import annotations

import numpy as np


def intersection_over_union(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def box_mask_iou(box_xyxy: list[float] | tuple[float, float, float, float] | np.ndarray, mask: np.ndarray) -> float:
    mask_array = np.asarray(mask).astype(bool)
    if mask_array.size == 0:
        return 0.0

    height, width = mask_array.shape[:2]
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=float).tolist()
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), 0, width - 1))
    y2 = int(np.clip(np.ceil(y2), 0, height - 1))

    if x2 < x1 or y2 < y1:
        return 0.0

    box_mask = np.zeros((height, width), dtype=bool)
    box_mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return intersection_over_union(box_mask, mask_array)
