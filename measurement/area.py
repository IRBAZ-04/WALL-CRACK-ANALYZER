"""
area.py
--------
Computes crack pixel area and the percentage of the analyzed wall
region that is affected/damaged.

Damage % = (crack_area_px / wall_area_px) * 100

wall_area_px is the number of pixels in the analyzed region (by
default the whole image, but you can pass a wall-boundary mask if
you also crop out sky/floor/non-wall regions).
"""

import numpy as np


def measure_area(binary_mask, wall_mask=None, cm_per_pixel=None):
    crack_area_px = int(np.count_nonzero(binary_mask))

    if wall_mask is not None:
        wall_area_px = int(np.count_nonzero(wall_mask))
    else:
        wall_area_px = binary_mask.shape[0] * binary_mask.shape[1]

    damage_pct = round((crack_area_px / wall_area_px) * 100, 4) if wall_area_px > 0 else 0.0

    result = {
        "crack_area_px": crack_area_px,
        "wall_area_px": wall_area_px,
        "damage_percentage": damage_pct,
        "crack_area_cm2": None,
    }
    if cm_per_pixel:
        result["crack_area_cm2"] = round(crack_area_px * (cm_per_pixel ** 2), 2)

    return result
