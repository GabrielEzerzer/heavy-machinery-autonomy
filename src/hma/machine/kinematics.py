"""Bucket-tip kinematics. Machine frame: origin on the ground at the swing axis, x forward, y left,
z up; metres, radians. Pitch: the pin-to-tip line's angle below the upper structure's forward axis.
Inverse kinematics prefers the tip in front of the swing axis and returns None when nothing fits."""

from __future__ import annotations

import math

import numpy as np

SWING_HEIGHT = 1.0
BOOM_PIVOT = np.array([0.14, -0.16443, 0.424])
BOOM_LENGTH = 4.6
ARM_LENGTH = 2.519
BUCKET_LENGTH = 1.23

JOINT_LIMITS = {
    "boom": (-1.2217304764, 0.767944870878),
    "arm": (0.523598775598, 2.65290046303),
    "bucket": (-0.575958653158, 2.49582083035),
}
LIMIT_TOLERANCE = 1e-9


def forward_kinematics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, float)
    swing, boom, arm, bucket = np.moveaxis(q, -1, 0)
    a1, a2, a3 = boom, boom + arm, boom + arm + bucket
    reach = (BOOM_PIVOT[0] + BOOM_LENGTH * np.cos(a1) + ARM_LENGTH * np.cos(a2)
             + BUCKET_LENGTH * np.cos(a3))
    side = BOOM_PIVOT[1]
    height = (SWING_HEIGHT + BOOM_PIVOT[2] - BOOM_LENGTH * np.sin(a1)
              - ARM_LENGTH * np.sin(a2) - BUCKET_LENGTH * np.sin(a3))
    c, s = np.cos(swing), np.sin(swing)
    position = np.stack([c * reach - s * side, s * reach + c * side, height], axis=-1)
    return position, np.arctan2(np.sin(a3), np.cos(a3))


def within_limits(q: np.ndarray) -> bool:
    return all(
        lo - LIMIT_TOLERANCE <= float(q[i]) <= hi + LIMIT_TOLERANCE
        for i, (lo, hi) in enumerate(JOINT_LIMITS.values(), start=1)
    )


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def inverse_kinematics(position: np.ndarray, pitch: float) -> np.ndarray | None:
    x, y, z = (float(v) for v in position)
    side = float(BOOM_PIVOT[1])
    radial_sq = x * x + y * y - side * side
    if radial_sq < 0.0:
        return None
    for radial in (math.sqrt(radial_sq), -math.sqrt(radial_sq)):
        q = _solve_plane(radial, z, pitch)
        if q is not None:
            q[0] = _wrap(math.atan2(y, x) - math.atan2(side, radial))
            return q
    return None


def _solve_plane(radial: float, z: float, pitch: float) -> np.ndarray | None:
    px = radial - BOOM_PIVOT[0] - BUCKET_LENGTH * math.cos(pitch)
    pz = SWING_HEIGHT + BOOM_PIVOT[2] - z - BUCKET_LENGTH * math.sin(pitch)
    cos_arm = (px * px + pz * pz - BOOM_LENGTH**2 - ARM_LENGTH**2) / (2 * BOOM_LENGTH * ARM_LENGTH)
    if abs(cos_arm) > 1.0:
        return None
    arm = math.acos(cos_arm)
    boom = _wrap(math.atan2(pz, px) - math.atan2(ARM_LENGTH * math.sin(arm),
                                                 BOOM_LENGTH + ARM_LENGTH * math.cos(arm)))
    q = np.array([0.0, boom, arm, _wrap(pitch - boom - arm)])
    return q if within_limits(q) else None
