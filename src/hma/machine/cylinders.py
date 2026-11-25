"""Boom, arm and bucket cylinders: pins in the model's link frames, sizes, relief pressures,
and cylinder length and lever arm (dL/dq) against joint angle. Pins are (x, y, z); every joint turns
about its link's y axis, and the bucket cylinder drives the bucket through a four-bar linkage."""

from __future__ import annotations

import math

import numpy as np

PINS = {
    "boom_foot": ("upper", (0.5561, -0.1644, 0.033)),
    "boom_rod": ("boom", (-0.999, 0.0, 1.7051)),
    "arm_head": ("boom", (-1.0725, 0.0, 2.6515)),
    "arm_rod": ("arm", (-0.2796, 0.0, -0.604)),
    "bucket_head": ("arm", (-0.5497, 0.0, 0.4449)),
    "side_link": ("arm", (-0.0306, 0.0, 2.1757)),
    "link_pin": ("bucket", (-0.348, 0.0, -0.1064)),
}
SIDE_LINK = 0.4922
H_LINK = 0.4656
CYLINDER_OFFSETS = {"boom": (-0.25, 0.25), "arm": (0.0,), "bucket": (0.0,)}
CYLINDER_PINS = {
    "boom": ("boom_foot", "boom_rod"),
    "arm": ("arm_head", "arm_rod"),
    "bucket": ("bucket_head", "coupler"),
}

BOOM_PIVOT = (0.14, 0.424)
ARM_PIVOT = (0.0, 4.6)
BUCKET_PIVOT = (0.0, 2.519)
BOOM_TURN = math.pi / 2

BORE = {"boom": 0.105, "arm": 0.115, "bucket": 0.100}
ROD = {"boom": 0.070, "arm": 0.080, "bucket": 0.070}
COUNT = {"boom": 2, "arm": 1, "bucket": 1}
CYLINDER_JOINTS = ("boom", "arm", "bucket")
MAIN_RELIEF = 34.3e6
PORT_RELIEF = {"boom": (37.3e6, 37.3e6), "arm": (37.3e6, 39.2e6), "bucket": (37.3e6, 39.2e6)}


def areas(joint: str) -> tuple[float, float]:
    bore = COUNT[joint] * math.pi / 4 * BORE[joint] ** 2
    return bore, bore - COUNT[joint] * math.pi / 4 * ROD[joint] ** 2


def xz(name: str) -> np.ndarray:
    x, _, z = PINS[name][1]
    return np.array([x, z])


def _turn(angle: np.ndarray, point: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * point[0] + s * point[1], -s * point[0] + c * point[1]])


def pin_on_child(pivot, angle, point) -> tuple[np.ndarray, np.ndarray]:
    arm = _turn(angle, point)
    pivot = np.asarray(pivot, float).reshape(2, *([1] * np.ndim(angle)))
    return pivot + arm, np.stack([arm[1], -arm[0]])


def coupler(q_bucket) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_bucket, float)
    a, da = pin_on_child(BUCKET_PIVOT, q, xz("link_pin"))
    b = xz("side_link").reshape(2, *([1] * q.ndim))
    ab = a - b
    d = np.hypot(ab[0], ab[1])
    along = (SIDE_LINK**2 - H_LINK**2 + d * d) / (2 * d)
    height = np.sqrt(SIDE_LINK**2 - along**2)
    unit = ab / d
    normal = np.stack([-unit[1], unit[0]])
    c = b + along * unit + height * normal
    cb, ca = c - b, c - a
    det = cb[0] * ca[1] - cb[1] * ca[0]
    rhs = ca[0] * da[0] + ca[1] * da[1]
    dc = np.stack([-cb[1] * rhs, cb[0] * rhs]) / det
    return c, dc


def cylinder_length(joint: str, q) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, float)
    if joint == "boom":
        x, dx = pin_on_child(BOOM_PIVOT, BOOM_TURN + q, xz("boom_rod"))
        fixed = xz("boom_foot")
    elif joint == "arm":
        x, dx = pin_on_child(ARM_PIVOT, q, xz("arm_rod"))
        fixed = xz("arm_head")
    elif joint == "bucket":
        x, dx = coupler(q)
        fixed = xz("bucket_head")
    else:
        raise ValueError(f"no cylinder drives {joint!r}; choose from {CYLINDER_JOINTS}")
    rel = x - fixed.reshape(2, *([1] * q.ndim))
    length = np.hypot(rel[0], rel[1])
    return length, (rel[0] * dx[0] + rel[1] * dx[1]) / length


def joint_torque_limits(joint: str, q, pressure_push, pressure_pull) -> tuple[np.ndarray, ...]:
    _, lever = cylinder_length(joint, q)
    bore, annulus = areas(joint)
    push = np.asarray(pressure_push, float) * bore * np.abs(lever)
    pull = np.asarray(pressure_pull, float) * annulus * np.abs(lever)
    extend_positive = lever > 0
    return np.where(extend_positive, push, pull), np.where(extend_positive, pull, push)
