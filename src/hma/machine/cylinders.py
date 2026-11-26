"""Boom, arm and bucket cylinders: pins in the model's link frames, sizes, relief pressures,
and cylinder length and lever arm (dL/dq) against joint angle. Pins are (x, y, z); every joint turns
about its link's y axis, and the bucket cylinder drives the bucket through a four-bar linkage."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from hma.machine import kinematics

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

BOOM_PIVOT = (float(kinematics.BOOM_PIVOT[0]), float(kinematics.BOOM_PIVOT[2]))
ARM_PIVOT = (0.0, kinematics.BOOM_LENGTH)
BUCKET_PIVOT = (0.0, kinematics.ARM_LENGTH)
BOOM_TURN = math.pi / 2

BORE = {"boom": 0.105, "arm": 0.115, "bucket": 0.100}
ROD = {"boom": 0.070, "arm": 0.080, "bucket": 0.070}
COUNT = {"boom": 2, "arm": 1, "bucket": 1}
CYLINDER_JOINTS = ("boom", "arm", "bucket")
MAIN_RELIEF = 34.3e6
PORT_RELIEF = {"boom": (37.3e6, 37.3e6), "arm": (37.3e6, 39.2e6), "bucket": (37.3e6, 39.2e6)}


SEAL_TEST_ROD = 0.040
SEAL_PRESSURE = np.array([6.3e6, 11.0e6, 16.0e6])
SEAL_SPEED = np.array([0.05, 0.15, 0.30, 0.45])
ROD_SEAL = np.array([[740.0, 1085.0, 1195.0], [630.0, 920.0, 990.0],
                     [650.0, 905.0, 950.0], [660.0, 890.0, 930.0]])
PISTON_SEAL = np.array([[120.0, 190.0, 285.0], [145.0, 195.0, 265.0],
                        [200.0, 245.0, 310.0], [210.0, 255.0, 315.0]])
SEAL_TABLES = np.stack([ROD_SEAL, PISTON_SEAL])
SEAL_SHARE = 0.5
SEAL_SCALES = SEAL_SHARE * np.array([[COUNT[j] * ROD[j] / SEAL_TEST_ROD for j in CYLINDER_JOINTS],
                                     [COUNT[j] * BORE[j] / SEAL_TEST_ROD for j in CYLINDER_JOINTS]])
_SPEED_INDEX = np.arange(len(SEAL_SPEED), dtype=float)
_PRESSURE_INDEX = np.arange(len(SEAL_PRESSURE), dtype=float)
_FIRST_STEP = SEAL_PRESSURE[1] - SEAL_PRESSURE[0]
_LAST_STEP = SEAL_PRESSURE[-1] - SEAL_PRESSURE[-2]


@njit(cache=True, error_model="numpy")
def _interp(x, xp):
    if x != x:
        return x
    last = xp.shape[0] - 1
    if x < xp[0]:
        return 0.0
    if x >= xp[last]:
        return float(last)
    j = 0
    while xp[j + 1] <= x:
        j += 1
    if xp[j] == x:
        return float(j)
    slope = 1.0 / (xp[j + 1] - xp[j])
    value = slope * (x - xp[j]) + j
    if value != value:
        value = slope * (x - xp[j + 1]) + (j + 1)
    return value


@njit(cache=True, error_model="numpy")
def _seal(head, rod, speed, speeds, pressures, tables, scales, first_step, last_step, out):
    columns = scales.shape[1]
    for k in range(out.shape[0]):
        i = _interp(abs(speed[k]), speeds)
        i0 = min(max(int(i), 0), speeds.shape[0] - 2)
        a = i - i0
        total = 0.0
        for kind in range(2):
            p = rod[k] if kind == 0 else abs(head[k] - rod[k])
            below, above = p - pressures[0], p - pressures[-1]
            below = below if below <= 0.0 or below != below else 0.0
            above = above if above >= 0.0 or above != above else 0.0
            j = _interp(p, pressures) + below / first_step + above / last_step
            j0 = min(max(int(math.floor(j)), 0), pressures.shape[0] - 2)
            b = j - j0
            table = tables[kind]
            low = table[i0, j0] * (1 - b) + table[i0, j0 + 1] * b
            high = table[i0 + 1, j0] * (1 - b) + table[i0 + 1, j0 + 1] * b
            force = low + a * (high - low)
            force = force if force >= 0.0 or force != force else 0.0
            term = scales[kind, k % columns] * force
            total = term if kind == 0 else total + term
        out[k] = total


def seal_friction(head, rod, speed) -> np.ndarray:
    head, rod, speed = np.broadcast_arrays(np.asarray(head, float), np.asarray(rod, float),
                                           np.asarray(speed, float))
    if head.shape[-1:] != (SEAL_SCALES.shape[1],):
        raise ValueError(f"the last axis must hold the {SEAL_SCALES.shape[1]} cylinders")
    out = np.empty(head.shape)
    _seal(head.ravel(), rod.ravel(), speed.ravel(), SEAL_SPEED, SEAL_PRESSURE, SEAL_TABLES,
          SEAL_SCALES, _FIRST_STEP, _LAST_STEP, out.reshape(-1))
    return out


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
