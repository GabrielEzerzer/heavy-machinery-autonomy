"""Tracks as rows of road wheels that MuJoCo's own collision and solver put on the ground, with a
travel drive per track. step() keeps the positive drive power of both tracks within the engine's
share: a force on each drive wheel in data.qfrc_applied, split by shares kept in data.userdata
with the power each track used in the last step, so a saved integration state replays exactly."""

from __future__ import annotations

import weakref
from typing import NamedTuple

import mujoco
import numpy as np

from hma.machine.mass import (
    GAUGE,
    IDLER,
    LINK_BAND,
    ROLLER,
    ROLLERS,
    SHOE_WIDTH,
    TRACK_CHAINS,
    TRACK_HEIGHT,
    TRAVEL_MOTOR,
    TUMBLER,
)

RAIL = 0.07
WHEEL_RADIUS = 0.35
WHEELS = ROLLERS + 2
WHEEL_MASSES = (TRAVEL_MOTOR, *[ROLLER] * ROLLERS, IDLER)
SIDES = ("left", "right")
TRACK = "track_"
TRAVEL = "travel_"

TRAVEL_SPEED = 5.5 / 3.6
TRACTION = 102.0e3 / 2
ENGINE_POWER = 65.0e3
DRIVE_EFFICIENCY = 0.7
TRAVEL_POWER = ENGINE_POWER * DRIVE_EFFICIENCY
BELT_ARMATURE = TRACK_CHAINS / 2 * WHEEL_RADIUS**2
WHEEL_INERTIA = 1e-3
BRAKE_SOLIMP = (0.9999, 0.9999, 0.001)
SHARE_TIME = 0.1
FULL_SHARE = 0.99

FRICTION = 0.565
FRICTION_RANGE = (0.55, 0.58)
FRICTION_REFERENCE = 40.0
CONTACT_SOLIMP = (0.999, 0.999, 0.001, 0.5, 2.0)
COUPLING_SOLREF = (0.01, 1.0)
COUPLING_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)

FRONT_LINE, REAR_LINE, SIDE_LINE = TUMBLER, -TUMBLER, GAUGE / 2 + RAIL
OUTLINE = (-TUMBLER - WHEEL_RADIUS, TUMBLER + WHEEL_RADIUS,
           -(GAUGE + SHOE_WIDTH) / 2, (GAUGE + SHOE_WIDTH) / 2)
SHELL_LIFT = LINK_BAND
SHELL_TOP = TRACK_HEIGHT
SUPPORT_GAP = 2e-4
SUPPORT_HALF_HEIGHT = 0.01
FRICTION_TENDON = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)
WHEEL_X = tuple(float(x) for x in np.linspace(REAR_LINE, FRONT_LINE, WHEELS))


def side_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def wheel_name(side: str, k: int) -> str:
    return f"{TRACK}{side}_{k}"


def shell_names(side: str) -> tuple[str, ...]:
    return (f"{TRACK}{side}_frame", f"{TRACK}{side}_rear", f"{TRACK}{side}_front",
            f"{TRACK}{side}_support")


def contact_names(side: str) -> tuple[str, ...]:
    return (*(wheel_name(side, k) for k in range(WHEELS)), *shell_names(side))


class Drives(NamedTuple):
    dofs: np.ndarray
    tendons: np.ndarray


_drives: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def drives(model: mujoco.MjModel) -> Drives:
    found = _drives.get(model)
    if found is None:
        from hma.machine.model import VALVE, machine_prefix

        dofs, tendons = [], []
        i = 0
        while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, machine_prefix(i) + "base") >= 0:
            prefix = machine_prefix(i)
            dofs.append([model.joint(prefix + wheel_name(s, 0)).dofadr[0] for s in SIDES])
            tendons.append([model.tendon(prefix + TRAVEL + s + VALVE).id for s in SIDES])
            i += 1
        found = Drives(np.array(dofs, np.intp), np.array(tendons, np.intp))
        _drives[model] = found
    return found


def tendon_friction(data: mujoco.MjData, tendons: np.ndarray) -> np.ndarray:
    tendons = np.asarray(tendons, np.intp)
    rows = data.efc_type == FRICTION_TENDON
    forces = np.bincount(data.efc_id[rows], weights=data.efc_force[rows],
                         minlength=int(tendons.max(initial=-1)) + 1)
    return forces[tendons]


def drive_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    lay = drives(model)
    return (tendon_friction(data, lay.tendons)
            + data.qfrc_applied[lay.dofs] / WHEEL_RADIUS)


def shares(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    lead = 0.5 + data.userdata[:len(drives(model).dofs), None]
    return np.concatenate([lead, 1.0 - lead], axis=1)


def split_power(belt_speeds: np.ndarray, used: np.ndarray, share: np.ndarray,
                rate: float) -> np.ndarray:
    full = used >= FULL_SHARE * share * TRAVEL_POWER
    weight = np.abs(belt_speeds) * np.where(full.any(axis=-1, keepdims=True), full, True)
    open_use = np.where(full, 0.0, used)
    rest = TRAVEL_POWER - open_use.sum(axis=-1, keepdims=True)
    total = weight.sum(axis=-1, keepdims=True)
    even = np.full_like(weight, 1.0 / weight.shape[-1])
    part = np.divide(weight, total, out=even, where=total > 0.0)
    target = (open_use + rest * part) / TRAVEL_POWER
    return share + rate * (target - share)


def drive_caps(belt_speeds: np.ndarray, share: np.ndarray) -> np.ndarray:
    return np.minimum(TRACTION, share * TRAVEL_POWER / np.maximum(np.abs(belt_speeds), 1e-12))


def used_power(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    n = len(drives(model).dofs)
    return data.userdata[n:3 * n].reshape(n, 2)


def limit_power(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    lay = drives(model)
    belt = data.qvel[lay.dofs] * WHEEL_RADIUS
    used = used_power(model, data)
    share = split_power(belt, used, shares(model, data), model.opt.timestep / SHARE_TIME)
    data.userdata[:len(lay.dofs)] = share[:, 0] - 0.5
    excess = drive_caps(belt, share) - TRACTION
    data.qfrc_applied[lay.dofs] = excess * np.sign(belt) * WHEEL_RADIUS


def step(model: mujoco.MjModel, data: mujoco.MjData, nstep: int = 1) -> None:
    for _ in range(nstep):
        limit_power(model, data)
        lay = drives(model)
        belt = data.qvel[lay.dofs] * WHEEL_RADIUS
        mujoco.mj_step(model, data)
        used_power(model, data)[:] = np.maximum(drive_forces(model, data) * belt, 0.0)
