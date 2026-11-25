"""Tracks as rows of road wheels that MuJoCo's own collision and solver put on the ground, with a
travel drive per track. step() caps the two drives' power, then runs mujoco.mj_step; it holds no
state of its own."""

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
BRAKE_SOLIMP = (0.9999, 0.9999, 0.001)

FRICTION = 0.565
FRICTION_RANGE = (0.55, 0.58)
FRICTION_REFERENCE = 35.0
CONTACT_SOLIMP = (0.999, 0.999, 0.001, 0.5, 2.0)
IMPRATIO = 1000.0
COUPLING_SOLREF = (0.01, 1.0)
COUPLING_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)

FRONT_LINE, REAR_LINE, SIDE_LINE = TUMBLER, -TUMBLER, GAUGE / 2 + RAIL
OUTLINE = (-TUMBLER - WHEEL_RADIUS, TUMBLER + WHEEL_RADIUS,
           -(GAUGE + SHOE_WIDTH) / 2, (GAUGE + SHOE_WIDTH) / 2)
SHELL_LIFT = LINK_BAND
SHELL_TOP = TRACK_HEIGHT
WHEEL_X = tuple(float(x) for x in np.linspace(REAR_LINE, FRONT_LINE, WHEELS))


def side_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def wheel_name(side: str, k: int) -> str:
    return f"{TRACK}{side}_{k}"


def shell_names(side: str) -> tuple[str, ...]:
    return (f"{TRACK}{side}_frame", f"{TRACK}{side}_rear", f"{TRACK}{side}_front")


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


def travel_caps(belt_speeds: np.ndarray) -> np.ndarray:
    total = np.abs(belt_speeds).sum(axis=-1, keepdims=True)
    cap = np.minimum(TRACTION, TRAVEL_POWER / np.maximum(total, 1e-12))
    return np.broadcast_to(cap, np.shape(belt_speeds))


def write_caps(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    lay = drives(model)
    model.tendon_frictionloss[lay.tendons] = travel_caps(data.qvel[lay.dofs] * WHEEL_RADIUS)


def step(model: mujoco.MjModel, data: mujoco.MjData, nstep: int = 1) -> None:
    for _ in range(nstep):
        write_caps(model, data)
        mujoco.mj_step(model, data)
