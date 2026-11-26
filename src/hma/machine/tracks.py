"""Tracks as rows of road wheels and belt rollers on MuJoCo's own contact, a travel drive per track.
step() caps the positive drive power at the engine's share by a force on each drive wheel while its
belt runs the way its valve turns, split by shares and last-step powers kept in data.userdata."""

from __future__ import annotations

import weakref
from typing import NamedTuple

import mujoco
import numpy as np
from numba import njit

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
BELT_ROLLERS = WHEELS - 1
BELT_MASS = 0.1
SIDES = ("left", "right")
TRACK = "track_"
TRAVEL = "travel_"

TRAVEL_SPEED = 5.5 / 3.6
TRACTION = 102.0e3 / 2
ENGINE_POWER = 65.0e3
DRIVE_EFFICIENCY = 0.7
TRAVEL_POWER = ENGINE_POWER * DRIVE_EFFICIENCY
BELT_ARMATURE = TRACK_CHAINS / 2 * WHEEL_RADIUS**2
COUPLED = WHEELS + BELT_ROLLERS
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
BELT_GAP = 2e-4
FRICTION_TENDON = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)
WHEEL_X = tuple(float(x) for x in np.linspace(REAR_LINE, FRONT_LINE, WHEELS))
BELT_X = tuple((a + b) / 2 for a, b in zip(WHEEL_X[:-1], WHEEL_X[1:], strict=True))


def side_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def wheel_name(side: str, k: int) -> str:
    return f"{TRACK}{side}_{k}"


def belt_name(side: str, k: int) -> str:
    return f"{TRACK}{side}_belt_{k}"


def coupled_names(side: str) -> tuple[str, ...]:
    return (*(wheel_name(side, k) for k in range(WHEELS)),
            *(belt_name(side, k) for k in range(BELT_ROLLERS)))


def shell_names(side: str) -> tuple[str, ...]:
    return f"{TRACK}{side}_frame", f"{TRACK}{side}_rear", f"{TRACK}{side}_front"


def contact_names(side: str) -> tuple[str, ...]:
    return (*coupled_names(side), *shell_names(side))


class Drives(NamedTuple):
    dofs: np.ndarray
    tendons: np.ndarray
    valves: np.ndarray


_drives: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def drives(model: mujoco.MjModel) -> Drives:
    found = _drives.get(model)
    if found is None:
        from hma.machine.model import VALVE, machine_prefix

        dofs, tendons, valves = [], [], []
        i = 0
        while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, machine_prefix(i) + "base") >= 0:
            prefix = machine_prefix(i)
            dofs.append([model.joint(prefix + wheel_name(s, 0)).dofadr[0] for s in SIDES])
            tendons.append([model.tendon(prefix + TRAVEL + s + VALVE).id for s in SIDES])
            valves.append([model.joint(prefix + TRAVEL + s + VALVE).dofadr[0] for s in SIDES])
            i += 1
        found = Drives(np.array(dofs, np.intp), np.array(tendons, np.intp),
                       np.array(valves, np.intp))
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


def used_power(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    n = len(drives(model).dofs)
    return data.userdata[n:3 * n].reshape(n, 2)


@njit(cache=True, error_model="numpy")
def _limit(qvel, userdata, qfrc, dofs, valves, rate, power, traction, radius, full_share):
    n = dofs.shape[0]
    belt, used, share = np.empty(2), np.empty(2), np.empty(2)
    weight, open_use = np.empty(2), np.empty(2)
    full = np.empty(2, np.bool_)
    for i in range(n):
        lead = 0.5 + userdata[i]
        share[0], share[1] = lead, 1.0 - lead
        for s in range(2):
            belt[s] = qvel[dofs[i, s]] * radius
            used[s] = userdata[n + 2 * i + s]
            full[s] = used[s] >= full_share * share[s] * power
        for s in range(2):
            weight[s] = abs(belt[s]) * (1.0 if full[s] or not (full[0] or full[1]) else 0.0)
            open_use[s] = 0.0 if full[s] else used[s]
        rest = power - (open_use[0] + open_use[1])
        total = weight[0] + weight[1]
        for s in range(2):
            part = weight[s] / total if total > 0.0 else 0.5
            target = (open_use[s] + rest * part) / power
            share[s] = share[s] + rate * (target - share[s])
        userdata[i] = share[0] - 0.5
        for s in range(2):
            speed = abs(belt[s])
            cap = share[s] * power / (speed if speed >= 1e-12 or speed != speed else 1e-12)
            excess = 0.0
            if qvel[valves[i, s]] * belt[s] > 0.0:
                excess = (traction if traction <= cap else cap) - traction
            sign = 1.0 if belt[s] > 0.0 else -1.0 if belt[s] < 0.0 else 0.0 * abs(belt[s])
            qfrc[dofs[i, s]] = excess * sign * radius


@njit(cache=True, error_model="numpy")
def _store_used(efc_type, efc_id, efc_force, kind, tendons, qfrc, dofs, belt, userdata, radius):
    n = dofs.shape[0]
    size = 0
    for i in range(n):
        for s in range(2):
            size = max(size, tendons[i, s] + 1)
    forces = np.zeros(size)
    for r in range(efc_type.shape[0]):
        if efc_type[r] == kind and efc_id[r] < size:
            forces[efc_id[r]] += efc_force[r]
    for i in range(n):
        for s in range(2):
            power = (forces[tendons[i, s]] + qfrc[dofs[i, s]] / radius) * belt[i, s]
            userdata[n + 2 * i + s] = power if power >= 0.0 or power != power else 0.0


def limit_power(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    lay = drives(model)
    _limit(data.qvel, data.userdata, data.qfrc_applied, lay.dofs, lay.valves,
           model.opt.timestep / SHARE_TIME, TRAVEL_POWER, TRACTION, WHEEL_RADIUS, FULL_SHARE)


def step(model: mujoco.MjModel, data: mujoco.MjData, nstep: int = 1) -> None:
    lay = drives(model)
    for _ in range(nstep):
        limit_power(model, data)
        belt = data.qvel[lay.dofs] * WHEEL_RADIUS
        mujoco.mj_step(model, data)
        _store_used(data.efc_type, data.efc_id, data.efc_force, FRICTION_TENDON, lay.tendons,
                    data.qfrc_applied, lay.dofs, belt, data.userdata, WHEEL_RADIUS)
