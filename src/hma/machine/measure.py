"""Measurements of the machine on rigid ground: hand statics against simulated tipping, and
physics throughput. Usage: python -m hma.machine.measure [tipping|throughput]"""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import numpy as np

from hma.machine.ground import Ground, apply_ground, settle
from hma.machine.model import (
    HOME_KEY,
    JOINT_NAMES,
    TRAVEL_NAMES,
    VALVE,
    load_scene,
    machine_prefix,
)
from hma.machine.statics import Machine, reach_load_point
from hma.machine.tracks import FRICTION, FRONT_LINE, REAR_LINE, SIDE_LINE, step

P = machine_prefix(0)
HOLD = 1e9
TIPPED = 0.5


def slope(along=0.0, across=0.0, friction=FRICTION):
    return Ground(grade=(math.tan(along), math.tan(across)), friction=friction)


def released(ground=None, pose=None, hold=False, heading=0.0):
    model = load_scene(1)
    apply_ground(model, 0, ground or Ground())
    if hold:
        for name in JOINT_NAMES:
            model.tendon_frictionloss[model.tendon(P + name + VALVE).id] = HOLD
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    if pose is not None:
        for name, value in zip(JOINT_NAMES, pose, strict=True):
            data.qpos[model.joint(P + name).qposadr[0]] = value
    if not settle(model, data, 0, heading=heading):
        raise RuntimeError("the machine did not come to rest")
    mujoco.mj_forward(model, data)
    return model, data


def base_frame(model, data):
    body = model.body(P + "base").id
    return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()


def machine_com(model, data):
    bodies = np.flatnonzero(model.body_rootid == model.body(P + "base").id)
    mass = model.body_mass[bodies]
    return float(mass.sum()), mass @ data.xipos[bodies] / mass.sum()


def pin(model, data):
    return data.xanchor[model.joint(P + "bucket").id].copy()


def bucket_tip(model, data):
    return data.site(P + "bucket_tip").xpos.copy()


def run(model, data, seconds, force=None, at=None, body="arm"):
    target = model.body(P + body).id
    for _ in range(round(seconds / model.opt.timestep)):
        data.qfrc_applied[:] = 0.0
        if force is not None:
            point = at(model, data)
            mujoco.mj_applyFT(model, data, force, np.zeros(3), point, target, data.qfrc_applied)
        step(model, data)
    data.qfrc_applied[:] = 0.0


def turned(model, data, start):
    _, rot = base_frame(model, data)
    return math.degrees(math.acos(np.clip(np.trace(start.T @ rot) / 2 - 0.5, -1.0, 1.0)))


def chart_pose(height, radius, swing=0.0):
    boom, arm = reach_load_point(Machine(), radius, height)
    return (swing, boom, arm, 0.0)


def hand_tipping_load(model, data, line_point, axis):
    mass, com = machine_com(model, data)
    load = pin(model, data)
    normal = np.cross(axis, [0.0, 0.0, 1.0])
    return mass * ((line_point - com) @ normal) / ((load - line_point) @ normal)


def tips_under(model, data, load_kg, seconds=3.0):
    _, start = base_frame(model, data)
    run(model, data, seconds, np.array([0.0, 0.0, -9.81 * load_kg]), pin)
    return turned(model, data, start)


CASES = {
    "front, flat, 6.0 m": (0.0, 6.0, 0.0, 0.0, 0.0),
    "rear, flat, 4.5 m": (0.0, 4.5, math.pi, 0.0, 0.0),
    "side, flat, 4.5 m": (0.0, 4.5, math.pi / 2, 0.0, 0.0),
    "front, 10 deg nose down, 6.0 m": (0.0, 6.0, 0.0, -math.radians(10.0), 0.0),
    "side, 10 deg cross slope, 4.5 m": (0.0, 4.5, math.pi / 2, 0.0, -math.radians(10.0)),
}


def tipping_case(name):
    height, radius, swing, along, across = CASES[name]
    pose = chart_pose(height, radius, swing)
    ground = slope(along, across)
    model, data = released(ground, pose=pose, hold=True)
    origin, rot = base_frame(model, data)
    if swing in (0.0, math.pi):
        line = origin + rot @ [FRONT_LINE if swing == 0.0 else REAR_LINE, 0.0, 0.0]
        axis = rot[:, 1]
    else:
        line, axis = origin + rot @ [0.0, SIDE_LINE, 0.0], rot[:, 0]
    return ground, pose, hand_tipping_load(model, data, line, axis)


def sim_tipping_load(name: str, tolerance: float = 0.005) -> tuple[float, float]:
    ground, pose, hand = tipping_case(name)
    low, high = 0.8 * hand, 1.2 * hand
    while high - low > tolerance * hand:
        middle = (low + high) / 2
        model, data = released(ground, pose=pose, hold=True)
        if tips_under(model, data, middle) > TIPPED:
            high = middle
        else:
            low = middle
    return hand, (low + high) / 2


def throughput(n: int, weld: bool, travel: bool, seconds: float = 20.0) -> float:
    from hma.hydraulics import Hydraulics, draw_machine

    model = load_scene(n, weld=weld)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [draw_machine(100 + i) for i in range(n)])
    rng = np.random.default_rng(0)
    tracks = np.array([[model.actuator(machine_prefix(i) + t).id for t in TRAVEL_NAMES]
                       for i in range(n)])
    limit = model.actuator_ctrlrange[tracks, 1]
    decisions = round(seconds * layer.control_hz)
    began = time.perf_counter()
    for _ in range(decisions):
        layer.act(rng.uniform(-1.0, 1.0, (n, len(JOINT_NAMES))))
        if travel:
            data.ctrl[tracks] = rng.uniform(-1.0, 1.0, tracks.shape) * limit
        for _ in range(layer.steps_per_decision):
            layer.step()
    return decisions * layer.steps_per_decision / (time.perf_counter() - began)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("tipping", "throughput"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.what == "tipping":
        print("load at the bucket pin that tips the machine, kg (hand statics / simulation)")
        for name in CASES:
            hand, sim = sim_tipping_load(name)
            print(f"  {name:34s} {hand:8.0f} {sim:8.0f} {100 * (sim / hand - 1):+6.1f}%")
        return
    print("physics steps per second through the hydraulics layer, random commands at 10 Hz")
    for n in (1, 8):
        for weld in (True, False):
            for travel in ((False,) if weld else (False, True)):
                rates = [throughput(n, weld, travel) for _ in range(args.repeats)]
                label = f"{n} machine{'s' if n > 1 else ''}, " + (
                    "welded" if weld else "released, " + ("travelling" if travel else "standing"))
                print(f"  {label:36s} {min(rates):8.0f} to {max(rates):8.0f}")


if __name__ == "__main__":
    main()
