"""Physics steps per second through hma.sim.step while every machine cuts: one welded machine,
and 8 machines in one scene, with the time split between physics, hydraulics and soil.
Usage: python -m hma.sim.bench"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from hma.hydraulics import NOMINAL, Hydraulics, draw_machine
from hma.machine import load_scene
from hma.machine.kinematics import inverse_kinematics
from hma.sim import reset, step
from hma.sim.dig import PITCH_DIG
from hma.soil import NOMINAL as NOMINAL_SOIL
from hma.soil import Soil, sample_soil

EPISODE = 30
EPISODES = 6
REPEATS = 3
DRAG = np.array([0.0, 0.1, 1.0, 0.0])
START = (np.array([6.6, 0.0, -0.15]), PITCH_DIG)


def _timed(clock: dict, key: str, call):
    def run(*args):
        start = time.perf_counter()
        call(*args)
        clock[key] += time.perf_counter() - start
    return run


def measure(machines, soils_params) -> dict:
    model = load_scene(len(machines), weld=True)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, machines)
    soils = [Soil(model, i, p) for i, p in enumerate(soils_params)]
    pose = inverse_kinematics(*START)
    commands = np.tile(DRAG, (len(machines), 1))
    best = None
    physics = mujoco.mj_step
    for _ in range(REPEATS):
        clock = {"physics": 0.0, "soil": 0.0}
        mujoco.mj_step = _timed(clock, "physics", physics)
        for soil in soils:
            soil.substep = _timed(clock, "soil", Soil.substep.__get__(soil))
            soil.control_step = _timed(clock, "soil", Soil.control_step.__get__(soil))
        total, cutting = 0.0, []
        try:
            for _ in range(EPISODES):
                for i, machine in enumerate(machines):
                    reset(layer, soils, i, machine, pose)
                start = time.perf_counter()
                for _ in range(EPISODE):
                    step(layer, soils, commands)
                    cutting.append(np.mean([s.resistance > 0.0 for s in soils]))
                total += time.perf_counter() - start
        finally:
            mujoco.mj_step = physics
            for soil in soils:
                del soil.substep, soil.control_step
        steps = EPISODES * EPISODE * layer.steps_per_decision
        result = {"rate": steps / total, "total": total, "cutting": float(np.mean(cutting)),
                  **clock}
        if best is None or result["rate"] > best["rate"]:
            best = result
    return best


def main() -> None:
    one = measure([NOMINAL], [NOMINAL_SOIL])
    eight = measure([draw_machine(s) for s in range(8)], [sample_soil(s) for s in range(8)])
    for name, r, n in (("1 machine", one, 1), ("8 machines", eight, 8)):
        share = {k: r[k] / r["total"] for k in ("physics", "soil")}
        rest = 1.0 - share["physics"] - share["soil"]
        print(f"{name:>10}: {r['rate']:7.0f} scene steps/s, {n * r['rate']:7.0f} machine steps/s; "
              f"physics {share['physics']:.0%}, hydraulics and loop {rest:.0%}, "
              f"soil {share['soil']:.0%}; cutting on {r['cutting']:.0%} of control steps")


if __name__ == "__main__":
    main()
