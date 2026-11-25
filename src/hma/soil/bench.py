"""Physics steps per second for one welded machine through step(), substeps and control steps
included: the bare model, the same machine moving in free air over soil, and cutting soil.
Usage: python -m hma.soil.bench"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from hma.machine.kinematics import inverse_kinematics
from hma.machine.model import HOME_KEY, JOINT_NAMES, load_scene, machine_prefix, write_ground
from hma.soil.demo import DEPTH, DRAG_FROM, PITCH_DIG
from hma.soil.params import NOMINAL
from hma.soil.soil import CONTROL_HZ, Soil, physics_steps, step

STEPS = 4000
REPEATS = 3


def _rate(run) -> float:
    best = 0.0
    for _ in range(REPEATS):
        start = time.perf_counter()
        run()
        best = max(best, STEPS / (time.perf_counter() - start))
    return best


def main() -> None:
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    soil = Soil(model, 0, NOMINAL)
    prefix = machine_prefix(0)
    arm = model.actuator(prefix + "arm").id
    joints = [model.joint(prefix + name).qposadr[0] for name in JOINT_NAMES]
    per_control = physics_steps(model, CONTROL_HZ)
    cutting = []

    def start(point: np.ndarray, command: float) -> None:
        mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
        data.qpos[joints] = inverse_kinematics(point, PITCH_DIG)
        soil.reset(model, data)
        data.ctrl[arm] = command

    def bare() -> None:
        start(np.array([DRAG_FROM, 0.0, -DEPTH]), -0.05)
        for _ in range(STEPS):
            mujoco.mj_step(model, data)

    def in_air() -> None:
        start(np.array([6.0, 0.0, 1.5]), 0.1)
        for _ in range(STEPS // per_control):
            step(model, data, [soil])

    def in_soil() -> None:
        start(np.array([DRAG_FROM, 0.0, -DEPTH]), -0.05)
        for _ in range(STEPS // per_control):
            step(model, data, [soil])
            cutting.append(soil.resistance > 0.0)

    heights = np.zeros(soil.grid.shape)

    def writes() -> None:
        for _ in range(STEPS):
            write_ground(model, 0, heights)

    rates = _rate(bare), _rate(in_air), _rate(in_soil), _rate(writes)
    print(f"bare model:      {rates[0]:9.0f} physics steps/s")
    print(f"free air:        {rates[1]:9.0f} physics steps/s through step()")
    print(f"cutting soil:    {rates[2]:9.0f} physics steps/s through step() "
          f"({np.mean(cutting):.0%} of control steps with resistance)")
    print(f"write_ground:    {1e6 / rates[3]:9.1f} us per call")


if __name__ == "__main__":
    main()
