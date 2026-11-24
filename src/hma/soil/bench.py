"""Physics steps per second for one welded machine: the bare model against the same model cutting
nominal soil, and the cost of one write_ground per control step.
Usage: python -m hma.soil.bench"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from hma.machine.kinematics import inverse_kinematics
from hma.machine.model import HOME_KEY, JOINT_NAMES, load_scene, machine_prefix, write_ground
from hma.soil.demo import DEPTH, DRAG_FROM, PITCH_DIG
from hma.soil.params import NOMINAL
from hma.soil.soil import Soil

STEPS = 4000
REPEATS = 3


def _set_pose(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    for name, value in zip(JOINT_NAMES, q, strict=True):
        data.qpos[model.joint(machine_prefix(0) + name).qposadr[0]] = value
    mujoco.mj_forward(model, data)


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
    arm = model.actuator(machine_prefix(0) + "arm").id
    buried = inverse_kinematics(np.array([DRAG_FROM, 0.0, -DEPTH]), PITCH_DIG)

    def bare() -> None:
        mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
        _set_pose(model, data, buried)
        data.ctrl[arm] = -0.05
        for _ in range(STEPS):
            mujoco.mj_step(model, data)

    soil = Soil(model, 0, NOMINAL)
    cutting = []

    def with_soil() -> None:
        mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
        _set_pose(model, data, buried)
        soil.reset(model, data)
        data.ctrl[arm] = -0.05
        for _ in range(STEPS):
            soil.substep(model, data)
            mujoco.mj_step(model, data)
            cutting.append(soil.resistance > 0.0)

    heights = np.zeros(soil.grid.shape)

    def writes() -> None:
        for _ in range(STEPS):
            write_ground(model, 0, heights)

    bare_rate, soil_rate, write_rate = _rate(bare), _rate(with_soil), _rate(writes)
    print(f"bare model:      {bare_rate:9.0f} physics steps/s")
    print(f"cutting soil:    {soil_rate:9.0f} physics steps/s "
          f"({np.mean(cutting):.0%} of steps with resistance)")
    print(f"write_ground:    {1e6 / write_rate:9.1f} us per call")


if __name__ == "__main__":
    main()
