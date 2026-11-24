"""Scripted motion of the 12 t class excavator through its joint ranges in the interactive viewer.
Usage: python -m hma.machine.demo [--machines N]"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator

import mujoco
import numpy as np

from hma.machine.model import (
    HOME_KEY,
    HOME_POSE,
    JOINT_NAMES,
    MACHINE_SPACING,
    load_scene,
    machine_prefix,
)

WAYPOINTS = np.array([
    HOME_POSE,
    [0.0, -1.1, 1.6, 0.4],
    [0.0, -1.1, 0.6, -0.5],
    [0.0, 0.6, 0.6, -0.5],
    [0.0, 0.6, 2.4, 1.0],
    [0.0, 0.2, 2.0, 2.4],
    [1.5, -0.4, 1.4, 2.0],
    [-1.5, -0.4, 1.4, 2.0],
    HOME_POSE,
])
SPEEDS = np.array([0.6, 0.3, 0.6, 0.8])
GAIN = 2.0
TOLERANCE = 0.01
CONTROL_STEPS = 10
WAYPOINT_TIMEOUT = 20.0


def motion(model: mujoco.MjModel, data: mujoco.MjData, n_machines: int) -> Iterator[int]:
    acts = np.array([[model.actuator(machine_prefix(i) + j).id for j in JOINT_NAMES]
                     for i in range(n_machines)])
    qpos = np.array([[model.joint(machine_prefix(i) + j).qposadr[0] for j in JOINT_NAMES]
                     for i in range(n_machines)])
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    for index, target in enumerate(WAYPOINTS[1:], start=1):
        deadline = data.time + WAYPOINT_TIMEOUT
        while np.abs(data.qpos[qpos] - target).max() > TOLERANCE and data.time < deadline:
            error = target - data.qpos[qpos]
            data.ctrl[acts] = np.clip(GAIN * error, -SPEEDS, SPEEDS)
            mujoco.mj_step(model, data, CONTROL_STEPS)
            yield index
    data.ctrl[:] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machines", type=int, default=1)
    n_machines = parser.parse_args().machines
    import mujoco.viewer

    model = load_scene(n_machines, weld=True)
    data = mujoco.MjData(model)
    steps = motion(model, data, n_machines)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [3.0, (n_machines - 1) * MACHINE_SPACING / 2, 1.5]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 16.0, 120.0, -20.0
        start = time.perf_counter()
        while viewer.is_running():
            with viewer.lock():
                if next(steps, None) is None:
                    break
            viewer.sync()
            ahead = data.time - (time.perf_counter() - start)
            if ahead > 0:
                time.sleep(ahead)


if __name__ == "__main__":
    main()
