"""Scripted motion of the 12 t class excavator through its joint ranges in the interactive viewer.
Usage: python -m hma.machine.demo [--machines N]"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Iterator

import mujoco
import numpy as np

from hma.machine.model import HOME_KEY, HOME_POSE, JOINT_NAMES, load_scene, machine_prefix
from hma.machine.render import apply_look, default_camera, draw_hydraulics, view_option

WAYPOINTS = np.array([
    HOME_POSE,
    [0.0, -1.1, 1.6, 0.4],
    [0.0, -1.1, 0.6, -0.5],
    [0.0, -0.1, 0.6, -0.5],
    [0.0, -0.5, 2.4, 1.0],
    [0.0, -0.4, 2.0, 2.4],
    [1.5, -0.4, 1.4, 2.0],
    [-1.5, -0.4, 1.4, 2.0],
    HOME_POSE,
])
SPEEDS = np.array([0.6, 0.3, 0.6, 0.8])
GAIN = 2.0
TOLERANCE = 0.01
CONTROL_PERIOD = 0.1
WAYPOINT_TIMEOUT = 20.0


def motion(model: mujoco.MjModel, data: mujoco.MjData, n_machines: int) -> Iterator[int]:
    acts = np.array([[model.actuator(machine_prefix(i) + j).id for j in JOINT_NAMES]
                     for i in range(n_machines)])
    qpos = np.array([[model.joint(machine_prefix(i) + j).qposadr[0] for j in JOINT_NAMES]
                     for i in range(n_machines)])
    control_steps = round(CONTROL_PERIOD / model.opt.timestep)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    for index, target in enumerate(WAYPOINTS[1:], start=1):
        deadline = data.time + WAYPOINT_TIMEOUT
        while np.abs(data.qpos[qpos] - target).max() > TOLERANCE and data.time < deadline:
            error = target - data.qpos[qpos]
            data.ctrl[acts] = np.clip(GAIN * error, -SPEEDS, SPEEDS)
            mujoco.mj_step(model, data, control_steps)
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
    before = set(threading.enumerate())
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            camera = default_camera(model, data)
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance, viewer.cam.azimuth = camera.distance, camera.azimuth
            viewer.cam.elevation = camera.elevation
            option = view_option()
            viewer.opt.sitegroup[:] = option.sitegroup
            viewer.opt.flags[:] = option.flags
            apply_look(viewer.user_scn)
        start = time.perf_counter()
        while viewer.is_running():
            with viewer.lock():
                if next(steps, None) is None:
                    break
                viewer.user_scn.ngeom = 0
                draw_hydraulics(model, data, viewer.user_scn)
            viewer.sync()
            ahead = data.time - (time.perf_counter() - start)
            if ahead > 0:
                time.sleep(ahead)
    for thread in set(threading.enumerate()) - before:
        thread.join()


if __name__ == "__main__":
    main()
