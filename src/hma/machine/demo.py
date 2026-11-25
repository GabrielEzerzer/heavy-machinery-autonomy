"""Scripted motion of the 12 t class excavator through its joint ranges in the interactive viewer,
driven by valve commands through the nominal hydraulics.
Usage: python -m hma.machine.demo [--machines N]"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Iterator

import mujoco
import numpy as np

from hma.hydraulics import NOMINAL, Hydraulics
from hma.hydraulics.circuit import valve_opening
from hma.machine.model import HOME_POSE, load_scene
from hma.machine.render import apply_look, default_camera, draw_hydraulics, view_option

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
RAMP = 0.02
TOLERANCE = 0.01
CONTROL_PERIOD = 0.1
WAYPOINT_TIMEOUT = 30.0


def motion(model: mujoco.MjModel, data: mujoco.MjData, n_machines: int) -> Iterator[int]:
    layer = Hydraulics(model, data, [NOMINAL] * n_machines, control_hz=1.0 / CONTROL_PERIOD)
    for index in range(n_machines):
        layer.reset(index, NOMINAL, HOME_POSE)
    for index, target in enumerate(WAYPOINTS[1:], start=1):
        deadline = data.time + WAYPOINT_TIMEOUT
        while data.time < deadline:
            seen = layer.observe()
            error = target - seen.positions
            if np.abs(error).max() <= TOLERANCE:
                break
            speed = np.clip(GAIN * error, -SPEEDS, SPEEDS)
            rows = zip(seen.positions, speed, seen.pressures, strict=True)
            layer.act(np.array([np.sign(s) * valve_opening(q, s, p, RAMP) for q, s, p in rows]))
            for _ in range(layer.steps_per_decision):
                layer.step()
            yield index
    layer.act(np.zeros((n_machines, len(HOME_POSE))))


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
