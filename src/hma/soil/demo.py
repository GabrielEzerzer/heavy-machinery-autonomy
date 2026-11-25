"""One scripted dig cycle on the welded machine in nominal soil: lower, drag, curl, lift, swing,
dump. Joints follow inverse-kinematics targets through proportional speed commands; a phase
ends without the swing while the soil holds the edge. Usage: python -m hma.soil.demo"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import mujoco
import numpy as np

from hma.machine.kinematics import inverse_kinematics
from hma.machine.model import JOINT_NAMES, SPEED_LIMITS, load_scene, machine_prefix
from hma.soil.params import NOMINAL
from hma.soil.soil import CONTROL_HZ, Soil, step, sync_viewer

PITCH_DIG = 1.9
PITCH_CARRY = 3.0
PITCH_DUMP = 0.9
DEPTH = 0.3
DRAG_FROM, DRAG_TO = 6.6, 4.6
DRAG_SPEED = 0.3
SWING_TO = 1.2
GAIN = 3.0
TOLERANCE = 0.01
SWING = JOINT_NAMES.index("swing")
PHASE_TIMEOUT = 15.0
START = (np.array([DRAG_FROM, 0.0, 0.5]), PITCH_DIG)
PHASES = (
    ("lower", (DRAG_FROM, 0.0, -DEPTH), PITCH_DIG),
    ("drag", (DRAG_TO, 0.0, -DEPTH), PITCH_DIG),
    ("curl", (DRAG_TO, 0.0, -DEPTH + 0.2), PITCH_CARRY),
    ("lift", (DRAG_TO, 0.0, 1.5), PITCH_CARRY),
    ("swing", (DRAG_TO * np.cos(SWING_TO), DRAG_TO * np.sin(SWING_TO), 1.5), PITCH_CARRY),
    ("dump", (DRAG_TO * np.cos(SWING_TO), DRAG_TO * np.sin(SWING_TO), 1.5), PITCH_DUMP),
)


def start_pose(model: mujoco.MjModel, data: mujoco.MjData, index: int = 0) -> None:
    q = inverse_kinematics(*START)
    for name, value in zip(JOINT_NAMES, q, strict=True):
        joint = model.joint(machine_prefix(index) + name)
        data.qpos[joint.qposadr[0]] = value
        data.qvel[joint.dofadr[0]] = 0.0
    mujoco.mj_forward(model, data)


def cycle(model: mujoco.MjModel, data: mujoco.MjData, soil: Soil,
          control_hz: float = CONTROL_HZ) -> Iterator[str]:
    prefix = machine_prefix(soil.index)
    qpos = [model.joint(prefix + j).qposadr[0] for j in JOINT_NAMES]
    acts = [model.actuator(prefix + j).id for j in JOINT_NAMES]
    limits = np.array([SPEED_LIMITS[j] for j in JOINT_NAMES])
    period = 1.0 / control_hz
    position, pitch = START
    for name, goal, goal_pitch in PHASES:
        goal = np.array(goal, float)
        deadline = data.time + PHASE_TIMEOUT
        while data.time < deadline:
            gap = goal - position
            stride = DRAG_SPEED * period if name == "drag" else np.inf
            position = position + gap * min(1.0, stride / max(np.linalg.norm(gap), 1e-12))
            pitch = goal_pitch if name != "drag" else pitch
            target = inverse_kinematics(position, pitch)
            error = target - data.qpos[qpos]
            waiting = np.abs(error)
            if any(soil.limits):
                waiting[SWING] = 0.0
            if np.linalg.norm(goal - position) < 1e-9 and waiting.max() < TOLERANCE:
                break
            data.ctrl[acts] = np.clip(GAIN * error, -limits, limits)
            step(model, data, [soil], control_hz)
            yield name
    data.ctrl[acts] = 0.0


def main() -> None:
    import mujoco.viewer

    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    soil = Soil(model, 0, NOMINAL)
    start_pose(model, data)
    soil.reset(model, data)
    steps = cycle(model, data, soil)
    before = set(threading.enumerate())
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [4.0, 1.5, 0.0]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 12.0, 150.0, -30.0
        start = time.perf_counter()
        while viewer.is_running():
            with viewer.lock():
                if next(steps, None) is None:
                    break
            sync_viewer(viewer, soil)
            viewer.sync()
            ahead = data.time - (time.perf_counter() - start)
            if ahead > 0:
                time.sleep(ahead)
    for thread in set(threading.enumerate()) - before:
        thread.join()


if __name__ == "__main__":
    main()
