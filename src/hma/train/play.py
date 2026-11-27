"""One propping episode in the interactive viewer: the baseline and a trained controller on the same
seed, side by side. Each runs in its own one-machine scene and is not stepped again once its episode
ends; a two-machine scene drawn from the same seed shows both."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator

import mujoco
import numpy as np

from hma.hydraulics.layer import JOINTS
from hma.machine.model import MACHINE_SPACING, machine_prefix
from hma.machine.render import apply_look, default_camera, draw_hydraulics, view_option
from hma.train.scene import PropScene
from hma.train.task import HELD_OUT_BASE, Settings

LABELS = ("baseline", "trained")
SIZES = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}


def episodes(rung: str, seed: int, policy=None, mode: str = "residual",
             settings: Settings | None = None) -> Iterator[tuple[list[PropScene], list]]:
    if seed >= HELD_OUT_BASE:
        raise ValueError("held-out seeds are never played")
    scenes = [PropScene(1, rung), PropScene(1, rung, mode, settings)]
    policies = [None, policy]
    obs = [scene.reset(0, seed)[None] for scene in scenes]
    outcomes: list[dict | None] = [None, None]
    yield scenes, outcomes
    while None in outcomes:
        for i, scene in enumerate(scenes):
            if outcomes[i] is not None:
                continue
            action = (policies[i](obs[i].astype(np.float32)) if policies[i]
                      else np.zeros((1, JOINTS)))
            obs[i], _, done, infos = scene.step(np.asarray(action, float))
            if done[0]:
                outcomes[i] = infos[0]
        yield scenes, outcomes


def why(info: dict, s: Settings) -> list[str]:
    checks = ((info["cause"] is not None, f"early end: {info['cause']}"),
              (info["hold_share"] < s.hold_share, f"hold share in band under {s.hold_share:.2f}"),
              (info["landing"] >= s.landing, f"a landing at {s.landing} m/s or faster"),
              (info["bucket_down"], "bucket on the ground at the end"),
              (not info["front_down"], "front of the tracks not set down"),
              (info["front_carried"] < s.carry, f"front carries under {s.carry:.0%} of its load"),
              (info["front_speed"] >= s.still, f"front moving at {s.still} m/s or faster"))
    return [reason for failed, reason in checks if failed]


def summary(label: str, info: dict, s: Settings) -> str:
    verdict = "pass" if info["passed"] else "fail (" + "; ".join(why(info, s)) + ")"
    return (f"{label}: {verdict}; hold share in band {info['hold_share']:.2f}, tilt error "
            f"{math.degrees(info['tilt_error']):.2f} deg, landing {info['landing']:.3f} m/s")


def mirror(display: PropScene, scenes: list[PropScene]):
    parts = []
    for slot, scene in enumerate(scenes):
        prefix, model = machine_prefix(slot), display.model
        for j in range(model.njnt):
            name = model.joint(j).name
            if not name.startswith(prefix):
                continue
            size = SIZES.get(int(model.jnt_type[j]), 1)
            start = int(model.jnt_qposadr[j])
            source = int(scene.model.joint(machine_prefix(0) + name[len(prefix):]).qposadr[0])
            shift = np.zeros(size)
            if size == 7:
                shift[1] = slot * MACHINE_SPACING
            parts.append((scene, slice(start, start + size), slice(source, source + size), shift))

    def copy() -> None:
        for scene, to, source, shift in parts:
            display.data.qpos[to] = scene.data.qpos[source] + shift
        mujoco.mj_forward(display.model, display.data)

    return copy


def texts(scenes: list[PropScene], outcomes: list) -> list[tuple]:
    out = []
    for label, scene, info, grid in zip(LABELS, scenes, outcomes,
                                        (mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                         mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT), strict=True):
        commanded, measured = np.degrees(scene.tilt(0))
        state = "running" if info is None else ("pass" if info["passed"] else "fail")
        out.append((mujoco.mjtFontScale.mjFONTSCALE_150, grid,
                    f"{label}\ncommanded tilt\nmeasured tilt",
                    f"{state}\n{commanded:.1f} deg\n{measured:.1f} deg"))
    return out


def watch(rung: str, seed: int, policy=None, mode: str = "residual",
          settings: Settings | None = None) -> list[dict]:
    import mujoco.viewer

    steps = episodes(rung, seed, policy, mode, settings)
    scenes, outcomes = next(steps)
    display = PropScene(2, rung)
    for slot in range(2):
        display.reset(slot, seed)
    copy = mirror(display, scenes)
    copy()
    before = set(threading.enumerate())
    with mujoco.viewer.launch_passive(display.model, display.data) as viewer:
        with viewer.lock():
            camera = default_camera(display.model, display.data)
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance, viewer.cam.azimuth = camera.distance, camera.azimuth
            viewer.cam.elevation = camera.elevation
            option = view_option()
            viewer.opt.sitegroup[:] = option.sitegroup
            viewer.opt.flags[:] = option.flags
            apply_look(viewer.user_scn)
        start, frames = time.perf_counter(), 0
        while viewer.is_running():
            running = None in outcomes
            with viewer.lock():
                if running:
                    scenes, outcomes = next(steps)
                    copy()
                viewer.user_scn.ngeom = 0
                draw_hydraulics(display.model, display.data, viewer.user_scn)
            viewer.set_texts(texts(scenes, outcomes))
            viewer.sync()
            if running and None not in outcomes:
                for label, scene, info in zip(LABELS, scenes, outcomes, strict=True):
                    print(summary(label, info, scene.settings), flush=True)
            frames += 1
            ahead = start + frames * scenes[0].period - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
    for thread in set(threading.enumerate()) - before:
        thread.join()
    return outcomes
