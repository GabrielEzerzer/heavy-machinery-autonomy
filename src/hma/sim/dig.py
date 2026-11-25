"""One dig cycle from the home pose (lower, drag, curl, lift, swing, dump) with the base controller
through the hydraulics layer into the soil, on the nominal pair and on drawn machines and soils.
Usage: python -m hma.sim.dig [--pairs N] [--free] [--view]"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from hma.hydraulics import NOMINAL, BaseController, Hydraulics, Machine, draw_machine
from hma.machine import HOME_POSE, JOINT_LIMITS, JOINT_NAMES, forward_kinematics, load_scene
from hma.machine.kinematics import BOOM_PIVOT
from hma.sim import reset, step
from hma.soil import NOMINAL as NOMINAL_SOIL
from hma.soil import Soil, SoilParams, sample_soil, sync_viewer

PASS_DEPTH = 0.3
DRAG_FROM, DRAG_TO = 6.6, 4.6
SWING_TO = 1.2
PITCH_DIG, PITCH_CARRY, PITCH_DUMP = 1.9, 3.0, 0.9
CARRY_HEIGHT = 1.5
APPROACH_HEIGHT = 0.5
PHASES = (
    ("approach", 5.0, (DRAG_FROM, 0.0, APPROACH_HEIGHT), PITCH_DIG),
    ("lower", 3.0, (DRAG_FROM, 0.0, -PASS_DEPTH), PITCH_DIG),
    ("drag", 8.0, (DRAG_TO, 0.0, -PASS_DEPTH), PITCH_DIG),
    ("curl", 3.0, (DRAG_TO, 0.0, -PASS_DEPTH + 0.2), PITCH_CARRY),
    ("lift", 3.0, (DRAG_TO, 0.0, CARRY_HEIGHT), PITCH_CARRY),
    ("swing", 4.0, (DRAG_TO, SWING_TO, CARRY_HEIGHT), PITCH_CARRY),
    ("dump", 6.0, (DRAG_TO, SWING_TO, CARRY_HEIGHT), PITCH_DUMP),
    ("hold", 3.0, (DRAG_TO, SWING_TO, CARRY_HEIGHT), PITCH_DUMP),
)
TRENCH = ((DRAG_TO - 0.6, DRAG_FROM + 0.6), (-0.9, 0.6))
PILE_RADIUS = 1.5
EXCESS_DEPTH = 0.2
STALL_LOAD = 0.99
STALL_SPEED = 0.01
STALL_TIME = 1.0
STOP_MARGIN = 0.02
DUMPED = 0.01


def _ease(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def tip_point(radial: float, swing: float, z: float) -> np.ndarray:
    c, s = math.cos(swing), math.sin(swing)
    return np.array([c * radial - s * BOOM_PIVOT[1], s * radial + c * BOOM_PIVOT[1], z])


@dataclass(frozen=True)
class CyclePath:
    start: tuple[float, float, float]
    start_pitch: float
    phases: tuple = PHASES

    @property
    def duration(self) -> float:
        return sum(seconds for _, seconds, _, _ in self.phases)

    def phase(self, t: float) -> str:
        for name, seconds, _, _ in self.phases:
            if t < seconds:
                return name
            t -= seconds
        return self.phases[-1][0]

    def __call__(self, t: float) -> tuple[np.ndarray, float]:
        point, pitch = np.array(self.start), self.start_pitch
        for _, seconds, goal, goal_pitch in self.phases:
            s = _ease(t / seconds)
            here = point + s * (np.array(goal) - point)
            pitch = pitch + s * (goal_pitch - pitch)
            if t < seconds:
                break
            t -= seconds
            point = np.array(goal)
        return tip_point(*here), pitch


def home_path() -> CyclePath:
    q = np.array(HOME_POSE)
    (x, _, z), pitch = forward_kinematics(np.array([0.0, *q[1:]]))
    return CyclePath((float(x), float(q[0]), float(z)), float(pitch))


DUMP_AT = tip_point(DRAG_TO, SWING_TO, CARRY_HEIGHT)


def admissible(path: CyclePath, soil: Soil, samples: int = 2000) -> bool:
    xs, ys = soil.grid.axes()
    for t in np.linspace(0.0, path.duration, samples):
        (x, y, z), _ = path(t)
        below = z < soil.grid.z0
        if below and not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
            return False
        if soil.grid.z0 - z > PASS_DEPTH + 1e-9:
            return False
    return True


def _region(soil: Soil, inside) -> np.ndarray:
    return inside(soil.cells[..., 0], soil.cells[..., 1])


def dig(machine: Machine, params: SoilParams, weld: bool = True, viewer=None) -> dict:
    model = load_scene(1, weld=weld)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [machine])
    soil = Soil(model, 0, params)
    path = home_path()
    controller = BaseController(path, layer.control_hz)
    reset(layer, [soil], 0, machine)
    start = soil.heights.copy()
    total = soil.total_volume()
    limits = np.array([[-math.inf, math.inf], *JOINT_LIMITS.values()])
    stalled_for = np.zeros(len(JOINT_NAMES))
    stalls, errors, filled, excess, outside, base = set(), [], 0.0, 0.0, False, []
    steps = round(path.duration * layer.control_hz)
    for k in range(steps):
        t = k / layer.control_hz
        step(layer, [soil], controller(t, layer.observe().positions[0])[None])
        state = layer.true_state()
        q, v, load = state.positions[0], state.velocities[0], state.loads[0]
        tip, _ = forward_kinematics(q)
        errors.append(np.linalg.norm(tip - path(t + 1.0 / layer.control_hz)[0]))
        filled = max(filled, soil.bucket_volume)
        excess = max(excess, soil.uncut_depth(data) - PASS_DEPTH)
        edge_z = data.site_xpos[soil.edges, 2].min()
        outside |= edge_z < soil.grid.z0 and not soil.edge_inside(data)
        free = (q - limits[:, 0] > STOP_MARGIN) & (limits[:, 1] - q > STOP_MARGIN)
        held = (np.abs(load) >= STALL_LOAD) & (np.abs(v) < STALL_SPEED) & free
        stalled_for = np.where(held, stalled_for + 1.0 / layer.control_hz, 0.0)
        for j in np.flatnonzero(stalled_for >= STALL_TIME):
            stalls.add(f"{JOINT_NAMES[j]} in {path.phase(t)}")
        base.append(data.xpos[soil.base].copy())
        if viewer is not None:
            viewer(model, data, soil)
    change = soil.heights - start
    (x0, x1), (y0, y1) = TRENCH
    trench = _region(soil, lambda x, y: (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
    pile = _region(soil, lambda x, y: np.hypot(x - DUMP_AT[0], y - DUMP_AT[1]) <= PILE_RADIUS)
    area = soil.grid.cell_area
    errors = np.array(errors)
    base = np.array(base)
    return {
        "machine": machine.seed, "soil": params,
        "trench_out": float(-change[trench].sum() * area),
        "pile": float(change[pile].sum() * area),
        "filled": filled, "left_in_bucket": soil.bucket_volume,
        "error_mean": float(errors.mean()), "error_max": float(errors.max()),
        "stalls": sorted(stalls), "dumped": soil.bucket_volume < DUMPED,
        "broke_rule": excess > EXCESS_DEPTH or outside,
        "excess": excess, "outside": outside,
        "volume_error": soil.total_volume() - total, "lost": soil.lost_volume,
        "base_moved": float(np.linalg.norm(base[-1] - base[0])),
        "deepest": float(-change.min()), "highest": float(change.max()),
    }


def _row(name: str, r: dict) -> str:
    stalls = ", ".join(r["stalls"]) or "-"
    return (f"{name:>10} {r['trench_out']:6.3f} {r['pile']:6.3f} {r['filled']:6.3f} "
            f"{r['left_in_bucket']:6.3f} {100 * r['error_mean']:6.1f} {100 * r['error_max']:6.1f} "
            f"{'yes' if r['broke_rule'] else 'no':>5} {stalls}")


def _viewer_sync():
    import mujoco.viewer

    state = {}

    def sync(model, data, soil):
        if "viewer" not in state:
            state["threads"] = set(threading.enumerate())
            state["viewer"] = mujoco.viewer.launch_passive(model, data)
            state["start"] = time.perf_counter() - data.time
        viewer = state["viewer"]
        if viewer.is_running():
            sync_viewer(viewer, soil)
            viewer.sync()
            ahead = data.time - (time.perf_counter() - state["start"])
            if ahead > 0:
                time.sleep(ahead)

    def close():
        if "viewer" in state:
            state["viewer"].close()
            for thread in set(threading.enumerate()) - state["threads"]:
                thread.join()

    return sync, close


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=30, help="drawn machine and soil pairs")
    parser.add_argument("--free", action="store_true", help="also run one cycle on a free base")
    parser.add_argument("--view", action="store_true", help="watch the nominal cycle")
    args = parser.parse_args()
    print(f"{'pair':>10} {'out m3':>6} {'pile':>6} {'filled':>6} {'left':>6} {'err cm':>6} "
          f"{'max':>6} {'broke':>5} stalls")
    if args.view:
        sync, close = _viewer_sync()
        try:
            nominal = dig(NOMINAL, NOMINAL_SOIL, viewer=sync)
        finally:
            close()
    else:
        nominal = dig(NOMINAL, NOMINAL_SOIL)
    print(_row("nominal", nominal))
    runs = []
    for seed in range(args.pairs):
        runs.append(dig(draw_machine(seed), sample_soil(seed)))
        print(_row(f"seed {seed}", runs[-1]))
    if runs:
        print(f"{'median':>10} {np.median([r['trench_out'] for r in runs]):6.3f} "
              f"{np.median([r['pile'] for r in runs]):6.3f} "
              f"{np.median([r['filled'] for r in runs]):6.3f} "
              f"{np.median([r['left_in_bucket'] for r in runs]):6.3f} "
              f"{100 * np.median([r['error_mean'] for r in runs]):6.1f} "
              f"{100 * np.median([r['error_max'] for r in runs]):6.1f}")
        print(f"stalled {sum(bool(r['stalls']) for r in runs)} of {len(runs)}, "
              f"never emptied the bucket {sum(not r['dumped'] for r in runs)}, "
              f"half or less of the soil out reached the pile "
              f"{sum(r['pile'] <= r['trench_out'] / 2 for r in runs)}, "
              f"broke the scoring rule {sum(r['broke_rule'] for r in runs)}, "
              f"largest volume error {max(abs(r['volume_error']) for r in runs):.1e} m3")
    if args.free:
        free = dig(NOMINAL, NOMINAL_SOIL, weld=False)
        print(_row("free base", free))
        print(f"free base moved {free['base_moved']:.3f} m")


if __name__ == "__main__":
    main()
