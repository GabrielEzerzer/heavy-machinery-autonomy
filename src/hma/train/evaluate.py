"""Scoring over the fixed development seeds, and throughput timing. Seeds run in fixed scenes and
are rerun when a neighbour's solver failure cuts them short; outcomes are grouped by thirds of every
drawn parameter and compared seed by seed against the baseline. Held-out seeds are refused."""

from __future__ import annotations

import csv
import math
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from hma.train.scene import CAUSES, PropScene
from hma.train.task import DEV_BASE, HELD_OUT_BASE, Settings, training_seed

DRAWN = ("target", "ramp_up", "hold", "ramp_down", "prop_x", "pitch", "friction", "step", "grade",
         "cross", "delay", "lag", "deadband", "sensor_latency", "counterweight", "link_mass",
         "seal_friction")
DEGREES = {"target", "pitch"}
FILLER = DEV_BASE - 1
ROUNDS = 4


def episode_row(scene, slot: int, info: dict, controller: str) -> dict:
    task, ground, machine = scene.tasks[slot], scene.grounds[slot], scene.machines[slot]
    return {
        "controller": controller, "rung": scene.rung, "scene_size": scene.n, "slot": slot,
        **{k: v for k, v in info.items() if k not in ("truncated", "cut_short")},
        "target": task.target, "ramp_up": task.ramp_up, "hold": task.hold,
        "ramp_down": task.ramp_down, "prop_x": task.prop_x, "pitch": task.pitch,
        "friction": ground.friction, "step": ground.step, "grade": ground.grade,
        "cross": ground.cross, "delay": float(np.mean(machine.delay)),
        "lag": float(np.mean(machine.lag)), "deadband": float(np.mean(machine.deadband)),
        "sensor_latency": machine.sensor_latency, "counterweight": machine.counterweight,
        "link_mass": float(np.mean(machine.link_mass)), "seal_friction": machine.seal_friction,
    }


def load_policy(checkpoint: Path):
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint / "model.zip", device="cpu")
    with open(checkpoint / "normalize.pkl", "rb") as f:
        normalize = pickle.load(f)
    normalize.training = False
    return lambda obs: model.predict(normalize.normalize_obs(obs), deterministic=True)[0]


def _filler(round_: int, slot: int, restart: int) -> int:
    return training_seed(np.random.default_rng((FILLER, round_, slot, restart)))


def _run_round(scene: PropScene, pending: dict[int, int], round_: int, policy,
               controller: str) -> list[dict]:
    restarts = [0] * scene.n
    obs = np.stack([scene.reset(slot, pending.get(slot, _filler(round_, slot, 0)))
                    for slot in range(scene.n)])
    live, rows = set(pending), []
    while live:
        actions = policy(obs.astype(np.float32)) if policy else np.zeros((scene.n, 4))
        obs, _, done, infos = scene.step(np.asarray(actions, float))
        for slot in np.flatnonzero(done):
            slot = int(slot)
            if slot in live:
                live.discard(slot)
                if not infos[slot]["cut_short"]:
                    rows.append(episode_row(scene, slot, infos[slot], controller))
                    del pending[slot]
            restarts[slot] += 1
            obs[slot] = scene.reset(slot, _filler(round_, slot, restarts[slot]))
    return rows


def run_group(rung: str, mode: str, seeds: list[int], machines: int, checkpoint: str | None,
              settings: Settings | None = None) -> list[dict]:
    policy = None
    if checkpoint:
        import torch

        torch.set_num_threads(1)
        policy = load_policy(Path(checkpoint))
    controller = "policy" if checkpoint else "baseline"
    scene = PropScene(machines, rung, mode, settings)
    pending, rows = dict(enumerate(seeds)), []
    for round_ in range(ROUNDS):
        rows += _run_round(scene, pending, round_, policy, controller)
        if not pending:
            return rows
    raise RuntimeError(f"seeds {sorted(pending.values())} were cut short in {ROUNDS} rounds")


def score(rung: str, mode: str, seeds: list[int], machines: int, procs: int,
          checkpoint: str | None, settings: Settings | None = None) -> list[dict]:
    if any(seed >= HELD_OUT_BASE for seed in seeds):
        raise ValueError("held-out seeds are never scored")
    if len(set(seeds)) != len(seeds):
        raise ValueError("each seed is scored once")
    groups = [seeds[k:k + machines] for k in range(0, len(seeds), machines)]
    args = [[rung] * len(groups), [mode] * len(groups), groups, [machines] * len(groups),
            [checkpoint] * len(groups), [settings] * len(groups)]
    if procs <= 1 or len(groups) == 1:
        parts = list(map(run_group, *args))
    else:
        with ProcessPoolExecutor(min(procs, len(groups)), mp_context=get_context("spawn")) as pool:
            parts = list(pool.map(run_group, *args))
    rows = [row for part in parts for row in part]
    order = {seed: k for k, seed in enumerate(seeds)}
    return sorted(rows, key=lambda r: (r["controller"], order[r["seed"]]))


def _summary(rows: list[dict]) -> str:
    if not rows:
        return "| 0 | - | - | - | - |"
    passed = 100.0 * np.mean([r["passed"] for r in rows])
    share = float(np.mean([r["hold_share"] for r in rows]))
    error = math.degrees(float(np.mean([r["tilt_error"] for r in rows])))
    causes = Counter(r["cause"] for r in rows if r["cause"])
    ends = ", ".join(f"{c} {causes[c]}" for c in CAUSES if causes[c]) or "none"
    return f"| {len(rows)} | {passed:.0f}% | {share:.2f} | {error:.2f} | {ends} |"


def _mean_and_error(values: np.ndarray) -> str:
    error = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    return f"{values.mean():+.3f} ± {error:.3f}"


def paired(rows: list[dict], reference: str = "baseline") -> list[str]:
    base = {r["seed"]: r for r in rows if r["controller"] == reference}
    lines = []
    for controller in sorted({r["controller"] for r in rows} - {reference}):
        pairs = [(base[r["seed"]], r) for r in rows
                 if r["controller"] == controller and r["seed"] in base]
        if not pairs:
            continue
        share = np.array([c["hold_share"] - b["hold_share"] for b, c in pairs])
        error = np.degrees([c["tilt_error"] - b["tilt_error"] for b, c in pairs])
        flips = Counter((b["passed"], c["passed"]) for b, c in pairs)
        lines += [f"### {controller} against {reference}, seed by seed", "",
                  f"{len(pairs)} seeds. Change in hold share in band "
                  f"{_mean_and_error(share)}; change in tilt error {_mean_and_error(error)} deg "
                  "(mean ± standard error of the paired differences). Passes gained "
                  f"{flips[(False, True)]}, lost {flips[(True, False)]}.", "",
                  f"| seed | pass ({reference}, {controller}) | hold share in band | "
                  "tilt error (deg) |", "|---|---|---|---|"]
        for b, c in pairs:
            lines.append(f"| {b['seed']} | {'yes' if b['passed'] else 'no'}, "
                         f"{'yes' if c['passed'] else 'no'} | {b['hold_share']:.2f}, "
                         f"{c['hold_share']:.2f} | {math.degrees(b['tilt_error']):.2f}, "
                         f"{math.degrees(c['tilt_error']):.2f} |")
        lines.append("")
    return lines


def _value(row: dict, name: str) -> float:
    return math.degrees(row[name]) if name in DEGREES else row[name]


def table(rows: list[dict]) -> str:
    lines = []
    for controller in sorted({r["controller"] for r in rows}):
        mine = [r for r in rows if r["controller"] == controller]
        lines += [f"### {controller}, rung {mine[0]['rung']}", "",
                  f"Scene size {mine[0]['scene_size']} machines; each seed keeps the slot its "
                  "place in the seed list gives it.", "",
                  "| group | episodes | pass | hold share in band | tilt error (deg) | "
                  "early ends |", "|---|---|---|---|---|---|", "| all " + _summary(mine), ""]
        lines += ["| drawn parameter (range) | episodes | pass | hold share in band | "
                  "tilt error (deg) | early ends |", "|---|---|---|---|---|---|"]
        for name in DRAWN:
            values = np.array([_value(r, name) for r in mine])
            if np.ptp(values) == 0:
                continue
            edges = np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0])
            for k in range(3):
                upper = values <= edges[k + 1] if k == 2 else values < edges[k + 1]
                picked = [r for r, keep in zip(mine, (values >= edges[k]) & upper, strict=True)
                          if keep]
                label = f"{name} {edges[k]:.3g} to {edges[k + 1]:.3g}"
                lines.append(f"| {label} " + _summary(picked))
        lines.append("")
    return "\n".join(lines + paired(rows))


def write(rows: list[dict], out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = out / f"{name}.md"
    path.write_text(table(rows), encoding="utf-8")
    return path


def throughput(rung: str, procs: int, machines: int, seconds: float, seed: int = 0) -> dict:
    from hma.train.vec import ProcVecEnv

    envs = ProcVecEnv(procs, machines, rung, "residual", seed)
    try:
        envs.reset()
        rng = np.random.default_rng(seed)
        for _ in range(3):
            envs.step_async(rng.uniform(-1.0, 1.0, (envs.num_envs, 4)))
            envs.step_wait()
        decisions, wall0 = 0, time.perf_counter()
        while time.perf_counter() - wall0 < seconds:
            envs.step_async(rng.uniform(-1.0, 1.0, (envs.num_envs, 4)))
            envs.step_wait()
            decisions += 1
        wall = time.perf_counter() - wall0
    finally:
        envs.close()
    return {"procs": procs, "machines": machines, "seconds": wall,
            "control_steps_per_s": decisions * envs.num_envs / wall,
            "per_process": decisions * machines / wall, "scene_decisions_per_s": decisions / wall}
