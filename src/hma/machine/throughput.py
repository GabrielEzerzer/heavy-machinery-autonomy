"""Physics steps per second through the hydraulics layer, standing, propping and carrying, for 1 and
8 machines. With --before <src dir> each run alternates with the same run on that source tree;
a tree without hma.machine.load carries its payload as a force at the bucket tip."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

MASS = 2000.0
POSES = {"standing": (0.0, -0.6, 1.9, 0.6), "propping": (0.0, -0.5, 1.6, 1.0),
         "carrying": (0.0, -1.0, 1.6, 2.3)}
SCENES = [(n, case) for n in (1, 8) for case in POSES]


def measure(n: int, case: str, seconds: float, warmup: float) -> tuple[float, int]:
    from hma.hydraulics.layer import Hydraulics
    from hma.hydraulics.params import NOMINAL
    from hma.machine import TIP_SITE, load_scene, machine_prefix

    try:
        from hma.machine.load import apply_load, block, hang_load
    except ImportError:
        apply_load = None
    model = load_scene(n)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [NOMINAL] * n)
    for i in range(n):
        layer.reset(i, NOMINAL, POSES[case])
        if case == "carrying" and apply_load is not None:
            apply_load(model, i, block(MASS), mujoco.MjData(model))
            hang_load(model, data, i)
    rng = np.random.default_rng(0)
    commands = np.zeros((n, 4))

    def decide() -> None:
        if case == "propping":
            commands[:, 1] = 1.0 / 3.0
        else:
            commands[:, 0] = rng.uniform(-1.0, 1.0, n)
        layer.act(commands)
        if case == "carrying" and apply_load is None:
            for i in range(n):
                body = model.body(machine_prefix(i) + "bucket").id
                tip = data.site_xpos[model.site(machine_prefix(i) + TIP_SITE).id]
                force = np.array([0.0, 0.0, -MASS * -model.opt.gravity[2]])
                data.xfrc_applied[body] = [*force, *np.cross(tip - data.xipos[body], force)]

    per = layer.steps_per_decision
    for _ in range(round(warmup / model.opt.timestep / per)):
        decide()
        for _ in range(per):
            layer.step()
    decisions = round(seconds / model.opt.timestep / per)
    start = time.perf_counter()
    for _ in range(decisions):
        decide()
        for _ in range(per):
            layer.step()
    return decisions * per / (time.perf_counter() - start), int(data.ncon)


def run_one(src: str | None, n: int, case: str, seconds: float, warmup: float) -> dict:
    env = dict(os.environ)
    if src is not None:
        env["PYTHONPATH"] = src
    out = subprocess.run([sys.executable, __file__, "--one", str(n), case, "--seconds",
                          str(seconds), "--warmup", str(warmup)], env=env, check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", help="src directory of the version to compare against")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=4.0)
    parser.add_argument("--one", nargs=2, metavar=("N", "CASE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        rate, ncon = measure(int(args.one[0]), args.one[1], args.seconds, args.warmup)
        print(json.dumps({"rate": rate, "ncon": ncon}))
        return
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("| Scene | Before | After | After / before, median (range) | Contacts after |")
    print("|---|---|---|---|---|")
    for n, case in SCENES:
        before, after, ncon = [], [], 0
        for _ in range(args.repeats):
            if args.before:
                before.append(run_one(args.before, n, case, args.seconds, args.warmup)["rate"])
            result = run_one(here, n, case, args.seconds, args.warmup)
            after.append(result["rate"])
            ncon = result["ncon"]
        ratios = np.array(after) / np.array(before) if before else np.array([np.nan])
        span = (f"{min(before):,.0f}-{max(before):,.0f}" if before else "-")
        print(f"| {n} machine{'s' if n > 1 else ''} {case} | {span} | "
              f"{min(after):,.0f}-{max(after):,.0f} | {np.median(ratios):.2f} "
              f"({ratios.min():.2f}-{ratios.max():.2f}) | {ncon} |", flush=True)


if __name__ == "__main__":
    main()
