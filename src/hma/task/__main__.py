# Command line: python -m hma.task "put the pallet of kerbstones at the near marker" [--run]
# Prints the specification and every check, asks the operator, then prints the plan or the refusal.
# With --run an approved prop plan becomes a propping task shown in the viewer, baseline beside it.
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable

from hma.task.body import MachineBody
from hma.task.cache import PlanCache
from hma.task.checks import CheckReport, Plan
from hma.task.model import CodexCliSource, ModelUnavailable, PlanSource
from hma.task.pipeline import plan_task
from hma.task.reach import MachineArm
from hma.task.scene import DEMO_SCENE


def ask_operator(plan: Plan) -> bool:
    print(f"\nProposed: {plan.describe()}")
    try:
        answer = input("Approve this plan? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def print_report(report: CheckReport, from_cache: bool, seconds: float | None) -> None:
    print("Source: cache" if from_cache else f"Source: model ({seconds:.1f} s)")
    if report.spec is not None:
        print("Specification:")
        for key, value in report.spec.model_dump(mode="json").items():
            print(f"  {key}: {value}")
    print("Checks:")
    for number, result in enumerate(report.results, start=1):
        print(f"  {number}. {result.name}: {'pass' if result.passed else 'FAIL'} - {result.reason}")
    for warning in report.warnings:
        print(f"Warning: {warning}")


def run_plan(plan: Plan, args: argparse.Namespace, show: Callable | None) -> int:
    from hma.task.prop import CannotRun, prop_task
    from hma.train.play import watch_checkpoint
    from hma.train.task import HELD_OUT_BASE, dev_seeds

    seed = dev_seeds(1)[0] if args.machine_seed is None else args.machine_seed
    if seed >= HELD_OUT_BASE:
        print("CANNOT RUN: held-out seeds are never played; give a development seed.")
        return 3
    try:
        task = prop_task(plan, DEMO_SCENE, seed)
    except CannotRun as err:
        print(f"CANNOT RUN: {err}")
        return 3
    print(
        f"Running: the bucket presses {task.prop_x:.2f} m ahead; tilt "
        f"{math.degrees(task.target):g} degrees, ramp up {task.ramp_up:g} s, hold {task.hold:g} s, "
        f"ramp down {task.ramp_down:g} s; {args.rung} rung, machine seed {seed}; "
        f"baseline beside {args.checkpoint or 'the baseline'}."
    )
    outcomes = (show or watch_checkpoint)(args.rung, seed, args.checkpoint, task)
    if None in outcomes:
        print("The viewer was closed before the episode ended.")
    return 0


def main(
    argv: list[str] | None = None,
    source: PlanSource | None = None,
    confirm: Callable[[Plan], bool] = ask_operator,
    show: Callable | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m hma.task")
    parser.add_argument("instruction", help="plain-English instruction")
    parser.add_argument("--cache", help="folder of cached plans (default: a user cache folder)")
    parser.add_argument("--model", help="model name for the Codex CLI (default: its own setting)")
    parser.add_argument("--run", action="store_true",
                        help="after approval, show the plan in the viewer (propping only)")
    parser.add_argument("--checkpoint",
                        help="with --run: a trained propping checkpoint (default: baseline twice)")
    parser.add_argument("--rung", choices=("nominal", "machines"), default="nominal",
                        help="with --run: one nominal machine, or one drawn from --machine-seed")
    parser.add_argument("--machine-seed", type=int,
                        help="with --run: a development seed (default: the first one)")
    args = parser.parse_args(argv)

    print(f"Instruction: {args.instruction}")
    try:
        outcome = plan_task(
            args.instruction,
            scene=DEMO_SCENE,
            source=source or CodexCliSource(model=args.model),
            cache=PlanCache(args.cache),
            reach=MachineArm(),
            body=MachineBody(),
            confirm=confirm,
            on_validated=print_report,
        )
    except (ModelUnavailable, ValueError) as err:
        print(f"No plan: {err}")
        return 2
    report = outcome.report
    if len(report.results) == 6:
        result = report.results[-1]
        print(f"  6. {result.name}: {'pass' if result.passed else 'FAIL'} - {result.reason}")
    if report.approved:
        print(f"APPROVED: {report.plan.describe()}")
        return run_plan(report.plan, args, show) if args.run else 0
    refusal = report.refusal
    print(f"REFUSED at {refusal.name}: {refusal.reason}" if refusal else "REFUSED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
