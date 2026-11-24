# Command line: python -m hma.task "dig 1 m down at the far marker"
# Prints the specification and every check, asks the operator, then prints the plan or the refusal.
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

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


def main(
    argv: list[str] | None = None,
    source: PlanSource | None = None,
    confirm: Callable[[Plan], bool] = ask_operator,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m hma.task")
    parser.add_argument("instruction", help="plain-English digging instruction")
    parser.add_argument("--cache", help="folder of cached plans (default: a user cache folder)")
    parser.add_argument("--model", help="model name for the Codex CLI (default: its own setting)")
    args = parser.parse_args(argv)

    print(f"Instruction: {args.instruction}")
    try:
        outcome = plan_task(
            args.instruction,
            scene=DEMO_SCENE,
            source=source or CodexCliSource(model=args.model),
            cache=PlanCache(args.cache),
            reach=MachineArm(),
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
        return 0
    refusal = report.refusal
    print(f"REFUSED at {refusal.name}: {refusal.reason}" if refusal else "REFUSED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
