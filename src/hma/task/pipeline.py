# From instruction to approved plan: cache or model, the checks, then the operator.
# A plan is cached only after checks one to five pass, and is checked again on every replay.
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from hma.task.body import Body
from hma.task.cache import PlanCache
from hma.task.checks import CheckReport, Plan, confirm_plan, validate
from hma.task.model import PlanSource
from hma.task.reach import Reach
from hma.task.scene import Scene


@dataclass
class Outcome:
    report: CheckReport
    from_cache: bool
    model_seconds: float | None


def plan_task(
    instruction: str,
    *,
    scene: Scene,
    source: PlanSource,
    cache: PlanCache,
    reach: Reach,
    body: Body,
    confirm: Callable[[Plan], bool],
    on_validated: Callable[[CheckReport, bool, float | None], None] | None = None,
) -> Outcome:
    raw = cache.get(instruction, scene)
    from_cache, seconds = raw is not None, None
    if raw is None:
        start = time.perf_counter()
        raw = source(instruction, scene)
        seconds = time.perf_counter() - start
    report = validate(raw, scene, reach, body)
    if report.validated and not from_cache:
        try:
            cache.put(instruction, scene, raw)
        except OSError:
            report.warnings.append("The plan was not cached: the cache folder cannot be written.")
    if on_validated is not None:
        on_validated(report, from_cache, seconds)
    return Outcome(confirm_plan(report, confirm), from_cache, seconds)
