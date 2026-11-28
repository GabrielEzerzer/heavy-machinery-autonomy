# From an approved prop plan to the propping environment's task, or a plain reason it cannot run.
# The plan sets the prop distance and the hold; tilt, ramps, bucket angle and starting height are
# fixed values from the middle of the ranges the propping controller was trained on.
from __future__ import annotations

import math

from hma.task.checks import Plan, where
from hma.task.scene import Scene
from hma.task.spec import TaskType
from hma.train import footing
from hma.train.scene import CONTROL_HZ
from hma.train.task import PropPath, Settings, Task, reachable

TARGET = math.radians(12.5)
RAMP_S = 4.5
RAMP_DOWN_S = 8.0
PITCH = math.radians(75.0)
LIFT_M = 1.0
LEVEL_M = 1e-9


class CannotRun(ValueError):
    pass


def prop_task(plan: Plan, scene: Scene, seed: int, settings: Settings | None = None) -> Task:
    s = settings or Settings()
    if plan.task_type is not TaskType.prop:
        raise CannotRun(
            f"Only propping can run in the simulator so far; this plan is a "
            f"{plan.task_type.value} task."
        )
    if plan.travels:
        raise CannotRun(
            f"The plan drives the machine {math.dist(plan.start.xy, plan.stand.xy):.1f} m first, "
            "and there is no controller for driving yet; only propping from where the machine "
            "stands can run."
        )
    lo, hi = s.hold
    if not lo <= plan.hold_s <= hi:
        raise CannotRun(
            f"A hold of {plan.hold_s:g} s is outside the {lo:g} to {hi:g} s the propping "
            "controller was trained on."
        )
    distance = math.dist(plan.stand.xy, plan.prop_xy)
    lo, hi = s.prop_x
    if not lo <= distance <= hi:
        raise CannotRun(
            f"The bucket would press {distance:.2f} m ahead of the machine; the propping "
            f"controller was trained on {lo:g} to {hi:g} m."
        )
    under = scene.ground(plan.stand.xy)
    for what, xy in (("the machine", plan.stand.xy), ("the bucket", plan.prop_xy)):
        ground = scene.ground(xy)
        if ground.slope_deg > 0:
            raise CannotRun(
                f"The ground under {what} at {where(xy)} slopes {ground.slope_deg:g} degrees; the "
                "propping controller has only been trained on level ground."
            )
        if abs(ground.height_m - under.height_m) > LEVEL_M:
            raise CannotRun(
                f"The ground under the bucket at {where(xy)} is "
                f"{abs(ground.height_m - under.height_m):.2f} m "
                f"{'higher' if ground.height_m > under.height_m else 'lower'} than under the "
                "machine; the propping controller has only been trained on level ground."
            )
    task = Task(seed=seed, target=TARGET, ramp_up=RAMP_S, hold=plan.hold_s,
                ramp_down=RAMP_DOWN_S, prop_x=distance, start_z=LIFT_M, pitch=PITCH, attempts=1,
                rest=s.rest, place=s.place, wait=s.wait, final=s.final)
    if not reachable(PropPath(task, footing.rear_pivot()), 1.0 / CONTROL_HZ):
        raise CannotRun(
            f"The arm cannot follow the propping path to the point {distance:.2f} m ahead with "
            f"the bucket at {math.degrees(PITCH):g} degrees and starting "
            f"{LIFT_M:g} m above the ground."
        )
    return task
