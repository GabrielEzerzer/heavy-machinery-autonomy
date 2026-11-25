# The six checks between the model's output and the machine, in order; the first failure stops.
# Every limit lives in trusted code; neither the instruction nor the model's output can change it.
# Each limit's source is listed in the README under "Language step".
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

import numpy as np
from pydantic import ValidationError

from hma.task.body import BUCKET_KG, Body, rating_cells
from hma.task.reach import Reach
from hma.task.scene import (
    Ground,
    Hazard,
    HazardKind,
    Load,
    Marker,
    Point2,
    Pose,
    Scene,
    distance_to_segment,
)
from hma.task.spec import Status, TaskSpec, TaskType, printable, spec_model

Point = tuple[float, float, float]

USES = {
    TaskType.carry: ("load_id", "marker_id", "stand_marker_id"),
    TaskType.prop: ("hold_s",),
    TaskType.turn: ("marker_id",),
    TaskType.travel: ("marker_id",),
}
NEEDS = {
    TaskType.carry: ("load_id", "marker_id"),
    TaskType.prop: ("hold_s",),
    TaskType.turn: ("marker_id",),
    TaskType.travel: ("marker_id",),
}
OPTIONAL = ("load_id", "marker_id", "stand_marker_id", "hold_s")


@dataclass(frozen=True)
class Limits:
    tipping_fraction: float = 0.75
    hydraulic_fraction: float = 0.87
    lift_slope_deg: float = 0.6
    travel_pitch_deg: float = 30.0
    travel_roll_deg: float = 10.0
    swing_pitch_deg: float = 15.0
    swing_roll_deg: float = 10.0
    step_m: float = 0.3
    edge_keep_back_m: float = 0.61
    boom_arm_deg: tuple[float, float] = (90.0, 110.0)
    bucket_ground_slope_deg: float = 10.0
    hold_s: tuple[float, float] = (1.0, 60.0)
    carry_height_m: float = 0.3
    min_move_m: float = 0.5
    prop_radius_m: tuple[float, float] = (2.5, 8.5)
    sample_step_m: float = 0.25
    turn_step_deg: float = 5.0


LIMITS = Limits()


@dataclass(frozen=True)
class Plan:
    task_type: TaskType
    start: Pose
    stand: Pose
    load: Load | None = None
    target: Marker | None = None
    hold_s: float | None = None
    prop_xy: Point2 | None = None

    @property
    def travels(self) -> bool:
        return self.start.xy != self.stand.xy

    def describe(self) -> str:
        place = (
            f"after travelling {math.dist(self.start.xy, self.stand.xy):.1f} m to "
            f"({self.stand.x_m:g}, {self.stand.y_m:g})"
            if self.travels
            else "from where the machine stands"
        )
        if self.task_type is TaskType.carry:
            return (
                f"carry the {self.load.label} ({self.load.id}, {self.load.mass_kg:g} kg) to the "
                f"{self.target.label} ({self.target.id}), {place}"
            )
        if self.task_type is TaskType.prop:
            return f"prop the machine on its bucket, hold {self.hold_s:g} s, set down"
        if self.task_type is TaskType.turn:
            return (
                f"prop on the bucket, turn the tracks {turn_angle(self):.0f} degrees to face "
                f"the {self.target.label} ({self.target.id}), set down"
            )
        return (
            f"travel {math.dist(self.start.xy, self.stand.xy):.1f} m to the {self.target.label} "
            f"({self.target.id})"
        )


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str


@dataclass
class CheckReport:
    spec: TaskSpec | None = None
    plan: Plan | None = None
    results: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    approved: bool = False

    @property
    def validated(self) -> bool:
        return len(self.results) >= 5 and all(r.passed for r in self.results[:5])

    @property
    def refusal(self) -> CheckResult | None:
        return next((r for r in self.results if not r.passed), None)


@dataclass(frozen=True)
class Frame:
    origin: np.ndarray
    axes: np.ndarray

    def to_machine(self, p: Point) -> Point:
        local = self.axes.T @ (np.asarray(p, float) - self.origin)
        return (float(local[0]), float(local[1]), float(local[2]))


def bearing(a: Point2, b: Point2) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def turn_angle(plan: Plan) -> float:
    to = bearing(plan.stand.xy, plan.target.xy)
    return abs((to - plan.stand.heading_deg + 180.0) % 360.0 - 180.0)


def frame_at(scene: Scene, pose: Pose) -> Frame:
    ground = scene.ground(pose.xy)
    s, d = math.radians(ground.slope_deg), math.radians(ground.downhill_deg)
    normal = np.array([math.sin(s) * math.cos(d), math.sin(s) * math.sin(d), math.cos(s)])
    h = math.radians(pose.heading_deg)
    forward = np.array([math.cos(h), math.sin(h), 0.0])
    forward -= normal * float(forward @ normal)
    forward /= np.linalg.norm(forward)
    axes = np.column_stack([forward, np.cross(normal, forward), normal])
    return Frame(np.array([pose.x_m, pose.y_m, ground.height_m]), axes)


def slope_tilt(ground: Ground, heading_deg: float) -> tuple[float, float]:
    t = math.tan(math.radians(ground.slope_deg))
    rel = math.radians(ground.downhill_deg - heading_deg)
    return (abs(math.degrees(math.atan(t * math.cos(rel)))),
            abs(math.degrees(math.atan(t * math.sin(rel)))))


def surface(ground: Ground) -> tuple[float, float]:
    return (0.0, 0.0) if ground.slope_deg == 0 else (ground.slope_deg, ground.downhill_deg % 360)


def footprint_points(body: Body, pose: Pose, step: float) -> list[Point2]:
    x0, x1, y0, y1 = body.footprint()
    nx, ny = max(1, math.ceil((x1 - x0) / step)), max(1, math.ceil((y1 - y0) / step))
    c, s = math.cos(math.radians(pose.heading_deg)), math.sin(math.radians(pose.heading_deg))
    found = []
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        for j in range(ny + 1):
            y = y0 + (y1 - y0) * j / ny
            found.append((pose.x_m + c * x - s * y, pose.y_m + s * x + c * y))
    return found


def ahead(pose: Pose, distance: float) -> Point2:
    h = math.radians(pose.heading_deg)
    return (pose.x_m + distance * math.cos(h), pose.y_m + distance * math.sin(h))


def turn_headings(plan: Plan, limits: Limits = LIMITS) -> list[float]:
    start = plan.stand.heading_deg
    delta = (bearing(plan.stand.xy, plan.target.xy) - start + 180.0) % 360.0 - 180.0
    n = max(1, math.ceil(abs(delta) / limits.turn_step_deg))
    return [start + delta * k / n for k in range(n + 1)]


def segment_gap(a: Point2, b: Point2, c: Point2, d: Point2) -> float:
    def side(p: Point2, q: Point2, r: Point2) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    if side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0:
        return 0.0
    return min(distance_to_segment(a, c, d), distance_to_segment(b, c, d),
               distance_to_segment(c, a, b), distance_to_segment(d, a, b))


def hazard_gap(hazard: Hazard, a: Point2, b: Point2 | None = None) -> float:
    return segment_gap(a, b if b is not None else a, hazard.points[0], hazard.points[-1])


def where(p: Point2) -> str:
    return f"({p[0]:.1f}, {p[1]:.1f})"


def check_schema(raw: str, scene: Scene) -> tuple[CheckResult, TaskSpec | None]:
    model = spec_model(scene)
    try:
        spec = model.model_validate_json(raw)
    except ValidationError as err:
        first = err.errors()[0]
        loc = first["loc"]
        if not loc:
            place = "output"
        elif loc[0] in model.model_fields:
            place = loc[0]
        else:
            place = "an unexpected field"
        msg = printable(first["msg"])[:200]
        return CheckResult("schema", False, f"Model output rejected: {place}: {msg}."), None
    if spec.status is not Status.plan:
        return CheckResult(
            "schema",
            False,
            f"The model returned no plan ({spec.status.value}): {spec.message}",
        ), spec
    return CheckResult("schema", True, "Output matches the specification."), spec


def check_fields(
    spec: TaskSpec, scene: Scene, limits: Limits = LIMITS
) -> tuple[CheckResult, Plan | None]:
    name = "fields and ranges"

    def refuse(reason: str) -> tuple[CheckResult, None]:
        return CheckResult(name, False, reason), None

    if spec.task_type is None:
        return refuse("The plan names no task type.")
    kind = spec.task_type
    for key in NEEDS[kind]:
        if getattr(spec, key) is None:
            return refuse(f"A {kind.value} task needs {key}, and the plan gives none.")
    for key in OPTIONAL:
        if key not in USES[kind] and getattr(spec, key) is not None:
            return refuse(f"A {kind.value} task takes no {key}, and the plan gives one.")
    start = scene.machine
    target = scene.marker(spec.marker_id) if spec.marker_id is not None else None
    stand = start
    if kind is TaskType.travel or spec.stand_marker_id is not None:
        goal = target if kind is TaskType.travel else scene.marker(spec.stand_marker_id)
        if math.dist(start.xy, goal.xy) < limits.min_move_m:
            if kind is TaskType.travel:
                return refuse(f"The machine already stands at the {goal.label} ({goal.id}).")
        else:
            stand = Pose(goal.x_m, goal.y_m, bearing(start.xy, goal.xy))
    if kind is TaskType.turn and math.dist(start.xy, target.xy) < limits.min_move_m:
        return refuse(f"The machine stands on the {target.label} ({target.id}); it cannot face it.")
    load = scene.load(spec.load_id) if spec.load_id is not None else None
    plan = Plan(kind, start, stand, load, target, spec.hold_s)
    return CheckResult(name, True, "Task type, names and numbers are inside limits."), plan


def find_prop_point(
    plan: Plan, scene: Scene, body: Body, limits: Limits = LIMITS
) -> Point2 | None:
    frame = frame_at(scene, plan.stand)
    h = math.radians(plan.stand.heading_deg)
    lo, hi = limits.boom_arm_deg
    centre = (lo + hi) / 2
    best: tuple[float, Point2] | None = None
    r = limits.prop_radius_m[0]
    while r <= limits.prop_radius_m[1] + 1e-9:
        xy = (plan.stand.x_m + r * math.cos(h), plan.stand.y_m + r * math.sin(h))
        local = frame.to_machine((*xy, scene.ground(xy).height_m))
        for angle in body.boom_arm_angles(*local):
            if lo <= angle <= hi and (best is None or abs(angle - centre) < best[0]):
                best = (abs(angle - centre), xy)
        r += 0.1
    return best[1] if best else None


def check_reach(
    plan: Plan, scene: Scene, reach: Reach, body: Body, limits: Limits = LIMITS
) -> tuple[CheckResult, Plan]:
    name = "reach"
    if plan.task_type is TaskType.travel:
        return CheckResult(name, True, "Travel moves nothing with the arm."), plan
    if plan.task_type is TaskType.carry:
        frame = frame_at(scene, plan.stand)
        for what, xy in (("load", plan.load.xy), ("set-down point", plan.target.xy)):
            local = frame.to_machine((*xy, scene.ground(xy).height_m))
            if not reach.reachable(*local):
                return CheckResult(
                    name,
                    False,
                    f"The {what} at {where(xy)} is {math.hypot(local[0], local[1]):.1f} m from "
                    "where the machine stands and out of the bucket's reach.",
                ), plan
        return CheckResult(name, True, "The load and the set-down point are within reach."), plan
    prop = find_prop_point(plan, scene, body, limits)
    if prop is None:
        lo, hi = limits.boom_arm_deg
        return CheckResult(
            name,
            False,
            "The bucket cannot rest on the ground ahead of the tracks with the angle between "
            f"boom and arm at {lo:g} to {hi:g} degrees.",
        ), plan
    return CheckResult(
        name, True, f"The bucket can prop the machine at {where(prop)}."
    ), replace(plan, prop_xy=prop)


def machine_problem(
    scene: Scene, body: Body, pose: Pose, activity: str, limits: Limits = LIMITS
) -> str | None:
    pitch_max, roll_max = {
        "travel": (limits.travel_pitch_deg, limits.travel_roll_deg),
        "lifting": (limits.lift_slope_deg, limits.lift_slope_deg),
        "propping": (limits.swing_pitch_deg, limits.swing_roll_deg),
    }[activity]
    grounds = [scene.ground(q) for q in footprint_points(body, pose, limits.sample_step_m)]
    tilts = [slope_tilt(g, pose.heading_deg) for g in grounds]
    pitch, roll = max(t[0] for t in tilts), max(t[1] for t in tilts)
    if pitch > pitch_max + 1e-9 or roll > roll_max + 1e-9:
        along = pitch > pitch_max + 1e-9
        return (
            f"At {where(pose.xy)} the ground under the tracks tilts the machine "
            f"{pitch if along else roll:.1f} degrees {'along' if along else 'across'} them; the "
            f"limit for {activity} is {pitch_max if along else roll_max:g} degrees."
        )
    if activity != "travel" and len({surface(g) for g in grounds}) > 1:
        return (
            f"At {where(pose.xy)} the tracks rest on ground of more than one slope; {activity} "
            "needs one even surface under them."
        )
    radius = body.plan_radius()
    arm_end = ahead(pose, body.travel_reach(limits.carry_height_m))
    for hazard in scene.hazards:
        if hazard.kind is not HazardKind.overhead:
            continue
        what, gap = "machine comes", hazard_gap(hazard, pose.xy) - radius
        if activity == "travel" and hazard_gap(hazard, pose.xy, arm_end) < gap:
            what, gap = "arm, held ahead for travel, comes", hazard_gap(hazard, pose.xy, arm_end)
        if gap < hazard.keep_out_m:
            return (
                f"At {where(pose.xy)} the {what} within {max(gap, 0):.1f} m of the "
                f"{hazard.label} ({hazard.id}); its keep-out distance is {hazard.keep_out_m:g} m."
            )
    for step in scene.steps:
        gap = step.distance(pose.xy) - radius
        if step.height_m > limits.step_m and gap < limits.edge_keep_back_m:
            return (
                f"At {where(pose.xy)} the machine is {max(gap, 0):.1f} m from a "
                f"{step.height_m:.2f} m step or edge; it must stay {limits.edge_keep_back_m:g} m "
                f"back from any step over {limits.step_m:g} m."
            )
        if activity != "travel" and gap < 0:
            return (
                f"At {where(pose.xy)} the ground under the machine is not one surface: a "
                f"{step.height_m:.2f} m step lies under it; {activity} needs even ground."
            )
    return None


def lift_problem(
    plan: Plan, scene: Scene, body: Body, point: Point, on_ground: bool, limits: Limits = LIMITS
) -> str | None:
    xy = (point[0], point[1])
    size = max(plan.load.radius_m, scene.bucket_width_m / 2)
    for hazard in scene.hazards:
        if hazard.kind is HazardKind.overhead:
            gap = min(hazard_gap(hazard, xy) - size, hazard_gap(hazard, plan.stand.xy, xy))
        elif on_ground:
            gap = hazard_gap(hazard, xy) - size
        else:
            continue
        if gap < hazard.keep_out_m:
            return (
                f"At {where(xy)} the bucket, arm or load comes within {max(gap, 0):.1f} m of the "
                f"{hazard.label} ({hazard.id}); its keep-out distance is {hazard.keep_out_m:g} m."
            )
    local = frame_at(scene, plan.stand).to_machine(point)
    radius = math.hypot(local[0], local[1])
    cells = rating_cells(radius)
    if cells is None:
        return (
            f"At {where(xy)}, {radius:.1f} m from the swing axis, the manufacturer's lifting "
            "chart does not rate every height, so nothing is lifted there."
        )
    limit, governs = math.inf, ""
    for cell in cells:
        at = f"at {cell.radius_m:g} m radius and {cell.height_m:g} m height"
        chart = f"the manufacturer's over-side rating of {cell.side_kg:.0f} kg {at}"
        options = [(cell.side_kg, chart)]
        model = body.lift_capacity(cell.radius_m, cell.height_m)
        if model is not None:
            options.append((limits.tipping_fraction * model.tipping_kg,
                            f"{limits.tipping_fraction:.0%} of the model's "
                            f"{model.tipping_kg:.0f} kg tipping load {at}"))
            options.append((limits.hydraulic_fraction * model.hydraulic_kg,
                            f"{limits.hydraulic_fraction:.0%} of the model's "
                            f"{model.hydraulic_kg:.0f} kg hydraulic lift capacity {at}"))
        for value, text in options:
            if value - BUCKET_KG < limit:
                limit, governs = value - BUCKET_KG, text
    if plan.load.mass_kg > limit:
        return (
            f"The {plan.load.label} ({plan.load.mass_kg:g} kg) is over the rated lift capacity of "
            f"{max(limit, 0):.0f} kg at {radius:.1f} m from the swing axis: {governs}, less the "
            f"{BUCKET_KG:.0f} kg bucket."
        )
    return None


def ground_point(scene: Scene, xy: Point2, lift: float = 0.0) -> Point:
    return (xy[0], xy[1], scene.ground(xy).height_m + lift)


def check_rules(
    plan: Plan, scene: Scene, body: Body, limits: Limits = LIMITS
) -> tuple[CheckResult, list[str]]:
    name = "safety rules"
    warnings: list[str] = []

    def refuse(reason: str) -> tuple[CheckResult, list[str]]:
        return CheckResult(name, False, reason), warnings

    if plan.travels:
        route_heading = Pose(plan.start.x_m, plan.start.y_m, plan.stand.heading_deg)
        for pose in (route_heading, plan.stand):
            if problem := machine_problem(scene, body, pose, "travel", limits):
                return refuse(problem)
    if plan.task_type is TaskType.travel:
        return CheckResult(name, True, "Start and end of the route are inside travel limits."), []
    activity = "lifting" if plan.task_type is TaskType.carry else "propping"
    if problem := machine_problem(scene, body, plan.stand, activity, limits):
        return refuse(problem)
    if plan.task_type is TaskType.turn:
        end = replace(plan.stand, heading_deg=turn_headings(plan, limits)[-1])
        if problem := machine_problem(scene, body, end, "propping", limits):
            return refuse(problem)
    if plan.task_type is TaskType.carry:
        if plan.load.mass_kg is None:
            return refuse(
                f"The mass of the {plan.load.label} ({plan.load.id}) is not known; a load is not "
                "lifted until its mass is known."
            )
        for xy in (plan.load.xy, plan.target.xy):
            if problem := lift_problem(plan, scene, body, ground_point(scene, xy), True, limits):
                return refuse(problem)
        warnings.append(
            "Keep people out of the working area; no load passes over a person."
        )
        return CheckResult(
            name, True, "The load is within the rated lift capacity at both ends, on level ground."
        ), warnings
    if plan.task_type is TaskType.prop:
        lo, hi = limits.hold_s
        if not lo <= plan.hold_s <= hi:
            return refuse(f"A hold of {plan.hold_s:g} s is outside {lo:g} to {hi:g} s.")
    prop = plan.prop_xy
    ground = scene.ground(prop)
    if ground.slope_deg > limits.bucket_ground_slope_deg:
        return refuse(
            f"The ground under the bucket at {where(prop)} slopes {ground.slope_deg:g} degrees; "
            f"the limit is {limits.bucket_ground_slope_deg:g} degrees."
        )
    for step in scene.steps:
        if step.distance(prop) < scene.bucket_width_m / 2:
            return refuse(
                f"A {step.height_m:.2f} m step lies under the bucket at {where(prop)}; the bucket "
                "must rest flat on one surface."
            )
    for hazard in scene.hazards:
        gap = (
            hazard_gap(hazard, plan.stand.xy, prop)
            if hazard.kind is HazardKind.overhead
            else hazard_gap(hazard, prop) - scene.bucket_width_m / 2
        )
        if gap < hazard.keep_out_m:
            return refuse(
                f"The bucket pressing at {where(prop)} comes within {max(gap, 0):.1f} m of the "
                f"{hazard.label} ({hazard.id}); its keep-out distance is {hazard.keep_out_m:g} m."
            )
    return CheckResult(
        name, True, "Ground and slope under the machine and the bucket are inside propping limits."
    ), warnings


def spaced(a: Point, b: Point, step: float) -> list[Point]:
    n = max(1, math.ceil(math.dist(a, b) / step))
    return [tuple(a[i] + (b[i] - a[i]) * k / n for i in range(3)) for k in range(n + 1)]


def travel_samples(plan: Plan, limits: Limits = LIMITS) -> list[Pose]:
    a, b = (*plan.start.xy, 0.0), (*plan.stand.xy, 0.0)
    return [Pose(p[0], p[1], plan.stand.heading_deg) for p in spaced(a, b, limits.sample_step_m)]


def lift_samples(plan: Plan, scene: Scene, limits: Limits = LIMITS) -> list[Point]:
    pick = ground_point(scene, plan.load.xy)
    drop = ground_point(scene, plan.target.xy)
    up = (pick[0], pick[1], pick[2] + limits.carry_height_m)
    over = (drop[0], drop[1], drop[2] + limits.carry_height_m)
    step = limits.sample_step_m
    return spaced(pick, up, step) + spaced(up, over, step)[1:] + spaced(over, drop, step)[1:]


def sweep(
    plan: Plan, scene: Scene, reach: Reach, body: Body, points: Iterable[Point],
    limits: Limits = LIMITS,
) -> str | None:
    frame = frame_at(scene, plan.stand)
    for p in points:
        xy = (p[0], p[1])
        if p[2] < scene.ground(xy).height_m - 1e-9:
            return f"At {where(xy)} the load's route passes {p[2]:.2f} m, below the ground."
        if not reach.reachable(*frame.to_machine(p)):
            return f"At {where(xy)}, {p[2]:.2f} m high, the load's route is out of reach."
        if problem := lift_problem(plan, scene, body, p, False, limits):
            return problem
    return None


def check_path(
    plan: Plan, scene: Scene, reach: Reach, body: Body, points: Iterable[Point] | None = None,
    limits: Limits = LIMITS,
) -> CheckResult:
    start = time.perf_counter()
    count = 0
    if plan.travels:
        poses = travel_samples(plan, limits)
        count += len(poses)
        for pose in poses:
            if problem := machine_problem(scene, body, pose, "travel", limits):
                return CheckResult("path", False, problem)
    if plan.task_type is TaskType.carry:
        points = list(lift_samples(plan, scene, limits) if points is None else points)
        count += len(points)
        if problem := sweep(plan, scene, reach, body, points, limits):
            return CheckResult("path", False, problem)
    if plan.task_type is TaskType.turn:
        headings = turn_headings(plan, limits)
        count += len(headings)
        for heading in headings:
            pose = replace(plan.stand, heading_deg=heading)
            if problem := machine_problem(scene, body, pose, "propping", limits):
                return CheckResult("path", False, f"Turning, at {heading:.0f} degrees: {problem}")
    elapsed = time.perf_counter() - start
    if count == 0:
        return CheckResult("path", True, "The task has no route to sample.")
    return CheckResult(
        "path", True, f"{count} samples along the route pass every check ({elapsed * 1000:.0f} ms)."
    )


def validate(raw: str, scene: Scene, reach: Reach, body: Body) -> CheckReport:
    report = CheckReport()
    result, spec = check_schema(raw, scene)
    report.spec = spec
    report.results.append(result)
    if not result.passed:
        return report
    result, plan = check_fields(spec, scene)
    report.results.append(result)
    if plan is None:
        return report
    result, plan = check_reach(plan, scene, reach, body)
    report.plan = plan
    report.results.append(result)
    if not result.passed:
        return report
    result, warnings = check_rules(plan, scene, body)
    report.results.append(result)
    report.warnings.extend(warnings)
    if not result.passed:
        return report
    report.results.append(check_path(plan, scene, reach, body))
    return report


def confirm_plan(report: CheckReport, confirm: Callable[[Plan], bool]) -> CheckReport:
    if not report.validated:
        return report
    agreed = confirm(report.plan) is True
    reason = (
        "The operator approved the plan." if agreed else "The operator did not approve the plan."
    )
    report.results.append(CheckResult("operator confirmation", agreed, reason))
    report.approved = agreed
    return report


def run_checks(
    raw: str, scene: Scene, reach: Reach, body: Body, confirm: Callable[[Plan], bool]
) -> CheckReport:
    return confirm_plan(validate(raw, scene, reach, body), confirm)
