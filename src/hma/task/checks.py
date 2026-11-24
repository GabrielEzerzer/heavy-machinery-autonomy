# The six checks between the model's output and the machine, in order; the first failure stops.
# Every limit lives in trusted code; neither the instruction nor the model's output can change it.
# Trench rules follow 29 CFR 1926 Subpart P (1926.651(j)(2), 1926.652(a)(1), Appendix F).
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

import numpy as np
from pydantic import ValidationError

from hma.task.reach import Reach
from hma.task.scene import Scene
from hma.task.spec import (
    Heading,
    SoilClass,
    SpoilSide,
    Status,
    TaskSpec,
    printable,
    spec_model,
)

Point = tuple[float, float, float]


@dataclass(frozen=True)
class Limits:
    depth_m: tuple[float, float] = (0.1, 10.0)
    length_m: tuple[float, float] = (0.5, 10.0)
    width_max_m: float = 3.0
    machine_clearance_m: float = 2.0
    default_length_m: float = 2.0
    default_heading: Heading = Heading.toward_machine
    default_spoil_side: SpoilSide = SpoilSide.right
    spoil_gap_m: float = 1.0
    spoil_width_m: float = 1.5
    approach_heights_m: tuple[float, ...] = (0.5, 1.0)
    sample_step_m: float = 0.25


LIMITS = Limits()
NO_ENTRY_DEPTH_M = 1.52
SPOIL_MIN_GAP_M = 0.61
ENGINEER_DESIGN_DEPTH_M = 6.1


@dataclass(frozen=True)
class Plan:
    marker_id: str
    marker_label: str
    depth_m: float
    length_m: float
    width_m: float
    heading: Heading
    spoil_side: SpoilSide
    soil_class: SoilClass
    centre: tuple[float, float]
    bearing_deg: float
    defaults: tuple[str, ...] = ()
    no_entry: bool = False

    @property
    def half_extents(self) -> tuple[float, float]:
        if self.heading is Heading.toward_machine:
            return (self.length_m / 2, self.width_m / 2)
        return (self.width_m / 2, self.length_m / 2)

    def spoil_band(self, gap: float, width: float) -> tuple[float, float]:
        hv = self.half_extents[1]
        near, far = hv + gap, hv + gap + width
        return (near, far) if self.spoil_side is SpoilSide.left else (-far, -near)

    def to_world(self, u: float, v: float, z: float = 0.0) -> Point:
        b = math.radians(self.bearing_deg)
        cx, cy = self.centre
        return (cx + u * math.cos(b) - v * math.sin(b), cy + u * math.sin(b) + v * math.cos(b), z)

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        b = math.radians(self.bearing_deg)
        dx, dy = x - self.centre[0], y - self.centre[1]
        return (dx * math.cos(b) + dy * math.sin(b), -dx * math.sin(b) + dy * math.cos(b))

    def corners(self, z: float = 0.0) -> list[Point]:
        hu, hv = self.half_extents
        return [self.to_world(su * hu, sv * hv, z) for su in (-1, 1) for sv in (-1, 1)]

    def describe(self) -> str:
        return (
            f"trench {self.depth_m:g} m deep, {self.length_m:g} m long, "
            f"{self.width_m:g} m wide at the {self.marker_label} ({self.marker_id}), "
            f"heading {self.heading.value}, spoil {self.spoil_side.value}, "
            f"soil {self.soil_class.value}{', NO ENTRY' if self.no_entry else ''}"
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


def _rect_gap(
    plan: Plan, x: float, y: float, u_band: tuple[float, float], v_band: tuple[float, float]
) -> float:
    u, v = plan.to_local(x, y)
    du = max(u_band[0] - u, u - u_band[1], 0.0)
    dv = max(v_band[0] - v, v - v_band[1], 0.0)
    return math.hypot(du, dv)


def check_schema(raw: str, scene: Scene) -> tuple[CheckResult, TaskSpec | None]:
    model = spec_model(scene)
    try:
        spec = model.model_validate_json(raw)
    except ValidationError as err:
        first = err.errors()[0]
        loc = first["loc"]
        if not loc:
            where = "output"
        elif loc[0] in model.model_fields:
            where = loc[0]
        else:
            where = "an unexpected field"
        msg = printable(first["msg"])[:200]
        return CheckResult("schema", False, f"Model output rejected: {where}: {msg}."), None
    if spec.status is not Status.plan:
        return CheckResult(
            "schema",
            False,
            f"The model returned no plan ({spec.status.value}): {spec.message}",
        ), spec
    return CheckResult("schema", True, "Output matches the specification."), spec


def check_ranges(
    spec: TaskSpec, scene: Scene, limits: Limits = LIMITS
) -> tuple[CheckResult, Plan | None]:
    name = "ranges and geofence"
    if spec.task_type is None:
        return CheckResult(name, False, "The plan names no task type."), None
    marker = scene.marker(spec.marker_id) if spec.marker_id is not None else None
    if marker is None:
        return CheckResult(name, False, "The plan names no marker that exists in the scene."), None
    if spec.depth_m is None:
        return CheckResult(name, False, "The plan gives no depth."), None

    defaults = []
    length = spec.length_m
    if length is None:
        length = limits.default_length_m
        defaults.append(f"length {length:g} m")
    width = spec.width_m
    if width is None:
        width = scene.bucket_width_m
        defaults.append(f"width {width:g} m (one bucket)")
    heading = spec.heading
    if heading is None:
        heading = limits.default_heading
        defaults.append(f"heading {heading.value}")
    spoil = spec.spoil_side
    if spoil is None:
        spoil = limits.default_spoil_side
        defaults.append(f"spoil {spoil.value}")

    lo, hi = limits.depth_m
    if not lo <= spec.depth_m <= hi:
        return CheckResult(
            name, False, f"Depth {spec.depth_m:g} m is outside {lo:g} to {hi:g} m."
        ), None
    lo, hi = limits.length_m
    if not lo <= length <= hi:
        return CheckResult(name, False, f"Length {length:g} m is outside {lo:g} to {hi:g} m."), None
    if width < scene.bucket_width_m:
        return CheckResult(
            name,
            False,
            f"Width {width:g} m is narrower than the {scene.bucket_width_m:g} m bucket.",
        ), None
    if width > limits.width_max_m:
        return CheckResult(
            name,
            False,
            f"Width {width:g} m is wider than {limits.width_max_m:g} m.",
        ), None

    plan = Plan(
        marker_id=marker.id,
        marker_label=marker.label,
        depth_m=spec.depth_m,
        length_m=length,
        width_m=width,
        heading=heading,
        spoil_side=spoil,
        soil_class=spec.soil_class or SoilClass.unknown,
        centre=marker.xy,
        bearing_deg=marker.bearing_deg,
        defaults=tuple(defaults),
    )
    hu, hv = plan.half_extents
    if _rect_gap(plan, 0.0, 0.0, (-hu, hu), (-hv, hv)) < limits.machine_clearance_m:
        return CheckResult(
            name,
            False,
            f"The trench comes within {limits.machine_clearance_m:g} m of the swing axis.",
        ), None
    for hazard in scene.hazards:
        if _rect_gap(plan, *hazard.xy, (-hu, hu), (-hv, hv)) < hazard.keep_out_m:
            return CheckResult(
                name,
                False,
                f"The trench enters the {hazard.keep_out_m:g} m keep-out zone around the "
                f"{hazard.label} ({hazard.id}).",
            ), None
    note = f" Defaults applied: {', '.join(defaults)}." if defaults else ""
    return CheckResult(name, True, "Marker, sizes and footprint are inside limits." + note), plan


def deepest_reachable(reach: Reach, x: float, y: float, floor: float = -12.0) -> float | None:
    if not reach.reachable(x, y, 0.0):
        return None
    if reach.reachable(x, y, floor):
        return floor
    top, bottom = 0.0, floor
    while top - bottom > 0.05:
        mid = (top + bottom) / 2
        top, bottom = (mid, bottom) if reach.reachable(x, y, mid) else (top, mid)
    return top


def check_reach(plan: Plan, reach: Reach) -> CheckResult:
    for corner in plan.corners(-plan.depth_m):
        if not reach.reachable(*corner):
            radius = math.hypot(corner[0], corner[1])
            deepest = deepest_reachable(reach, corner[0], corner[1])
            limit = (
                f"the bucket reaches about {-deepest:.1f} m deep there"
                if deepest is not None
                else "the bucket cannot reach that far out"
            )
            return CheckResult(
                "reach",
                False,
                f"A trench corner {plan.depth_m:g} m deep, {radius:.1f} m from the machine, "
                f"is out of reach: {limit}.",
            )
    return CheckResult("reach", True, "All four trench corners are reachable at full depth.")


def check_trench_rules(
    plan: Plan, scene: Scene, limits: Limits = LIMITS
) -> tuple[CheckResult, Plan, list[str]]:
    name = "trench rules"
    warnings: list[str] = []
    if plan.depth_m > ENGINEER_DESIGN_DEPTH_M:
        return (
            CheckResult(
                name,
                False,
                f"Depth {plan.depth_m:g} m is over {ENGINEER_DESIGN_DEPTH_M:g} m (20 ft): "
                "protection must be designed by a registered professional engineer "
                "(29 CFR 1926 Subpart P, Appendix F).",
            ),
            plan,
            warnings,
        )
    if limits.spoil_gap_m < SPOIL_MIN_GAP_M:
        return (
            CheckResult(
                name,
                False,
                f"Spoil would sit {limits.spoil_gap_m:g} m from the edge; the minimum is "
                f"{SPOIL_MIN_GAP_M:g} m (2 ft) (29 CFR 1926.651(j)(2)).",
            ),
            plan,
            warnings,
        )
    hu, _ = plan.half_extents
    band = plan.spoil_band(limits.spoil_gap_m, limits.spoil_width_m)
    obstacles = [(0.0, 0.0, limits.machine_clearance_m, "machine")]
    obstacles += [(*h.xy, h.keep_out_m, f"{h.label} ({h.id})") for h in scene.hazards]
    for x, y, radius, label in obstacles:
        if _rect_gap(plan, x, y, (-hu, hu), band) < radius:
            return (
                CheckResult(
                    name,
                    False,
                    f"Spoil on the {plan.spoil_side.value} would land inside the "
                    f"{radius:g} m keep-out zone around the {label}.",
                ),
                plan,
                warnings,
            )
    if plan.depth_m >= NO_ENTRY_DEPTH_M:
        plan = replace(plan, no_entry=True)
        warnings.append(
            f"NO ENTRY: at {plan.depth_m:g} m the trench is {NO_ENTRY_DEPTH_M:g} m (5 ft) "
            "or deeper; "
            "nobody enters without a protective system (29 CFR 1926.652(a)(1))."
        )
    if plan.soil_class is SoilClass.unknown:
        warnings.append(
            "Soil class not given: treat it as Type C until a competent person classifies it "
            "(29 CFR 1926 Subpart P, Appendix A)."
        )
    return (
        CheckResult(
            name,
            True,
            f"Spoil {limits.spoil_gap_m:g} m from the edge (minimum {SPOIL_MIN_GAP_M:g} m); "
            "depth under "
            f"{ENGINEER_DESIGN_DEPTH_M:g} m.",
        ),
        plan,
        warnings,
    )


def sample_points(plan: Plan, limits: Limits = LIMITS) -> list[Point]:
    step = limits.sample_step_m
    hu, hv = plan.half_extents

    def axis(lo: float, hi: float) -> np.ndarray:
        return np.linspace(lo, hi, max(2, math.ceil((hi - lo) / step) + 1))

    us, vs = axis(-hu, hu), axis(-hv, hv)
    heights = list(axis(-plan.depth_m, 0.0)) + list(limits.approach_heights_m)
    points = [plan.to_world(u, v, z) for z in heights for u in us for v in vs]
    near, far = plan.spoil_band(limits.spoil_gap_m, limits.spoil_width_m)
    dump_height = limits.approach_heights_m[-1]
    points += [plan.to_world(u, (near + far) / 2, dump_height) for u in us]
    return points


def sweep(points: Iterable[Point], reach: Reach) -> Point | None:
    return next((p for p in points if not reach.reachable(*p)), None)


def check_path(plan: Plan, reach: Reach, points: Iterable[Point] | None = None) -> CheckResult:
    points = list(sample_points(plan) if points is None else points)
    start = time.perf_counter()
    bad = sweep(points, reach)
    elapsed = time.perf_counter() - start
    if bad is not None:
        x, y, z = bad
        return CheckResult(
            "path",
            False,
            f"Point ({x:.2f}, {y:.2f}, {z:.2f}) m on the dig is out of reach.",
        )
    return CheckResult(
        "path",
        True,
        f"{len(points)} points through the trench, approach and spoil are reachable "
        f"({elapsed * 1000:.0f} ms).",
    )


def validate(raw: str, scene: Scene, reach: Reach) -> CheckReport:
    report = CheckReport()
    result, spec = check_schema(raw, scene)
    report.spec = spec
    report.results.append(result)
    if not result.passed:
        return report
    result, plan = check_ranges(spec, scene)
    report.results.append(result)
    if plan is None:
        return report
    report.plan = plan
    report.results.append(check_reach(plan, reach))
    if not report.results[-1].passed:
        return report
    result, plan, warnings = check_trench_rules(plan, scene)
    report.results.append(result)
    report.warnings.extend(warnings)
    report.plan = plan
    if not result.passed:
        return report
    report.results.append(check_path(plan, reach))
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
    raw: str, scene: Scene, reach: Reach, confirm: Callable[[Plan], bool]
) -> CheckReport:
    return confirm_plan(validate(raw, scene, reach), confirm)
