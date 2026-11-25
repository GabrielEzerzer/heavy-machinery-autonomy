# The scene the language model sees: the machine's pose, markers, loads, hazards and the ground.
# Site frame: x and y in metres on the ground plan, z up; headings and bearings in degrees
# counter-clockwise from +x. Ground outside every zone is level at height 0; later zones win.
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import cached_property

Point2 = tuple[float, float]
EDGE_PIECE_M = 0.25
EDGE_PROBE_M = 1e-3
STEP_TOLERANCE_M = 0.01


class HazardKind(StrEnum):
    overhead = "overhead"
    buried = "buried"


@dataclass(frozen=True)
class Pose:
    x_m: float
    y_m: float
    heading_deg: float

    @property
    def xy(self) -> Point2:
        return (self.x_m, self.y_m)


@dataclass(frozen=True)
class Marker:
    id: str
    label: str
    x_m: float
    y_m: float

    @property
    def xy(self) -> Point2:
        return (self.x_m, self.y_m)


@dataclass(frozen=True)
class Load:
    id: str
    label: str
    mass_kg: float | None
    x_m: float
    y_m: float
    radius_m: float

    @property
    def xy(self) -> Point2:
        return (self.x_m, self.y_m)


@dataclass(frozen=True)
class Hazard:
    id: str
    label: str
    kind: HazardKind
    points: tuple[Point2, ...]
    keep_out_m: float

    def __post_init__(self) -> None:
        if len(self.points) not in (1, 2) or self.keep_out_m < 0:
            raise ValueError(f"hazard {self.id} needs one or two points and a keep-out >= 0")

    def distance(self, p: Point2) -> float:
        a = self.points[0]
        return distance_to_segment(p, a, self.points[-1])


@dataclass(frozen=True)
class Zone:
    id: str
    label: str
    corners: tuple[Point2, ...]
    height_m: float
    slope_deg: float
    downhill_deg: float

    def __post_init__(self) -> None:
        if len(self.corners) < 3 or not 0 <= self.slope_deg < 90:
            raise ValueError(f"zone {self.id} needs three corners and a slope in [0, 90)")

    def contains(self, p: Point2) -> bool:
        x, y = p
        inside = False
        n = len(self.corners)
        for i in range(n):
            (x1, y1), (x2, y2) = self.corners[i], self.corners[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside

    def height(self, p: Point2) -> float:
        b = math.radians(self.downhill_deg)
        x0, y0 = self.corners[0]
        along = (p[0] - x0) * math.cos(b) + (p[1] - y0) * math.sin(b)
        return self.height_m - math.tan(math.radians(self.slope_deg)) * along


@dataclass(frozen=True)
class Ground:
    height_m: float
    slope_deg: float
    downhill_deg: float
    zone_id: str | None


@dataclass(frozen=True)
class Step:
    a: Point2
    b: Point2
    height_m: float

    def distance(self, p: Point2) -> float:
        return distance_to_segment(p, self.a, self.b)


@dataclass(frozen=True)
class Scene:
    machine: Pose
    markers: tuple[Marker, ...]
    loads: tuple[Load, ...]
    hazards: tuple[Hazard, ...]
    zones: tuple[Zone, ...]
    reach_m: float
    bucket_width_m: float

    def marker(self, marker_id: str) -> Marker | None:
        return next((m for m in self.markers if m.id == marker_id), None)

    def load(self, load_id: str) -> Load | None:
        return next((m for m in self.loads if m.id == load_id), None)

    def ground(self, p: Point2) -> Ground:
        zone = next((z for z in reversed(self.zones) if z.contains(p)), None)
        if zone is None:
            return Ground(0.0, 0.0, 0.0, None)
        return Ground(zone.height(p), zone.slope_deg, zone.downhill_deg, zone.id)

    @cached_property
    def steps(self) -> tuple[Step, ...]:
        found = []
        for zone in self.zones:
            n = len(zone.corners)
            for i in range(n):
                a, b = zone.corners[i], zone.corners[(i + 1) % n]
                length = math.dist(a, b)
                pieces = max(1, math.ceil(length / EDGE_PIECE_M))
                nx, ny = (b[1] - a[1]) / length, -(b[0] - a[0]) / length
                for k in range(pieces):
                    p = lerp2(a, b, k / pieces)
                    q = lerp2(a, b, (k + 1) / pieces)
                    m = lerp2(p, q, 0.5)
                    out = (m[0] + nx * EDGE_PROBE_M, m[1] + ny * EDGE_PROBE_M)
                    inn = (m[0] - nx * EDGE_PROBE_M, m[1] - ny * EDGE_PROBE_M)
                    rise = abs(self.ground(out).height_m - self.ground(inn).height_m)
                    if rise > STEP_TOLERANCE_M:
                        found.append(Step(p, q, rise))
        return tuple(found)

    def as_data(self) -> dict:
        return asdict(self)


def lerp2(a: Point2, b: Point2, t: float) -> Point2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def distance_to_segment(p: Point2, a: Point2, b: Point2) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    span = dx * dx + dy * dy
    t = 0.0 if span == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / span))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def _box(x0: float, x1: float, y0: float, y1: float) -> tuple[Point2, ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


RAMP_SLOPE_DEG = 12.0
PAD_HEIGHT_M = round(6.0 * math.tan(math.radians(RAMP_SLOPE_DEG)), 3)

DEMO_SCENE = Scene(
    machine=Pose(0.0, 0.0, 0.0),
    markers=(
        Marker("M1", "near marker", 4.5, -1.5),
        Marker("M2", "far marker", 6.0, 1.5),
        Marker("M3", "pad marker", 17.0, 0.0),
        Marker("M4", "cable marker", 3.3, 4.5),
        Marker("M5", "slope marker", 0.0, 9.0),
        Marker("M6", "bank marker", -8.0, 0.0),
        Marker("M7", "line marker", 6.0, -8.0),
    ),
    loads=(
        Load("L1", "pallet of kerbstones", 800.0, 5.0, -3.0, 0.6),
        Load("L2", "concrete pipe section", 4000.0, 6.5, -1.0, 0.7),
        Load("L3", "unmarked crate", None, 4.0, -4.5, 0.5),
    ),
    hazards=(
        Hazard("H1", "overhead power line", HazardKind.overhead, ((-20.0, -13.0), (25.0, -13.0)),
               6.0),
        Hazard("H2", "buried cable", HazardKind.buried, ((3.0, 3.0), (3.0, 8.0)), 0.5),
    ),
    zones=(
        Zone("Z1", "ramp", _box(8.0, 14.0, -4.0, 4.0), 0.0, RAMP_SLOPE_DEG, 180.0),
        Zone("Z2", "raised pad", _box(14.0, 22.0, -5.0, 5.0), PAD_HEIGHT_M, 0.0, 0.0),
        Zone("Z3", "side slope", _box(-4.0, 6.0, 6.0, 13.0), 0.0, 8.0, 90.0),
        Zone("Z4", "steep bank", _box(-12.0, -4.0, -3.0, 3.0), 0.0, 35.0, 180.0),
    ),
    reach_m=8.1,
    bucket_width_m=1.0,
)
