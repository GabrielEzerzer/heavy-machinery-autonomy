"""Mass, centre of mass and inertia of each body, summed from the components in PARTS. Each part is
a solid or thin-walled box, a ring, a rod, a point, a track loop or a mesh surface in its body's
frame; the README gives the source or estimate behind every mass and position."""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from hma.machine import cylinders

ASSETS = Path(__file__).parent / "assets"

OPERATING_WEIGHT = 12000.0
COUNTERWEIGHT_MASS = 2450.0
COUNTERWEIGHT_BOX = ((-1.91227, -0.0003785, 0.381216), (0.297132, 1.3, 0.5))
FRONT = {"boom": 1220.0, "arm": 600.0, "bucket": 400.0}
FUEL = 250.0 * 0.82
OPERATOR = 75.0
BOOM_CYLINDERS = 240.0
TRACK_CHAINS = 1540.0
TRAVEL_MOTOR = 180.0
IDLER = 100.0
ROLLER = 22.0
ROLLERS = 7
SIBLING_SPLIT = {"lower_frame": 2620.0, "swing_frame": 1240.0, "upper_rest": 2750.0}

TUMBLER = 1.44
GAUGE = 1.99
SHOE_WIDTH = 0.5
TRACK_HEIGHT = 0.79
LINK_BAND = 0.06

BOOM_PIVOT = np.array(cylinders.BOOM_PIVOT)
HOME_BOOM = -0.6
BOOM_FOOT = np.array(cylinders.PINS["boom_foot"][1])
BOOM_ROD = np.array(cylinders.PINS["boom_rod"][1])
BOOM_CYLINDER_Y = (-0.25, 0.25)
BARREL_LENGTH, ROD_LENGTH = 1.25, 1.15
BARREL_SHARE = 65.0 / 120.0


class Solid(NamedTuple):
    mass: float
    com: np.ndarray
    inertia: np.ndarray


def _shared() -> float:
    named = (sum(FRONT.values()) + COUNTERWEIGHT_MASS + FUEL + OPERATOR + BOOM_CYLINDERS
             + TRACK_CHAINS)
    return (OPERATING_WEIGHT - named) / sum(SIBLING_SPLIT.values())


def split(name: str) -> float:
    return SIBLING_SPLIT[name] * _shared()


def point(mass: float, at) -> Solid:
    return Solid(mass, np.asarray(at, float), np.zeros((3, 3)))


def box(mass: float, centre, half) -> Solid:
    a, b, c = np.asarray(half, float) ** 2
    return Solid(mass, np.asarray(centre, float), mass / 3.0 * np.diag([b + c, a + c, a + b]))


def shell_box(mass: float, centre, half) -> Solid:
    h = np.asarray(half, float)
    areas = np.array([h[1] * h[2], h[0] * h[2], h[0] * h[1]])
    parts = []
    for axis in range(3):
        plate = h.copy()
        plate[axis] = 0.0
        for side in (-1.0, 1.0):
            offset = np.zeros(3)
            offset[axis] = side * h[axis]
            parts.append(box(mass * areas[axis] / (2 * areas.sum()), np.asarray(centre) + offset,
                             plate))
    return combine(parts)


def ring(mass: float, centre, radius: float) -> Solid:
    r2 = radius * radius
    return Solid(mass, np.asarray(centre, float), mass * np.diag([r2 / 2, r2 / 2, r2]))


def rod(mass: float, start, end) -> Solid:
    start, end = np.asarray(start, float), np.asarray(end, float)
    d = end - start
    return Solid(mass, (start + end) / 2, mass / 12.0 * (d @ d * np.eye(3) - np.outer(d, d)))


def track_loop(mass: float, y: float) -> Solid:
    bottom, top = LINK_BAND, TRACK_HEIGHT - LINK_BAND
    radius, middle = (top - bottom) / 2, (top + bottom) / 2
    straight = 2 * TUMBLER
    s = (np.arange(4000) + 0.5) / 4000 * 2 * (straight + math.pi * radius)
    pts = np.zeros((s.size, 3))
    for k, length in enumerate(s):
        if length < straight:
            pts[k] = (-TUMBLER + length, y, bottom)
            continue
        length -= straight
        if length < math.pi * radius:
            a = length / radius
            pts[k] = (TUMBLER + radius * math.sin(a), y, middle - radius * math.cos(a))
            continue
        length -= math.pi * radius
        if length < straight:
            pts[k] = (TUMBLER - length, y, top)
            continue
        a = (length - straight) / radius
        pts[k] = (-TUMBLER - radius * math.sin(a), y, middle + radius * math.cos(a))
    each = mass / s.size
    width = each * SHOE_WIDTH**2 / 12.0 * np.diag([1.0, 0.0, 1.0])
    return combine([Solid(each, p, width) for p in pts])


def _triangles(path: Path) -> np.ndarray:
    verts, faces = [], []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(c.split("/")[0]) - 1 for c in line.split()[1:4]])
    return np.asarray(verts)[np.asarray(faces)]


@cache
def _unit_shell(mesh: str) -> Solid:
    tri = _triangles(ASSETS / f"{mesh}.obj")
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = area.sum()
    com = (tri.sum(axis=1) * area[:, None]).sum(axis=0) / (3 * total)
    rel = tri - com
    s = rel.sum(axis=1)
    second = (np.einsum("t,tvi,tvj->ij", area, rel, rel) + np.einsum("t,ti,tj->ij", area, s, s))
    second /= 12 * total
    return Solid(1.0, com, np.trace(second) * np.eye(3) - second)


def mesh_shell(mass: float, mesh: str) -> Solid:
    unit = _unit_shell(mesh)
    return Solid(mass, unit.com.copy(), unit.inertia * mass)


def about(solid: Solid, at: np.ndarray) -> np.ndarray:
    d = solid.com - at
    return solid.inertia + solid.mass * (d @ d * np.eye(3) - np.outer(d, d))


def combine(parts: list[Solid]) -> Solid:
    mass = sum(p.mass for p in parts)
    com = sum(p.mass * p.com for p in parts) / mass
    return Solid(mass, com, sum(about(p, com) for p in parts))


def _turn(angle: float, x: float, z: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * x + s * z, -s * x + c * z])


def _boom_cylinders() -> tuple[list[Solid], list[Solid]]:
    angle = cylinders.BOOM_TURN + HOME_BOOM
    rod_pin = BOOM_PIVOT + _turn(angle, BOOM_ROD[0], BOOM_ROD[2])
    foot = BOOM_FOOT[[0, 2]]
    axis = (rod_pin - foot) / np.linalg.norm(rod_pin - foot)
    barrel_end = foot + BARREL_LENGTH * axis
    rod_start = _turn(-angle, *(rod_pin - ROD_LENGTH * axis - BOOM_PIVOT))
    barrels, rods = [], []
    each = BOOM_CYLINDERS / len(BOOM_CYLINDER_Y)
    for dy in BOOM_CYLINDER_Y:
        y = BOOM_FOOT[1] + dy
        barrels.append(rod(each * BARREL_SHARE, (foot[0], y, foot[1]),
                           (barrel_end[0], y, barrel_end[1])))
        rods.append(rod(each * (1 - BARREL_SHARE), (rod_start[0], dy, rod_start[1]),
                        (BOOM_ROD[0], dy, BOOM_ROD[2])))
    return barrels, rods


def _lower() -> list[Solid]:
    frame = split("lower_frame")
    motors, idlers, rollers = 2 * TRAVEL_MOTOR, 2 * IDLER, 2 * ROLLERS * ROLLER
    carriers, sides = 2 * 15.0, 2 * 300.0
    body = frame - motors - idlers - rollers - carriers - sides
    parts = [box(body, (0.0, 0.0, 0.665), (0.55, 0.7, 0.215))]
    for side in (1.0, -1.0):
        y = side * GAUGE / 2
        parts += [
            shell_box(sides / 2, (0.0, y, 0.4), (1.3, 0.12, 0.2)),
            box(motors / 2, (-TUMBLER, side * 0.95, 0.35), (0.25, 0.2, 0.25)),
            box(idlers / 2, (TUMBLER, y, 0.35), (0.25, 0.08, 0.25)),
            rod(rollers / 2, (-1.08, y, 0.12), (1.08, y, 0.12)),
            point(carriers / 2, (0.0, y, 0.66)),
            track_loop(TRACK_CHAINS / 2, y),
        ]
    return parts


def _upper(barrels: list[Solid]) -> list[Solid]:
    rest = split("upper_rest")
    named = {
        "engine": (534.0, box, (-1.25, 0.0, 0.55), (0.3, 0.75, 0.35)),
        "pumps": (100.0, box, (-1.25, -0.95, 0.45), (0.2, 0.15, 0.15)),
        "hydraulic_tank": (193.0, box, (-0.45, -0.9, 0.6), (0.3, 0.25, 0.4)),
        "fuel_tank": (60.0 + FUEL, box, (0.15, -0.9, 0.55), (0.3, 0.25, 0.4)),
        "control_valve": (90.0, box, (-0.4, -0.25, 0.35), (0.2, 0.2, 0.15)),
        "swing_drive": (150.0, box, (-0.15, 0.4, 0.35), (0.15, 0.15, 0.3)),
        "batteries": (50.0, box, (0.65, -0.9, 0.35), (0.15, 0.2, 0.12)),
        "cab": (450.0, shell_box, (0.65, 0.725, 0.97), (0.5, 0.505, 0.77)),
    }
    parts = [shape(mass, centre, half) for mass, shape, centre, half in named.values()]
    covers, bearing = 300.0, 200.0
    other = rest - sum(v[0] for v in named.values()) + FUEL - covers - bearing
    centre, half = COUNTERWEIGHT_BOX
    return parts + barrels + [
        box(split("swing_frame"), (-0.5, 0.0, 0.03), (1.1, 0.95, 0.15)),
        ring(bearing, (0.0, 0.0, -0.08), 0.5),
        mesh_shell(covers, "body"),
        box(other, (-0.5, 0.0, 0.03), (1.1, 0.95, 0.15)),
        point(OPERATOR, (0.6, 0.72, 0.75)),
        box(COUNTERWEIGHT_MASS, centre, half),
    ]


def body_parts() -> dict[str, list[Solid]]:
    barrels, rods = _boom_cylinders()
    return {
        "base": _lower(),
        "upper": _upper(barrels),
        "boom": [mesh_shell(FRONT["boom"], "boom"), *rods],
        "arm": [mesh_shell(FRONT["arm"], "arm")],
        "bucket": [mesh_shell(FRONT["bucket"], "bucket")],
    }


@cache
def body_solids() -> dict[str, Solid]:
    return {name: combine(parts) for name, parts in body_parts().items()}


def full_inertia(solid: Solid) -> list[float]:
    i = solid.inertia
    return [i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]
