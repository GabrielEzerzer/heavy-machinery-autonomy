"""Collision shapes for the excavator: each link's mesh is cut by planes into convex regions, and
each piece collides as the convex hull of the mesh inside its region. Shapes carry no mass. Regions
are lists of half-spaces (n, d) meaning n . p <= d in the link frame."""

from __future__ import annotations

import itertools
from functools import cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent / "assets"
EPS = 1e-9
BIG = 10.0

LINK_MESH = {"base": "crawler", "upper": "body", "boom": "boom", "arm": "arm", "bucket": "bucket"}
BUCKET_CENTRE = (-0.42, 0.5)
BUCKET_INNER_Y = 0.30
TEETH_Y = (-0.279, -0.139, 0.001, 0.141, 0.281)
TOOTH_HALF_WIDTH = 0.026
TOOTH_ROOT_Z = 0.93
LIP_Z = 1.10


def box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> list:
    out = []
    for k in range(3):
        n = np.zeros(3)
        n[k] = 1.0
        out += [(n, hi[k]), (-n, -lo[k])]
    return out


def _sector(degrees: tuple[float, float], centre: tuple[float, float]) -> list:
    out = []
    for angle, sign in zip(np.radians(degrees), (-1.0, 1.0), strict=True):
        n = sign * np.array([-np.sin(angle), 0.0, np.cos(angle)])
        out.append((n, float(n @ [centre[0], 0.0, centre[1]])))
    return out


def _bucket_pieces() -> dict[str, list]:
    inner = box((-BIG, -BUCKET_INNER_Y, -BIG), (BIG, BUCKET_INNER_Y, BIG))
    pieces = {}
    cuts = (-180.0, -115.0, -90.0, -70.0, 70.0, 100.0, 140.0, 180.0)
    below_teeth = [(np.array([0.0, 0.0, 1.0]), TOOTH_ROOT_Z)]
    for k, (a0, a1) in enumerate(itertools.pairwise(cuts)):
        pieces[f"shell_{k}"] = inner + _sector((a0, a1), BUCKET_CENTRE)
        if a1 > 0.0 and a0 < 140.0:
            pieces[f"shell_{k}"] += below_teeth
    pieces["edge"] = inner + box((-BIG, -BIG, TOOTH_ROOT_Z), (BIG, BIG, LIP_Z))
    pieces["side_right"] = box((-BIG, -BIG, -BIG), (BIG, -BUCKET_INNER_Y, BIG))
    pieces["side_left"] = box((-BIG, BUCKET_INNER_Y, -BIG), (BIG, BIG, BIG))
    for k, y in enumerate(TEETH_Y):
        pieces[f"tooth_{k}"] = box((-BIG, y - TOOTH_HALF_WIDTH, TOOTH_ROOT_Z),
                                   (BIG, y + TOOTH_HALF_WIDTH, BIG))
    return pieces


def _slabs(cuts: tuple[float, ...], names: tuple[str, ...], axis: int = 2) -> dict[str, list]:
    pieces = {}
    for name, (lo, hi) in zip(names, itertools.pairwise(cuts), strict=True):
        low, high = [-BIG] * 3, [BIG] * 3
        low[axis], high[axis] = lo, hi
        pieces[name] = box(tuple(low), tuple(high))
    return pieces


def _boom_pieces() -> dict[str, list]:
    return _slabs((-BIG, 0.25, 0.85, 1.4, 2.05, 3.1, 4.15, BIG),
                  ("foot", "lower", "knee", "bracket", "upper", "upper_far", "tip"))


def _upper_pieces() -> dict[str, list]:
    return {
        "rear": box((-BIG, -BIG, -BIG), (-0.72, BIG, 1.0)),
        "cab": box((-0.72, 0.21, 0.0), (BIG, 1.3, 1.8)),
        "roof_guard": box((-0.72, 0.21, 1.8), (BIG, 1.3, 2.27)),
        "right": box((-0.72, -BIG, 0.07), (0.15, -0.54, 1.0)),
        "deck_right": box((-0.72, -BIG, -BIG), (BIG, -0.555, 0.07)),
        "deck_middle": box((-0.72, -0.555, -BIG), (BIG, 0.21, 0.07)),
        "step_box": box((0.15, -BIG, 0.07), (0.66, -0.54, 0.45)),
        "step_front_outer": box((0.66, -BIG, 0.07), (BIG, -0.9, 0.45)),
        "step_front_inner": box((0.66, -0.9, 0.07), (BIG, -0.54, 0.45)),
    }


def _base_pieces() -> dict[str, list]:
    return {
        "track_right": box((-BIG, -1.3, -BIG), (BIG, -0.65, 0.9)),
        "track_left": box((-BIG, 0.65, -BIG), (BIG, 1.3, 0.9)),
        "step_right_rear": box((-BIG, -BIG, -BIG), (-0.2, -1.3, 0.9)),
        "step_right_front": box((-0.2, -BIG, -BIG), (BIG, -1.3, 0.9)),
        "step_left_rear": box((-BIG, 1.3, -BIG), (-0.2, BIG, 0.9)),
        "step_left_front": box((-0.2, 1.3, -BIG), (BIG, BIG, 0.9)),
        "belly": box((-BIG, -0.65, 0.42), (BIG, 0.65, 0.64)),
    }


PIECES = {
    "base": _base_pieces(),
    "upper": _upper_pieces(),
    "boom": _boom_pieces(),
    "arm": _slabs((-BIG, 0.6, BIG), ("foot", "beam")),
    "bucket": _bucket_pieces(),
}
GROUND_PIECES = {("base", "belly"), *(("upper", name) for name in PIECES["upper"])}


@cache
def link_triangles(link: str) -> np.ndarray:
    verts, faces = [], []
    for line in (ASSETS / f"{LINK_MESH[link]}.obj").read_text(encoding="ascii").splitlines():
        kind, *fields = line.split() or [""]
        if kind == "v":
            verts.append([float(x) for x in fields[:3]])
        elif kind == "f":
            corners = [int(f.split("/")[0]) - 1 for f in fields]
            faces += [(corners[0], a, b) for a, b in zip(corners[1:], corners[2:], strict=False)]
    return np.array(verts)[np.array(faces)]


def _inside(points: np.ndarray, region: list, tol: float = 1e-7) -> np.ndarray:
    keep = np.ones(len(points), bool)
    for n, d in region:
        keep &= points @ n <= d + tol
    return keep


def _region_edges(region: list) -> list[tuple[np.ndarray, np.ndarray]]:
    corners = []
    for (i, a), (j, b), (k, c) in itertools.combinations(enumerate(region), 3):
        m = np.array([a[0], b[0], c[0]])
        if abs(np.linalg.det(m)) < EPS:
            continue
        p = np.linalg.solve(m, [a[1], b[1], c[1]])
        if _inside(p[None], region)[0]:
            corners.append((p, {i, j, k}))
    edges = []
    for (p, s), (q, t) in itertools.combinations(corners, 2):
        if len(s & t) >= 2 and np.linalg.norm(p - q) > EPS:
            edges.append((p, q))
    return edges


def _segment_triangle_hits(p: np.ndarray, q: np.ndarray, tri: np.ndarray) -> np.ndarray:
    d = q - p
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > EPS
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = p - tri[:, 0]
    u = f * np.einsum("ij,ij->i", s, h)
    qv = np.cross(s, e1)
    v = f * (qv @ d)
    t = f * np.einsum("ij,ij->i", e2, qv)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
    return p + t[hit, None] * d


def clipped_points(tri: np.ndarray, region: list) -> np.ndarray:
    verts = tri.reshape(-1, 3)
    points = [verts[_inside(verts, region)]]
    starts = tri.reshape(-1, 3)
    ends = np.roll(tri, -1, axis=1).reshape(-1, 3)
    for n, d in region:
        a, b = starts @ n - d, ends @ n - d
        cross = (a * b < 0)
        t = a[cross] / (a[cross] - b[cross])
        hit = starts[cross] + t[:, None] * (ends[cross] - starts[cross])
        points.append(hit[_inside(hit, region)])
    for p, q in _region_edges(region):
        points.append(_segment_triangle_hits(p, q, tri))
    return np.unique(np.vstack(points).round(9), axis=0)


@cache
def link_pieces(link: str) -> dict[str, np.ndarray]:
    tri = link_triangles(link)
    out = {}
    for name, region in PIECES[link].items():
        points = clipped_points(tri, region) if region else np.unique(tri.reshape(-1, 3), axis=0)
        if len(points) < 4:
            raise ValueError(f"{link}/{name}: region holds no mesh")
        out[name] = points
    return out


def piece_names(link: str) -> list[str]:
    return [f"{link}_{name}" for name in PIECES[link]]


def add_collision_geoms(spec, bit: int, solref, solimp, friction, margin: float) -> None:
    import mujoco

    for link in PIECES:
        body = spec.body(link)
        for name, points in link_pieces(link).items():
            mesh = f"{link}_{name}_hull"
            spec.add_mesh(name=mesh, uservert=points.ravel().tolist())
            touches_ground = (link, name) in GROUND_PIECES or link in ("boom", "arm", "bucket")
            body.add_geom(name=f"{link}_{name}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
                          mass=0.0, group=3, rgba=[0.9, 0.3, 0.3, 0.35],
                          contype=bit if touches_ground else 0, conaffinity=bit, priority=1,
                          friction=list(friction), solref=list(solref), solimp=list(solimp),
                          margin=margin)
