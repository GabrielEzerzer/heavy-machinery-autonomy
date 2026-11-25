"""One soil per machine: a height grid cut by the bucket's edge; bucket fill, spill, dump, settling.
Call substep() before every mj_step and control_step() once per control period, or use step().
Resistance is friction on three tendons set by the bucket's pose; soil weight is a bucket force."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from hma.machine.model import (
    EDGE_SITES,
    GROUND,
    GROUND_DEPTH,
    SOIL_AXES,
    TIP_SITE,
    TRACK_FEET,
    VALVE,
    ground_shape,
    machine_prefix,
    write_ground,
)
from hma.soil.force import V_CAP, depth_sums, penetration, raw_force, smooth_cap
from hma.soil.ground import Grid, settle
from hma.soil.params import NOMINAL, SoilParams

CONTROL_HZ = 10.0
BUCKET_CAPACITY = 0.45
DUMP_ANGLE = math.pi / 4
DUMP_TIME = 0.5
EDGE_SAMPLES = 7
SPILL_AHEAD = 2.0
SETTLE_PASSES = 10
ANCHOR_DISTANCE = 1000.0
MAX_EDGE_SPEED = 10.0
GROUND_REACH = 2
PROTECT_TOLERANCE = 1e-3
UP = (0.0, 0.0, 1.0)


def physics_steps(model: mujoco.MjModel, control_hz: float) -> int:
    steps = round(1.0 / (control_hz * model.opt.timestep))
    if steps < 1 or not math.isclose(steps * control_hz * model.opt.timestep, 1.0):
        raise ValueError(f"control rate {control_hz} Hz is not a whole number of physics steps")
    return steps


def joint_load(model: mujoco.MjModel, data: mujoco.MjData, index: int, joint: str) -> float:
    tendon = model.tendon(machine_prefix(index) + joint + VALVE).id
    rows = ((data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)
            & (data.efc_id == tendon))
    return -float(data.efc_force[rows].sum())


class Soil:
    def __init__(self, model: mujoco.MjModel, index: int = 0, params: SoilParams = NOMINAL):
        prefix = machine_prefix(index)
        self.index = index
        self.params = params
        hfield, geom = model.hfield(prefix + GROUND), model.geom(prefix + GROUND)
        nrow, ncol = ground_shape()
        self.grid = Grid(x0=float(geom.pos[0] - hfield.size[0]),
                         y0=float(geom.pos[1] - hfield.size[1]),
                         z0=float(geom.pos[2] + GROUND_DEPTH),
                         cell=float(2 * hfield.size[0] / (ncol - 1)), shape=(nrow, ncol))
        self.hfield_id = hfield.id
        self.tip = model.site(prefix + TIP_SITE).id
        self.edges = [model.site(prefix + name).id for name in EDGE_SITES]
        self.width = float(np.linalg.norm(model.site_pos[self.edges[1]]
                                          - model.site_pos[self.edges[0]]))
        self.bucket = model.body(prefix + "bucket").id
        self.base = model.body(prefix + "base").id
        self.tendons = [model.tendon(prefix + name).id for name in SOIL_AXES]
        self.anchors = [int(model.body_mocapid[model.body(prefix + name).id]) for name in SOIL_AXES]
        tip = model.site_pos[self.tip]
        self.reach = float(np.linalg.norm(tip))
        along = tip / np.linalg.norm(tip)
        offset = model.body_ipos[self.bucket] - (model.body_ipos[self.bucket] @ along) * along
        self.mouth = -offset / np.linalg.norm(offset)
        feet = [g for g in range(model.ngeom)
                if model.geom(g).name.startswith(prefix + TRACK_FEET)]
        reach = model.geom_size[feet, 0][:, None]
        self.footprint = (np.min(model.geom_pos[feet, :2] - reach, axis=0),
                          np.max(model.geom_pos[feet, :2] + reach, axis=0))
        xs, ys = self.grid.axes()
        self.cells = np.stack(np.meshgrid(xs, ys), axis=-1)
        self.fractions = [(k + 0.5) / EDGE_SAMPLES for k in range(EDGE_SAMPLES)]
        self.heights = np.zeros(self.grid.shape)
        self.loose = np.zeros(self.grid.shape)
        self.protected = np.zeros(self.grid.shape, bool)
        self.bucket_volume = 0.0
        self.lost_volume = 0.0
        self.resistance = 0.0
        self.limits = (0.0, 0.0, 0.0)
        self.axes = np.zeros((3, 3))
        self._edge = ([0.0] * 3, [0.0] * 3)
        self._base_pose: list[float] | None = None

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData,
              heights: np.ndarray | None = None) -> None:
        heights = np.zeros(self.grid.shape) if heights is None else np.array(heights, float)
        write_ground(model, self.index, heights)
        self.heights = heights
        self.loose = np.zeros(self.grid.shape)
        self.bucket_volume = 0.0
        self.lost_volume = 0.0
        mujoco.mj_forward(model, data)
        self._edge = data.site_xpos[self.edges].tolist()
        self._base_pose = None
        self._protect(data)
        self._set_resistance(model, data, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        data.xfrc_applied[self.bucket] = 0.0

    def ground_volume(self) -> float:
        return float(self.heights.sum()) * self.grid.cell_area

    def total_volume(self) -> float:
        return self.ground_volume() + self.bucket_volume + self.lost_volume

    def edge_force(self, data: mujoco.MjData) -> np.ndarray:
        force = np.zeros(3)
        friction = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON
        for tendon, anchor in zip(self.tendons, self.anchors, strict=True):
            line = data.site_xpos[self.tip] - data.mocap_pos[anchor]
            force += float(data.efc_force[friction & (data.efc_id == tendon)].sum()) * line / (
                np.linalg.norm(line))
        return force

    def substep(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        a, b = data.site_xpos[self.edges].tolist()
        p, q = self._edge
        self._edge = (a, b)
        if max(math.dist(a, p), math.dist(b, q)) > MAX_EDGE_SPEED * model.opt.timestep:
            p, q = a, b
        frame = data.site_xmat[self.tip].tolist()
        lip = (frame[2], frame[5], frame[8])
        span = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        travel = self._travel(span)
        r = data.xmat[self.bucket].tolist()
        m = self.mouth
        mouth = tuple(r[3 * i] * m[0] + r[3 * i + 1] * m[1] + r[3 * i + 2] * m[2] for i in range(3))
        limits, in_soil = (0.0, 0.0, 0.0), False
        if self.grid.clearance(self.heights, (p, q, a, b), GROUND_REACH) <= 0.0:
            self._cut((p, q, b, a), span, mouth)
            limits, in_soil = self._limits(a, b, span, travel, lip)
        if self.bucket_volume > 0.0 and not in_soil and -lip[2] > math.sin(DUMP_ANGLE):
            tilt = (math.asin(min(-lip[2], 1.0)) - DUMP_ANGLE) / (math.pi / 2 - DUMP_ANGLE)
            poured = min(self.bucket_volume,
                         BUCKET_CAPACITY * tilt * model.opt.timestep / DUMP_TIME)
            facing = 1.0 if travel[0] * mouth[0] + travel[1] * mouth[1] >= 0.0 else -1.0
            below = self.reach * facing * np.array(travel)
            self.lost_volume += self.grid.deposit(
                self.heights, self.loose, self.protected, poured,
                np.array(a) - below / 2, np.array(span), below)
            self.bucket_volume -= poured
        self._set_resistance(model, data, limits, travel)
        self._weigh(data)

    def control_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        settle(self.heights, self.loose, self.protected, self.params.friction_angle,
               self.grid.cell, SETTLE_PASSES)
        self._protect(data)
        write_ground(model, self.index, self.heights)

    def _cut(self, corners: tuple, span: tuple, mouth: tuple) -> None:
        removed = self.grid.cut(self.heights, self.loose, self.protected, corners)
        if removed <= 0.0:
            return
        p, q, b, a = corners
        moved = [(a[i] + b[i] - p[i] - q[i]) / 2 for i in range(3)]
        forward = moved[0] * mouth[0] + moved[1] * mouth[1] + moved[2] * mouth[2] > 0.0
        held = min(self.bucket_volume + removed, BUCKET_CAPACITY) if forward else self.bucket_volume
        taken = held - self.bucket_volume
        self.bucket_volume = held
        norm = math.hypot(moved[0], moved[1])
        stride = self.grid.cell / norm if norm > 1e-12 else 0.0
        ahead = np.array([moved[0] * stride, moved[1] * stride, 0.0])
        self.lost_volume += self.grid.deposit(
            self.heights, self.loose, self.protected, removed - taken,
            np.array(a) + SPILL_AHEAD * ahead, np.array(span), ahead)

    def _weigh(self, data: mujoco.MjData) -> None:
        data.xfrc_applied[self.bucket] = 0.0
        data.xfrc_applied[self.bucket, 2] = -self.params.unit_weight * self.bucket_volume

    def _depth(self, x: float, y: float, z: float) -> tuple[float, float]:
        top, loose = self.grid.surface(self.heights, self.loose, x, y)
        return (top - z, max(top - z - loose, 0.0)) if top > z else (0.0, 0.0)

    @staticmethod
    def _deeper(*samples: tuple[float, float]) -> tuple[float, float]:
        return max(d for d, _ in samples), max(f for _, f in samples)

    def _price(self, profile: list, rake: float, attack: float, load: float) -> float:
        sums = depth_sums([d for d, _ in profile], [d - f for d, f in profile], self.width)
        return smooth_cap(raw_force(self.params, sums, rake, attack, load))

    def _limits(self, a: list[float], b: list[float], span: tuple, travel: tuple,
                lip: tuple) -> tuple[tuple[float, float, float], bool]:
        hx, hy = self.grid.cell * travel[0], self.grid.cell * travel[1]
        sx, sy = (-hy, hx) if hx * span[1] - hy * span[0] >= 0.0 else (hy, -hx)
        side = self._deeper(self._depth(a[0] - sx, a[1] - sy, a[2]),
                            self._depth(b[0] + sx, b[1] + sy, b[2]))
        cut, sweep, firm = [], [], 0.0
        for f in self.fractions:
            x, y, z = a[0] + f * span[0], a[1] + f * span[1], a[2] + f * span[2]
            at = self._depth(x, y, z)
            ahead, behind = self._depth(x + hx, y + hy, z), self._depth(x - hx, y - hy, z)
            cut.append(self._deeper(ahead, behind, at))
            sweep.append(self._deeper(side, at))
            firm += at[1]
        if max(d for d, _ in cut + sweep) <= 0.0:
            return (0.0, 0.0, 0.0), False
        plate = lip[0] * travel[0] + lip[1] * travel[1]
        rake = math.atan2(-lip[2], plate)
        load = self.params.unit_weight * self.bucket_volume
        vertical = penetration(self.params, firm * self.width / len(self.fractions))
        return (self._price(cut, rake, plate, load), self._price(sweep, rake, plate, load),
                smooth_cap(vertical, V_CAP)), True

    @staticmethod
    def _travel(span: tuple) -> tuple[float, float, float]:
        norm = math.hypot(span[0], span[1])
        if norm < 1e-9:
            return (1.0, 0.0, 0.0)
        return (-span[1] / norm, span[0] / norm, 0.0)

    def _set_resistance(self, model: mujoco.MjModel, data: mujoco.MjData,
                        limits: tuple[float, float, float], travel: tuple) -> None:
        self.limits = limits
        self.resistance = limits[0]
        model.tendon_frictionloss[self.tendons] = limits
        self.axes = np.array([travel, (-travel[1], travel[0], 0.0), UP])
        data.mocap_pos[self.anchors] = data.site_xpos[self.tip] - ANCHOR_DISTANCE * self.axes

    def _protect(self, data: mujoco.MjData) -> None:
        pose = data.xpos[self.base].tolist() + data.xmat[self.base].tolist()
        if self._base_pose is not None and max(
                abs(x - y) for x, y in zip(pose, self._base_pose, strict=True)) < PROTECT_TOLERANCE:
            return
        self._base_pose = pose
        frame = data.xmat[self.base].reshape(3, 3)
        local = (self.cells - data.xpos[self.base][:2]) @ frame[:2, :2]
        lo, hi = self.footprint
        self.protected = np.all((local >= lo) & (local <= hi), axis=-1)


def step(model: mujoco.MjModel, data: mujoco.MjData, soils: list[Soil],
         control_hz: float = CONTROL_HZ) -> None:
    for _ in range(physics_steps(model, control_hz)):
        for soil in soils:
            soil.substep(model, data)
        mujoco.mj_step(model, data)
    for soil in soils:
        soil.control_step(model, data)


def sync_viewer(viewer, soil: Soil) -> None:
    viewer.update_hfield(soil.hfield_id)
