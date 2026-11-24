"""One soil per machine: a height grid cut by the bucket's edge; bucket fill, spill, dump, settling.
Call substep() before every mj_step and control_step() once per control period, or use step().
Resistance is friction on three tendons along the edge's motion; soil weight is a bucket force."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from hma.machine.model import (
    EDGE_SITES,
    GROUND,
    GROUND_DEPTH,
    JOINT_NAMES,
    SOIL_AXES,
    TIP_SITE,
    TRACK_FEET,
    VALVE,
    ground_shape,
    machine_prefix,
    write_ground,
)
from hma.soil.force import depth_sums, raw_force, smooth_cap
from hma.soil.ground import Grid, settle
from hma.soil.params import NOMINAL, SoilParams

CONTROL_HZ = 10.0
BUCKET_CAPACITY = 0.45
DUMP_ANGLE = math.pi / 4
EDGE_SAMPLES = 7
SPEED_FADE = 0.01
TRAVEL_BLEND = 0.1
SPILL_AHEAD = 2.0
SETTLE_PASSES = 10
ANCHOR_DISTANCE = 1000.0


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
        self.arm_dofs = [model.joint(prefix + j).dofadr[0] for j in JOINT_NAMES]
        self.valve_dofs = [model.joint(prefix + j + VALVE).dofadr[0] for j in JOINT_NAMES]
        feet = [g for g in range(model.ngeom)
                if model.geom(g).name.startswith(prefix + TRACK_FEET)]
        reach = model.geom_size[feet, 0][:, None]
        self.footprint = (np.min(model.geom_pos[feet, :2] - reach, axis=0),
                          np.max(model.geom_pos[feet, :2] + reach, axis=0))
        xs, ys = self.grid.axes()
        self.cells = np.stack(np.meshgrid(xs, ys), axis=-1)
        self.fractions = [(k + 0.5) / EDGE_SAMPLES for k in range(EDGE_SAMPLES)]
        self.jac = np.zeros((3, model.nv))
        self.heights = np.zeros(self.grid.shape)
        self.loose = np.zeros(self.grid.shape)
        self.protected = np.zeros(self.grid.shape, bool)
        self.bucket_volume = 0.0
        self.lost_volume = 0.0
        self.resistance = 0.0
        self.direction = np.zeros(3)
        self._edge = ([0.0] * 3, [0.0] * 3)

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
        self._protect(data)
        self._set_resistance(model, data, 0.0, (0.0, 0.0, 0.0))
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
        frame = data.site_xmat[self.tip].tolist()
        lip = (frame[2], frame[5], frame[8])
        r = data.xmat[self.bucket].tolist()
        m = self.mouth
        mouth = [r[3 * i] * m[0] + r[3 * i + 1] * m[1] + r[3 * i + 2] * m[2] for i in range(3)]
        moved = [(a[i] + b[i] - p[i] - q[i]) / 2 for i in range(3)]
        removed = self.grid.cut(self.heights, self.loose, self.protected, (p, q, b, a))
        span = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
        if removed > 0.0:
            forward = moved[0] * mouth[0] + moved[1] * mouth[1] + moved[2] * mouth[2] > 0.0
            taken = min(removed, BUCKET_CAPACITY - self.bucket_volume) if forward else 0.0
            self.bucket_volume += taken
            norm = math.hypot(moved[0], moved[1])
            stride = self.grid.cell / norm if norm > 1e-12 else 0.0
            ahead = np.array([moved[0] * stride, moved[1] * stride, 0.0])
            self.lost_volume += self.grid.deposit(self.heights, self.loose, removed - taken,
                                                  np.array(a) + SPILL_AHEAD * ahead, span, ahead)
        actual, intended = self._edge_motion(model, data)
        speed = math.hypot(*intended)
        steer = SPEED_FADE / speed if speed > 1e-12 else 0.0
        u = [actual[i] + steer * intended[i] for i in range(3)]
        norm = math.hypot(*u)
        direction = (u[0] / norm, u[1] / norm, u[2] / norm) if norm > 1e-12 else (0.0, 0.0, 0.0)
        lx, ly = self.grid.cell * direction[0], self.grid.cell * direction[1]
        depths = []
        for f in self.fractions:
            z = a[2] + f * span[2]
            top = self.grid.height(self.heights, a[0] + f * span[0] + lx, a[1] + f * span[1] + ly)
            depths.append(max(top - z, 0.0))
        in_soil = max(depths) > 0.0
        travel = self._travel(span)
        if self.bucket_volume > 0.0 and not in_soil and -lip[2] > math.sin(DUMP_ANGLE):
            below = self.reach * travel
            self.lost_volume += self.grid.deposit(self.heights, self.loose, self.bucket_volume,
                                                  np.array(a) - below / 2, span, below)
            self.bucket_volume = 0.0
        force = 0.0
        if in_soil and speed > 1e-12:
            force = math.tanh(speed / SPEED_FADE) * self._resistance(depths, direction, travel, lip)
        self._set_resistance(model, data, force, direction)
        data.xfrc_applied[self.bucket] = 0.0
        data.xfrc_applied[self.bucket, 2] = -self.params.unit_weight * self.bucket_volume

    def control_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        settle(self.heights, self.loose, self.params.friction_angle, self.grid.cell,
               SETTLE_PASSES)
        self._protect(data)
        write_ground(model, self.index, self.heights)

    def _edge_motion(self, model: mujoco.MjModel,
                     data: mujoco.MjData) -> tuple[list[float], list[float]]:
        mujoco.mj_jacSite(model, data, self.jac, None, self.tip)
        intended = self.jac[:, self.arm_dofs] @ data.qvel[self.valve_dofs]
        return (self.jac @ data.qvel).tolist(), intended.tolist()

    def _resistance(self, depths: list[float], direction: tuple[float, float, float],
                    travel: np.ndarray, lip: tuple[float, float, float]) -> float:
        sums = depth_sums(depths, self.width)
        attack = direction[0] * lip[0] + direction[1] * lip[1] + direction[2] * lip[2]
        load = self.params.unit_weight * self.bucket_volume
        along = direction[0] * travel[0] + direction[1] * travel[1]
        forward = 0.5 * (1.0 + math.tanh(along / TRAVEL_BLEND))
        plate_along = -(lip[0] * travel[0] + lip[1] * travel[1])
        force = 0.0
        for sign, weight in ((1.0, forward), (-1.0, 1.0 - forward)):
            rake = math.atan2(-lip[2], -sign * plate_along)
            force += weight * smooth_cap(raw_force(self.params, sums, rake, attack, load))
        return force

    @staticmethod
    def _travel(span: np.ndarray) -> np.ndarray:
        norm = math.hypot(span[0], span[1])
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return np.array([span[1] / norm, -span[0] / norm, 0.0])

    def _set_resistance(self, model: mujoco.MjModel, data: mujoco.MjData, force: float,
                        direction: tuple[float, float, float]) -> None:
        self.resistance = force
        self.direction = np.array(direction)
        model.tendon_frictionloss[self.tendons] = force
        if force > 0.0:
            x, y, z = direction
            side = (y, -x, 0.0) if math.hypot(x, y) > 1e-6 else (0.0, z, -y)
            n = math.hypot(*side)
            sx, sy, sz = side[0] / n, side[1] / n, side[2] / n
            frame = np.array([direction, (sx, sy, sz),
                              (y * sz - z * sy, z * sx - x * sz, x * sy - y * sx)])
            data.mocap_pos[self.anchors] = data.site_xpos[self.tip] - ANCHOR_DISTANCE * frame

    def _protect(self, data: mujoco.MjData) -> None:
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
