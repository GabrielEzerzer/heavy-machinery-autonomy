"""The fixed hydraulic circuit: actuator areas, which pump feeds which function in which direction,
valve and relief sizing, and tables of cylinder length and lever arm against joint angle. Arrays
follow JOINT_NAMES; the last axis of a chamber array is (head or port A, rod or port B)."""

from __future__ import annotations

import math

import numpy as np

from hma.machine import JOINT_LIMITS, JOINT_NAMES
from hma.machine.cylinders import CYLINDER_JOINTS, areas, cylinder_length

LITRE_PER_MIN = 1e-3 / 60.0
PUMPS = 2
PUMP_FLOW = 105.0 * LITRE_PER_MIN
ENGINE_POWER = 65.0e3
PUMP_EFFICIENCY = 0.85
PUMP_RESPONSE = 0.1
STANDBY_FLOW = 20.0 * LITRE_PER_MIN
STANDBY_PRESSURE = 1.0e6
SWING_TORQUE = 34.0e3
SWING_RELIEF = 32.3e6
SWING_SPEED = 13.7 * 2.0 * math.pi / 60.0
SWING_VOLUMETRIC = 0.95
SWING_DISPLACEMENT = PUMP_FLOW * SWING_VOLUMETRIC / SWING_SPEED
SWING_MECHANICAL = SWING_TORQUE / (SWING_DISPLACEMENT * SWING_RELIEF)
TANK_PRESSURE = 0.3e6
CAVITATION_PRESSURE = -0.09e6
VALVE_DROP = 1.5e6
RETURN_DROP = {("boom", "retract"): 5.0e6, ("arm", "extend"): 100.0e6,
               ("arm", "retract"): 14.0e6, ("bucket", "extend"): 12.0e6}
REGENERATION_DROP = {"boom": 30.0e6, "arm": 100.0e6, "bucket": 12.0e6}
RELIEF_OVERRIDE = 1.0e6
RELIEF_KNEE = 0.2e6
MAKEUP_DROP = 0.2e6
ROOT_SMOOTHING = 1.0e4
BRAKE_DELAY = 5.0
JOINT_SIGN = np.array([1.0, -1.0, 1.0, 1.0])

AREA = np.array([(SWING_DISPLACEMENT, SWING_DISPLACEMENT), *(areas(j) for j in CYLINDER_JOINTS)])
FLOW_AREA = AREA / np.array([[SWING_VOLUMETRIC], [1.0], [1.0], [1.0]])
STROKE_AREA = AREA * np.array([[0.0], [1.0], [1.0], [1.0]])

FEEDS = {
    "swing": {"extend": (0.0, 1.0), "retract": (0.0, 1.0)},
    "boom": {"extend": (1.0, 1.0), "retract": (1.0, 0.0)},
    "arm": {"extend": (1.0, 1.0), "retract": (1.0, 1.0)},
    "bucket": {"extend": (1.0, 0.0), "retract": (1.0, 0.0)},
}
REGENERATION = {"boom": "retract", "arm": "extend", "bucket": "extend"}
PRIORITY = ("swing", "arm", 1)


def _rated_flow(joint: str) -> float:
    return PUMP_FLOW


def _tables() -> dict[str, np.ndarray]:
    demand = np.zeros((2, len(JOINT_NAMES), PUMPS))
    meter_in = np.zeros((2, len(JOINT_NAMES), PUMPS, 2))
    meter_out = np.zeros((2, len(JOINT_NAMES), 2))
    regeneration = np.zeros((2, len(JOINT_NAMES)))
    for k, joint in enumerate(JOINT_NAMES):
        for d, direction in enumerate(("extend", "retract")):
            inlet, outlet = (0, 1) if direction == "extend" else (1, 0)
            feeds = np.array(FEEDS[joint][direction]) * _rated_flow(joint)
            demand[d, k] = feeds
            meter_in[d, k, :, inlet] = feeds / math.sqrt(VALVE_DROP)
            back = feeds.sum() * FLOW_AREA[k, outlet] / FLOW_AREA[k, inlet]
            regenerates = REGENERATION.get(joint) == direction
            drop = RETURN_DROP.get((joint, direction), VALVE_DROP)
            meter_out[d, k, outlet] = back / math.sqrt(drop)
            if regenerates:
                regeneration[d, k] = back / math.sqrt(REGENERATION_DROP[joint])
    return {"demand": demand, "meter_in": meter_in, "meter_out": meter_out,
            "regeneration": regeneration}


TABLES = _tables()
RELIEF_SLOPE = PUMP_FLOW / RELIEF_OVERRIDE
MAKEUP_GAIN = PUMP_FLOW / math.sqrt(MAKEUP_DROP)
BYPASS_GAIN = STANDBY_FLOW / math.sqrt(STANDBY_PRESSURE - TANK_PRESSURE)

NODES = PUMPS + 2 * len(JOINT_NAMES)
TANK = NODES
OPEN_BYPASS = len(JOINT_NAMES)
OPEN_ALWAYS = len(JOINT_NAMES) + PUMPS


def chamber(k: int, side: int) -> int:
    return PUMPS + 2 * k + side


def _orifices() -> list[tuple[int, int, int, float, float]]:
    edges = []
    for k in range(len(JOINT_NAMES)):
        for pump in range(PUMPS):
            for side in range(2):
                edges.append((pump, chamber(k, side), k, *TABLES["meter_in"][:, k, pump, side]))
    for k in range(len(JOINT_NAMES)):
        for side in range(2):
            edges.append((chamber(k, side), TANK, k, *TABLES["meter_out"][:, k, side]))
        edges.append((chamber(k, 0), chamber(k, 1), k, 0.0, TABLES["regeneration"][1, k]))
        edges.append((chamber(k, 1), chamber(k, 0), k, TABLES["regeneration"][0, k], 0.0))
    for pump in range(PUMPS):
        edges.append((pump, TANK, OPEN_BYPASS + pump, BYPASS_GAIN, BYPASS_GAIN))
    for k in range(len(JOINT_NAMES)):
        for side in range(2):
            edges.append((TANK, chamber(k, side), OPEN_ALWAYS, MAKEUP_GAIN, MAKEUP_GAIN))
    return edges


def _vented() -> np.ndarray:
    return np.array([float(dst == TANK) for _, dst in _reliefs()])


def _reliefs() -> list[tuple[int, int]]:
    edges = [(chamber(0, 0), chamber(0, 1)), (chamber(0, 1), chamber(0, 0))]
    edges += [(chamber(k, side), TANK) for k in range(1, len(JOINT_NAMES)) for side in range(2)]
    return edges + [(pump, TANK) for pump in range(PUMPS)]


ORIFICES = _orifices()
RELIEFS = _reliefs()
RELIEF_VENT = _vented() * TANK_PRESSURE
LEAKS = [(chamber(k, side), TANK) for k in range(len(JOINT_NAMES)) for side in range(2)]


def _incidence() -> np.ndarray:
    pairs = [e[:2] for e in ORIFICES] + RELIEFS + LEAKS
    incidence = np.zeros((len(pairs), NODES + 1))
    for row, (src, dst) in enumerate(pairs):
        incidence[row, src] += 1.0
        incidence[row, dst] -= 1.0
    return incidence


INCIDENCE = _incidence()
OPENINGS = OPEN_ALWAYS + 1


def _gain_map() -> np.ndarray:
    gain = np.zeros((2 * OPENINGS, len(ORIFICES)))
    for e, (_, _, column, extend, retract) in enumerate(ORIFICES):
        gain[column, e] = extend - retract
        gain[OPENINGS + column, e] = retract
    return gain


GAIN_MAP = _gain_map()
PRIORITY_EDGES = np.array([e for e, (src, _, column, _, _) in enumerate(ORIFICES)
                           if column == JOINT_NAMES.index(PRIORITY[1]) and src == PRIORITY[2]])
DEMAND_MASK = (TABLES["demand"] > 0.0).astype(float)

GEOMETRY_MARGIN = 0.1
GEOMETRY_POINTS = 4001


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts, steps, tables = [], [], []
    stroke = np.zeros((len(JOINT_NAMES), 2))
    for k, joint in enumerate(CYLINDER_JOINTS):
        lo, hi = JOINT_LIMITS[joint]
        q = np.linspace(lo - GEOMETRY_MARGIN, hi + GEOMETRY_MARGIN, GEOMETRY_POINTS)
        starts.append(q[0])
        steps.append(q[1] - q[0])
        tables.append(np.stack(cylinder_length(joint, q), axis=-1))
        ends = cylinder_length(joint, np.array([lo, hi]))[0]
        stroke[k + 1] = ends.min(), ends.max()
    return np.array(starts), 1.0 / np.array(steps), np.concatenate(tables), stroke


GEOMETRY_START, GEOMETRY_SCALE, GEOMETRY_TABLE, STROKE = _geometry()
GEOMETRY_BASE = np.arange(len(CYLINDER_JOINTS)) * GEOMETRY_POINTS
GEOMETRY_STEP = np.diff(GEOMETRY_TABLE, axis=0, append=GEOMETRY_TABLE[-1:])
GEOMETRY_LAST = GEOMETRY_POINTS - 1.0 - 1e-9


def geometry(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, float)
    u = (q[..., 1:] - GEOMETRY_START) * GEOMETRY_SCALE
    np.minimum(np.maximum(u, 0.0, out=u), GEOMETRY_LAST, out=u)
    index = u.astype(np.intp)
    u -= index
    index += GEOMETRY_BASE
    value = GEOMETRY_TABLE[index]
    value += u[..., None] * GEOMETRY_STEP[index]
    length = np.empty(q.shape)
    lever = np.ones(q.shape)
    length[..., 0] = q[..., 0]
    length[..., 1:] = value[..., 0]
    lever[..., 1:] = value[..., 1]
    return length, lever


def _outlet_drops() -> tuple[np.ndarray, np.ndarray]:
    drops = np.full((2, len(JOINT_NAMES)), VALVE_DROP)
    regeneration = np.zeros((2, len(JOINT_NAMES)))
    for k, joint in enumerate(JOINT_NAMES):
        for d, direction in enumerate(("extend", "retract")):
            drops[d, k] = RETURN_DROP.get((joint, direction), VALVE_DROP)
            if REGENERATION.get(joint) == direction:
                regeneration[d, k] = 1.0 / math.sqrt(REGENERATION_DROP[joint])
    return drops, regeneration


OUTLET_DROP, REGENERATING = _outlet_drops()


def full_speed(q: np.ndarray, direction: np.ndarray) -> np.ndarray:
    _, lever = geometry(q)
    extend = np.asarray(direction, float) * JOINT_SIGN > 0
    d = np.where(extend, 0, 1)
    flow = TABLES["demand"][d, np.arange(len(JOINT_NAMES))].sum(-1)
    inlet = np.where(extend, FLOW_AREA[:, 0], FLOW_AREA[:, 1])
    return flow / (inlet * np.abs(lever))


def valve_opening(q: np.ndarray, speed: np.ndarray, pressures: np.ndarray,
                  ramp: float) -> np.ndarray:
    speed = np.asarray(speed, float)
    pressures = np.asarray(pressures, float)
    joints = np.arange(len(JOINT_NAMES))
    extend = speed * JOINT_SIGN > 0
    d = np.where(extend, 0, 1)
    inlet = np.where(extend, 0, 1)
    share = np.where(speed == 0.0, 0.0, np.abs(speed) / full_speed(q, speed))
    p_in = pressures[joints, inlet]
    p_out = pressures[joints, 1 - inlet]
    drop = OUTLET_DROP[d, joints]
    outlet = (np.sqrt(np.maximum(p_out - TANK_PRESSURE, 0.0) / drop)
              + REGENERATING[d, joints] * np.sqrt(np.maximum(p_out - p_in, 0.0)))
    overrun = (p_out > TANK_PRESSURE) & (
        p_out * AREA[joints, 1 - inlet] > p_in * AREA[joints, inlet])
    fed = (DEMAND_MASK[d, joints] > 0.0) & (share > 0.0)[:, None]
    pushed = fed & ~overrun[:, None]
    bypass = np.zeros(len(JOINT_NAMES))
    for pump in range(PUMPS):
        if not pushed[:, pump].any():
            continue
        pressure = np.max(np.where(fed[:, pump], p_in, TANK_PRESSURE)) + VALVE_DROP
        owner = np.argmax(np.where(pushed[:, pump], share, -1.0))
        bypass[owner] += BYPASS_GAIN * math.sqrt(max(pressure - TANK_PRESSURE, 0.0))
    bypass /= TABLES["demand"][d, joints].sum(-1)
    pushing = share + np.minimum(share / ramp, 1.0) * bypass * (1.0 - share) / (1.0 + bypass)
    braking = share / np.maximum(outlet, 1e-6)
    return np.minimum(np.where(overrun, braking, pushing), 1.0)
