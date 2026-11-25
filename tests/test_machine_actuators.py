import mujoco
import numpy as np
import pytest

from hma.machine import (
    FORCE_CAPS,
    HOME_KEY,
    JOINT_LIMITS,
    JOINT_NAMES,
    LINK_MASSES,
    load_scene,
    machine_prefix,
)
from hma.machine.load import apply_load, block, hang_load
from hma.machine.model import SPEED_LIMITS
from scenes import dig_pit

PAYLOAD = 1200.0
SLUNG_CURL = 1.45
DRIFT_LIMIT_DEG_S = 0.1
PREFIX = machine_prefix(0)


class Machine:
    def __init__(self, pose=None, payload=0.0):
        self.model = dig_pit(load_scene(1, weld=True))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(HOME_KEY).id)
        if pose is not None:
            for name, value in zip(JOINT_NAMES, pose, strict=True):
                self.data.qpos[self.joint(name).qposadr[0]] = value
        if payload:
            apply_load(self.model, 0, block(payload), mujoco.MjData(self.model))
            hang_load(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def joint(self, name):
        return self.model.joint(PREFIX + name)

    def q(self, name):
        return float(self.data.qpos[self.joint(name).qposadr[0]])

    def v(self, name):
        return float(self.data.qvel[self.joint(name).dofadr[0]])

    def command(self, name, speed):
        self.data.ctrl[self.model.actuator(PREFIX + name).id] = speed

    def run(self, seconds, load=None):
        for _ in range(round(seconds / self.model.opt.timestep)):
            self.data.qfrc_applied[:] = 0.0
            if load is not None:
                self.data.qfrc_applied[self.joint(load[0]).dofadr[0]] += load[1]
            mujoco.mj_step(self.model, self.data)


def test_stalled_joint_resumes_at_commanded_speed_without_lunging():
    machine = Machine()
    machine.command("arm", 0.3)
    machine.run(2.0, load=("arm", -1.5 * FORCE_CAPS["arm"]))
    stalled_at = machine.q("arm")
    peak = 0.0
    for _ in range(10):
        machine.run(0.1)
        peak = max(peak, machine.v("arm"))
    assert peak <= 0.3 * 1.02
    assert machine.v("arm") == pytest.approx(0.3, rel=0.02)
    assert machine.q("arm") - stalled_at == pytest.approx(0.3, abs=0.03)
    machine.command("arm", 0.0)
    machine.run(0.5)
    held = machine.q("arm")
    machine.run(1.0)
    assert abs(machine.q("arm") - held) < 1e-3


def test_swing_stops_within_what_its_force_cap_allows():
    machine = Machine()
    machine.command("swing", SPEED_LIMITS["swing"])
    machine.run(1.0)
    speed, start = machine.v("swing"), machine.q("swing")
    mass_matrix = np.zeros((machine.model.nv, machine.model.nv))
    mujoco.mj_fullM(machine.model, machine.data, mass_matrix)
    dof = machine.joint("swing").dofadr[0]
    braking = speed**2 * mass_matrix[dof, dof] / (2 * FORCE_CAPS["swing"])
    machine.command("swing", 0.0)
    machine.run(4.0)
    assert machine.q("swing") - start == pytest.approx(braking, rel=0.05)
    assert abs(machine.v("swing")) < 1e-3


@pytest.mark.parametrize(("name", "step"), [("swing", 0.8), ("boom", -0.5), ("arm", -0.6),
                                            ("bucket", 1.0)])
def test_ten_hertz_proportional_control_settles_on_each_joint(name, step):
    machine = Machine()
    target = machine.q(name) + step
    limit = SPEED_LIMITS[name]
    errors = []
    for _ in range(100):
        machine.command(name, float(np.clip(2.0 * (target - machine.q(name)), -limit, limit)))
        machine.run(0.1)
        errors.append(machine.q(name) - target)
    assert abs(errors[-1]) < 1e-3
    assert max(abs(e) for e in errors[-20:]) < 2e-3


@pytest.mark.parametrize("pose", [(0.0, 0.0, 0.6, 0.0), (0.0, 0.3, 0.8, 2.0)])
def test_zero_command_drift_under_full_payload(pose):
    machine = Machine(pose, payload=PAYLOAD)
    machine.run(0.5)
    before = np.array([machine.q(n) for n in JOINT_NAMES])
    machine.run(2.0)
    after = np.array([machine.q(n) for n in JOINT_NAMES])
    assert np.degrees(np.abs(after - before) / 2.0).max() < DRIFT_LIMIT_DEG_S


@pytest.mark.parametrize("name", ["boom", "arm", "bucket"])
def test_each_joint_lifts_and_holds_the_loaded_arm_across_its_range(name):
    lo, hi = JOINT_LIMITS[name]
    if name == "bucket":
        lo = SLUNG_CURL
    pose = dict(zip(JOINT_NAMES, (0.0, 0.0, 0.8, 2.3), strict=True))
    pose[name] = lo + 0.15
    machine = Machine(tuple(pose.values()), payload=PAYLOAD)
    speed = SPEED_LIMITS[name]
    budget = 1.3 * (hi - lo) / speed
    for direction, goal in ((1.0, hi - 0.15), (-1.0, lo + 0.15)):
        machine.command(name, direction * speed)
        elapsed = 0.0
        while direction * (goal - machine.q(name)) > 0.0 and elapsed < budget:
            machine.run(0.1)
            elapsed += 0.1
        assert direction * (goal - machine.q(name)) <= 0.0
    machine.command(name, 0.0)
    machine.run(0.5)
    held = machine.q(name)
    machine.run(2.0)
    assert np.degrees(abs(machine.q(name) - held) / 2.0) < DRIFT_LIMIT_DEG_S


def boom_arm_bucket_motion(mass_scale):
    machine = Machine((0.0, -0.6, 1.9, 2.0), payload=PAYLOAD)
    for body in LINK_MASSES:
        index = machine.model.body(PREFIX + body).id
        machine.model.body_mass[index] *= mass_scale
        machine.model.body_inertia[index] *= mass_scale
    trace = []
    for k in range(40):
        for name, phase in (("boom", 0.0), ("arm", 1.0), ("bucket", 2.0)):
            machine.command(name, 0.5 * SPEED_LIMITS[name] * np.sin(0.4 * k + phase))
        machine.run(0.1)
        trace.append([machine.q(n) for n in JOINT_NAMES])
    return np.array(trace)


def test_joint_motion_below_the_force_cap_does_not_depend_on_link_mass():
    nominal = boom_arm_bucket_motion(1.0)
    for scale in (0.9, 1.1):
        np.testing.assert_allclose(boom_arm_bucket_motion(scale), nominal, atol=2e-3)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("name", ["boom", "arm", "bucket"])
def test_full_speed_into_a_joint_limit_stops_at_the_limit(name, direction):
    machine = Machine()
    machine.command(name, direction * SPEED_LIMITS[name])
    machine.run(5.0)
    lo, hi = JOINT_LIMITS[name]
    assert lo - 5e-4 <= machine.q(name) <= hi + 5e-4
    assert abs(machine.q(name) - (hi if direction > 0 else lo)) < 5e-3
