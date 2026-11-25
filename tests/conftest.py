import json
from pathlib import Path

import pytest

from hma.task import DEMO_SCENE, Capacity, MachineArm, MachineBody

MOCK = Path(__file__).resolve().parents[1] / "mock" / "model_outputs"

BASE_SPEC = {
    "status": "plan",
    "message": "Carry the pallet of kerbstones to the near marker.",
    "task_type": "carry",
    "load_id": "L1",
    "marker_id": "M1",
    "stand_marker_id": None,
    "hold_s": None,
}
NO_TASK = {key: None for key in ("task_type", "load_id", "marker_id", "stand_marker_id", "hold_s")}


def spec_json(**changes) -> str:
    return json.dumps({**BASE_SPEC, **changes})


def task_json(**fields) -> str:
    return json.dumps({**BASE_SPEC, **NO_TASK, **fields})


def recorded(name: str) -> dict:
    return json.loads((MOCK / f"{name}.json").read_text(encoding="utf-8"))


class Everywhere:
    def reachable(self, x, y, z):
        return True


class StandIn:
    def __init__(self, tipping=None, hydraulic=None, radius=2.5, angle=100.0):
        self.tipping, self.hydraulic, self.radius, self.angle = tipping, hydraulic, radius, angle

    def lift_capacity(self, radius_m, height_m):
        if self.tipping is None:
            return None
        return Capacity(self.tipping, self.hydraulic)

    def plan_radius(self):
        return self.radius

    def boom_arm_angles(self, x, y, z):
        return (self.angle,) if 3.0 <= x <= 6.0 else ()


@pytest.fixture
def scene():
    return DEMO_SCENE


@pytest.fixture(scope="session")
def arm():
    return MachineArm()


@pytest.fixture(scope="session")
def body():
    return MachineBody()
