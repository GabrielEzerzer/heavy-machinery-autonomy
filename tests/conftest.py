import json
from pathlib import Path

import pytest

from hma.task import DEMO_SCENE, StandInArm

MOCK = Path(__file__).resolve().parents[1] / "mock" / "model_outputs"

BASE_SPEC = {
    "status": "plan",
    "message": "Dig 1 m down at the far marker.",
    "task_type": "trench",
    "marker_id": "M3",
    "depth_m": 1,
    "length_m": None,
    "width_m": None,
    "heading": None,
    "spoil_side": None,
    "soil_class": "unknown",
}


def spec_json(**changes) -> str:
    return json.dumps({**BASE_SPEC, **changes})


def recorded(name: str) -> dict:
    return json.loads((MOCK / f"{name}.json").read_text(encoding="utf-8"))


class Everywhere:
    def reachable(self, x, y, z):
        return True


@pytest.fixture
def scene():
    return DEMO_SCENE


@pytest.fixture
def arm():
    return StandInArm()
