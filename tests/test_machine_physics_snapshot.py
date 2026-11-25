"""The compiled scene's physics must match the recorded snapshot, so visual changes cannot leak in.
After a deliberate physics change, rewrite it: python tests/test_machine_physics_snapshot.py"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hma.machine import load_scene

SNAPSHOT = Path(__file__).resolve().parents[1] / "mock" / "physics" / "excavator_scene.json"
SCENES = {"free_two_machines": {"n_machines": 2}, "welded": {"n_machines": 1, "weld": True}}
FIELDS = [
    "body_parentid", "body_mass", "body_ipos", "body_iquat", "body_inertia", "body_pos",
    "body_quat", "body_contype", "body_conaffinity", "body_gravcomp",
    "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited", "qpos0",
    "jnt_stiffness", "jnt_solref", "jnt_solimp", "jnt_margin",
    "dof_armature", "dof_damping", "dof_frictionloss",
    "tendon_frictionloss", "tendon_solref_fri", "tendon_solimp_fri", "tendon_invweight0",
    "tendon_stiffness", "tendon_damping", "tendon_range", "wrap_type", "wrap_objid", "wrap_prm",
    "actuator_trntype", "actuator_trnid", "actuator_gainprm", "actuator_biasprm",
    "actuator_ctrlrange", "actuator_forcerange",
    "eq_type", "eq_obj1id", "eq_obj2id", "eq_data", "eq_active0",
    "site_bodyid", "site_pos", "site_quat",
    "key_qpos", "nbody", "njnt", "nv", "ntendon", "nu", "neq", "nmocap",
]
GEOM_FIELDS = [
    "geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat", "geom_contype",
    "geom_conaffinity", "geom_condim", "geom_friction", "geom_solref", "geom_solimp",
    "geom_margin", "geom_gap", "geom_priority",
]
OPTION_FIELDS = ["timestep", "integrator", "tolerance", "gravity", "cone", "solver", "iterations"]


def snapshot(model: mujoco.MjModel) -> dict[str, list]:
    out = {name: np.asarray(getattr(model, name)).tolist() for name in FIELDS}
    contact = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    out["contact_geom_names"] = [model.geom(i).name for i in np.flatnonzero(contact)]
    for name in GEOM_FIELDS:
        out[name] = np.asarray(getattr(model, name))[contact].tolist()
    for name in OPTION_FIELDS:
        out["opt_" + name] = np.asarray(getattr(model.opt, name)).tolist()
    return out


def current() -> dict[str, dict[str, list]]:
    return {name: snapshot(load_scene(**kwargs)) for name, kwargs in SCENES.items()}


@pytest.mark.parametrize("scene", SCENES)
def test_scene_physics_matches_the_recorded_snapshot(scene):
    recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))[scene]
    now = snapshot(load_scene(**SCENES[scene]))
    assert now.keys() == recorded.keys()
    for name, value in recorded.items():
        assert now[name] == value, name


if __name__ == "__main__":
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(current()), encoding="utf-8")
