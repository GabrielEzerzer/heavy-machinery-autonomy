import math

import mujoco
import numpy as np
import pytest

from hma.machine import (
    JOINT_LIMITS,
    JOINT_NAMES,
    TIP_SITE,
    forward_kinematics,
    inverse_kinematics,
    load_scene,
    machine_prefix,
)

LISTED_REACH = 8.27
LISTED_REACH_AT_GROUND = 8.14
LISTED_DEPTH = 5.57


def random_poses(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = [rng.uniform(-math.pi, math.pi, n)]
    cols += [rng.uniform(*JOINT_LIMITS[j], n) for j in JOINT_NAMES[1:]]
    return np.stack(cols, axis=1)


def test_forward_kinematics_matches_mujoco_tip_site():
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    prefix = machine_prefix(0)
    for q in random_poses(50):
        for name, value in zip(JOINT_NAMES, q, strict=True):
            data.qpos[model.joint(prefix + name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        position, _ = forward_kinematics(q)
        np.testing.assert_allclose(data.site(prefix + TIP_SITE).xpos, position, atol=1e-9)


def tip_in_front(q: np.ndarray) -> bool:
    position, _ = forward_kinematics(q)
    heading = np.array([math.cos(q[0]), math.sin(q[0])])
    return float(position[:2] @ heading) > 0.0


def test_round_trip_across_workspace():
    poses = random_poses(2000, seed=1)
    positions, pitches = forward_kinematics(poses)
    for q, position, pitch in zip(poses, positions, pitches, strict=True):
        solved = inverse_kinematics(position, pitch)
        assert solved is not None
        if tip_in_front(q):
            np.testing.assert_allclose(solved, q, atol=1e-7)
        again, again_pitch = forward_kinematics(solved)
        np.testing.assert_allclose(again, position, atol=1e-9)
        assert abs(again_pitch - pitch) < 1e-9


def test_round_trip_at_joint_limits():
    for corner in np.ndindex(2, 2, 2):
        limits = [JOINT_LIMITS[j][c] for j, c in zip(JOINT_NAMES[1:], corner, strict=True)]
        q = np.array([0.3, *limits])
        position, pitch = forward_kinematics(q)
        solved = inverse_kinematics(position, pitch)
        assert solved is not None
        if tip_in_front(q):
            np.testing.assert_allclose(solved, q, atol=1e-7)
        np.testing.assert_allclose(forward_kinematics(solved)[0], position, atol=1e-9)


@pytest.mark.parametrize(
    ("position", "pitch"),
    [
        ((20.0, 0.0, 0.0), 0.5),
        ((0.0, 0.0, 15.0), 0.0),
        ((0.05, 0.05, 1.0), 0.0),
        ((0.0, 0.0, -7.0), 1.5),
    ],
)
def test_inverse_kinematics_refuses_points_out_of_reach(position, pitch):
    assert inverse_kinematics(np.array(position), pitch) is None


def test_round_trip_with_the_tip_over_the_swing_axis():
    boom, arm = JOINT_LIMITS["boom"][1], 2.38673775
    lo, hi = JOINT_LIMITS["bucket"]

    def forward_x(bucket):
        return forward_kinematics(np.array([0.0, boom, arm, bucket]))[0][0]

    assert forward_x(lo) * forward_x(hi) < 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if forward_x(lo) * forward_x(mid) > 0.0 else (lo, mid)
    position, pitch = forward_kinematics(np.array([0.0, boom, arm, lo]))
    assert abs(position[0]) < 1e-12
    solved = inverse_kinematics(position, pitch)
    assert solved is not None
    again, again_pitch = forward_kinematics(solved)
    np.testing.assert_allclose(again, position, atol=1e-9)
    assert abs(again_pitch - pitch) < 1e-9


@pytest.mark.parametrize(
    ("position", "pitch"),
    [((math.nan, 0.0, 0.0), 0.0), ((3.0, math.inf, 0.0), 0.0), ((3.0, 0.0, 0.0), math.nan)],
)
def test_inverse_kinematics_refuses_values_that_are_not_finite(position, pitch):
    assert inverse_kinematics(np.array(position), pitch) is None


@pytest.mark.parametrize(
    ("position", "pitch"),
    [(np.zeros(2), 0.0), (np.zeros((2, 3)), 0.0), (np.zeros(3), np.zeros(2))],
)
def test_inverse_kinematics_takes_one_point_only(position, pitch):
    with pytest.raises(ValueError):
        inverse_kinematics(position, pitch)


@pytest.mark.parametrize(
    "q",
    [
        (0.2, 0.9, 1.2, 0.5),
        (0.2, -1.3, 1.2, 0.5),
        (0.2, 0.0, 0.4, 0.5),
        (0.2, 0.0, 2.8, 0.5),
        (0.2, 0.0, 1.2, -0.7),
        (0.2, 0.0, 1.2, 2.6),
    ],
)
def test_inverse_kinematics_refuses_points_outside_joint_limits(q):
    position, pitch = forward_kinematics(np.array(q))
    assert inverse_kinematics(position, pitch) is None


def test_reach_and_depth_against_listed_figures():
    grids = [np.linspace(*JOINT_LIMITS[j], 241) for j in JOINT_NAMES[1:]]
    boom, arm, bucket = np.meshgrid(*grids, indexing="ij")
    q = np.stack([np.zeros_like(boom), boom, arm, bucket], axis=-1)
    position, _ = forward_kinematics(q)
    radius = np.hypot(position[..., 0], position[..., 1])
    at_ground = radius[np.abs(position[..., 2]) < 0.02].max()
    assert abs(radius.max() / LISTED_REACH - 1) < 0.05
    assert abs(at_ground / LISTED_REACH_AT_GROUND - 1) < 0.05
    assert abs(-position[..., 2].min() / LISTED_DEPTH - 1) < 0.05
