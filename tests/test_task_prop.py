"""An approved prop plan becomes the propping task with its distance and hold, every plan the
controller was not trained for is refused with a reason and opens no window, and a scene reset onto
a given task runs that task with the machine still drawn from the seed."""

import argparse
import math
import re
from dataclasses import replace

import numpy as np
import pytest

from conftest import task_json
from hma.task import DEMO_SCENE
from hma.task.__main__ import main, run_plan
from hma.task.checks import LIMITS, Plan
from hma.task.prop import LIFT_M, PITCH, RAMP_S, TARGET, CannotRun, prop_task
from hma.task.scene import Pose, Scene, Zone
from hma.task.spec import TaskType
from hma.train.play import episodes, summary
from hma.train.scene import PropScene
from hma.train.task import DEV_BASE, HELD_OUT_BASE, Settings

SEED = DEV_BASE + 3
HERE = DEMO_SCENE.machine
PROP = Plan(TaskType.prop, HERE, HERE, hold_s=5.0, prop_xy=(5.0, 0.0))
INSTRUCTION = "prop yourself up on the bucket and hold for five seconds"


def flat_with(*zones):
    return Scene(HERE, (), (), (), zones, 1.0)


def box(x0, x1, height=0.0, slope=0.0):
    return Zone("Z", "zone", ((x0, -2.0), (x1, -2.0), (x1, 2.0), (x0, 2.0)), height, slope, 0.0)


def prop_source(hold_s=5):
    return lambda instruction, scene: task_json(task_type="prop", hold_s=hold_s)


def test_the_checks_look_for_a_prop_point_only_where_the_controller_trained():
    assert LIMITS.prop_radius_m == Settings().prop_x


def test_the_fixed_values_lie_inside_the_ranges_the_controller_trained_on():
    s = Settings()
    for value, (lo, hi) in ((TARGET, s.target), (RAMP_S, s.ramp), (PITCH, s.pitch),
                            (LIFT_M, s.lift)):
        assert lo <= value <= hi


def test_a_prop_plan_becomes_a_task_with_its_distance_and_hold():
    task = prop_task(replace(PROP, prop_xy=(4.2, 0.0), hold_s=7.5), DEMO_SCENE, SEED)
    assert (task.prop_x, task.hold, task.seed) == (4.2, 7.5, SEED)
    assert (task.target, task.ramp_up, task.ramp_down, task.pitch, task.start_z) == (
        TARGET, RAMP_S, RAMP_S, PITCH, LIFT_M)


@pytest.mark.parametrize(
    ("plan", "scene", "reason"),
    [
        (replace(PROP, task_type=TaskType.carry), DEMO_SCENE, "Only propping can run"),
        (replace(PROP, stand=Pose(2.0, 0.0, 0.0), prop_xy=(7.0, 0.0)), DEMO_SCENE,
         "drives the machine 2.0 m first, and there is no controller for driving"),
        (replace(PROP, hold_s=2.0), DEMO_SCENE, "A hold of 2 s is outside the 3 to 10 s"),
        (replace(PROP, hold_s=12.0), DEMO_SCENE, "A hold of 12 s is outside the 3 to 10 s"),
        (replace(PROP, prop_xy=(3.0, 0.0)), DEMO_SCENE, "press 3.00 m ahead"),
        (replace(PROP, prop_xy=(6.6, 0.0)), DEMO_SCENE, "press 6.60 m ahead"),
        (PROP, flat_with(box(-3.0, 3.0, slope=5.0)),
         "under the machine at (0.0, 0.0) slopes 5 degrees"),
        (PROP, flat_with(box(4.0, 8.0, slope=5.0)),
         "under the bucket at (5.0, 0.0) slopes 5 degrees"),
        (PROP, flat_with(box(4.0, 8.0, height=0.3)), "0.30 m higher than under the machine"),
        (replace(PROP, prop_xy=(3.5, 0.0)), DEMO_SCENE,
         "The arm cannot follow the propping path to the point 3.50 m ahead"),
    ],
)
def test_a_plan_the_controller_was_not_trained_for_is_refused_with_its_reason(plan, scene, reason):
    with pytest.raises(CannotRun, match=re.escape(reason)):
        prop_task(plan, scene, SEED)


def test_without_run_the_command_never_opens_the_viewer(capsys, tmp_path):
    def never(*args):
        raise AssertionError("the viewer opened")

    code = main([INSTRUCTION, "--cache", str(tmp_path)], source=prop_source(),
                confirm=lambda plan: True, show=never)
    assert code == 0 and capsys.readouterr().out.rstrip().endswith(
        "APPROVED: prop the machine on its bucket, hold 5 s, set down")


@pytest.mark.parametrize(("hold_s", "extra", "reason"), [
    (20, [], "CANNOT RUN: A hold of 20 s is outside the 3 to 10 s"),
    (5, ["--machine-seed", str(HELD_OUT_BASE)], "CANNOT RUN: held-out seeds are never played"),
])
def test_a_plan_that_cannot_run_opens_no_window_and_exits_non_zero(capsys, tmp_path, hold_s,
                                                                    extra, reason):
    opened = []
    code = main([INSTRUCTION, "--run", "--cache", str(tmp_path), *extra],
                source=prop_source(hold_s), confirm=lambda plan: True,
                show=lambda *args: opened.append(args))
    out = capsys.readouterr().out
    assert code == 3 and not opened
    assert "APPROVED" in out and reason in out


def test_the_demo_instruction_goes_from_approval_to_a_task_in_the_viewer(capsys, tmp_path):
    shown = []
    code = main([INSTRUCTION, "--run", "--cache", str(tmp_path)], source=prop_source(),
                confirm=lambda plan: True, show=lambda *args: shown.append(args) or (None, None))
    out = capsys.readouterr().out
    ((rung, seed, checkpoint, task),) = shown
    assert code == 0 and (rung, seed, checkpoint) == ("nominal", DEV_BASE, None)
    assert task.prop_x == pytest.approx(4.3) and task.hold == 5.0
    assert "Running: the bucket presses 4.30 m ahead" in out


def test_a_runnable_plan_runs_both_machines_to_the_end_and_prints_both_outcomes(capsys):
    def headless(rung, seed, checkpoint, task):
        *_, (scenes, outcomes) = episodes(rung, seed, task=task)
        for label, scene, info in zip(("baseline", "trained"), scenes, outcomes, strict=True):
            assert scene.tasks[0] is task
            print(summary(label, info, scene.settings))
        return outcomes

    args = argparse.Namespace(machine_seed=None, rung="nominal", checkpoint=None)
    assert run_plan(replace(PROP, prop_xy=(6.5, 0.0)), args, headless) == 0
    out = capsys.readouterr().out
    assert "the bucket presses 6.50 m ahead" in out and "hold 5 s" in out
    assert "baseline: pass" in out and "trained: pass" in out


def test_a_given_task_is_the_one_the_scene_runs_and_the_machine_still_comes_from_the_seed():
    plain = PropScene(1, "machines")
    plain.reset(0, SEED)
    given = PropScene(1, "machines")
    given.reset(0, SEED, plain.tasks[0])
    zero = np.zeros((1, 4))
    for _ in range(30):
        a, b = plain.step(zero), given.step(zero)
        assert all(np.array_equal(x, y) for x, y in zip(a[:3], b[:3], strict=True))
    assert np.array_equal(plain.data.qpos, given.data.qpos)

    task = replace(plain.tasks[0], prop_x=5.0, hold=4.0, target=math.radians(8.0))
    other = PropScene(1, "machines")
    other.reset(0, SEED, task)
    assert other.tasks[0] is task and other.seeds[0] == SEED
    assert repr(other.machines[0]) == repr(plain.machines[0])
    assert other.tilt(0)[0] == 0.0
    other.k[0] = round((task.hold_span[0] + task.hold / 2) / other.period)
    assert other.tilt(0)[0] == pytest.approx(task.target)


def test_a_given_task_is_refused_on_the_rough_ground_rung():
    scene = PropScene(1, "ground")
    with pytest.raises(ValueError, match="level ground"):
        scene.reset(0, SEED, prop_task(PROP, DEMO_SCENE, SEED))
