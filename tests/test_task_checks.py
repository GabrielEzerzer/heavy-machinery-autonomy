import time

from conftest import Everywhere, recorded, spec_json
from hma.task import run_checks, spec_model, validate
from hma.task.checks import (
    Limits,
    check_path,
    check_ranges,
    check_reach,
    check_schema,
    check_trench_rules,
    sample_points,
)


def plan_for(scene, **changes):
    spec = spec_model(scene).model_validate_json(spec_json(**changes))
    result, plan = check_ranges(spec, scene)
    assert result.passed, result.reason
    return plan


def test_schema_accepts_a_recorded_plan(scene):
    result, spec = check_schema(recorded("stage_far_marker")["output"], scene)
    assert result.passed
    assert spec.marker_id == "M3" and spec.depth_m == 1


def test_schema_rejects_an_injected_extra_field(scene):
    result, spec = check_schema(recorded("extra_field")["output"], scene)
    assert not result.passed and spec is None
    assert "limits_off" in result.reason


def test_schema_rejects_a_wrong_type(scene):
    result, _ = check_schema(recorded("wrong_type")["output"], scene)
    assert not result.passed
    assert "depth_m" in result.reason


def test_schema_rejects_unknown_marker_text_and_out_of_bounds(scene):
    for raw in (
        spec_json(marker_id="M9"),
        spec_json(depth_m=51),
        spec_json(depth_m=-1),
        "Sure! Here is the plan: {}",
        spec_json(message="x" * 501),
    ):
        assert not check_schema(raw, scene)[0].passed, raw


def test_schema_strips_control_characters_from_the_message(scene):
    _, spec = check_schema(spec_json(message="Dig\x1b[2J now\n"), scene)
    assert spec.message == "Dig [2J now "


def test_schema_stops_on_a_clarifying_question(scene):
    result, spec = check_schema(recorded("no_place")["output"], scene)
    assert not result.passed and spec.status == "clarify"
    assert "which marker" in result.reason.lower()


def test_ranges_accepts_the_stage_plan_and_applies_defaults(scene):
    plan = plan_for(scene)
    assert (plan.length_m, plan.width_m) == (2.0, 1.0)
    assert plan.heading == "toward_machine" and plan.spoil_side == "right"
    assert len(plan.defaults) == 4


def test_ranges_refuses_sizes_outside_limits(scene):
    model = spec_model(scene)
    for changes, words in (
        ({"width_m": 0.5}, "narrower than the 1 m bucket"),
        ({"width_m": 4}, "wider"),
        ({"length_m": 20}, "Length 20"),
        ({"depth_m": 12}, "Depth 12"),
        ({"depth_m": None}, "no depth"),
        ({"marker_id": None}, "no marker"),
    ):
        result, plan = check_ranges(model.model_validate_json(spec_json(**changes)), scene)
        assert not result.passed and plan is None
        assert words in result.reason, result.reason


def test_geofence_refuses_hazard_and_machine_footprints(scene):
    model = spec_model(scene)
    for changes, words in (
        ({"marker_id": "H1"}, "buried cable"),
        ({"marker_id": "M2", "heading": "across"}, "buried cable"),
        ({"marker_id": "M1"}, "swing axis"),
    ):
        result, _ = check_ranges(model.model_validate_json(spec_json(**changes)), scene)
        assert not result.passed and words in result.reason, result.reason


def test_reach_accepts_the_stage_trench(scene, arm):
    assert check_reach(plan_for(scene), arm).passed


def test_reach_refuses_eight_metres_with_the_depth_the_bucket_can_reach(scene, arm):
    result = check_reach(plan_for(scene, depth_m=8), arm)
    assert not result.passed
    assert "8 m deep" in result.reason and "reaches about" in result.reason


def test_trench_rules_shallow_deep_and_too_deep(scene):
    result, plan, warnings = check_trench_rules(plan_for(scene, soil_class="B"), scene)
    assert result.passed and not plan.no_entry and warnings == []

    result, plan, warnings = check_trench_rules(plan_for(scene, depth_m=2), scene)
    assert result.passed and plan.no_entry
    assert any("NO ENTRY" in w and "1926.652" in w for w in warnings)

    result, _, _ = check_trench_rules(plan_for(scene, depth_m=7), scene)
    assert not result.passed and "registered professional engineer" in result.reason


def test_trench_rules_refuse_spoil_too_close_or_on_a_hazard(scene):
    result, _, _ = check_trench_rules(plan_for(scene), scene, Limits(spoil_gap_m=0.3))
    assert not result.passed and "0.61" in result.reason

    result, _, _ = check_trench_rules(plan_for(scene, spoil_side="left"), scene)
    assert not result.passed and "buried cable" in result.reason


def test_path_accepts_the_stage_trench_quickly(scene, arm):
    plan = plan_for(scene)
    start = time.perf_counter()
    result = check_path(plan, arm)
    assert result.passed
    assert time.perf_counter() - start < 1.0
    assert len(sample_points(plan)) > 100


def test_path_refuses_a_point_in_the_volume_out_of_reach(scene):
    class NoDeeperThanHalfMetre:
        def reachable(self, x, y, z):
            return z >= -0.5

    result = check_path(plan_for(scene), NoDeeperThanHalfMetre())
    assert not result.passed and "out of reach" in result.reason


def test_path_sweeps_given_points(scene, arm):
    assert not check_path(plan_for(scene), arm, points=[(5.5, 0, -1), (30, 0, 0)]).passed


def test_confirmation_is_required(scene, arm):
    raw = recorded("stage_far_marker")["output"]
    assert run_checks(raw, scene, arm, lambda plan: True).approved
    report = run_checks(raw, scene, arm, lambda plan: False)
    assert not report.approved and report.refusal.name == "operator confirmation"
    assert not run_checks(raw, scene, arm, lambda plan: "yes").approved


def test_confirmation_is_not_asked_for_a_refused_plan(scene, arm):
    asked = []
    report = run_checks(recorded("eight_metres")["output"], scene, arm, asked.append)
    assert not report.approved and asked == []
    assert report.refusal.name == "reach"


def test_checks_stop_at_the_first_failure(scene, arm):
    report = validate(spec_json(depth_m=7), scene, Everywhere())
    assert [r.passed for r in report.results] == [True, True, True, False]
    assert len(validate(spec_json(depth_m=7), scene, arm).results) == 3
