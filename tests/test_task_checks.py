import dataclasses
import json
import math
import time

import pytest

from conftest import Everywhere, StandIn, recorded, spec_json, task_json
from hma.task import Hazard, Load, Marker, Pose, Zone, run_checks, spec_model, validate
from hma.task.checks import (
    check_fields,
    check_path,
    check_reach,
    check_rules,
    check_schema,
    lift_samples,
)
from hma.task.scene import HazardKind

CHAIN = ["schema", "fields and ranges", "reach", "safety rules", "path"]
WIDE = ((-40.0, -40.0), (40.0, -40.0), (40.0, 40.0), (-40.0, 40.0))


def box(x0, x1, y0, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def with_zone(scene, *zones):
    return dataclasses.replace(scene, zones=scene.zones + zones)


def with_hazard(scene, *hazards):
    return dataclasses.replace(scene, hazards=scene.hazards + hazards)


def plan_for(scene, raw=None, **changes):
    spec = spec_model(scene).model_validate_json(raw or spec_json(**changes))
    result, plan = check_fields(spec, scene)
    assert result.passed, result.reason
    return plan


def reached(scene, raw, reach, body):
    result, plan = check_reach(plan_for(scene, raw), scene, reach, body)
    assert result.passed, result.reason
    return plan


def rules(scene, raw, body, reach=None):
    return check_rules(reached(scene, raw, reach or Everywhere(), body), scene, body)[0]


def test_schema_accepts_a_recorded_plan(scene):
    result, spec = check_schema(recorded("carry_pallet")["output"], scene)
    assert result.passed
    assert (spec.task_type, spec.load_id, spec.marker_id) == ("carry", "L1", "M1")


def test_schema_rejects_an_injected_extra_field(scene):
    result, spec = check_schema(recorded("extra_field")["output"], scene)
    assert not result.passed and spec is None
    assert "unexpected field" in result.reason and "limits_off" not in result.reason


def test_schema_error_never_echoes_a_model_key(scene):
    key = "\x1b[2J\x1b[H" + "APPROVED " * 2500
    raw = json.dumps({**json.loads(spec_json()), key: 1})
    result, _ = check_schema(raw, scene)
    assert not result.passed and len(result.reason) < 300
    assert result.reason.isprintable() and "APPROVED" not in result.reason


def test_schema_rejects_a_wrong_type(scene):
    result, _ = check_schema(recorded("wrong_type")["output"], scene)
    assert not result.passed and "hold_s" in result.reason


def test_schema_rejects_unknown_names_text_and_out_of_bounds(scene):
    for raw in (
        spec_json(marker_id="M9"),
        spec_json(load_id="L9"),
        spec_json(marker_id="H1"),
        spec_json(stand_marker_id="L1"),
        spec_json(task_type="trench"),
        task_json(task_type="prop", hold_s=3601),
        task_json(task_type="prop", hold_s=-1),
        "Sure! Here is the plan: {}",
        spec_json(message="x" * 501),
    ):
        assert not check_schema(raw, scene)[0].passed, raw


def test_schema_strips_control_characters_from_the_message(scene):
    _, spec = check_schema(spec_json(message="Lift\x1b[2J now\n"), scene)
    assert spec.message == "Lift [2J now "


def test_schema_stops_on_a_clarifying_question(scene):
    result, spec = check_schema(recorded("no_place")["output"], scene)
    assert not result.passed and spec.status == "clarify"


def test_schema_stops_on_a_refusal(scene):
    result, spec = check_schema(recorded("dig_trench")["output"], scene)
    assert not result.passed and spec.status == "refuse"


def test_fields_build_a_plan_and_a_standing_place(scene):
    plan = plan_for(scene)
    assert plan.load.id == "L1" and plan.target.id == "M1" and not plan.travels
    plan = plan_for(scene, stand_marker_id="M2")
    assert plan.travels and plan.stand.xy == (6.0, 1.5)


def test_fields_refuse_missing_extra_and_pointless_fields(scene):
    model = spec_model(scene)
    at_m2 = dataclasses.replace(scene, machine=Pose(6.0, 1.5, 0.0))
    for raw, words, where in (
        (spec_json(task_type=None), "no task type", scene),
        (spec_json(load_id=None), "needs load_id", scene),
        (task_json(task_type="prop"), "needs hold_s", scene),
        (task_json(task_type="travel", marker_id="M2", hold_s=5), "takes no hold_s", scene),
        (task_json(task_type="prop", hold_s=5, marker_id="M1"), "takes no marker_id", scene),
        (task_json(task_type="travel", marker_id="M2"), "already stands", at_m2),
        (task_json(task_type="turn", marker_id="M2"), "cannot face it", at_m2),
    ):
        result, plan = check_fields(model.model_validate_json(raw), where)
        assert not result.passed and plan is None
        assert words in result.reason, result.reason


def test_reach_accepts_the_pallet_and_refuses_a_far_set_down(scene, arm, body):
    assert check_reach(plan_for(scene), scene, arm, body)[0].passed
    result, _ = check_reach(plan_for(scene, marker_id="M3"), scene, arm, body)
    assert not result.passed and "17.0 m" in result.reason
    assert "out of the bucket's reach" in result.reason


def test_reach_measures_from_where_the_machine_will_stand(scene, arm, body):
    result, _ = check_reach(plan_for(scene, stand_marker_id="M5"), scene, arm, body)
    assert not result.passed and "12.9 m from where the machine stands" in result.reason
    assert check_reach(plan_for(scene, stand_marker_id="M2"), scene, arm, body)[0].passed


def test_reach_finds_a_prop_point_with_boom_and_arm_at_90_to_110_degrees(scene, arm, body):
    raw = task_json(task_type="prop", hold_s=10)
    result, plan = check_reach(plan_for(scene, raw), scene, arm, body)
    assert result.passed and plan.prop_xy[1] == 0.0 and 3.0 < plan.prop_xy[0] < 8.0
    result, plan = check_reach(plan_for(scene, raw), scene, arm, StandIn(angle=120.0))
    assert not result.passed and "90 to 110 degrees" in result.reason


def test_reach_passes_travel(scene, arm, body):
    raw = task_json(task_type="travel", marker_id="M3")
    assert check_reach(plan_for(scene, raw), scene, arm, body)[0].passed


def test_rated_capacity_from_the_manufacturer_chart(scene, body):
    assert rules(scene, spec_json(), body).passed
    result = rules(scene, spec_json(load_id="L2"), body)
    assert not result.passed
    assert "4000 kg) is over the rated lift capacity of 1130 kg" in result.reason
    assert "over-side rating of 1530 kg at 7.04 m radius and 1.5 m height" in result.reason
    assert "less the 400 kg bucket" in result.reason


def test_a_stronger_model_never_raises_the_chart_rating(scene):
    result = rules(scene, spec_json(load_id="L2"), StandIn(tipping=1e6, hydraulic=1e6))
    assert not result.passed and "over-side rating" in result.reason


def test_rated_capacity_from_the_model_tipping_side(scene):
    result = rules(scene, spec_json(), StandIn(tipping=1500.0, hydraulic=1e6))
    assert not result.passed and "capacity of 725 kg" in result.reason
    assert "75% of the model's 1500 kg tipping load" in result.reason
    assert rules(scene, spec_json(), StandIn(tipping=2000.0, hydraulic=1e6)).passed


def test_rated_capacity_from_the_model_hydraulic_side(scene):
    result = rules(scene, spec_json(), StandIn(tipping=1e6, hydraulic=1300.0))
    assert not result.passed and "capacity of 731 kg" in result.reason
    assert "87% of the model's 1300 kg hydraulic" in result.reason
    assert rules(scene, spec_json(), StandIn(tipping=1e6, hydraulic=2000.0)).passed


def test_nothing_is_lifted_where_the_chart_does_not_rate_every_height(scene, body):
    close = dataclasses.replace(
        scene, loads=scene.loads + (Load("L9", "kerb unit", 300.0, 2.9, -0.5, 0.3),)
    )
    result = rules(close, spec_json(load_id="L9"), body)
    assert not result.passed and "does not rate every height" in result.reason


def test_a_load_of_unknown_mass_is_not_lifted(scene, body):
    result = rules(scene, spec_json(load_id="L3"), body)
    assert not result.passed and "not known" in result.reason


def test_lifting_needs_level_ground(scene, body):
    flat = with_zone(scene, Zone("T", "test", box(-5, 5, -5, 5), 0.0, 0.5, 0.0))
    assert rules(flat, spec_json(), body).passed
    tilted = with_zone(scene, Zone("T", "test", box(-5, 5, -5, 5), 0.0, 2.0, 0.0))
    result = rules(tilted, spec_json(), body)
    assert not result.passed and "limit for lifting is 0.6 degrees" in result.reason


def test_lifting_needs_one_surface_under_the_machine(scene, body):
    kerb = with_zone(scene, Zone("T", "kerb", box(-3, -1.5, -3, 3), 0.2, 0.0, 0.0))
    result = rules(kerb, spec_json(), body)
    assert not result.passed and "not one surface" in result.reason
    tilted_ahead = with_zone(scene, Zone("T", "slope", box(1, 10, -10, 10), 0.0, 0.5, 0.0))
    result = rules(tilted_ahead, spec_json(), body)
    assert not result.passed and "more than one slope; lifting" in result.reason


def test_the_whole_track_footprint_must_be_level_for_lifting(scene, body):
    slope_ahead = with_zone(scene, Zone("T", "slope", box(1, 10, -10, 10), 0.0, 20.0, 0.0))
    result = rules(slope_ahead, spec_json(), body)
    assert not result.passed and "under the tracks tilts the machine 20.0 degrees" in result.reason
    clear = with_zone(scene, Zone("T", "slope", box(2, 10, -10, 10), 0.0, 20.0, 0.0))
    result = rules(clear, spec_json(), body)
    assert "tracks" not in result.reason


def edge_rise(scene, a, b, pieces=40):
    nx, ny = b[1] - a[1], a[0] - b[0]
    n = math.hypot(nx, ny)
    rises = []
    for k in range(pieces + 1):
        t = k / pieces
        m = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        out = scene.ground((m[0] + nx / n * 1e-4, m[1] + ny / n * 1e-4)).height_m
        inn = scene.ground((m[0] - nx / n * 1e-4, m[1] - ny / n * 1e-4)).height_m
        rises.append(abs(out - inn))
    return max(rises)


def test_demo_ground_joins_up_where_the_machine_drives_on(scene):
    for a, b in (
        ((-4.0, -2.99), (-4.0, 2.99)),
        ((-8.0, -2.99), (-8.0, 2.99)),
        ((8.0, -3.99), (8.0, 3.99)),
        ((14.0, -3.99), (14.0, 3.99)),
        ((-3.99, 6.0), (5.99, 6.0)),
    ):
        assert edge_rise(scene, a, b) < 0.01, (a, b)
    assert edge_rise(scene, (14.0, 4.01), (14.0, 4.99)) > 1.0
    bank = [scene.ground((-4.0 - 0.5 * i, 0.0)).height_m for i in range(1, 9)]
    assert all(h2 < h1 < 0 for h1, h2 in zip(bank, bank[1:], strict=False))
    assert scene.ground((-6.0, 0.0)).height_m == pytest.approx(-2 * math.tan(math.radians(35)))
    assert scene.ground((-10.0, 0.0)).height_m == pytest.approx(bank[-1])


def test_travel_slope_along_and_across(scene, body):
    assert rules(scene, task_json(task_type="travel", marker_id="M5"), body).passed
    result = rules(scene, task_json(task_type="travel", marker_id="M6"), body)
    assert not result.passed and "35.0 degrees along" in result.reason
    side = with_zone(scene, Zone("T", "side", box(4, 8, -1, 4), 0.0, 12.0, 90.0))
    result = rules(side, task_json(task_type="travel", marker_id="M2"), body)
    assert not result.passed and "across" in result.reason
    assert "limit for travel is 10" in result.reason


def test_propping_slope_limits(scene, body):
    raw = task_json(task_type="prop", hold_s=10)
    along = with_zone(scene, Zone("T", "slope", WIDE, 0.0, 9.0, 0.0))
    assert rules(along, raw, body).passed
    steep = with_zone(scene, Zone("T", "slope", WIDE, 0.0, 20.0, 0.0))
    result = rules(steep, raw, body)
    assert not result.passed and "limit for propping is 15 degrees" in result.reason
    part = with_zone(scene, Zone("T", "slope", box(1, 10, -10, 10), 0.0, 5.0, 0.0))
    result = rules(part, raw, body)
    assert not result.passed and "more than one slope; propping" in result.reason


def test_overhead_keep_out_for_machine_bucket_and_load(scene, body):
    result = rules(scene, task_json(task_type="travel", marker_id="M7"), body)
    assert not result.passed and "overhead power line (H1)" in result.reason
    line = Hazard("T", "test line", HazardKind.overhead, ((2.0, -6.0), (9.0, -6.0)), 3.0)
    result = rules(with_hazard(scene, line), spec_json(), body)
    assert not result.passed and "bucket, arm or load" in result.reason
    over_arm = Hazard("T", "test line", HazardKind.overhead, ((2.0, -0.2), (2.0, -0.6)), 0.5)
    result = rules(with_hazard(scene, over_arm), spec_json(), body)
    assert not result.passed and "test line" in result.reason
    behind = Hazard("T", "pole", HazardKind.overhead, ((-3.0, 0.0),), 0.5)
    result = rules(with_hazard(scene, behind), spec_json(), body)
    assert not result.passed and "the machine comes within 0.4 m of the pole" in result.reason
    raw = task_json(task_type="travel", marker_id="M2")
    result = rules(with_hazard(scene, behind), raw, body)
    assert not result.passed and "At (0.0, 0.0) the machine comes within" in result.reason
    shadow = Hazard("T", "pole", HazardKind.overhead, ((3.5, -2.1),), 0.5)
    result = rules(with_hazard(scene, shadow), spec_json(), body)
    assert not result.passed
    assert "bucket, arm or load comes within 0.0 m of the pole" in result.reason


def test_overhead_keep_out_over_the_propping_arm(scene, body):
    raw = task_json(task_type="prop", hold_s=10)
    assert rules(scene, raw, body).passed
    across = Hazard("T", "test line", HazardKind.overhead, ((4.0, -1.0), (4.0, 1.0)), 0.5)
    result = rules(with_hazard(scene, across), raw, body)
    assert not result.passed and "bucket pressing" in result.reason and "test line" in result.reason


def test_overhead_keep_out_counts_the_arm_held_ahead_for_travel(scene, body):
    raw = task_json(task_type="travel", marker_id="M2")
    plan = plan_for(scene, raw)
    h = math.radians(plan.stand.heading_deg)
    front = (6.0 + 9.0 * math.cos(h), 1.5 + 9.0 * math.sin(h))
    result = rules(with_hazard(scene, Hazard("T", "pole", HazardKind.overhead, (front,), 6.0)),
                   raw, body)
    assert not result.passed and "arm, held ahead for travel, comes within 0.9 m" in result.reason
    behind = (-12.0 * math.cos(h), -12.0 * math.sin(h))
    clear = with_hazard(scene, Hazard("T", "pole", HazardKind.overhead, (behind,), 6.0))
    assert rules(clear, raw, body).passed


def test_buried_keep_out_where_the_bucket_meets_the_ground(scene, body):
    result = rules(scene, spec_json(marker_id="M4"), body)
    assert not result.passed and "buried cable (H2)" in result.reason
    under_arm = Hazard("T", "pipe", HazardKind.buried, ((2.0, -0.2), (2.0, -0.6)), 0.5)
    assert rules(with_hazard(scene, under_arm), spec_json(), body).passed
    raw = task_json(task_type="prop", hold_s=10)
    x = reached(scene, raw, Everywhere(), body).prop_xy[0]
    ahead = Hazard("T", "pipe", HazardKind.buried, ((x, -3.0), (x, 3.0)), 0.5)
    result = rules(with_hazard(scene, ahead), raw, body)
    assert not result.passed and "bucket pressing" in result.reason


def test_edges_keep_the_machine_back(scene, body):
    assert rules(scene, task_json(task_type="travel", marker_id="M3"), body).passed
    narrow = dataclasses.replace(
        scene, zones=tuple(
            dataclasses.replace(z, corners=box(14, 20, -5, 5)) if z.id == "Z2" else z
            for z in scene.zones
        ),
    )
    result = rules(narrow, task_json(task_type="travel", marker_id="M3"), body)
    assert not result.passed and "must stay 0.61 m back" in result.reason


def test_hold_time_limits(scene, body):
    assert rules(scene, task_json(task_type="prop", hold_s=60), body).passed
    for hold in (600, 0.5):
        result = rules(scene, task_json(task_type="prop", hold_s=hold), body)
        assert not result.passed and "outside 1 to 60 s" in result.reason


def test_ground_under_the_bucket(scene, body):
    raw = task_json(task_type="prop", hold_s=10)
    steep = with_zone(scene, Zone("T", "bank", box(3, 9, -3, 3), 0.0, 12.0, 0.0))
    result = rules(steep, raw, body)
    assert not result.passed and "under the bucket" in result.reason
    assert "12 degrees" in result.reason
    kerb_edge = with_zone(scene, Zone("T", "kerb", box(4.4, 9, -3, 3), 0.1, 0.0, 0.0))
    reach_plan = reached(kerb_edge, raw, Everywhere(), StandIn(angle=100.0))
    result = check_rules(dataclasses.replace(reach_plan, prop_xy=(4.5, 0.0)), kerb_edge, body)[0]
    assert not result.passed and "flat on one surface" in result.reason


def test_turn_uses_the_propping_rules(scene, body):
    raw = task_json(task_type="turn", marker_id="M2")
    assert rules(scene, raw, body).passed
    steep = with_zone(scene, Zone("T", "slope", WIDE, 0.0, 11.0, 90.0))
    result = rules(steep, raw, body)
    assert not result.passed and "across" in result.reason


def test_a_turn_is_checked_at_the_heading_it_ends_on(scene, body):
    along = with_zone(scene, Zone("T", "slope", WIDE, 0.0, 14.0, 0.0))
    result = rules(along, task_json(task_type="turn", marker_id="M5"), body)
    assert not result.passed and "14.0 degrees across them" in result.reason


def test_a_turn_is_checked_at_every_heading_it_passes(scene, body):
    along = with_zone(scene, Zone("T", "slope", WIDE, 0.0, 14.0, 0.0))
    plan = reached(along, task_json(task_type="turn", marker_id="M6"), Everywhere(), body)
    result = check_path(plan, along, Everywhere(), body)
    assert not result.passed and "Turning, at" in result.reason
    assert "degrees across them; the limit for propping is 10" in result.reason
    plan = reached(scene, task_json(task_type="turn", marker_id="M6"), Everywhere(), body)
    assert check_path(plan, scene, Everywhere(), body).passed
    plan = reached(along, task_json(task_type="turn", marker_id="M1"), Everywhere(), body)
    assert check_path(plan, along, Everywhere(), body).passed


def test_path_accepts_the_pallet_route_quickly(scene, arm, body):
    plan = reached(scene, spec_json(), arm, body)
    start = time.perf_counter()
    assert check_path(plan, scene, arm, body).passed
    assert time.perf_counter() - start < 2.0
    assert len(lift_samples(plan, scene)) > 10


def test_path_refuses_a_point_out_of_reach(scene, body):
    class NoFurtherThanFiveMetres:
        def reachable(self, x, y, z):
            return x * x + y * y <= 25.5

    plan = reached(scene, spec_json(marker_id="M2"), Everywhere(), body)
    result = check_path(plan, scene, NoFurtherThanFiveMetres(), body)
    assert not result.passed and "out of reach" in result.reason


def test_path_refuses_a_route_below_the_ground(scene, arm, body):
    plan = reached(scene, spec_json(), arm, body)
    result = check_path(plan, scene, arm, body, points=[(5.0, -2.0, -0.2)])
    assert not result.passed and "below the ground" in result.reason


def test_path_refuses_a_keep_out_between_the_ends(scene, arm, body):
    mid = Hazard("T", "lamp post", HazardKind.overhead, ((5.5, -0.75),), 1.0)
    hazy = with_hazard(scene, mid)
    plan = reached(hazy, spec_json(marker_id="M2"), arm, body)
    assert check_rules(plan, hazy, body)[0].passed
    result = check_path(plan, hazy, arm, body)
    assert not result.passed and "lamp post" in result.reason


def test_path_refuses_capacity_between_the_ends(scene, arm, body):
    across = dataclasses.replace(
        scene,
        loads=scene.loads + (Load("L9", "kerb unit", 300.0, 4.0, -2.4, 0.3),),
        markers=scene.markers + (Marker("M9", "other side", 4.0, 2.4),),
    )
    plan = reached(across, spec_json(load_id="L9", marker_id="M9"), arm, body)
    assert check_rules(plan, across, body)[0].passed
    result = check_path(plan, across, arm, body)
    assert not result.passed and "does not rate every height" in result.reason


def test_path_refuses_a_step_on_the_travel_route(scene, body):
    for height, passes in ((0.2, True), (0.5, False)):
        step = with_zone(scene, Zone("T", "step", box(-3, 4, 4, 5), height, 0.0, 0.0))
        plan = reached(step, task_json(task_type="travel", marker_id="M5"), Everywhere(), body)
        assert check_rules(plan, step, body)[0].passed
        result = check_path(plan, step, Everywhere(), body)
        assert result.passed is passes, result.reason


def test_confirmation_is_required(scene, arm, body):
    raw = recorded("carry_pallet")["output"]
    approved = run_checks(raw, scene, arm, body, lambda plan: True)
    assert approved.approved and approved.validated
    report = run_checks(raw, scene, arm, body, lambda plan: False)
    assert not report.approved and report.refusal.name == "operator confirmation"
    assert report.validated
    assert not run_checks(raw, scene, arm, body, lambda plan: "yes").approved


def test_confirmation_is_not_asked_for_a_refused_plan(scene, arm, body):
    asked = []
    report = run_checks(recorded("heavy_pipe")["output"], scene, arm, body, asked.append)
    assert not report.approved and asked == []
    assert report.refusal.name == "safety rules"


@pytest.mark.parametrize(
    ("raw", "stop"),
    [
        (spec_json(load_id="L9"), "schema"),
        (spec_json(load_id=None), "fields and ranges"),
        (spec_json(marker_id="M3"), "reach"),
        (spec_json(load_id="L2"), "safety rules"),
    ],
)
def test_each_check_refuses_in_the_chain(scene, arm, body, raw, stop):
    report = validate(raw, scene, arm, body)
    assert report.refusal.name == stop
    assert [r.name for r in report.results] == CHAIN[: CHAIN.index(stop) + 1]


def test_path_refuses_in_the_chain(scene, arm, body):
    hazy = with_hazard(scene, Hazard("T", "lamp post", HazardKind.overhead, ((5.5, -0.75),), 1.0))
    report = validate(spec_json(marker_id="M2"), hazy, arm, body)
    assert report.refusal.name == "path" and [r.name for r in report.results] == CHAIN


def test_every_check_runs_in_order(scene, arm, body):
    report = validate(spec_json(), scene, arm, body)
    assert [r.name for r in report.results] == CHAIN and report.validated


def test_scene_rejects_malformed_hazards_and_zones():
    with pytest.raises(ValueError):
        Hazard("X", "x", HazardKind.overhead, (), 1.0)
    with pytest.raises(ValueError):
        Zone("X", "x", ((0, 0), (1, 0)), 0.0, 0.0, 0.0)
    assert Load("X", "x", None, 0, 0, 0.5).mass_kg is None
