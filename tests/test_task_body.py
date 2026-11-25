import pytest

from hma.machine import statics
from hma.task.body import SIDE_ROWS, Capacity, rating_cells
from hma.task.capacity import cells, compute
from hma.task.capacity_table import TABLE


def side(cells):
    return min(c.side_kg for c in cells)


def test_a_chart_radius_reads_that_column_at_every_height():
    cells = rating_cells(4.5)
    assert {c.radius_m for c in cells} == {4.5}
    column = [float(v[1].lstrip("*")) for k, v in statics.CHART.items() if k[1] == 4.5]
    assert side(cells) == min(column)


def test_between_chart_radii_both_neighbours_count():
    assert side(rating_cells(5.0)) == 1850.0
    assert {c.radius_m for c in rating_cells(5.0)} == {4.5, 5.43, 6.0}
    assert side(rating_cells(6.6)) == 1530.0


def test_no_rating_where_any_height_is_unrated():
    assert rating_cells(3.0) is None
    assert rating_cells(7.1) is None


def test_the_rating_never_rises_with_radius_from_the_first_rated_radius():
    radii = [4.5 + 0.05 * i for i in range(50)]
    ratings = [side(rating_cells(r)) for r in radii]
    assert all(a >= b for a, b in zip(ratings, ratings[1:], strict=False))


def test_every_row_includes_the_at_max_reach_column():
    assert all(len(cells) >= 2 for cells in SIDE_ROWS.values())
    assert SIDE_ROWS[1.5][-1].radius_m == 7.04


def test_machine_body_geometry(body):
    assert 2.0 < body.plan_radius() < 3.0
    x0, x1, y0, y1 = body.footprint()
    assert -2.0 < x0 < -1.4 and 1.4 < x1 < 2.0 and -1.5 < y0 < -1.1 and 1.1 < y1 < 1.5
    assert 7.0 < body.travel_reach(0.3) < 9.0
    angles = body.boom_arm_angles(5.0, 0.0, 0.0)
    assert angles and all(0.0 < a < 180.0 for a in angles)
    assert body.boom_arm_angles(20.0, 0.0, 0.0) == ()


def test_machine_body_rates_a_chart_cell_from_the_capacity_table(body):
    assert body.lift_capacity(6.0, 0.0) == Capacity(*TABLE[(0.0, 6.0)])
    assert body.lift_capacity(7.04, 1.5) is None


def test_the_capacity_table_matches_the_machine_statics():
    stale = "the machine changed: run python -m hma.task.capacity"
    fresh = compute()
    assert fresh.keys() == TABLE.keys() == set(cells()), stale
    for key, value in fresh.items():
        if value is None:
            assert TABLE[key] is None, (key, stale)
        else:
            assert TABLE[key] == pytest.approx(value, abs=0.5), (key, stale)
