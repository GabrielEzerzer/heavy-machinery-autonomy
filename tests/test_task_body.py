from hma.task.body import SIDE_CHART, chart_rating


def test_a_chart_point_reads_the_lowest_rating_at_that_radius_over_every_height():
    assert chart_rating(4.5) == min(dict(row)[4.5] for row in SIDE_CHART.values())
    assert chart_rating(3.0) == 3350.0


def test_between_chart_radii_the_lower_neighbour_rating_applies():
    assert chart_rating(5.0) == 1850.0
    assert chart_rating(6.6) == 1530.0


def test_no_rating_outside_the_chart():
    assert chart_rating(2.9) is None
    assert chart_rating(7.1) is None


def test_the_rating_never_rises_with_radius_from_three_metres():
    radii = [3.0 + 0.05 * i for i in range(81)]
    ratings = [chart_rating(r) for r in radii]
    assert all(a >= b for a, b in zip(ratings, ratings[1:], strict=False))


def test_machine_body_geometry(body):
    assert 2.0 < body.plan_radius() < 3.0
    assert body.lift_capacity(5.0, 0.0) is None
    angles = body.boom_arm_angles(5.0, 0.0, 0.0)
    assert angles and all(0.0 < a < 180.0 for a in angles)
    assert body.boom_arm_angles(20.0, 0.0, 0.0) == ()
