import math

import numpy as np
import pytest

from hma.soil.ground import MIN_FOOTPRINT, Grid, settle

GRID = Grid(x0=0.0, y0=0.0, z0=0.0, cell=0.1, shape=(21, 31))


def blank():
    return np.zeros(GRID.shape), np.zeros(GRID.shape), np.zeros(GRID.shape, bool)


def test_an_edge_sliding_along_itself_or_straight_down_cuts_nothing():
    heights, loose, protected = blank()
    p, q = (0.5, 0.5, -0.3), (0.5, 1.1, -0.3)
    a, b = (0.5, 0.8, -0.3), (0.5, 1.4, -0.3)
    assert GRID.cut(heights, loose, protected, (p, q, b, a)) == 0.0
    down = ((0.5, 0.5, -0.6), (0.5, 1.1, -0.6))
    assert GRID.cut(heights, loose, protected, (p, q, down[1], down[0])) == 0.0
    assert not heights.any()


def test_a_long_stride_in_one_step_lowers_every_cell_it_sweeps():
    heights, loose, protected = blank()
    p, q = (0.2, 0.5, -0.2), (0.2, 1.1, -0.2)
    a, b = (2.6, 0.5, -0.2), (2.6, 1.1, -0.2)
    removed = GRID.cut(heights, loose, protected, (p, q, b, a))
    swept = heights[5:12, 2:27]
    assert np.all(swept == -0.2)
    assert np.count_nonzero(heights) == swept.size
    assert removed == pytest.approx(0.2 * swept.size * GRID.cell_area)


def test_a_tilted_edge_cuts_to_its_own_height_along_its_length():
    heights, loose, protected = blank()
    p, q = (1.0, 0.0, -0.1), (1.0, 2.0, -0.5)
    a, b = (1.2, 0.0, -0.1), (1.2, 2.0, -0.5)
    GRID.cut(heights, loose, protected, (p, q, b, a))
    assert heights[0, 10] == pytest.approx(-0.1)
    assert heights[20, 10] == pytest.approx(-0.5)
    assert heights[10, 11] == pytest.approx(-0.3)


def test_protected_cells_keep_their_height():
    heights, loose, protected = blank()
    protected[:, :15] = True
    p, q = (0.2, 0.5, -0.2), (0.2, 1.1, -0.2)
    a, b = (2.6, 0.5, -0.2), (2.6, 1.1, -0.2)
    GRID.cut(heights, loose, protected, (p, q, b, a))
    assert not heights[:, :15].any()
    assert heights[5:12, 15:27].min() == -0.2


def test_cutting_takes_loose_spoil_first():
    heights, loose, protected = blank()
    heights[:] = loose[:] = 0.1
    GRID.cut(heights, loose, protected, ((1.0, 0.5, 0.05), (1.0, 1.1, 0.05),
                                         (1.5, 1.1, 0.05), (1.5, 0.5, 0.05)))
    assert loose[8, 12] == pytest.approx(0.05)


def test_deposit_inside_the_grid_keeps_all_volume_and_reports_nothing_lost():
    heights, loose, protected = blank()
    lost = GRID.deposit(heights, loose, protected, 0.3, np.array([1.0, 1.0, 0.0]),
                        np.array([0.6, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]))
    assert lost == 0.0
    assert heights.sum() * GRID.cell_area == pytest.approx(0.3)
    assert np.array_equal(heights, loose)


def test_deposit_off_the_grid_reports_what_left_it():
    heights, loose, protected = blank()
    lost = GRID.deposit(heights, loose, protected, 0.3, np.array([2.8, 1.0, 0.0]),
                        np.array([0.6, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]))
    assert 0.0 < lost < 0.3
    assert heights.sum() * GRID.cell_area + lost == pytest.approx(0.3)


def test_deposit_skips_protected_cells_and_keeps_the_volume_on_the_rest():
    heights, loose, protected = blank()
    protected[:, :12] = True
    lost = GRID.deposit(heights, loose, protected, 0.3, np.array([1.0, 1.0, 0.0]),
                        np.array([0.6, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]))
    assert lost == 0.0
    assert not heights[protected].any()
    assert heights.sum() * GRID.cell_area == pytest.approx(0.3, abs=1e-12)
    heights[:] = 0.0
    protected[:, :25] = True
    lost = GRID.deposit(heights, loose, protected, 0.2, np.array([1.0, 1.0, 0.0]),
                        np.array([0.6, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]))
    assert lost == 0.0
    assert not heights[protected].any()
    assert heights[:, 25].any()
    assert heights.sum() * GRID.cell_area == pytest.approx(0.2, abs=1e-12)


def test_a_deposit_along_its_own_span_still_covers_the_minimum_footprint():
    heights, loose, protected = blank()
    lost = GRID.deposit(heights, loose, protected, 0.2, np.array([1.0, 1.0, 0.0]),
                        np.array([0.0, 0.6, 0.0]), np.array([0.0, 0.1, 0.0]))
    rows, cols = np.nonzero(heights)
    assert lost == 0.0
    assert (cols.max() - cols.min()) * GRID.cell >= MIN_FOOTPRINT
    assert (rows.max() - rows.min()) * GRID.cell >= 0.6
    assert heights.sum() * GRID.cell_area == pytest.approx(0.2, abs=1e-12)
    heights, loose, protected = blank()
    GRID.deposit(heights, loose, protected, 0.2, np.array([1.0, 1.0, 0.0]),
                 np.zeros(3), np.zeros(3))
    rows, cols = np.nonzero(heights)
    assert min(np.ptp(rows), np.ptp(cols)) * GRID.cell >= MIN_FOOTPRINT
    assert heights.sum() * GRID.cell_area == pytest.approx(0.2, abs=1e-12)


def test_loose_spoil_sheds_to_its_repose_angle_and_keeps_its_volume():
    heights, loose, protected = blank()
    heights[10, 15] = loose[10, 15] = 1.0
    repose = 0.6
    for _ in range(200):
        settle(heights, loose, protected, repose, GRID.cell, 10)
    assert heights.sum() == pytest.approx(1.0)
    assert heights.max() < 0.5
    steep = np.abs(np.diff(heights, axis=1)).max(), np.abs(np.diff(heights, axis=0)).max()
    assert max(steep) <= GRID.cell * math.tan(repose) + 1e-3


def test_undisturbed_walls_stand():
    heights, loose, protected = blank()
    heights[:, :10] = -1.0
    before = heights.copy()
    settle(heights, loose, protected, 0.3, GRID.cell, 50)
    assert np.array_equal(heights, before)


def test_spoil_on_a_wall_top_sheds_without_eroding_the_wall():
    heights, loose, protected = blank()
    heights[:, :10] = -1.0
    heights[:, 10] += 0.3
    loose[:, 10] = 0.3
    for _ in range(50):
        settle(heights, loose, protected, 0.5, GRID.cell, 10)
    assert heights.sum() == pytest.approx(-10 * 21 + 0.3 * 21)
    assert np.all(heights[:, 11:] >= 0.0)
    assert np.all(heights[:, 10] >= 0.0)


def test_spoil_does_not_shed_onto_protected_cells():
    heights, loose, protected = blank()
    heights[10, 15] = loose[10, 15] = 1.0
    protected[:, 16:] = True
    for _ in range(50):
        settle(heights, loose, protected, 0.5, GRID.cell, 10)
    assert not heights[protected].any()
    assert heights.sum() == pytest.approx(1.0)


def test_deposit_onto_a_fully_protected_grid_reports_all_volume_lost():
    heights, loose, protected = blank()
    protected[:] = True
    lost = GRID.deposit(heights, loose, protected, 0.2, np.array([1.0, 1.0, 0.0]),
                        np.array([0.6, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]))
    assert lost == 0.2
    assert not heights.any() and not loose.any()
