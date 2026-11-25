"""Test scenes: each machine's tracks stand on two strips of ground with a pit around and between
them, so the arm and bucket move through their full range at any swing without touching ground."""

from hma.machine.ground import Box, Ground, apply_ground

PIT_DEPTH = 6.0
STRIP_X = 1.7
STRIP_Y = (0.6, 1.5)
PIT = Ground(level=-PIT_DEPTH, boxes=tuple(
    Box(0.0, side * (STRIP_Y[0] + STRIP_Y[1]) / 2, 0.0, STRIP_X, (STRIP_Y[1] - STRIP_Y[0]) / 2,
        PIT_DEPTH) for side in (-1, 1)))


def dig_pit(model, n_machines=1):
    for i in range(n_machines):
        apply_ground(model, i, PIT)
    return model
