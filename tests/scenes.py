"""Test scenes: each machine's tracks stand on two strips of ground with a pit around and between
them, so the arm and bucket move through their full range at any swing without touching ground."""

import numpy as np

from hma.machine import ground_axes, write_ground
from hma.machine.model import GROUND_DEPTH

STRIP_X = 1.7
STRIP_Y = (0.6, 1.5)


def dig_pit(model, n_machines=1):
    xs, ys = ground_axes()
    strips = ((np.abs(ys)[:, None] >= STRIP_Y[0]) & (np.abs(ys)[:, None] <= STRIP_Y[1])
              & (np.abs(xs)[None, :] <= STRIP_X))
    heights = np.where(strips, 0.0, -GROUND_DEPTH)
    for i in range(n_machines):
        write_ground(model, i, heights)
    return model
