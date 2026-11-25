"""Height-grid operations in world coordinates: surface lookup, cutting along a swept edge,
depositing volume over an area, and shedding loose spoil steeper than its repose angle. Every
change moves volume between cells or reports it as lost, so totals are conserved by construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hma.machine.model import GROUND_DEPTH, GROUND_TOP

INSIDE_TOLERANCE = 1e-9
SHED_SHARE = 0.125
SETTLE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    z0: float
    cell: float
    shape: tuple[int, int]

    @property
    def cell_area(self) -> float:
        return self.cell * self.cell

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        nrow, ncol = self.shape
        return self.x0 + self.cell * np.arange(ncol), self.y0 + self.cell * np.arange(nrow)

    def height(self, heights: np.ndarray, x: float, y: float) -> float:
        nrow, ncol = self.shape
        fx, fy = (x - self.x0) / self.cell, (y - self.y0) / self.cell
        if not (0.0 <= fx <= ncol - 1 and 0.0 <= fy <= nrow - 1):
            return -math.inf
        cx, cy = min(int(fx), ncol - 2), min(int(fy), nrow - 2)
        tx, ty = fx - cx, fy - cy
        low = heights[cy, cx] * (1 - tx) + heights[cy, cx + 1] * tx
        high = heights[cy + 1, cx] * (1 - tx) + heights[cy + 1, cx + 1] * tx
        return float(low * (1 - ty) + high * ty) + self.z0

    def clearance(self, heights: np.ndarray, points: tuple, reach: int) -> float:
        nrow, ncol = self.shape
        xs, ys = [c[0] for c in points], [c[1] for c in points]
        c0 = max(math.floor((min(xs) - self.x0) / self.cell) - reach, 0)
        c1 = min(math.ceil((max(xs) - self.x0) / self.cell) + reach, ncol - 1)
        r0 = max(math.floor((min(ys) - self.y0) / self.cell) - reach, 0)
        r1 = min(math.ceil((max(ys) - self.y0) / self.cell) + reach, nrow - 1)
        if c0 > c1 or r0 > r1:
            return math.inf
        top = float(heights[r0:r1 + 1, c0:c1 + 1].max())
        return min(c[2] for c in points) - self.z0 - top

    def cut(self, heights: np.ndarray, loose: np.ndarray, protected: np.ndarray,
            corners: tuple) -> float:
        nrow, ncol = self.shape
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        local = [c[2] - self.z0 for c in corners]
        c0 = max(math.ceil((min(xs) - self.x0) / self.cell - INSIDE_TOLERANCE), 0)
        c1 = min(math.floor((max(xs) - self.x0) / self.cell + INSIDE_TOLERANCE), ncol - 1)
        r0 = max(math.ceil((min(ys) - self.y0) / self.cell - INSIDE_TOLERANCE), 0)
        r1 = min(math.floor((max(ys) - self.y0) / self.cell + INSIDE_TOLERANCE), nrow - 1)
        if c0 > c1 or r0 > r1:
            return 0.0
        window = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        old = heights[window]
        if min(local) >= old.max():
            return 0.0
        gx = (self.x0 + self.cell * np.arange(c0, c1 + 1))[None, :]
        gy = (self.y0 + self.cell * np.arange(r0, r1 + 1))[:, None]
        path = np.full(old.shape, np.inf)
        for i, j, k in ((0, 1, 2), (0, 2, 3)):
            px, py = xs[i], ys[i]
            e1x, e1y, e2x, e2y = xs[j] - px, ys[j] - py, xs[k] - px, ys[k] - py
            det = e1x * e2y - e2x * e1y
            if abs(det) < 1e-12:
                continue
            dx, dy = gx - px, gy - py
            u = (dx * e2y - e2x * dy) / det
            v = (e1x * dy - dx * e1y) / det
            inside = ((u >= -INSIDE_TOLERANCE) & (v >= -INSIDE_TOLERANCE)
                      & (u + v <= 1 + INSIDE_TOLERANCE))
            z = local[i] + u * (local[j] - local[i]) + v * (local[k] - local[i])
            path = np.where(inside, np.minimum(path, z), path)
        new = np.where(protected[window], old, np.minimum(old, np.maximum(path, -GROUND_DEPTH)))
        lowered = old - new
        heights[window] = new
        loose[window] = np.maximum(loose[window] - lowered, 0.0)
        return float(lowered.sum()) * self.cell_area

    def deposit(self, heights: np.ndarray, loose: np.ndarray, volume: float, origin: np.ndarray,
                span: np.ndarray, depth: np.ndarray) -> float:
        if volume <= 0.0:
            return 0.0
        n_span = int(np.ceil(np.linalg.norm(span[:2]) * 2 / self.cell)) + 1
        n_depth = int(np.ceil(np.linalg.norm(depth[:2]) * 2 / self.cell)) + 1
        s, t = np.meshgrid(np.linspace(0, 1, n_span), np.linspace(0, 1, n_depth))
        points = origin[:2] + s.reshape(-1, 1) * span[:2] + t.reshape(-1, 1) * depth[:2]
        col = np.rint((points[:, 0] - self.x0) / self.cell).astype(int)
        row = np.rint((points[:, 1] - self.y0) / self.cell).astype(int)
        nrow, ncol = self.shape
        inside = (col >= 0) & (col < ncol) & (row >= 0) & (row < nrow)
        share = volume / len(points)
        flat = np.bincount(row[inside] * ncol + col[inside], minlength=nrow * ncol)
        added = (flat.reshape(self.shape) * (share / self.cell_area))
        room = np.maximum(GROUND_TOP - heights, 0.0)
        kept = np.minimum(added, room)
        heights += kept
        loose += kept
        if inside.all() and (added <= room).all():
            return 0.0
        return volume - float(kept.sum()) * self.cell_area


def settle(heights: np.ndarray, loose: np.ndarray, repose: float, cell: float,
           passes: int) -> None:
    rows, cols = np.nonzero(loose > 0.0)
    if rows.size == 0:
        return
    window = (slice(max(rows.min() - 1, 0), rows.max() + 2),
              slice(max(cols.min() - 1, 0), cols.max() + 2))
    h, spoil = heights[window], loose[window]
    limit = cell * np.tan(repose)
    for _ in range(passes):
        east = np.maximum(h[:, :-1] - h[:, 1:] - limit, 0.0) * SHED_SHARE
        west = np.maximum(h[:, 1:] - h[:, :-1] - limit, 0.0) * SHED_SHARE
        north = np.maximum(h[:-1, :] - h[1:, :] - limit, 0.0) * SHED_SHARE
        south = np.maximum(h[1:, :] - h[:-1, :] - limit, 0.0) * SHED_SHARE
        out = np.zeros_like(h)
        out[:, :-1] += east
        out[:, 1:] += west
        out[:-1, :] += north
        out[1:, :] += south
        scale = np.where(out > spoil, spoil / np.where(out > 0.0, out, 1.0), 1.0)
        east *= scale[:, :-1]
        west *= scale[:, 1:]
        north *= scale[:-1, :]
        south *= scale[1:, :]
        if max(east.max(), west.max(), north.max(), south.max()) < SETTLE_TOLERANCE:
            break
        moved = np.zeros_like(h)
        moved[:, :-1] -= east
        moved[:, 1:] += east
        moved[:, 1:] -= west
        moved[:, :-1] += west
        moved[:-1, :] -= north
        moved[1:, :] += north
        moved[1:, :] -= south
        moved[:-1, :] += south
        h += moved
        np.maximum(spoil + moved, 0.0, out=spoil)
    heights[window] = h
    loose[window] = spoil
