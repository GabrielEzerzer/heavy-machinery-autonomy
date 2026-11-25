"""A scene with both layers: machines through the hydraulics layer, each optionally on its soil.
Per physics step every soil sets its resistance and bucket weight, then the hydraulics layer writes
the joint speeds, the physics steps and the layer reads loads that include the soil."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hma.hydraulics import Hydraulics, Machine
from hma.soil import Soil


def step(layer: Hydraulics, soils: Sequence[Soil], commands: np.ndarray) -> None:
    layer.act(commands)
    model, data = layer.model, layer.data
    for _ in range(layer.steps_per_decision):
        for soil in soils:
            soil.substep(model, data)
        layer.step()
    for soil in soils:
        soil.control_step(model, data)


def reset(layer: Hydraulics, soils: Sequence[Soil], index: int, machine: Machine,
          pose: Sequence[float] | None = None, heights: np.ndarray | None = None) -> None:
    own = [soil for soil in soils if soil.index == index]
    for soil in own:
        soil.reset(layer.model, layer.data, heights)
    layer.reset(index, machine, pose)
    for soil in own:
        soil.reset(layer.model, layer.data, heights)


__all__ = ["reset", "step"]
