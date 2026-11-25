"""Soil for the excavator: a cuttable height grid, bucket fill and dump, and a digging
resistance from the fundamental earthmoving equation, with six soil parameters drawn per episode."""

from hma.soil.force import FORCE_CAP, cutting_force
from hma.soil.params import NOMINAL, SoilParams, sample_soil
from hma.soil.soil import (
    BUCKET_CAPACITY,
    CONTROL_HZ,
    DUMP_ANGLE,
    Soil,
    physics_steps,
    step,
    sync_viewer,
)

__all__ = [
    "BUCKET_CAPACITY",
    "CONTROL_HZ",
    "DUMP_ANGLE",
    "FORCE_CAP",
    "NOMINAL",
    "Soil",
    "SoilParams",
    "cutting_force",
    "physics_steps",
    "sample_soil",
    "step",
    "sync_viewer",
]
