# The scene the language model sees: markers, one hazard, and the machine's reach and bucket.
# Frame: origin on the ground at the swing axis, x forward, y left, z up, metres.
# Bearings are degrees from straight ahead, positive to the machine's left.
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Marker:
    id: str
    label: str
    distance_m: float
    bearing_deg: float
    hazard: bool = False
    keep_out_m: float = 0.0

    @property
    def xy(self) -> tuple[float, float]:
        b = math.radians(self.bearing_deg)
        return (self.distance_m * math.cos(b), self.distance_m * math.sin(b))


@dataclass(frozen=True)
class Scene:
    markers: tuple[Marker, ...]
    reach_m: float
    bucket_width_m: float

    def marker(self, marker_id: str) -> Marker | None:
        return next((m for m in self.markers if m.id == marker_id), None)

    @property
    def hazards(self) -> tuple[Marker, ...]:
        return tuple(m for m in self.markers if m.hazard)

    def as_data(self) -> dict:
        return asdict(self)


DEMO_SCENE = Scene(
    markers=(
        Marker("M1", "near marker", 2.5, -20.0),
        Marker("M2", "middle marker", 4.0, 15.0),
        Marker("M3", "far marker", 5.5, 0.0),
        Marker("H1", "buried cable", 4.5, 35.0, hazard=True, keep_out_m=1.0),
    ),
    reach_m=8.1,
    bucket_width_m=1.0,
)
