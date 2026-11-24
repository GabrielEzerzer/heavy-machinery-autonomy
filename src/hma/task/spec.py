# The task specification the model fills in, and the strict JSON schema handed to it.
# Bounds are wide on purpose: the model must be able to write down an unsafe request,
# so that the validator, not the model, refuses it.
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, create_model

from hma.task.scene import Scene

MESSAGE_MAX_CHARS = 500


class Status(StrEnum):
    plan = "plan"
    clarify = "clarify"
    refuse = "refuse"


class TaskType(StrEnum):
    trench = "trench"


class Heading(StrEnum):
    toward_machine = "toward_machine"
    across = "across"


class SpoilSide(StrEnum):
    left = "left"
    right = "right"


class SoilClass(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    unknown = "unknown"


def printable(text: str) -> str:
    return "".join(c if c.isprintable() else " " for c in text)


def _printable(text: str) -> str:
    if len(text) > MESSAGE_MAX_CHARS:
        raise ValueError(f"message is longer than {MESSAGE_MAX_CHARS} characters")
    return printable(text)


Metres = Annotated[float, Field(ge=0, le=50, allow_inf_nan=False)]


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: Status
    message: Annotated[str, AfterValidator(_printable)]
    task_type: TaskType | None
    marker_id: str | None
    depth_m: Metres | None
    length_m: Metres | None
    width_m: Metres | None
    heading: Heading | None
    spoil_side: SpoilSide | None
    soil_class: SoilClass | None


@lru_cache(maxsize=8)
def spec_model(scene: Scene) -> type[TaskSpec]:
    marker_ids = StrEnum("MarkerId", {m.id: m.id for m in scene.markers})
    return create_model("TaskSpec", __base__=TaskSpec, marker_id=(marker_ids | None, ...))


def strict_schema(scene: Scene) -> dict:
    schema = spec_model(scene).model_json_schema()
    _require_strict(schema)
    for definition in schema.get("$defs", {}).values():
        _require_strict(definition)
    return schema


def _require_strict(node: dict) -> None:
    if node.get("type") != "object":
        return
    properties = set(node.get("properties", {}))
    if set(node.get("required", [])) != properties:
        raise ValueError("strict schema needs every property required")
    if node.get("additionalProperties") is not False:
        raise ValueError("strict schema needs additionalProperties false")
