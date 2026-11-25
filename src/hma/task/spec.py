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
    carry = "carry"
    prop = "prop"
    turn = "turn"
    travel = "travel"


def printable(text: str) -> str:
    return "".join(c if c.isprintable() else " " for c in text)


def _printable(text: str) -> str:
    if len(text) > MESSAGE_MAX_CHARS:
        raise ValueError(f"message is longer than {MESSAGE_MAX_CHARS} characters")
    return printable(text)


Seconds = Annotated[float, Field(ge=0, le=3600, allow_inf_nan=False)]


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: Status
    message: Annotated[str, AfterValidator(_printable)]
    task_type: TaskType | None
    load_id: str | None
    marker_id: str | None
    stand_marker_id: str | None
    hold_s: Seconds | None


@lru_cache(maxsize=8)
def spec_model(scene: Scene) -> type[TaskSpec]:
    marker_ids = StrEnum("MarkerId", {m.id: m.id for m in scene.markers})
    load_ids = StrEnum("LoadId", {m.id: m.id for m in scene.loads})
    return create_model(
        "TaskSpec",
        __base__=TaskSpec,
        load_id=(load_ids | None, ...),
        marker_id=(marker_ids | None, ...),
        stand_marker_id=(marker_ids | None, ...),
    )


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
