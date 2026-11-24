# A folder of validated model outputs keyed by a hash of instruction and scene.
# Replayed outputs still pass through every check; the cache only saves the model call.
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from hma.task.model import normalise
from hma.task.scene import Scene


def default_cache_dir() -> Path:
    if override := os.environ.get("HMA_CACHE_DIR"):
        return Path(override)
    if sys.platform == "win32" and (local := os.environ.get("LOCALAPPDATA")):
        return Path(local, "hma", "plans")
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base, "hma", "plans")


def cache_key(instruction: str, scene: Scene) -> str:
    payload = json.dumps(
        {"instruction": normalise(instruction), "scene": scene.as_data()}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PlanCache:
    def __init__(self, folder: Path | str | None = None) -> None:
        self.folder = Path(folder) if folder is not None else default_cache_dir()

    def path(self, instruction: str, scene: Scene) -> Path:
        return self.folder / f"{cache_key(instruction, scene)}.json"

    def get(self, instruction: str, scene: Scene) -> str | None:
        try:
            record = json.loads(self.path(instruction, scene).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        output = record.get("output") if isinstance(record, dict) else None
        return output if isinstance(output, str) else None

    def put(self, instruction: str, scene: Scene, output: str) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.path(instruction, scene)
        tmp = target.with_suffix(".tmp")
        record = {"instruction": instruction, "output": output}
        tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
        tmp.replace(target)
