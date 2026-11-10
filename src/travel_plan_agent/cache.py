"""A small disk cache for search results, so re-running a plan doesn't use up the free API quota."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, directory: Path, ttl_hours: float) -> None:
        self._dir = directory
        self._ttl_seconds = ttl_hours * 3600

    def _path(self, namespace: str, params: dict[str, Any]) -> Path:
        digest = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
        return self._dir / namespace / f"{digest[:32]}.json"

    def get(self, namespace: str, params: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path(namespace, params)
        if not path.exists() or time.time() - path.stat().st_mtime > self._ttl_seconds:
            return None
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data

    def set(self, namespace: str, params: dict[str, Any], value: dict[str, Any]) -> None:
        path = self._path(namespace, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
