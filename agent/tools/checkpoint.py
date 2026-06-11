from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.tools.base import Tool

_CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class FileCheckpoint:
    checkpoint_id: str
    path: Path
    existed: bool
    content_path: Path | None
    sha256: str


class FileCheckpointStore:
    """Local, session-independent snapshots for reversible file mutations."""

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root or Path.cwd()).expanduser().resolve()
        self._checkpoint_dir = self._root / ".akashic" / "checkpoints"

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    def create(self, path: Path) -> FileCheckpoint:
        target = path.expanduser().resolve()
        checkpoint_id = self._new_checkpoint_id()
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        existed = target.exists()
        content_name = f"{checkpoint_id}.bin"
        content_path = self._checkpoint_dir / content_name
        payload = b""
        if existed:
            if not target.is_file():
                raise IsADirectoryError(f"checkpoint target is not a file: {target}")
            payload = target.read_bytes()
            content_path.write_bytes(payload)
        sha256 = hashlib.sha256(payload).hexdigest()

        meta = {
            "checkpoint_id": checkpoint_id,
            "path": str(target),
            "existed": existed,
            "content_file": content_name if existed else "",
            "sha256": sha256,
            "size_bytes": len(payload),
            "created_at_ms": int(time.time() * 1000),
        }
        self._meta_path(checkpoint_id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return FileCheckpoint(
            checkpoint_id=checkpoint_id,
            path=target,
            existed=existed,
            content_path=content_path if existed else None,
            sha256=sha256,
        )

    def load(self, checkpoint_id: str) -> FileCheckpoint:
        clean_id = self._validate_checkpoint_id(checkpoint_id)
        meta_path = self._meta_path(clean_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        path = Path(str(meta["path"])).expanduser().resolve()
        existed = bool(meta.get("existed"))
        content_file = str(meta.get("content_file") or "")
        content_path = self._checkpoint_dir / content_file if content_file else None
        return FileCheckpoint(
            checkpoint_id=str(meta["checkpoint_id"]),
            path=path,
            existed=existed,
            content_path=content_path,
            sha256=str(meta.get("sha256") or ""),
        )

    def _meta_path(self, checkpoint_id: str) -> Path:
        return self._checkpoint_dir / f"{self._validate_checkpoint_id(checkpoint_id)}.json"

    @staticmethod
    def _new_checkpoint_id() -> str:
        return f"{time.strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:12]}"

    @staticmethod
    def _validate_checkpoint_id(checkpoint_id: str) -> str:
        clean_id = str(checkpoint_id or "").strip()
        if not _CHECKPOINT_ID_RE.fullmatch(clean_id):
            raise ValueError("invalid checkpoint_id")
        return clean_id


def _default_checkpoint_store(allowed_dir: Path | None = None) -> FileCheckpointStore:
    return FileCheckpointStore(allowed_dir or Path.cwd())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class RestoreCheckpointTool(Tool):
    name = "restore_checkpoint"
    description = (
        "Restore a file to the content captured before write_file/edit_file changed it. "
        "Use this to undo an unsafe or incorrect file mutation by checkpoint_id."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "checkpoint_id": {
                "type": "string",
                "description": "checkpoint_id returned by write_file or edit_file",
            }
        },
        "required": ["checkpoint_id"],
    }

    def __init__(
        self,
        allowed_dir: Path | None = None,
        checkpoint_store: FileCheckpointStore | None = None,
    ) -> None:
        self._allowed_dir = allowed_dir
        self._checkpoint_store = checkpoint_store or _default_checkpoint_store(allowed_dir)

    async def execute(self, checkpoint_id: str, **kwargs: Any) -> str:
        try:
            checkpoint = self._checkpoint_store.load(checkpoint_id)
            target = checkpoint.path
            if self._allowed_dir is not None and not _is_relative_to(
                target, self._allowed_dir
            ):
                return f"checkpoint target outside allowed directory: {target}"

            if checkpoint.existed:
                if checkpoint.content_path is None or not checkpoint.content_path.exists():
                    return f"checkpoint content missing: {checkpoint_id}"
                payload = checkpoint.content_path.read_bytes()
                if checkpoint.sha256:
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest != checkpoint.sha256:
                        return f"checkpoint content checksum mismatch: {checkpoint_id}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                return f"restored checkpoint_id={checkpoint_id} path={target}"

            if target.exists():
                if not target.is_file():
                    return f"checkpoint target is not a file: {target}"
                target.unlink()
                return f"removed file created after checkpoint: {target}"
            return f"checkpoint already restored: {target}"
        except Exception as exc:
            return f"restore_checkpoint failed: {exc}"
