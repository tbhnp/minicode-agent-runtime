from __future__ import annotations

import asyncio
import re
from pathlib import Path

from agent.tools.checkpoint import FileCheckpointStore, RestoreCheckpointTool
from agent.tools.filesystem import EditFileTool, WriteFileTool


def _checkpoint_id(output: str) -> str:
    match = re.search(r"checkpoint_id=([A-Za-z0-9_.-]+)", output)
    assert match is not None, output
    return match.group(1)


def test_write_file_creates_checkpoint_and_restore_recovers_previous_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("old content", encoding="utf-8")
    store = FileCheckpointStore(tmp_path)
    writer = WriteFileTool(allowed_dir=tmp_path, checkpoint_store=store)
    restore = RestoreCheckpointTool(allowed_dir=tmp_path, checkpoint_store=store)

    output = asyncio.run(writer.execute("demo.txt", "new content"))
    checkpoint_id = _checkpoint_id(output)

    assert target.read_text(encoding="utf-8") == "new content"

    restore_output = asyncio.run(restore.execute(checkpoint_id=checkpoint_id))

    assert "restored" in restore_output
    assert target.read_text(encoding="utf-8") == "old content"


def test_edit_file_creates_checkpoint_and_restore_recovers_previous_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("alpha beta", encoding="utf-8")
    store = FileCheckpointStore(tmp_path)
    editor = EditFileTool(allowed_dir=tmp_path, checkpoint_store=store)
    restore = RestoreCheckpointTool(allowed_dir=tmp_path, checkpoint_store=store)

    output = asyncio.run(editor.execute("demo.txt", "beta", "gamma"))
    checkpoint_id = _checkpoint_id(output)

    assert target.read_text(encoding="utf-8") == "alpha gamma"

    asyncio.run(restore.execute(checkpoint_id=checkpoint_id))

    assert target.read_text(encoding="utf-8") == "alpha beta"


def test_restore_checkpoint_removes_file_created_after_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"
    store = FileCheckpointStore(tmp_path)
    writer = WriteFileTool(allowed_dir=tmp_path, checkpoint_store=store)
    restore = RestoreCheckpointTool(allowed_dir=tmp_path, checkpoint_store=store)

    output = asyncio.run(writer.execute("created.txt", "temporary"))
    checkpoint_id = _checkpoint_id(output)

    assert target.exists()

    restore_output = asyncio.run(restore.execute(checkpoint_id=checkpoint_id))

    assert "removed" in restore_output
    assert not target.exists()


def test_restore_checkpoint_rejects_paths_outside_allowed_dir(tmp_path: Path) -> None:
    allowed_dir = tmp_path / "workspace"
    allowed_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("old", encoding="utf-8")
    store = FileCheckpointStore(tmp_path)
    checkpoint = store.create(outside)
    outside.write_text("new", encoding="utf-8")
    restore = RestoreCheckpointTool(allowed_dir=allowed_dir, checkpoint_store=store)

    output = asyncio.run(restore.execute(checkpoint_id=checkpoint.checkpoint_id))

    assert "outside allowed directory" in output
    assert outside.read_text(encoding="utf-8") == "new"
