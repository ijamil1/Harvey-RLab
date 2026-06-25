from __future__ import annotations

import json
import posixpath
import shlex
from importlib import resources
from pathlib import PurePosixPath
from typing import Any


SKILL_NAMES = ("docx", "xlsx")
WORKSPACE = "/workspace"
DOCUMENTS_DIR = "/workspace/documents"
OUTPUT_DIR = "/workspace/output"
LAB_DIR = "/workspace/.lab"


def _resource_root():
    return resources.files("harvey_lab_classic").joinpath("resources")


def _safe_relative_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("document path must be a non-empty string")
    normalized = posixpath.normpath(path.strip().replace("\\", "/"))
    if normalized.startswith("../") or normalized == ".." or normalized.startswith("/"):
        raise ValueError(f"document path must stay under documents: {path!r}")
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid document path: {path!r}")
    return normalized


async def _run_checked(sandbox: Any, command: str, timeout: int | None = None) -> str:
    result = await sandbox.execute(command, timeout=timeout)
    if getattr(result, "exit_code", 0):
        stderr = getattr(result, "stderr", "") or getattr(result, "stdout", "")
        raise RuntimeError(stderr.strip() or f"sandbox command failed: {command}")
    return getattr(result, "stdout", "") or ""


async def _upload_text(sandbox: Any, path: str, text: str) -> None:
    parent = posixpath.dirname(path)
    await _run_checked(sandbox, f"mkdir -p {shlex.quote(parent)}")
    await sandbox.upload_bytes(path, text.encode("utf-8"), path.rsplit("/", 1)[-1])


async def _upload_resource_file(sandbox: Any, local: Any, remote_path: str) -> None:
    parent = posixpath.dirname(remote_path)
    await _run_checked(sandbox, f"mkdir -p {shlex.quote(parent)}")
    await sandbox.upload_bytes(
        remote_path,
        local.read_bytes(),
        remote_path.rsplit("/", 1)[-1],
    )


async def stage_rollout_context(sandbox: Any, row: dict[str, Any]) -> None:
    await _run_checked(
        sandbox,
        "rm -rf /workspace/documents /workspace/output /workspace/skills /workspace/.lab "
        "&& mkdir -p /workspace/documents /workspace/output /workspace/skills /workspace/.lab",
        timeout=60,
    )

    document_paths: list[str] = []
    for original_path, text in sorted(row["documents"].items()):
        relative_path = _safe_relative_path(str(original_path))
        document_paths.append(f"{DOCUMENTS_DIR}/{relative_path}")
        await _upload_text(sandbox, f"{DOCUMENTS_DIR}/{relative_path}", str(text))

    metadata = {
        "text_backed_documents": sorted(document_paths),
        "expected_deliverables": list(row["deliverables"]),
    }
    await _upload_text(
        sandbox,
        f"{LAB_DIR}/metadata.json",
        json.dumps(metadata, ensure_ascii=False, allow_nan=False),
    )

    tool_server = _resource_root().joinpath("tool_server.py")
    await _upload_resource_file(sandbox, tool_server, f"{LAB_DIR}/tool_server.py")

    skills_root = _resource_root().joinpath("skills")
    for skill_name in SKILL_NAMES:
        skill_root = skills_root.joinpath(skill_name)
        await _upload_resource_file(
            sandbox,
            skill_root.joinpath("SKILL.md"),
            f"/workspace/skills/{skill_name}/SKILL.md",
        )
        scripts_root = skill_root.joinpath("scripts")
        for child in scripts_root.iterdir():
            if child.is_file():
                await _upload_resource_file(
                    sandbox,
                    child,
                    f"/workspace/skills/{skill_name}/scripts/{child.name}",
                )


async def collect_deliverables(sandbox: Any, expected: list[str]) -> dict[str, Any]:
    payload = {"tool": "collect", "expected_deliverables": expected}
    command = (
        "python /workspace/.lab/tool_server.py "
        + shlex.quote(json.dumps(payload, ensure_ascii=False))
    )
    stdout = await _run_checked(sandbox, command, timeout=180)
    response = json.loads(stdout)
    if not isinstance(response, dict):
        raise RuntimeError("sandbox collect returned non-object JSON")
    if not response.get("ok", False):
        raise RuntimeError(str(response.get("result", "deliverable collection failed")))
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("sandbox collect result was not an object")
    return result
