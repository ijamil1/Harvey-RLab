from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


CONTROL_DIR_REFERENCES = (".rlm_control", ".rlm-control", "rlm_control")


class ReadOnlyDict(dict):
    @staticmethod
    def _blocked(*args, **kwargs):
        raise TypeError("mapping is read-only")

    __setitem__ = _blocked
    __delitem__ = _blocked
    clear = _blocked
    pop = _blocked
    popitem = _blocked
    setdefault = _blocked
    update = _blocked
    __ior__ = _blocked


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_workspace_path(path: str | os.PathLike[str], workspace: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    if not _is_under(candidate, workspace):
        raise ValueError(f"path is outside the workspace: {path}")
    return candidate


def _reject_control_dir_reference(value: str | os.PathLike[str]) -> None:
    text = os.fspath(value).lower()
    if any(reference in text for reference in CONTROL_DIR_REFERENCES):
        raise PermissionError("access denied to the RLM control directory")


def _parse_file(path: Path, timeout: int = 120) -> str:
    suffix = path.suffix.lower()
    if suffix in {".docx", ".xlsx"}:
        result = subprocess.run(
            ["parse-doc", suffix.lstrip("."), str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "document parser failed").strip())
        return result.stdout
    return path.read_text(encoding="utf-8", errors="replace")


def load_runtime_namespace(
    bootstrap_path: str | os.PathLike[str] = "/workspace/.lab/bootstrap.json",
    *,
    workspace_root: str | os.PathLike[str] = "/workspace",
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output_root = workspace / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    bootstrap = Path(bootstrap_path)
    payload = json.loads(bootstrap.read_text(encoding="utf-8"))
    bootstrap.unlink()

    documents = ReadOnlyDict(payload["documents"])
    skills = ReadOnlyDict(payload["skills"])
    instructions = str(payload["instructions"])
    expected_deliverables = list(payload["expected_deliverables"])

    def read(
        path: str | os.PathLike[str],
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        _reject_control_dir_reference(path)
        resolved = _resolve_workspace_path(path, workspace)
        if not resolved.exists():
            return f"Error: file not found: {resolved}"
        if resolved.is_dir():
            return f"Error: {resolved} is a directory"
        try:
            content = _parse_file(resolved)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"
        if offset is None and limit is None:
            return content
        lines = content.splitlines()
        start = max(0, offset or 0)
        end = len(lines) if limit is None else start + max(0, limit)
        return "\n".join(lines[start:end])

    def write(path: str | os.PathLike[str], content: object) -> str:
        _reject_control_dir_reference(path)
        raw_path = Path(path)
        resolved = (
            _resolve_workspace_path(raw_path, workspace)
            if raw_path.is_absolute()
            else _resolve_workspace_path(output_root / raw_path, workspace)
        )
        if _is_under(resolved, workspace / ".lab") or _is_under(
            resolved, workspace / "skills"
        ):
            raise PermissionError(f"write denied for protected path: {path}")
        text = "" if content is None else str(content)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(text, encoding="utf-8")
        return f"Wrote {len(text)} bytes to {path}"

    def bash(command: str, timeout: int | None = None) -> dict[str, Any]:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command is required")
        _reject_control_dir_reference(command)
        timeout_seconds = int(timeout or os.environ.get("RLM_SHELL_TIMEOUT", "60"))
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": False,
                "ok": result.returncode == 0,
            }
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
            )
            stderr = (
                exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
            )
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": None,
                "timed_out": True,
                "ok": False,
            }

    def collect_deliverables() -> dict[str, Any]:
        parsed: dict[str, str] = {}
        missing: list[str] = []
        errors: dict[str, str] = {}
        for name in expected_deliverables:
            path = output_root / name
            if not path.exists() or not path.is_file() or path.is_symlink():
                missing.append(name)
                continue
            try:
                parsed[name] = _parse_file(path)
            except Exception as exc:
                errors[name] = str(exc)
        return {
            "deliverables": parsed,
            "missing_deliverables": missing,
            "deliverable_errors": errors,
        }

    return {
        "documents": documents,
        "skills": skills,
        "instructions": instructions,
        "expected_deliverables": expected_deliverables,
        "read": read,
        "write": write,
        "bash": bash,
        "_collect_deliverables": collect_deliverables,
    }
