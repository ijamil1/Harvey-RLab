#!/usr/bin/env python3
from __future__ import annotations

import glob as globlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE = Path("/workspace").resolve()
DOCUMENTS = WORKSPACE / "documents"
OUTPUT = WORKSPACE / "output"
SKILLS = WORKSPACE / "skills"
LAB = WORKSPACE / ".lab"
METADATA_PATH = LAB / "metadata.json"
CONTROL_REFERENCES = ("/workspace/.lab", ".lab", "rlm_control", ".rlm_control")
SUPPORTED_PARSE_SUFFIXES = {".docx", ".xlsx"}


def _metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {"text_backed_documents": [], "expected_deliverables": []}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _ok(result: object) -> int:
    sys.stdout.write(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    return 0


def _error(message: str) -> int:
    sys.stdout.write(json.dumps({"ok": False, "result": f"Error: {message}"}))
    return 0


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _reject_control_reference(value: object) -> None:
    text = str(value).lower()
    if any(reference in text for reference in CONTROL_REFERENCES):
        raise PermissionError("access denied to environment implementation files")


def _workspace_path(path: str | os.PathLike[str]) -> Path:
    _reject_control_reference(path)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORKSPACE / candidate
    resolved = candidate.resolve(strict=False)
    if not _is_under(resolved, WORKSPACE):
        raise PermissionError(f"path is outside /workspace: {path}")
    return resolved


def _resolve_read_path(path: str) -> Path:
    if not path:
        raise ValueError("file_path is required")
    _reject_control_reference(path)
    raw = Path(path)
    if raw.is_absolute():
        return _workspace_path(path)
    for root in (WORKSPACE, DOCUMENTS, OUTPUT):
        candidate = (root / raw).resolve(strict=False)
        if _is_under(candidate, WORKSPACE) and candidate.exists():
            return candidate
    return (DOCUMENTS / raw).resolve(strict=False)


def _resolve_write_path(path: str) -> Path:
    if not path:
        raise ValueError("file_path is required")
    _reject_control_reference(path)
    raw = Path(path)
    if raw.is_absolute():
        resolved = _workspace_path(path)
        if not _is_under(resolved, OUTPUT):
            raise PermissionError("write tool may only write under /workspace/output")
        return resolved
    return (OUTPUT / raw).resolve(strict=False)


def _resolve_edit_path(path: str) -> Path:
    resolved = _resolve_read_path(path)
    if _is_under(resolved, DOCUMENTS) or _is_under(resolved, SKILLS):
        raise PermissionError("edit denied for read-only task or skill files")
    if not _is_under(resolved, WORKSPACE):
        raise PermissionError(f"path is outside /workspace: {path}")
    return resolved


def _resolve_search_root(path: str | None) -> Path:
    if not path:
        return DOCUMENTS
    return _resolve_read_path(path)


def _parse_file(path: Path, timeout: int = 120) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_PARSE_SUFFIXES:
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


def _is_text_backed_document(path: Path) -> bool:
    docs = set(str(item) for item in _metadata().get("text_backed_documents", []))
    return str(path.resolve(strict=False)) in docs


def read(payload: dict[str, Any]) -> str:
    path = _resolve_read_path(str(payload.get("file_path", "")))
    if not path.exists():
        return f"Error: file not found: {payload.get('file_path', '')}"
    if path.is_dir():
        return f"Error: {path} is a directory"
    if _is_text_backed_document(path):
        content = path.read_text(encoding="utf-8", errors="replace")
    else:
        content = _parse_file(path)
    offset = payload.get("offset")
    limit = payload.get("limit")
    if offset is None and limit is None:
        return content
    lines = content.split("\n")
    start = max(0, int(offset or 0))
    end = len(lines) if limit is None else start + max(0, int(limit))
    return "\n".join(lines[start:end])


def write(payload: dict[str, Any]) -> str:
    path = _resolve_write_path(str(payload.get("file_path", "")))
    if path.suffix.lower() in SUPPORTED_PARSE_SUFFIXES:
        return "Error: write only creates plain text files; use the skill scripts for Office deliverables"
    if not _is_under(path, OUTPUT):
        raise PermissionError("write tool may only write under /workspace/output")
    text = "" if payload.get("content") is None else str(payload.get("content"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"Wrote {len(text.encode('utf-8'))} bytes to {path}"


def edit(payload: dict[str, Any]) -> str:
    path = _resolve_edit_path(str(payload.get("file_path", "")))
    if not path.exists():
        return f"Error: file not found: {payload.get('file_path', '')}"
    if path.is_dir():
        return f"Error: {path} is a directory"
    old = str(payload.get("old_string", ""))
    new = str(payload.get("new_string", ""))
    replace_all = bool(payload.get("replace_all", False))
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    if count == 0:
        return f"Error: old_string not found in {path}"
    if count > 1 and not replace_all:
        return (
            f"Error: old_string found {count} times in {path}. "
            "Use replace_all=true to replace all."
        )
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    path.write_text(updated, encoding="utf-8")
    return f"Replaced {count if replace_all else 1} occurrence(s) in {path}"


def glob(payload: dict[str, Any]) -> str:
    pattern = str(payload.get("pattern", ""))
    if not pattern:
        return "Error: pattern is required"
    root = _resolve_search_root(payload.get("path"))
    if not root.exists():
        return f"Error: path does not exist: {payload.get('path')}"
    matches = sorted(
        (
            item
            for item in root.glob(pattern)
            if item.is_file()
            and _is_under(item, root)
            and not _is_under(item, LAB)
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return f"No files matching {pattern!r} in {root}"
    return "\n".join(str(item.relative_to(root)) for item in matches[:100])


def grep(payload: dict[str, Any]) -> str:
    pattern = str(payload.get("pattern", ""))
    if not pattern:
        return "Error: pattern is required"
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"
    root = _resolve_search_root(payload.get("path"))
    if not root.exists():
        return f"Error: path does not exist: {payload.get('path')}"
    if root.is_file():
        files = [root]
    else:
        files = [
            Path(item)
            for item in globlib.glob(str(root / str(payload.get("glob") or "**/*")), recursive=True)
        ]
    output_mode = str(payload.get("output_mode") or "files_with_matches")
    results: list[str] = []
    for path in files:
        if not path.is_file() or not _is_under(path, root) or _is_under(path, LAB):
            continue
        try:
            text = read({"file_path": str(path)})
        except Exception:
            continue
        matches = list(regex.finditer(text))
        if not matches:
            continue
        rel = str(path.relative_to(root)) if path != root else str(path)
        if output_mode == "files_with_matches":
            results.append(rel)
        elif output_mode == "count":
            results.append(f"{rel}: {len(matches)}")
        elif output_mode == "content":
            for index, line in enumerate(text.split("\n"), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{index}: {line}")
        else:
            return f"Error: unsupported output_mode: {output_mode}"
    return "\n".join(results[:250]) if results else f"No matches for {pattern!r}"


def _reject_unsafe_bash(command: str) -> None:
    _reject_control_reference(command)
    if ".." in command:
        raise PermissionError("bash command may not reference parent directories")
    absolute_paths = re.findall(r"(?<![\w.-])/(?!workspace(?:/|$))[\w./@%+=:,~-]*", command)
    if absolute_paths:
        raise PermissionError(
            "bash command may only reference absolute paths under /workspace"
        )


def bash(payload: dict[str, Any]) -> str:
    command = str(payload.get("command") or "")
    if not command.strip():
        return "Error: command is required"
    _reject_unsafe_bash(command)
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    output = result.stdout or ""
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"
    if result.returncode:
        output += f"\n(exit code {result.returncode})"
    return output or "(no output)"


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    expected = payload.get("expected_deliverables")
    if not isinstance(expected, list):
        expected = _metadata().get("expected_deliverables", [])
    parsed: dict[str, str] = {}
    missing: list[str] = []
    errors: dict[str, str] = {}
    for item in expected:
        name = str(item)
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            missing.append(name)
            continue
        path = OUTPUT / name
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


TOOLS = {
    "bash": bash,
    "read": read,
    "write": write,
    "edit": edit,
    "glob": glob,
    "grep": grep,
    "collect": collect,
}


def main() -> int:
    if len(sys.argv) != 2:
        return _error("usage: tool_server.py '<json payload>'")
    try:
        payload = json.loads(sys.argv[1])
        if not isinstance(payload, dict):
            return _error("payload must be a JSON object")
        tool = str(payload.get("tool") or "")
        if tool not in TOOLS:
            return _error(f"unknown tool: {tool}")
        return _ok(TOOLS[tool](payload))
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
