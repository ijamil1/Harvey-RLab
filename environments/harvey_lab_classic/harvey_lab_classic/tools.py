from __future__ import annotations

import json
import shlex
from typing import Any


TOOL_SERVER = "/workspace/.lab/tool_server.py"


async def _call_tool_server(
    sandbox: Any,
    payload: dict[str, Any],
    *,
    timeout: int = 120,
    state: dict[str, Any] | None = None,
) -> str:
    command = "python " + TOOL_SERVER + " " + shlex.quote(
        json.dumps(payload, ensure_ascii=False)
    )
    result = await sandbox.execute(command, timeout=timeout, working_dir="/workspace")
    if state is not None:
        state.setdefault("lab_sandbox_commands", []).append(
            {
                "tool": payload.get("tool"),
                "returncode": getattr(result, "exit_code", None),
                "stdout": getattr(result, "stdout", "") or "",
                "stderr": getattr(result, "stderr", "") or "",
            }
        )
    if getattr(result, "exit_code", 0):
        stderr = getattr(result, "stderr", "") or getattr(result, "stdout", "")
        return f"Error: sandbox tool failed: {stderr.strip()}"
    try:
        response = json.loads(getattr(result, "stdout", "") or "{}")
    except json.JSONDecodeError as exc:
        return f"Error: invalid sandbox tool response: {exc}"
    if not isinstance(response, dict):
        return "Error: invalid sandbox tool response"
    output = response.get("result", "")
    return str(output)


def _metric(state: dict[str, Any], key: str) -> None:
    metrics = state.setdefault("tool_metrics", {})
    metrics[key] = int(metrics.get(key, 0)) + 1


async def bash(command: str, sandbox: Any, state: dict[str, Any]) -> str:
    """Execute a bash command inside the Prime sandbox workspace."""
    _metric(state, "bash_commands")
    return await _call_tool_server(
        sandbox,
        {"tool": "bash", "command": command},
        timeout=120,
        state=state,
    )


async def read(
    file_path: str,
    sandbox: Any,
    state: dict[str, Any],
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Read a workspace file, parsing generated docx/xlsx files when needed."""
    _metric(state, "read_calls")
    return await _call_tool_server(
        sandbox,
        {
            "tool": "read",
            "file_path": file_path,
            "offset": offset,
            "limit": limit,
        },
        timeout=180,
        state=state,
    )


async def write(file_path: str, content: str, sandbox: Any, state: dict[str, Any]) -> str:
    """Write plain text under /workspace/output."""
    _metric(state, "files_written")
    return await _call_tool_server(
        sandbox,
        {"tool": "write", "file_path": file_path, "content": content},
        timeout=60,
        state=state,
    )


async def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    sandbox: Any,
    state: dict[str, Any],
    replace_all: bool = False,
) -> str:
    """Replace exact text in a writable workspace file."""
    _metric(state, "files_edited")
    return await _call_tool_server(
        sandbox,
        {
            "tool": "edit",
            "file_path": file_path,
            "old_string": old_string,
            "new_string": new_string,
            "replace_all": replace_all,
        },
        timeout=60,
        state=state,
    )


async def glob(
    pattern: str,
    sandbox: Any,
    state: dict[str, Any],
    path: str | None = None,
) -> str:
    """Find files inside /workspace using a glob pattern."""
    _metric(state, "glob_searches")
    return await _call_tool_server(
        sandbox,
        {"tool": "glob", "pattern": pattern, "path": path},
        timeout=60,
        state=state,
    )


async def grep(
    pattern: str,
    sandbox: Any,
    state: dict[str, Any],
    path: str | None = None,
    glob: str | None = None,
    output_mode: str = "files_with_matches",
) -> str:
    """Search file contents inside /workspace with a regex pattern."""
    _metric(state, "grep_searches")
    return await _call_tool_server(
        sandbox,
        {
            "tool": "grep",
            "pattern": pattern,
            "path": path,
            "glob": glob,
            "output_mode": output_mode,
        },
        timeout=90,
        state=state,
    )


ALL_TOOLS = (bash, read, write, edit, glob, grep)
