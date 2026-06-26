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
    """Execute a bash command and return its output.

    Use for running scripts, file manipulation, and sandbox-local shell
    operations. Each command runs from /workspace; filesystem changes persist
    between calls, but shell process state such as cd or exported variables does
    not.

    Args:
        command: The bash command to execute.
    """
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
    """Read a file from the task documents, output directory, or workspace.

    Relative paths are resolved by checking /workspace, then
    /workspace/documents, then /workspace/output; if the file does not already
    exist, the path defaults to /workspace/documents. Source task documents are
    text-backed files. Generated .docx and .xlsx files are parsed automatically.
    Use offset and limit for large files.

    Args:
        file_path: Relative path (resolved against /workspace,
            /workspace/documents, then /workspace/output) or an absolute path
            under /workspace.
        offset: Line number to start reading from (0-based). Optional.
        limit: Maximum number of lines to return. Optional.
    """
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
    """Write a plain-text scratch or output file under /workspace.

    Relative paths are resolved under /workspace. Use output/<filename> or an
    absolute /workspace/output/<filename> path for final deliverables that must
    be collected for scoring. For Office deliverables (.docx or .xlsx), use the
    file-type skill manuals; do not write raw text to a binary extension.
    Writing to task documents, skill files, or hidden lab files is denied.
    Creates parent directories if needed.

    Args:
        file_path: Relative path under /workspace (e.g., 'scratch.md' or
            'output/response.md') or an absolute path under /workspace.
        content: Plain-text content to write.
    """
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
    """Perform exact string replacement in a file you have already created or read.

    Relative paths resolve like read paths, but edits are denied for read-only
    task documents under /workspace/documents and skill files under
    /workspace/skills. The old_string must appear exactly once unless
    replace_all is true. Use for incremental refinement, not first-time writes.

    Args:
        file_path: Path to a writable file under /workspace; read-only task
            documents and skill files cannot be edited.
        old_string: The exact text to find and replace.
        new_string: The replacement text.
        replace_all: If true, replace all occurrences. Default false.
    """
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
    """Find files matching a glob pattern, sorted by modification time.

    Defaults to searching /workspace/documents. If path is provided, it resolves
    like a read path and may point to another directory under /workspace. Prefer
    this over `bash find` or `bash ls` for file discovery.

    Args:
        pattern: Glob pattern to match (e.g., '**/*.docx', 'src/**/*.py').
        path: Directory to search in. Defaults to /workspace/documents.
    """
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
    """Search file contents using regex patterns.

    Defaults to searching /workspace/documents. If path is provided, it resolves
    like a read path and may point to a file or directory under /workspace.
    Returns matching file paths, matching lines, or match counts.

    Args:
        pattern: Regex pattern to search for.
        path: File or directory to search in. Defaults to /workspace/documents.
        glob: Glob pattern to filter files (e.g., '*.py', '*.docx').
        output_mode: Output format. 'content' shows matching lines,
            'files_with_matches' shows file paths, 'count' shows match counts.
            Default: 'files_with_matches'.
    """
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
