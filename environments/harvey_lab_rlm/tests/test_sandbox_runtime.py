from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harvey_lab_rlm.sandbox_runtime import load_runtime_namespace


def write_bootstrap(root: Path) -> Path:
    bootstrap = root / ".lab" / "bootstrap.json"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text(
        json.dumps(
            {
                "instructions": "Create memo.docx.",
                "documents": {"facts.docx": "Fact text"},
                "skills": {"docx": "Skill text"},
                "expected_deliverables": ["memo.docx", "schedule.xlsx"],
            }
        ),
        encoding="utf-8",
    )
    return bootstrap


def test_runtime_loads_native_read_only_mappings_and_deletes_bootstrap(
    tmp_path: Path,
) -> None:
    bootstrap = write_bootstrap(tmp_path)

    namespace = load_runtime_namespace(bootstrap, workspace_root=tmp_path)

    assert isinstance(namespace["documents"], dict)
    assert isinstance(namespace["skills"], dict)
    assert namespace["documents"]["facts.docx"] == "Fact text"
    assert namespace["instructions"] == "Create memo.docx."
    assert namespace["expected_deliverables"] == ["memo.docx", "schedule.xlsx"]
    assert not bootstrap.exists()
    with pytest.raises(TypeError):
        namespace["documents"]["new"] = "blocked"


def test_runtime_helpers_execute_locally_and_persist_caller_state(tmp_path: Path) -> None:
    namespace = load_runtime_namespace(
        write_bootstrap(tmp_path), workspace_root=tmp_path
    )

    write = namespace["write"]
    read = namespace["read"]
    bash = namespace["bash"]
    assert write("notes.txt", "hello") == "Wrote 5 bytes to notes.txt"
    assert read(tmp_path / "output" / "notes.txt") == "hello"
    result = bash("python -c 'print(6 * 7)'")
    assert result["ok"] is True
    assert result["stdout"].strip() == "42"


def test_collect_deliverables_uses_exact_names_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = load_runtime_namespace(
        write_bootstrap(tmp_path), workspace_root=tmp_path
    )
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    (output / "Memo.docx").write_bytes(b"wrong case")
    (output / "memo.docx").write_bytes(b"valid placeholder")

    def fake_run(*args, **kwargs):
        class Result:
            returncode = 0
            stdout = "Parsed memo"
            stderr = ""

        return Result()

    monkeypatch.setattr("harvey_lab_rlm.sandbox_runtime.subprocess.run", fake_run)
    result = namespace["_collect_deliverables"]()

    assert result == {
        "deliverables": {"memo.docx": "Parsed memo"},
        "missing_deliverables": ["schedule.xlsx"],
        "deliverable_errors": {},
    }


def test_collect_deliverables_reports_parser_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = load_runtime_namespace(
        write_bootstrap(tmp_path), workspace_root=tmp_path
    )
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    (output / "memo.docx").write_bytes(b"broken")

    def fake_run(*args, **kwargs):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "invalid package"

        return Result()

    monkeypatch.setattr("harvey_lab_rlm.sandbox_runtime.subprocess.run", fake_run)
    result = namespace["_collect_deliverables"]()

    assert result["deliverables"] == {}
    assert result["missing_deliverables"] == ["schedule.xlsx"]
    assert result["deliverable_errors"] == {"memo.docx": "invalid package"}


def test_runtime_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    namespace = load_runtime_namespace(
        write_bootstrap(tmp_path), workspace_root=tmp_path
    )
    with pytest.raises(ValueError, match="outside"):
        namespace["read"](Path(os.sep) / "etc" / "hosts")
