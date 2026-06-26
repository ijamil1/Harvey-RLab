from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_tool_server():
    path = (
        Path(__file__).resolve().parents[1]
        / "harvey_lab_classic"
        / "resources"
        / "tool_server.py"
    )
    spec = importlib.util.spec_from_file_location("classic_tool_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def server(tmp_path, monkeypatch):
    module = load_tool_server()
    workspace = tmp_path / "workspace"
    documents = workspace / "documents"
    output = workspace / "output"
    skills = workspace / "skills"
    lab = workspace / ".lab"
    for path in (documents, output, skills, lab):
        path.mkdir(parents=True)
    monkeypatch.setattr(module, "WORKSPACE", workspace.resolve())
    monkeypatch.setattr(module, "DOCUMENTS", documents.resolve())
    monkeypatch.setattr(module, "OUTPUT", output.resolve())
    monkeypatch.setattr(module, "SKILLS", skills.resolve())
    monkeypatch.setattr(module, "LAB", lab.resolve())
    monkeypatch.setattr(module, "METADATA_PATH", lab / "metadata.json")
    return module


def test_read_returns_text_backed_source_without_parsing(server) -> None:
    source = server.DOCUMENTS / "source.docx"
    source.write_text("already extracted", encoding="utf-8")
    server.METADATA_PATH.write_text(
        json.dumps(
            {
                "text_backed_documents": [str(source.resolve())],
                "expected_deliverables": ["memo.docx"],
            }
        ),
        encoding="utf-8",
    )

    assert server.read({"file_path": "source.docx"}) == "already extracted"


def test_write_rejects_office_extensions(server) -> None:
    result = server.write({"file_path": "memo.docx", "content": "not a docx"})

    assert result.startswith("Error: write only creates plain text")
    assert not (server.WORKSPACE / "memo.docx").exists()


def test_write_allows_workspace_and_output_plain_text(server) -> None:
    scratch = server.write({"file_path": "scratch/notes.md", "content": "notes"})
    final = server.write({"file_path": "output/response.md", "content": "answer"})

    assert scratch.startswith("Wrote 5 bytes")
    assert final.startswith("Wrote 6 bytes")
    assert (server.WORKSPACE / "scratch" / "notes.md").read_text(encoding="utf-8") == "notes"
    assert (server.OUTPUT / "response.md").read_text(encoding="utf-8") == "answer"


def test_write_rejects_read_only_workspace_areas(server) -> None:
    with pytest.raises(PermissionError):
        server.write({"file_path": "documents/source.txt", "content": "changed"})
    with pytest.raises(PermissionError):
        server.write({"file_path": "skills/docx/SKILL.md", "content": "changed"})


def test_edit_rejects_documents(server) -> None:
    (server.DOCUMENTS / "source.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(PermissionError):
        server.edit(
            {
                "file_path": "source.txt",
                "old_string": "secret",
                "new_string": "changed",
            }
        )


def test_bash_rejects_paths_outside_workspace(server) -> None:
    with pytest.raises(PermissionError):
        server.bash({"command": "cat /etc/passwd"})


def test_collect_requires_exact_output_basename(server, monkeypatch) -> None:
    (server.OUTPUT / "memo.docx").write_text("fake", encoding="utf-8")
    monkeypatch.setattr(server, "_parse_file", lambda path: "parsed")

    result = server.collect({"expected_deliverables": ["memo.docx", "other.docx"]})

    assert result["deliverables"] == {"memo.docx": "parsed"}
    assert result["missing_deliverables"] == ["other.docx"]


def test_collect_ignores_extra_output_scratch_files(server, monkeypatch) -> None:
    (server.OUTPUT / "memo.docx").write_text("fake", encoding="utf-8")
    (server.OUTPUT / "scratch.md").write_text("notes", encoding="utf-8")
    (server.OUTPUT / "broken_extra.docx").write_text("not a docx", encoding="utf-8")
    parsed_paths: list[str] = []

    def parse_expected(path: Path) -> str:
        parsed_paths.append(path.name)
        if path.name != "memo.docx":
            raise RuntimeError("extra file should not be parsed")
        return "parsed"

    monkeypatch.setattr(server, "_parse_file", parse_expected)

    result = server.collect({"expected_deliverables": ["memo.docx"]})

    assert result["deliverables"] == {"memo.docx": "parsed"}
    assert result["missing_deliverables"] == []
    assert result["deliverable_errors"] == {}
    assert parsed_paths == ["memo.docx"]
