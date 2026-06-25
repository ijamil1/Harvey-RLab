from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from harvey_lab_classic.dataset import normalize_lab_row
from harvey_lab_classic.prompts import build_system_prompt
from harvey_lab_classic.resources import collect_deliverables, stage_rollout_context
from harvey_lab_classic.scoring import HarveyLabScorer
from harvey_lab_classic.tools import ALL_TOOLS, read, write


def sample_row() -> dict:
    return {
        "task_id": "corp/test",
        "practice_area": "corp",
        "title": "Test task",
        "work_type": "memo",
        "tags": ["test"],
        "instructions": "Draft the requested memo.",
        "deliverables": ["memo.docx"],
        "criteria": [
            {
                "id": "c1",
                "title": "Addresses issue",
                "match_criteria": "The memo addresses the issue.",
                "deliverables": ["memo.docx"],
            }
        ],
        "documents": {"source.docx": "Extracted source document text."},
    }


class FakeSandbox:
    def __init__(self, stdout: str | None = None) -> None:
        self.stdout = stdout or json.dumps({"ok": True, "result": "ok"})
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.deleted = False

    async def execute(self, command: str, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(exit_code=0, stdout=self.stdout, stderr="")

    async def upload_bytes(self, path: str, content: bytes, filename: str | None = None):
        self.uploads[path] = content

    async def delete(self):
        self.deleted = True


def test_dataset_normalization_preserves_first_user_message() -> None:
    row = normalize_lab_row(sample_row())

    assert row["prompt"] == [{"role": "user", "content": "Draft the requested memo."}]
    assert row["deliverables"] == ["memo.docx"]
    assert row["documents"] == {"source.docx": "Extracted source document text."}


def test_system_prompt_lists_skill_paths_without_inlining_manuals() -> None:
    prompt = build_system_prompt()

    assert "/workspace/skills/docx/SKILL.md" in prompt
    assert "/workspace/skills/xlsx/SKILL.md" in prompt
    assert "Run-merging gotcha" not in prompt
    assert "Banker conventions" not in prompt


@pytest.mark.asyncio
async def test_resource_staging_uploads_documents_skills_and_tool_server() -> None:
    sandbox = FakeSandbox()
    await stage_rollout_context(sandbox, normalize_lab_row(sample_row()))

    assert sandbox.uploads["/workspace/documents/source.docx"] == (
        b"Extracted source document text."
    )
    assert "/workspace/.lab/tool_server.py" in sandbox.uploads
    assert "/workspace/.lab/metadata.json" in sandbox.uploads
    assert "/workspace/skills/docx/SKILL.md" in sandbox.uploads
    assert "/workspace/skills/xlsx/SKILL.md" in sandbox.uploads
    assert any(path.endswith("/skills/docx/scripts/validate.py") for path in sandbox.uploads)
    assert any(path.endswith("/skills/xlsx/scripts/validate.py") for path in sandbox.uploads)

    metadata = json.loads(sandbox.uploads["/workspace/.lab/metadata.json"])
    assert metadata["text_backed_documents"] == ["/workspace/documents/source.docx"]
    assert metadata["expected_deliverables"] == ["memo.docx"]


@pytest.mark.asyncio
async def test_exposed_tools_route_through_sandbox_execute_only() -> None:
    sandbox = FakeSandbox(json.dumps({"ok": True, "result": "document text"}))
    state: dict = {}

    output = await read("source.docx", sandbox=sandbox, state=state)

    assert output == "document text"
    assert len(sandbox.commands) == 1
    assert "python /workspace/.lab/tool_server.py" in sandbox.commands[0]
    assert state["tool_metrics"]["read_calls"] == 1
    assert sorted(tool.__name__ for tool in ALL_TOOLS) == [
        "bash",
        "edit",
        "glob",
        "grep",
        "read",
        "write",
    ]


@pytest.mark.asyncio
async def test_write_tool_routes_plain_text_to_sandbox() -> None:
    sandbox = FakeSandbox(json.dumps({"ok": True, "result": "Wrote 5 bytes"}))
    state: dict = {}

    output = await write("note.md", "hello", sandbox=sandbox, state=state)

    assert output == "Wrote 5 bytes"
    assert '"tool": "write"' in sandbox.commands[0]
    assert state["tool_metrics"]["files_written"] == 1


@pytest.mark.asyncio
async def test_collect_deliverables_copies_sandbox_parse_result() -> None:
    payload = {
        "ok": True,
        "result": {
            "deliverables": {"memo.docx": "parsed memo"},
            "missing_deliverables": [],
            "deliverable_errors": {},
        },
    }
    sandbox = FakeSandbox(json.dumps(payload))

    result = await collect_deliverables(sandbox, ["memo.docx"])

    assert result["deliverables"] == {"memo.docx": "parsed memo"}
    assert result["missing_deliverables"] == []
    assert result["deliverable_errors"] == {}
    assert not sandbox.deleted


class PassingJudge:
    async def evaluate(self, **kwargs):
        return {"verdict": "pass", "reasoning": "sufficient"}


@pytest.mark.asyncio
async def test_scorer_matches_missing_deliverable_auto_fail_contract() -> None:
    state = normalize_lab_row(sample_row())
    state["expected_deliverables"] = ["memo.docx"]
    state["deliverables"] = {}
    state["deliverable_errors"] = {}
    state["missing_deliverables"] = ["memo.docx"]

    reward = await HarveyLabScorer(judge=PassingJudge()).score_rollout(state)

    assert reward == 0.0
    assert state["criterion_results"][0]["verdict"] == "fail"
    assert state["lab_metrics"]["lab_missing_deliverables"] == 1.0


@pytest.mark.asyncio
async def test_scorer_judges_available_deliverables() -> None:
    state = normalize_lab_row(sample_row())
    state["expected_deliverables"] = ["memo.docx"]
    state["deliverables"] = {"memo.docx": "parsed memo"}
    state["deliverable_errors"] = {}
    state["missing_deliverables"] = []

    reward = await HarveyLabScorer(judge=PassingJudge()).score_rollout(state)

    assert reward == 1.0
    assert state["criterion_results"][0]["verdict"] == "pass"
    assert state["lab_metrics"]["lab_judge_calls"] == 1.0
