from __future__ import annotations

from pathlib import Path

from datasets import Dataset
import pytest
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import UserMessage

from harvey_lab_rlm.environment import HarveyLabRLMEnv, SANDBOX_DOCKER_IMAGE
from harvey_lab_rlm.prompts import ROOT_PROMPT, SUB_LLM_SYSTEM_PROMPT


class NoopJudge:
    async def evaluate(self, **kwargs):
        return {"verdict": "pass", "reasoning": "ok"}


def dataset_builder() -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "Draft memo.docx."}],
                "task_id": "test/task",
                "practice_area": "test",
                "title": "Test",
                "work_type": "draft",
                "tags": [],
                "instructions": "Draft memo.docx.",
                "deliverables": ["memo.docx"],
                "criteria": [
                    {
                        "id": "C-001",
                        "title": "Exists",
                        "deliverables": ["memo.docx"],
                        "match_criteria": "PASS if it exists.",
                    }
                ],
                "documents": {"facts.txt": "facts"},
            }
        ]
    )


def test_environment_exposes_only_python_repl_to_root_model() -> None:
    env = HarveyLabRLMEnv(
        dataset_builder=dataset_builder,
        judge=NoopJudge(),
    )

    assert env.sandbox_docker_image == SANDBOX_DOCKER_IMAGE
    assert [tool.name for tool in env.tool_defs] == ["call_python_repl"]
    assert env.root_tool_names == ["llm_batch"]
    assert env.sub_tool_names == []
    assert env.prompt_builder.build_system_prompt() == (
        f"<SCAFFOLDING>\n{ROOT_PROMPT}\n</SCAFFOLDING>\n\n"
    )
    assert env.prompt_builder.build_sub_llm_system_prompt() == SUB_LLM_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_ready_result_is_copied_to_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HarveyLabRLMEnv(
        dataset_builder=dataset_builder,
        judge=NoopJudge(),
    )

    async def fake_execute(self, code, state):
        return {
            "answer": {"ready": True},
            "deliverables": {"memo.docx": "parsed"},
            "missing_deliverables": [],
            "deliverable_errors": {},
        }

    monkeypatch.setattr(RLMEnv, "_execute_code", fake_execute)
    state = {}
    result = await env._execute_code("answer['ready'] = True", state)

    assert result["answer"]["ready"] is True
    assert state["deliverables"] == {"memo.docx": "parsed"}
    assert state["missing_deliverables"] == []
    assert state["deliverable_errors"] == {}


@pytest.mark.asyncio
async def test_sub_llm_request_is_wrapped_with_task_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HarveyLabRLMEnv(
        dataset_builder=dataset_builder,
        judge=NoopJudge(),
    )
    captured = {}

    async def fake_request(self, **kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(RLMEnv, "_run_sub_llm_request", fake_request)
    await env._run_sub_llm_request(
        state_ref={"instructions": "Draft the exact memo."},
        client=object(),
        sub_model="test-model",
        messages=[UserMessage(content="Review this excerpt. Output format: JSON.")],
        batch_id="batch",
        request_id="request",
        parent_turn=1,
    )

    content = captured["messages"][0].content
    assert "Draft the exact memo." in content
    assert "Review this excerpt." in content
    assert "requested output format" in content


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_setup", [False, True])
async def test_setup_deletes_short_lived_staging_root(
    monkeypatch: pytest.MonkeyPatch,
    fail_setup: bool,
) -> None:
    env = HarveyLabRLMEnv(
        dataset_builder=dataset_builder,
        judge=NoopJudge(),
    )
    state = dict(dataset_builder()[0])
    captured_staging_root: Path | None = None

    def fake_stage(root, row):
        bootstrap = root / ".lab" / "bootstrap.json"
        bootstrap.parent.mkdir(parents=True)
        bootstrap.write_text("{}", encoding="utf-8")

    async def fake_setup(self, state, **kwargs):
        nonlocal captured_staging_root
        captured_staging_root = Path(state["info"]["context_dir"])
        assert captured_staging_root.is_dir()
        assert (captured_staging_root / ".lab" / "bootstrap.json").is_file()
        if fail_setup:
            raise RuntimeError("setup failed")

    monkeypatch.setattr(
        "harvey_lab_rlm.environment.stage_rollout_context",
        fake_stage,
    )
    monkeypatch.setattr(RLMEnv, "setup_state", fake_setup)

    if fail_setup:
        with pytest.raises(RuntimeError, match="setup failed"):
            await env.setup_state(state)
    else:
        await env.setup_state(state)

    assert captured_staging_root is not None
    assert not captured_staging_root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_cleanup", [False, True])
async def test_cleanup_deletes_host_rollout_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_cleanup: bool,
) -> None:
    env = HarveyLabRLMEnv(
        dataset_builder=dataset_builder,
        judge=NoopJudge(),
    )
    rollout_dir = tmp_path / "rlm_rollout_test"
    (rollout_dir / "rlm_fs").mkdir(parents=True)
    (rollout_dir / "rlm_control").mkdir()
    state = {"rlm_rollout_dir": str(rollout_dir)}

    async def fake_cleanup(self, state):
        if fail_cleanup:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(RLMEnv, "cleanup_rlm_state", fake_cleanup)

    if fail_cleanup:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await env.cleanup_rlm_state(state)
    else:
        await env.cleanup_rlm_state(state)

    assert not rollout_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_strips_runtime_paths_but_keeps_scoring_text() -> None:
    env = HarveyLabRLMEnv(
        dataset_builder=dataset_builder,
        judge=NoopJudge(),
    )
    state = {
        "rlm_rollout_dir": "/tmp/private",
        "rlm_fs_root_remote": "/workspace",
        "sandbox_state": {"ready": True},
        "deliverables": {"memo.docx": "parsed"},
    }

    await env.strip_runtime_paths(state)

    assert state == {"deliverables": {"memo.docx": "parsed"}}
