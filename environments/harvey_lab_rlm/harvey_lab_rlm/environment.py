from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import Message, Messages, State, UserMessage

from .dataset import LAB_FIELDS, make_dataset_builder, normalize_lab_row
from .judge import CriterionJudge, DeepSeekCriterionJudge
from .prompts import HarveyLabPromptBuilder, build_sub_llm_user_prompt
from .resources import stage_rollout_context, write_bootstrap
from .rubric import HarveyLabRubric
from .worker import customize_python_worker_script


SANDBOX_DOCKER_IMAGE = "irfanjamil10/harvey-lab-rlm-sandbox:0.1.0"
SANDBOX_WAIT_FOR_CREATION_MAX_ATTEMPTS = 240
SANDBOX_TIMEOUT_MINUTES = 17
ROLLOUT_TIMEOUT_SECONDS = 20 * 60


class HarveyLabTimingRubric(vf.Rubric):
    def __init__(self) -> None:
        super().__init__()
        self.add_metric(self.lab_rollout_duration_seconds)
        self.add_metric(self.lab_sandbox_lifetime_seconds)

    async def lab_rollout_duration_seconds(self, state: State) -> float:
        timing = state.get("timing")
        generation = getattr(timing, "generation", None)
        return float(getattr(generation, "duration", 0.0) or 0.0)

    async def lab_sandbox_lifetime_seconds(self, state: State) -> float:
        return float(state.get("lab_sandbox_lifetime_seconds", 0.0) or 0.0)


class HarveyLabRLMEnv(RLMEnv):
    def __init__(
        self,
        *,
        dataset_builder: Callable[[], Dataset],
        judge: CriterionJudge,
        judge_parallelism: int = 6,
        max_turns: int = 200,
        sub_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("timeout_seconds", ROLLOUT_TIMEOUT_SECONDS)
        rubric = HarveyLabRubric(judge=judge, parallelism=judge_parallelism)
        super().__init__(
            dataset=dataset_builder,
            rubric=rubric,
            max_turns=max_turns,
            tools=[],
            root_tools=[],
            sub_tools=[],
            sub_model=sub_model,
            sub_llm_max_turns=1,
            enable_sub_llms=True,
            enable_summarization=False,
            repl_language="python",
            system_prompt="Harvey LAB prompt contract is installed separately.",
            pip_install_packages="",
            include_sub_llm_in_trajectory=False,
            retain_filesystem_after_rollout=False,
            sandbox_docker_image=SANDBOX_DOCKER_IMAGE,
            sandbox_timeout_minutes=SANDBOX_TIMEOUT_MINUTES,
            **kwargs,
        )
        self._executor.sandbox_wait_for_creation_max_attempts = (
            SANDBOX_WAIT_FOR_CREATION_MAX_ATTEMPTS
        )
        self.prompt_builder = HarveyLabPromptBuilder()
        self.add_rubric(HarveyLabTimingRubric())

    async def setup_state(self, state: State, **kwargs: Any) -> None:
        original_info = state.get("info")
        metadata = original_info if isinstance(original_info, dict) else {}
        source = {
            field: metadata.get(field, state.get(field))
            for field in LAB_FIELDS
        }
        row = normalize_lab_row(source)
        for key, value in row.items():
            if key != "prompt":
                state[key] = value
        state["expected_deliverables"] = list(row["deliverables"])
        state["deliverables"] = {}
        state["missing_deliverables"] = list(row["deliverables"])
        state["deliverable_errors"] = {}
        state["rlm_fs_root_remote"] = "/workspace"

        with tempfile.TemporaryDirectory(prefix="harvey-lab-rlm-") as temp_dir:
            staging_root = Path(temp_dir)
            stage_rollout_context(staging_root, row)
            info = dict(original_info) if isinstance(original_info, dict) else {}
            info["context_dir"] = str(staging_root)
            state["info"] = info
            state["lab_sandbox_lifetime_start_time"] = time.time()
            try:
                await super().setup_state(state, **kwargs)
            finally:
                if original_info is None:
                    state.pop("info", None)
                else:
                    state["info"] = original_info

    def customize_worker_script(self, script: str, state: State) -> str:
        return customize_python_worker_script(script)

    async def _upload_message_history(self, state: State) -> None:
        return None

    async def _execute_code(self, code: str, state: State) -> dict[str, Any]:
        result = await super()._execute_code(code, state)
        answer = result.get("answer")
        if isinstance(answer, dict) and answer.get("ready", False):
            parsed = result.get("deliverables")
            missing = result.get("missing_deliverables")
            errors = result.get("deliverable_errors")
            state["deliverables"] = dict(parsed) if isinstance(parsed, dict) else {}
            state["missing_deliverables"] = (
                list(missing) if isinstance(missing, list) else []
            )
            state["deliverable_errors"] = (
                dict(errors) if isinstance(errors, dict) else {}
            )
        return result

    async def _run_sub_llm_request(
        self,
        *,
        state_ref: State,
        client,
        sub_model: str,
        messages: Messages,
        batch_id: str,
        request_id: str,
        parent_turn: int,
    ) -> dict[str, Any]:
        wrapped: list[Message] = []
        found_user = False
        for message in messages:
            if getattr(message, "role", None) == "user" and isinstance(
                getattr(message, "content", None), str
            ):
                wrapped.append(
                    UserMessage(
                        content=build_sub_llm_user_prompt(
                            str(state_ref.get("instructions", "")),
                            str(message.content),
                        )
                    )
                )
                found_user = True
            else:
                wrapped.append(message)
        if not found_user:
            wrapped.append(
                UserMessage(
                    content=build_sub_llm_user_prompt(
                        str(state_ref.get("instructions", "")),
                        "Provide a concise JSON analysis of the supplied task objective.",
                    )
                )
            )
        return await super()._run_sub_llm_request(
            state_ref=state_ref,
            client=client,
            sub_model=sub_model,
            messages=wrapped,
            batch_id=batch_id,
            request_id=request_id,
            parent_turn=parent_turn,
        )

    async def _recover_from_code_timeout(self, state: State) -> bool:
        try:
            session = self._executor._get_session(state)
            if session.sandbox_id is None:
                return False
            with tempfile.TemporaryDirectory(prefix="harvey-lab-bootstrap-") as temp:
                local_path = Path(temp) / "bootstrap.json"
                write_bootstrap(
                    local_path,
                    instructions=str(state["instructions"]),
                    documents=dict(state["documents"]),
                    expected_deliverables=list(state["expected_deliverables"]),
                )
                await self._executor._upload_file_with_retry(
                    session.sandbox_id,
                    "/workspace/.lab/bootstrap.json",
                    str(local_path),
                    "LAB bootstrap recovery upload",
                )
        except Exception:
            self.logger.exception("Failed to restore LAB bootstrap after timeout")
            return False
        return await super()._recover_from_code_timeout(state)

    @vf.cleanup
    async def cleanup_rlm_state(self, state: State) -> None:
        rollout_dir = state.get("rlm_rollout_dir")
        try:
            await super().cleanup_rlm_state(state)
        finally:
            if rollout_dir:
                await asyncio.to_thread(shutil.rmtree, rollout_dir, True)
            started_at = state.get("lab_sandbox_lifetime_start_time")
            if isinstance(started_at, (int, float)):
                state["lab_sandbox_lifetime_seconds"] = time.time() - float(started_at)

    @vf.cleanup(priority=-100)
    async def strip_runtime_paths(self, state: State) -> None:
        for key in (
            "rlm_rollout_dir",
            "rlm_fs_root",
            "rlm_control_dir",
            "rlm_paths",
            "rlm_fs_staging_root",
            "rlm_control_dir_local",
            "rlm_fs_root_remote",
            "rlm_control_dir_remote",
            "rlm_paths_remote",
            "rlm_fs_source",
            "sandbox_state",
            "interception_url",
            "root_tool_url",
            "lab_sandbox_lifetime_start_time",
        ):
            state.pop(key, None)


def _load_project_dotenv() -> None:
    dotenv_path = Path.cwd() / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path, override=False)


def load_environment(
    dataset_name: str = "irfanjamil/Harvey-LAB",
    split: str = "train",
    max_turns: int = 200,
    sub_model: str | None = None,
    judge_model: str = "deepseek-v4-flash",
    judge_parallelism: int = 6,
    *,
    judge: CriterionJudge | None = None,
    **kwargs: Any,
) -> vf.Environment:
    _load_project_dotenv()
    vf.ensure_keys(["PRIME_API_KEY", "DEEPSEEK_API_KEY"])
    resolved_judge = judge or DeepSeekCriterionJudge(
        model=judge_model,
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    return HarveyLabRLMEnv(
        dataset_builder=make_dataset_builder(dataset_name, split),
        judge=resolved_judge,
        judge_parallelism=judge_parallelism,
        max_turns=max_turns,
        sub_model=sub_model,
        **kwargs,
    )
