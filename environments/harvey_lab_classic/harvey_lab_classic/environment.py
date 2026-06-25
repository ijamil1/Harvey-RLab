from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import verifiers as vf

from .dataset import LAB_FIELDS, make_task_rows_builder, normalize_lab_row
from .judge import CriterionJudge, DeepSeekCriterionJudge
from .prompts import build_system_prompt
from .resources import collect_deliverables, stage_rollout_context
from .scoring import HarveyLabScorer
from .tools import ALL_TOOLS


SANDBOX_DOCKER_IMAGE = "irfanjamil10/harvey-lab-rlm-sandbox:0.1.0"
SANDBOX_TIMEOUT_MINUTES = 17


class HarveyLabClassicTaskset(vf.Taskset):
    def __init__(
        self,
        *,
        dataset_name: str,
        split: str,
        judge: CriterionJudge,
        judge_parallelism: int = 6,
    ) -> None:
        self.scorer = HarveyLabScorer(judge=judge, parallelism=judge_parallelism)
        super().__init__(
            source=make_task_rows_builder(dataset_name, split),
            taskset_id="harvey-lab-classic",
        )

    @vf.reward(priority=100)
    async def lab_reward(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return await self.scorer.score_rollout(state)

    @vf.metric(priority=0)
    async def lab_criteria_passed(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("lab_metrics", {}).get("lab_criteria_passed", 0.0))

    @vf.metric(priority=0)
    async def lab_criteria_total(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("lab_metrics", {}).get("lab_criteria_total", 0.0))

    @vf.metric(priority=0)
    async def lab_criterion_pass_rate(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("lab_metrics", {}).get("lab_criterion_pass_rate", 0.0))

    @vf.metric(priority=0)
    async def lab_all_pass(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("lab_metrics", {}).get("lab_all_pass", 0.0))

    @vf.metric(priority=0)
    async def lab_missing_deliverables(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("lab_metrics", {}).get("lab_missing_deliverables", 0.0))

    @vf.metric(priority=0)
    async def lab_deliverable_errors(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("lab_metrics", {}).get("lab_deliverable_errors", 0.0))

    @vf.metric(priority=0)
    async def lab_judge_calls(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("lab_metrics", {}).get("lab_judge_calls", 0.0))


class HarveyLabClassicHarness(vf.Harness):
    def __init__(
        self,
        *,
        max_turns: int = 200,
        sandbox_image: str = SANDBOX_DOCKER_IMAGE,
    ) -> None:
        self.classic_toolset = vf.Toolset(
            tools=ALL_TOOLS,
            sandbox={
                "image": sandbox_image,
                "cpu_cores": 1.0,
                "memory_gb": 2.0,
                "disk_size_gb": 5.0,
                "network_access": False,
                "timeout_minutes": SANDBOX_TIMEOUT_MINUTES,
                "scope": "rollout",
            },
        )
        super().__init__(
            system_prompt=build_system_prompt(),
            toolsets=[self.classic_toolset],
            max_turns=max_turns,
        )

    async def setup_state(self, task: vf.Task, state: vf.State) -> vf.State:
        source = {field: task.get(field) for field in LAB_FIELDS}
        row = normalize_lab_row(source)
        for key, value in row.items():
            if key != "prompt":
                state[key] = value
        state["expected_deliverables"] = list(row["deliverables"])
        state["deliverables"] = {}
        state["missing_deliverables"] = list(row["deliverables"])
        state["deliverable_errors"] = {}
        state["tool_metrics"] = {
            "bash_commands": 0,
            "read_calls": 0,
            "files_written": 0,
            "files_edited": 0,
            "glob_searches": 0,
            "grep_searches": 0,
        }

        state = await super().setup_state(task, state)
        sandbox = await self.runtime.resolve_tool_sandbox(
            self.classic_toolset,
            task,
            state,
        )
        state["_lab_classic_sandbox"] = sandbox
        state["lab_sandbox_lifetime_start_time"] = time.time()
        await stage_rollout_context(sandbox, row)
        return state

    @vf.update(priority=100)
    async def collect_classic_deliverables(
        self, task: vf.Task, state: vf.State
    ) -> None:
        _ = task
        sandbox = state.get("_lab_classic_sandbox")
        if sandbox is None or state.get("_lab_classic_deliverables_collected"):
            return
        expected = state.get("expected_deliverables")
        if not isinstance(expected, list):
            expected = []
        result = await collect_deliverables(sandbox, [str(item) for item in expected])
        state["deliverables"] = dict(result.get("deliverables") or {})
        state["missing_deliverables"] = list(result.get("missing_deliverables") or [])
        state["deliverable_errors"] = dict(result.get("deliverable_errors") or {})
        state["_lab_classic_deliverables_collected"] = True
        await sandbox.delete()
        started_at = state.get("lab_sandbox_lifetime_start_time")
        if isinstance(started_at, (int, float)):
            state["lab_sandbox_lifetime_seconds"] = time.time() - float(started_at)
        state.pop("_lab_classic_sandbox", None)

    @vf.metric(priority=0)
    async def lab_sandbox_lifetime_seconds(
        self, task: vf.Task, state: vf.State
    ) -> float:
        _ = task
        return float(state.get("lab_sandbox_lifetime_seconds", 0.0) or 0.0)

    @vf.metric(priority=0)
    async def lab_tool_calls(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        metrics = state.get("tool_metrics", {})
        if not isinstance(metrics, Mapping):
            return 0.0
        return float(sum(int(value) for value in metrics.values()))

    @vf.cleanup(priority=-100)
    async def strip_classic_runtime_state(
        self, task: vf.Task, state: vf.State
    ) -> None:
        _ = task
        sandbox = state.pop("_lab_classic_sandbox", None)
        if sandbox is not None:
            await sandbox.delete()
        state.pop("lab_sandbox_lifetime_start_time", None)
        state.pop("_lab_classic_deliverables_collected", None)


def _load_project_dotenv() -> None:
    dotenv_path = Path.cwd() / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path, override=False)


def load_environment(
    dataset_name: str = "irfanjamil/Harvey-LAB",
    split: str = "train",
    max_turns: int = 200,
    judge_model: str = "deepseek-v4-flash",
    judge_parallelism: int = 6,
    sandbox_image: str = SANDBOX_DOCKER_IMAGE,
    *,
    judge: CriterionJudge | None = None,
    **kwargs: Any,
) -> vf.Environment:
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported environment arguments: {unknown}")
    _load_project_dotenv()
    vf.ensure_keys(["PRIME_API_KEY", "DEEPSEEK_API_KEY"])
    resolved_judge = judge or DeepSeekCriterionJudge(
        model=judge_model,
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    taskset = HarveyLabClassicTaskset(
        dataset_name=dataset_name,
        split=split,
        judge=resolved_judge,
        judge_parallelism=judge_parallelism,
    )
    harness = HarveyLabClassicHarness(
        max_turns=max_turns,
        sandbox_image=sandbox_image,
    )
    return vf.Env(taskset=taskset, harness=harness)
