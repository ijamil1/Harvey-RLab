from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .judge import CriterionJudge


class HarveyLabScorer:
    def __init__(self, *, judge: CriterionJudge, parallelism: int = 6) -> None:
        self.judge = judge
        self.parallelism = max(1, int(parallelism))

    async def score_rollout(self, state: dict[str, Any]) -> float:
        criteria = state.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("rollout state has no LAB criteria")
        parsed = state.get("deliverables")
        deliverables = parsed if isinstance(parsed, dict) else {}
        errors = state.get("deliverable_errors")
        deliverable_errors = errors if isinstance(errors, dict) else {}
        expected = state.get("expected_deliverables")
        if not isinstance(expected, list):
            expected = list(deliverables)

        semaphore = asyncio.Semaphore(self.parallelism)
        judge_call_count = 0

        async def score_one(criterion: Mapping[str, Any]) -> dict[str, str]:
            nonlocal judge_call_count
            required = criterion.get("deliverables")
            if not isinstance(required, list) or not required:
                required = expected
            unavailable = [
                name
                for name in required
                if name not in deliverables or name in deliverable_errors
            ]
            if unavailable:
                return {
                    "id": str(criterion["id"]),
                    "title": str(criterion["title"]),
                    "verdict": "fail",
                    "reasoning": (
                        "Required deliverable text was unavailable: "
                        + ", ".join(unavailable)
                    ),
                }

            sections = [f"## {name}\n{deliverables[name]}" for name in required]
            async with semaphore:
                judge_call_count += 1
                result = await self.judge.evaluate(
                    task_title=str(state.get("title", "")),
                    task_instructions=str(state.get("instructions", "")),
                    criterion=criterion,
                    agent_output="\n\n".join(sections),
                )
            verdict = str(result.get("verdict", "")).lower()
            reasoning = result.get("reasoning")
            if verdict not in {"pass", "fail"}:
                raise ValueError(f"invalid judge verdict: {verdict!r}")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("judge reasoning must be a non-empty string")
            return {
                "id": str(criterion["id"]),
                "title": str(criterion["title"]),
                "verdict": verdict,
                "reasoning": reasoning.strip(),
            }

        results = await asyncio.gather(*(score_one(c) for c in criteria))
        passed = sum(result["verdict"] == "pass" for result in results)
        total = len(results)
        reward = passed / total
        missing = state.get("missing_deliverables")
        missing_count = len(missing) if isinstance(missing, list) else 0

        state["criterion_results"] = results
        state["lab_metrics"] = {
            "lab_criteria_passed": float(passed),
            "lab_criteria_total": float(total),
            "lab_criterion_pass_rate": float(reward),
            "lab_all_pass": float(passed == total),
            "lab_missing_deliverables": float(missing_count),
            "lab_deliverable_errors": float(len(deliverable_errors)),
            "lab_judge_calls": float(judge_call_count),
        }
        return float(reward)
