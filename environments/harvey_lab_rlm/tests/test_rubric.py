from __future__ import annotations

import asyncio

import pytest

from harvey_lab_rlm.rubric import HarveyLabRubric


class FakeJudge:
    def __init__(self, verdicts: dict[str, str], *, fail: bool = False):
        self.verdicts = verdicts
        self.fail = fail
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0

    async def evaluate(self, **kwargs) -> dict:
        if self.fail:
            raise RuntimeError("judge unavailable")
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        criterion_id = kwargs["criterion"]["id"]
        return {
            "verdict": self.verdicts[criterion_id],
            "reasoning": f"Reason for {criterion_id}",
        }


def state_fixture() -> dict:
    return {
        "title": "Draft memo",
        "instructions": "Draft the memo.",
        "deliverables": {"memo.docx": "Memo text", "schedule.xlsx": "Sheet text"},
        "missing_deliverables": [],
        "deliverable_errors": {},
        "criteria": [
            {
                "id": "C-001",
                "title": "Memo criterion",
                "deliverables": ["memo.docx"],
                "match_criteria": "PASS if memo text is present.",
            },
            {
                "id": "C-002",
                "title": "Workbook criterion",
                "deliverables": ["schedule.xlsx"],
                "match_criteria": "PASS if sheet text is present.",
            },
            {
                "id": "C-003",
                "title": "Both criterion",
                "deliverables": ["memo.docx", "schedule.xlsx"],
                "match_criteria": "PASS if both are present.",
            },
        ],
    }


@pytest.mark.asyncio
async def test_partial_credit_and_deliverable_scoping() -> None:
    judge = FakeJudge({"C-001": "pass", "C-002": "fail", "C-003": "pass"})
    rubric = HarveyLabRubric(judge=judge, parallelism=2)
    state = state_fixture()

    await rubric.score_rollout(state)

    assert state["reward"] == pytest.approx(2 / 3)
    assert [r["id"] for r in state["criterion_results"]] == [
        "C-001",
        "C-002",
        "C-003",
    ]
    assert state["metrics"]["lab_criteria_passed"] == 2.0
    assert state["metrics"]["lab_criteria_total"] == 3.0
    assert state["metrics"]["lab_all_pass"] == 0.0
    assert judge.max_active <= 2
    assert "schedule.xlsx" not in judge.calls[0]["agent_output"]
    assert "memo.docx" not in judge.calls[1]["agent_output"]
    assert "memo.docx" in judge.calls[2]["agent_output"]
    assert "schedule.xlsx" in judge.calls[2]["agent_output"]


@pytest.mark.asyncio
async def test_missing_deliverable_auto_fails_without_judge_call() -> None:
    judge = FakeJudge({"C-001": "pass", "C-002": "pass", "C-003": "pass"})
    rubric = HarveyLabRubric(judge=judge, parallelism=3)
    state = state_fixture()
    state["deliverables"].pop("schedule.xlsx")
    state["missing_deliverables"] = ["schedule.xlsx"]

    await rubric.score_rollout(state)

    assert [call["criterion"]["id"] for call in judge.calls] == ["C-001"]
    assert state["reward"] == pytest.approx(1 / 3)
    assert state["criterion_results"][1]["verdict"] == "fail"
    assert state["criterion_results"][2]["verdict"] == "fail"


@pytest.mark.asyncio
async def test_judge_infrastructure_error_propagates() -> None:
    rubric = HarveyLabRubric(judge=FakeJudge({}, fail=True), parallelism=2)

    with pytest.raises(RuntimeError, match="judge unavailable"):
        await rubric.score_rollout(state_fixture())
