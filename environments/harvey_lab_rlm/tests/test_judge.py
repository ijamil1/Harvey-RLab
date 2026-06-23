from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from harvey_lab_rlm.judge import DeepSeekCriterionJudge


class FakeChatCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="not json from provider")
                )
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


@pytest.mark.asyncio
async def test_logs_raw_judge_response_on_parse_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    judge = DeepSeekCriterionJudge(
        model="test-judge",
        api_key="unused",
        max_attempts=1,
    )
    judge.client = FakeClient()

    with caplog.at_level(logging.WARNING, logger="harvey_lab_rlm.judge"):
        with pytest.raises(ValueError, match="judge response did not contain JSON"):
            await judge.evaluate(
                task_title="Draft memo",
                task_instructions="Draft the memo.",
                criterion={
                    "id": "C-001",
                    "title": "Memo quality",
                    "match_criteria": "PASS if memo is good.",
                },
                agent_output="Memo text",
            )

    assert "Failed to parse judge response as JSON" in caplog.text
    assert "test-judge" in caplog.text
    assert "C-001" in caplog.text
    assert "not json from provider" in caplog.text
