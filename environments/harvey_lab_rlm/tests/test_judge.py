from __future__ import annotations

from types import SimpleNamespace

import pytest

from harvey_lab_rlm.judge import DeepSeekCriterionJudge


class FakeChatCompletions:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.response_text),
                )
            ]
        )


class FakeClient:
    def __init__(self, response_text: str) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions(response_text))


@pytest.mark.asyncio
async def test_raises_raw_judge_response_on_parse_failure() -> None:
    judge = DeepSeekCriterionJudge(
        model="test-judge",
        api_key="unused",
        max_attempts=1,
    )
    judge.client = FakeClient("not json from provider")

    with pytest.raises(ValueError) as exc_info:
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

    message = str(exc_info.value)
    assert "judge response did not contain JSON" in message
    assert "model='test-judge'" in message
    assert "attempt=1/1" in message
    assert "criterion_id='C-001'" in message
    assert "criterion_title='Memo quality'" in message
    assert "finish_reason='stop'" in message
    assert "response_chars=22" in message
    assert "raw_response='not json from provider'" in message


@pytest.mark.asyncio
async def test_raises_raw_judge_response_on_invalid_json_contract() -> None:
    judge = DeepSeekCriterionJudge(
        model="test-judge",
        api_key="unused",
        max_attempts=1,
    )
    judge.client = FakeClient('{"verdict":"maybe","reasoning":"unclear"}')

    with pytest.raises(ValueError) as exc_info:
        await judge.evaluate(
            task_title="Draft memo",
            task_instructions="Draft the memo.",
            criterion={
                "id": "C-002",
                "title": "Strict verdict",
                "match_criteria": "PASS or FAIL only.",
            },
            agent_output="Memo text",
        )

    message = str(exc_info.value)
    assert "invalid judge verdict: 'maybe'" in message
    assert "criterion_id='C-002'" in message
    assert 'raw_response=\'{"verdict":"maybe","reasoning":"unclear"}\'' in message
