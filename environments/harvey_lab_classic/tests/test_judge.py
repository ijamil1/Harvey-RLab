from __future__ import annotations

from types import SimpleNamespace

import pytest

from harvey_lab_classic.judge import DeepSeekCriterionJudge


class FakeChatCompletions:
    def __init__(self, *response_texts: str) -> None:
        self.response_texts = list(response_texts)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response_text = self.response_texts.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=response_text),
                )
            ]
        )


class FakeClient:
    def __init__(self, *response_texts: str) -> None:
        self.completions = FakeChatCompletions(*response_texts)
        self.chat = SimpleNamespace(completions=self.completions)


async def no_sleep(seconds: float) -> None:
    pass


@pytest.mark.asyncio
async def test_uses_4096_judge_output_token_limit() -> None:
    judge = DeepSeekCriterionJudge(
        model="test-judge",
        api_key="unused",
        max_attempts=1,
    )
    client = FakeClient('{"verdict":"pass","reasoning":"sufficient"}')
    judge.client = client

    await judge.evaluate(
        task_title="Draft memo",
        task_instructions="Draft the memo.",
        criterion={
            "id": "C-000",
            "title": "Memo quality",
            "match_criteria": "PASS if memo is good.",
        },
        agent_output="Memo text",
    )

    assert client.completions.calls[0]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_empty_json_response_retry_increases_limit_and_adds_retry_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import harvey_lab_classic.judge as judge_module

    monkeypatch.setattr(judge_module.asyncio, "sleep", no_sleep)
    judge = DeepSeekCriterionJudge(
        model="test-judge",
        api_key="unused",
        max_attempts=2,
    )
    client = FakeClient("", '{"verdict":"pass","reasoning":"sufficient"}')
    judge.client = client

    result = await judge.evaluate(
        task_title="Draft memo",
        task_instructions="Draft the memo.",
        criterion={
            "id": "C-000",
            "title": "Memo quality",
            "match_criteria": "PASS if memo is good.",
        },
        agent_output="Memo text",
    )

    assert result == {"verdict": "pass", "reasoning": "sufficient"}
    assert [call["max_tokens"] for call in client.completions.calls] == [4096, 5120]
    first_prompt = client.completions.calls[0]["messages"][0]["content"]
    retry_prompt = client.completions.calls[1]["messages"][0]["content"]
    assert "Do not overthink" not in first_prompt
    assert "Do not overthink" in retry_prompt
