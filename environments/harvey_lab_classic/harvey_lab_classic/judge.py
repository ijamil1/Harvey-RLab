from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any, Protocol

from openai import AsyncOpenAI

DEFAULT_MAX_TOKENS = 4096
RETRY_MAX_TOKENS_MULTIPLIER = 1.25


class CriterionJudge(Protocol):
    async def evaluate(
        self,
        *,
        task_title: str,
        task_instructions: str,
        criterion: Mapping[str, Any],
        agent_output: str,
    ) -> dict[str, str]: ...


class DeepSeekCriterionJudge:
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        *,
        api_key: str | None = None,
        timeout_seconds: float = 600,
        max_attempts: int = 3,
    ) -> None:
        self.model = model
        self.max_attempts = max(1, max_attempts)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def evaluate(
        self,
        *,
        task_title: str,
        task_instructions: str,
        criterion: Mapping[str, Any],
        agent_output: str,
    ) -> dict[str, str]:
        prompt = (
            "You are evaluating a legal work product against one criterion.\n\n"
            f"TASK TITLE\n{task_title}\n\n"
            f"TASK INSTRUCTIONS\n{task_instructions}\n\n"
            f"AGENT OUTPUT\n{agent_output}\n\n"
            f"CRITERION\n{criterion['title']}\n\n"
            f"{criterion['match_criteria']}\n\n"
            "Return JSON only with exactly two keys: "
            '{"verdict":"pass"|"fail","reasoning":"brief explanation"}.'
        )
        retry_instruction = (
            "\n\nDo not overthink. Return only the required JSON object."
        )
        last_error: Exception | None = None
        max_tokens = DEFAULT_MAX_TOKENS
        for attempt in range(self.max_attempts):
            try:
                request_prompt = prompt if attempt == 0 else prompt + retry_instruction
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": request_prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content or ""
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                try:
                    result = self._parse_result(text)
                except ValueError as exc:
                    if self._is_empty_json_response_error(exc, text):
                        max_tokens = int(max_tokens * RETRY_MAX_TOKENS_MULTIPLIER)
                    raise self._diagnostic_parse_error(
                        exc,
                        text=text,
                        finish_reason=finish_reason,
                        criterion=criterion,
                        attempt=attempt + 1,
                    ) from exc
                return result
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    await asyncio.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _is_empty_json_response_error(exc: ValueError, text: str) -> bool:
        return not text and str(exc) == "judge response did not contain JSON"

    @staticmethod
    def _parse_result(text: str) -> dict[str, str]:
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("judge response did not contain JSON")
            result = json.loads(text[start : end + 1])
        if not isinstance(result, dict):
            raise ValueError("judge response must be a JSON object")
        verdict = str(result.get("verdict", "")).lower()
        reasoning = result.get("reasoning")
        if verdict not in {"pass", "fail"}:
            raise ValueError(f"invalid judge verdict: {verdict!r}")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("judge reasoning must be a non-empty string")
        return {"verdict": verdict, "reasoning": reasoning.strip()}

    def _diagnostic_parse_error(
        self,
        exc: ValueError,
        *,
        text: str,
        finish_reason: str | None,
        criterion: Mapping[str, Any],
        attempt: int,
    ) -> ValueError:
        raw_response_limit = 4000
        raw_response = text[:raw_response_limit]
        truncated = len(text) > raw_response_limit
        return ValueError(
            f"{exc} "
            f"(model={self.model!r}, "
            f"attempt={attempt}/{self.max_attempts}, "
            f"criterion_id={criterion.get('id')!r}, "
            f"criterion_title={criterion.get('title')!r}, "
            f"finish_reason={finish_reason!r}, "
            f"response_chars={len(text)}, "
            f"raw_response_truncated={truncated}, "
            f"raw_response={raw_response!r})"
        )
