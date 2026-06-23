from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, Protocol

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


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
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content or ""
                try:
                    result = self._parse_result(text)
                except ValueError:
                    logger.warning(
                        "Failed to parse judge response as JSON "
                        "(model=%s, attempt=%d/%d, criterion_id=%s, "
                        "criterion_title=%r, response_chars=%d): %r",
                        self.model,
                        attempt + 1,
                        self.max_attempts,
                        criterion.get("id"),
                        criterion.get("title"),
                        len(text),
                        text,
                    )
                    raise
                return result
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    await asyncio.sleep(2**attempt)
        assert last_error is not None
        raise last_error

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
