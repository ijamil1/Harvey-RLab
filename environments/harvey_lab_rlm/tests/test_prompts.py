from __future__ import annotations

from harvey_lab_rlm.prompts import (
    ROOT_PROMPT,
    SUB_LLM_SYSTEM_PROMPT,
    build_sub_llm_user_prompt,
)


def test_root_prompt_matches_runtime_contract() -> None:
    required = [
        "call_python_repl",
        "<repl>",
        "documents",
        "skills",
        "instructions",
        "expected_deliverables",
        "read",
        "write",
        "bash",
        "llm_batch",
        "Work only within `/workspace`",
        "do not read, write, list, inspect, execute",
        "direct Python filesystem APIs",
        "subprocesses",
        "shell commands",
        "/workspace/output",
        'answer["ready"] = True',
        'answer["content"]',
        "exact",
        "no tools",
        "persistent",
    ]
    for text in required:
        assert text in ROOT_PROMPT

    obsolete = [
        "context_dir",
        "extra_data",
        "query_llm",
        "query_llm_batch",
        "finish(",
        "/workspace/documents",
    ]
    for text in obsolete:
        assert text not in ROOT_PROMPT


def test_sub_llm_system_prompt_is_focused_and_output_constrained() -> None:
    for text in [
        "focused",
        "stateless",
        "Analyze the task objective",
        "excerpts",
        "intermediate data",
        "supplied in the user message",
        "Return only the requested output format",
        "concise JSON",
        "`analysis` and `evidence`",
    ]:
        assert text in SUB_LLM_SYSTEM_PROMPT


def test_sub_llm_user_prompt_preserves_unicode_braces_and_quotes() -> None:
    instructions = 'Analyze § 2 for "Acme" and preserve {defined_terms}.'
    delegated = (
        'Document: "facts.docx"\nExcerpt: café — {x}\n'
        "Question: identify the obligation.\n"
        'Output format: {"obligation": "string"}'
    )

    rendered = build_sub_llm_user_prompt(instructions, delegated)

    assert instructions in rendered
    assert delegated in rendered
    assert "LAB TASK OBJECTIVE" in rendered
    assert "DELEGATED SUBTASK" in rendered
    assert "requested output format" in rendered
