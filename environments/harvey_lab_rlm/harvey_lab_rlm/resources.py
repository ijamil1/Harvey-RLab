from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


SKILL_NAMES = ("docx", "xlsx")


def _resource_skill_root():
    return resources.files("harvey_lab_rlm").joinpath("resources", "skills")


def _read_skill_manual(name: str) -> str:
    return (
        _resource_skill_root()
        .joinpath(name, "RLM_SKILL.md")
        .read_text(encoding="utf-8")
    )


def _copy_skill_scripts(name: str, destination: Path) -> None:
    packaged = _resource_skill_root().joinpath(name, "scripts")
    destination.mkdir(parents=True, exist_ok=True)
    for child in packaged.iterdir():
        if child.is_file():
            (destination / child.name).write_bytes(child.read_bytes())


def write_bootstrap(
    path: Path,
    *,
    instructions: str,
    documents: dict[str, str],
    expected_deliverables: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instructions": instructions,
        "documents": documents,
        "skills": {name: _read_skill_manual(name) for name in SKILL_NAMES},
        "expected_deliverables": expected_deliverables,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def stage_rollout_context(root: Path, row: dict[str, Any]) -> None:
    lab_dir = root / ".lab"
    write_bootstrap(
        lab_dir / "bootstrap.json",
        instructions=row["instructions"],
        documents=row["documents"],
        expected_deliverables=row["deliverables"],
    )
    runtime_source = resources.files("harvey_lab_rlm").joinpath(
        "sandbox_runtime.py"
    )
    (lab_dir / "lab_runtime.py").write_bytes(runtime_source.read_bytes())
    for skill_name in SKILL_NAMES:
        _copy_skill_scripts(
            skill_name,
            root / "skills" / skill_name / "scripts",
        )
    (root / "output").mkdir(parents=True, exist_ok=True)
