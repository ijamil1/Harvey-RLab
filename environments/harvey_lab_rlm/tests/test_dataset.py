from __future__ import annotations

import json

import pytest

from harvey_lab_rlm.dataset import normalize_lab_row


def valid_row() -> dict:
    return {
        "task_id": "corporate/draft-memo",
        "practice_area": "corporate",
        "title": "Draft memo",
        "work_type": "draft",
        "tags": ["memo"],
        "instructions": 'Draft the requested memo for "Acme".',
        "deliverables": ["memo.docx"],
        "criteria": [
            {
                "id": "C-001",
                "title": "Contains conclusion",
                "deliverables": ["memo.docx"],
                "match_criteria": "PASS if a conclusion is present.",
            }
        ],
        "documents": {"facts.docx": "Unicode facts: café — § 2."},
    }


def test_normalize_row_preserves_task_fields_and_builds_user_prompt() -> None:
    row = normalize_lab_row(valid_row())

    assert row["prompt"] == [
        {"role": "user", "content": 'Draft the requested memo for "Acme".'}
    ]
    assert row["documents"] == {"facts.docx": "Unicode facts: café — § 2."}
    assert row["criteria"][0]["id"] == "C-001"
    assert row["deliverables"] == ["memo.docx"]


def test_normalize_row_decodes_json_fields() -> None:
    source = valid_row()
    source["criteria"] = json.dumps(source["criteria"])
    source["documents"] = json.dumps(source["documents"])

    row = normalize_lab_row(source)

    assert isinstance(row["criteria"], list)
    assert isinstance(row["documents"], dict)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("instructions", "", "instructions"),
        ("deliverables", [], "deliverables"),
        ("deliverables", ["nested/memo.docx"], "basename"),
        ("deliverables", ["memo.pdf"], ".docx or .xlsx"),
        ("documents", ["not-a-mapping"], "documents"),
        ("documents", {"facts.docx": 123}, "document text"),
        ("criteria", [], "criteria"),
        (
            "criteria",
            [{"id": "C-001", "title": "Missing match criteria"}],
            "match_criteria",
        ),
        (
            "criteria",
            [
                {
                    "id": "C-001",
                    "title": "Wrong file",
                    "match_criteria": "PASS",
                    "deliverables": ["other.docx"],
                }
            ],
            "unknown deliverable",
        ),
    ],
)
def test_normalize_row_rejects_invalid_contract(
    field: str, value: object, message: str
) -> None:
    source = valid_row()
    source[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        normalize_lab_row(source)
