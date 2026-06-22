from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from datasets.table import InMemoryTable
from huggingface_hub import HfApi, hf_hub_download


SUPPORTED_DELIVERABLE_SUFFIXES = {".docx", ".xlsx"}
LAB_FIELDS = (
    "task_id",
    "practice_area",
    "title",
    "work_type",
    "tags",
    "instructions",
    "deliverables",
    "criteria",
    "documents",
)


def _decode_json_field(value: object, field: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must contain valid JSON") from exc


def _validate_deliverable_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("deliverables must contain non-empty strings")
    name = value.strip()
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"deliverable must be a basename: {name!r}")
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in SUPPORTED_DELIVERABLE_SUFFIXES:
        raise ValueError(f"deliverable must end in .docx or .xlsx: {name!r}")
    return name


def normalize_lab_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)

    instructions = normalized.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")
    instructions = instructions.strip()

    raw_deliverables = normalized.get("deliverables")
    if not isinstance(raw_deliverables, list) or not raw_deliverables:
        raise ValueError("deliverables must be a non-empty list")
    deliverables = [_validate_deliverable_name(item) for item in raw_deliverables]
    if len(set(deliverables)) != len(deliverables):
        raise ValueError("deliverables must not contain duplicate filenames")

    documents = _decode_json_field(normalized.get("documents"), "documents")
    if not isinstance(documents, dict):
        raise TypeError("documents must be a JSON object mapping filenames to text")
    for filename, text in documents.items():
        if not isinstance(filename, str) or not filename:
            raise TypeError("document names must be non-empty strings")
        if not isinstance(text, str):
            raise TypeError(f"document text must be a string for {filename!r}")

    criteria = _decode_json_field(normalized.get("criteria"), "criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("criteria must be a non-empty list")
    normalized_criteria: list[dict[str, Any]] = []
    deliverable_set = set(deliverables)
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise TypeError(f"criterion {index} must be a JSON object")
        result = dict(criterion)
        for field in ("id", "title", "match_criteria"):
            value = result.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"criterion {index} requires non-empty {field}")
            result[field] = value.strip()
        criterion_deliverables = result.get("deliverables")
        if criterion_deliverables is not None:
            if not isinstance(criterion_deliverables, list):
                raise TypeError(
                    f"criterion {result['id']} deliverables must be a list"
                )
            resolved = [
                _validate_deliverable_name(item) for item in criterion_deliverables
            ]
            unknown = sorted(set(resolved) - deliverable_set)
            if unknown:
                raise ValueError(
                    f"criterion {result['id']} references unknown deliverable: "
                    + ", ".join(unknown)
                )
            result["deliverables"] = resolved
        normalized_criteria.append(result)

    normalized.update(
        {
            "instructions": instructions,
            "deliverables": deliverables,
            "criteria": normalized_criteria,
            "documents": dict(documents),
            "prompt": [{"role": "user", "content": instructions}],
        }
    )
    return normalized


def make_dataset_builder(
    dataset_name: str = "irfanjamil/Harvey-LAB",
    split: str = "train",
) -> Callable[[], Dataset]:
    def build() -> Dataset:
        dataset = _load_json_metadata_compatible_dataset(dataset_name, split)
        return dataset.map(
            lambda row: {
                "prompt": [
                    {
                        "role": "user",
                        "content": str(row["instructions"]).strip(),
                    }
                ]
            },
            desc="Building Harvey LAB prompts",
        )

    return build


def _load_json_metadata_compatible_dataset(
    dataset_name: str, split: str
) -> Dataset:
    """Load LAB Parquet while bypassing datasets<5's unsupported Json feature."""
    match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\[(.*)\])?", split)
    if match is None:
        raise ValueError(f"unsupported split expression: {split!r}")
    split_name, slice_expression = match.groups()
    prefix = f"data/{split_name}-"
    repo_files = HfApi().list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = sorted(
        filename
        for filename in repo_files
        if filename.startswith(prefix) and filename.endswith(".parquet")
    )
    if not parquet_files:
        raise ValueError(
            f"no Parquet files found for split {split_name!r} in {dataset_name}"
        )

    tables: list[pa.Table] = []
    for filename in parquet_files:
        local_path = hf_hub_download(
            dataset_name,
            filename,
            repo_type="dataset",
        )
        table = pq.read_table(Path(local_path))
        for column_name in ("criteria", "documents"):
            index = table.schema.get_field_index(column_name)
            if index >= 0:
                table = table.set_column(
                    index,
                    column_name,
                    table[column_name].cast(pa.string()),
                )
        tables.append(table.replace_schema_metadata(None))

    dataset = Dataset(InMemoryTable(pa.concat_tables(tables)), split=split_name)
    if slice_expression is not None:
        start_text, separator, stop_text = slice_expression.partition(":")
        if not separator:
            index = int(start_text)
            return dataset.select([index])
        start = int(start_text) if start_text else 0
        stop = int(stop_text) if stop_text else len(dataset)
        return dataset.select(range(start, min(stop, len(dataset))))
    return dataset
