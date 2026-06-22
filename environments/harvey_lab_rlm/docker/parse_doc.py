#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

import openpyxl


def parse_docx(path: str) -> str:
    result = subprocess.run(
        [
            "pandoc",
            path,
            "-t",
            "markdown",
            "--wrap=none",
            "--track-changes=accept",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pandoc failed")
    return result.stdout


def parse_xlsx(path: str) -> str:
    workbook = openpyxl.load_workbook(path, data_only=False, read_only=True)
    sections: list[str] = []
    for sheet in workbook.worksheets:
        sections.append(f"=== Sheet: {sheet.title} ===")
        for row in sheet.iter_rows(values_only=True):
            values = ["" if value is None else str(value) for value in row]
            sections.append("\t".join(values).rstrip())
    return "\n".join(sections)


PARSERS = {"docx": parse_docx, "xlsx": parse_xlsx}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in PARSERS:
        print("usage: parse-doc {docx|xlsx} <path>", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(PARSERS[sys.argv[1]](sys.argv[2]))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
