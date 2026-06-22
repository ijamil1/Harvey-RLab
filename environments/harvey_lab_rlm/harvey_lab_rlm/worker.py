from __future__ import annotations


IMPORT_ANCHOR = "import traceback\nfrom pathlib import Path\n"
NAMESPACE_ANCHOR = (
    '    "answer": answer,\n'
    "}\n"
    "for tool_name in ROOT_TOOL_NAMES:\n"
)
RESULT_ANCHOR = (
    '    result["answer"] = namespace.get("answer", {"ready": False, "content": ""})\n'
    "\n"
    '    with open(ANSWER_FILE, "w", encoding="utf-8") as f:\n'
)


def customize_python_worker_script(script: str) -> str:
    missing = [
        name
        for name, anchor in (
            ("import", IMPORT_ANCHOR),
            ("namespace", NAMESPACE_ANCHOR),
            ("result", RESULT_ANCHOR),
        )
        if anchor not in script
    ]
    if missing:
        raise RuntimeError(
            "Unsupported Verifiers RLM worker template; missing worker template "
            "anchors: " + ", ".join(missing)
        )

    script = script.replace(
        IMPORT_ANCHOR,
        IMPORT_ANCHOR
        + "import shutil\n"
        + 'sys.path.insert(0, "/workspace/.lab")\n'
        + "from lab_runtime import load_runtime_namespace\n",
        1,
    )
    script = script.replace('    "extra_data": extra_data,\n', "", 1)
    script = script.replace(
        NAMESPACE_ANCHOR,
        '    "answer": answer,\n'
        "}\n"
        'lab_namespace = load_runtime_namespace("/workspace/.lab/bootstrap.json")\n'
        'shutil.rmtree("/workspace/.lab")\n'
        "namespace.update(lab_namespace)\n"
        "del lab_namespace\n"
        "for tool_name in ROOT_TOOL_NAMES:\n",
        1,
    )
    script = script.replace(
        RESULT_ANCHOR,
        '    result["answer"] = namespace.get("answer", {"ready": False, "content": ""})\n'
        '    if result["answer"].get("ready", False):\n'
        '        result.update(namespace["_collect_deliverables"]())\n'
        "\n"
        '    with open(ANSWER_FILE, "w", encoding="utf-8") as f:\n',
        1,
    )
    return script
