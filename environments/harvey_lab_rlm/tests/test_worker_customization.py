from __future__ import annotations

import pytest

from harvey_lab_rlm.worker import customize_python_worker_script


BASE_WORKER = """import sys
import traceback
from pathlib import Path

extra_data = fs_root

namespace: dict[str, object] = {
    "__name__": "__main__",
    "extra_data": extra_data,
    "answer": answer,
}
for tool_name in ROOT_TOOL_NAMES:
    namespace[tool_name] = _make_root_tool(tool_name)

    result["answer"] = namespace.get("answer", {"ready": False, "content": ""})

    with open(ANSWER_FILE, "w", encoding="utf-8") as f:
        json.dump(result["answer"], f)
"""


def test_worker_customization_injects_runtime_and_deliverable_collection() -> None:
    script = customize_python_worker_script(BASE_WORKER)

    assert "load_runtime_namespace" in script
    assert 'namespace.update(load_runtime_namespace("/workspace/.lab/bootstrap.json"))' in script
    assert 'result.update(namespace["_collect_deliverables"]())' in script
    assert '"extra_data": extra_data' not in script


def test_worker_customization_fails_on_upstream_template_drift() -> None:
    with pytest.raises(RuntimeError, match="worker template"):
        customize_python_worker_script("print('changed upstream')")


def test_worker_customization_matches_pinned_verifiers_template() -> None:
    from verifiers.envs.experimental.rlm_env import (
        RLMWorkerPaths,
        _render_worker_script,
    )

    paths = RLMWorkerPaths(
        base_dir="/workspace/.rlm-control",
        command_fifo="/workspace/.rlm-control/command",
        response_fifo="/workspace/.rlm-control/response",
        ready_flag="/workspace/.rlm-control/ready",
        worker_path="/workspace/.rlm-control/worker.py",
        worker_pid_file="/workspace/.rlm-control/pid",
        context_file="/workspace/.rlm-control/context.json",
        answer_file="/workspace/.rlm-control/answer.json",
        log_file="/workspace/.rlm-control/worker.log",
    )
    upstream = _render_worker_script(paths, repl_language="python")

    customized = customize_python_worker_script(upstream)

    assert 'namespace.update(load_runtime_namespace("/workspace/.lab/bootstrap.json"))' in customized
    assert 'result.update(namespace["_collect_deliverables"]())' in customized
