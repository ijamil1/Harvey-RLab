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
    assert 'lab_namespace = load_runtime_namespace("/workspace/.lab/bootstrap.json")' in script
    assert 'shutil.rmtree("/workspace/.lab")' in script
    assert "namespace.update(lab_namespace)" in script
    assert script.index("load_runtime_namespace") < script.index(
        'shutil.rmtree("/workspace/.lab")'
    )
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
        base_dir="/tmp/rlm_test/rlm_control",
        command_fifo="/tmp/rlm_test/rlm_control/command",
        response_fifo="/tmp/rlm_test/rlm_control/response",
        ready_flag="/tmp/rlm_test/rlm_control/ready",
        worker_path="/tmp/rlm_test/rlm_control/worker.py",
        worker_pid_file="/tmp/rlm_test/rlm_control/pid",
        context_file="/tmp/rlm_test/rlm_control/context.json",
        answer_file="/tmp/rlm_test/rlm_control/answer.json",
        log_file="/tmp/rlm_test/rlm_control/worker.log",
    )
    upstream = _render_worker_script(paths, repl_language="python")

    customized = customize_python_worker_script(upstream)

    assert 'lab_namespace = load_runtime_namespace("/workspace/.lab/bootstrap.json")' in customized
    assert 'shutil.rmtree("/workspace/.lab")' in customized
    assert "namespace.update(lab_namespace)" in customized
    assert 'result.update(namespace["_collect_deliverables"]())' in customized
