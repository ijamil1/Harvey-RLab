# harvey-lab-rlm

### Overview
- **Environment ID**: `harvey-lab-rlm`
- **Short description**: Verifiers implementation of [Harvey LAB](https://github.com/harveyai/harvey-labs), an open-source benchmark for evaluating LLM agents' ability to perform legal work in realistic environments, using a Recursive Language Model runtime.
- **RLM runtime**: the root model receives the task instructions and one model-facing tool, `call_python_repl`. That tool drives a persistent Python worker inside a Prime sandbox. The worker exposes LAB-specific helpers and can delegate focused, stateless sub-LLM requests through `llm_batch`.
- **Tags**: legal, rlm, sandbox, multi-turn, train, eval

### Datasets
- **Primary dataset**: [`irfanjamil/Harvey-LAB`](https://huggingface.co/datasets/irfanjamil/Harvey-LAB)
- **Training split**: `train`
- **Evaluation split**: `eval`
- **Dataset selection**: fixed by the environment and not configurable through environment arguments.
- **Document representation**: source documents are staged into the Python namespace as a read-only `documents` dictionary mapping document name to document text, not as user-editable workspace files.

### Environment Specifics
- **Type**: multi-turn RLM environment for legal tasks.
- **Root model-facing tool**: `call_python_repl`.
- **Python namespace**: `instructions`, `documents`, `skills`, `expected_deliverables`, sandbox-local `read`, `write`, `bash`, host-routed `llm_batch`, and the mutable `answer` completion signal.
- **Sub-LLMs**: stateless, tool-free calls made only when root-model code invokes `llm_batch`. They receive the task objective plus the context the root model explicitly includes.
- **Tool execution**: root Python code and sandbox-local helpers execute in a Prime sandbox, with one sandbox per rollout.
- **Sandbox image**: fixed by the environment and not configurable through environment arguments.
- **Output expectations**: exact `.docx`/`.xlsx` filenames directly under `/workspace/output`.
- **Rubric overview**: reward follows the method used in the original Harvey LAB: expected deliverables are parsed inside the sandbox with `parse-doc`, then the fixed `deepseek-v4-flash` judge model evaluates the agent's outputs against the task's fixed set of criteria.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run harvey-lab-rlm
```

Configure model and sampling:

```bash
prime eval run harvey-lab-rlm -m openai/gpt-4.1-mini -n 1 -r 1
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- `prime eval run` saves results automatically; do not add upload opt-out flags unless that is intentional.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `200` | Maximum root-model turns. |
| `sub_model` | str \| null | `null` | Model used by `llm_batch`; when unset, sub-LLM calls reuse the root model. |
| `judge_parallelism` | int | `6` | Concurrent criterion judge calls. |

### Secrets

`PRIME_API_KEY` and `DEEPSEEK_API_KEY` may be exported in the shell, or placed in a `.env` file in the current working directory used to launch Python. Exported environment variables take precedence over `.env` values.

### Sandbox Image

The environment owns the sandbox image and always uses `irfanjamil10/harvey-lab-rlm-sandbox:0.1.0`. The image contains Pandoc, LibreOffice Writer/Calc, document fonts, `parse-doc`, and the Python dependencies required by the bundled DOCX/XLSX skills.

### User Costs

Eval costs come from four sources:

- Prime sandbox usage: one rollout-scoped sandbox is created for each rollout. The RLM sandbox uses 1 CPU core, 2 GB RAM, and 5 GB disk, with a fixed 17 minute sandbox lifetime limit and a 20 minute outer rollout timeout. Sandbox pricing information is available in the [Prime Intellect sandbox docs](https://docs.primeintellect.ai/sandboxes/overview).
- Inference for the root model being evaluated. Lowering `max_turns` reduces the maximum number of root-model turns per rollout, which can limit inference cost and usually shortens rollout duration, reducing sandbox lifetime and sandbox cost.
- Inference for sub-LLM delegation through `llm_batch`. These calls happen only when the root model invokes `llm_batch` from the Python worker. If `sub_model` is unset, the root model is reused; if `sub_model` is set, sub-LLM cost follows that model's pricing and scales with the number of delegated requests.
- Inference for the fixed `deepseek-v4-flash` LLM-as-a-judge reward model. Each rollout is associated with a list of task criteria, and the judge evaluates each criterion individually. A rollout with N criteria therefore uses N judge calls, so judge inference cost scales with rollouts times criteria per rollout.

The cost per rollout is capped by the root-model turn limit, the sandbox lifetime limit, the outer rollout timeout, and the use of a relatively cheap fixed judge model. The environment code does not add any other model-inference calls.

### Reward

A rollout's reward is the fraction of task criteria judged passing. This scalar is the training signal used for RL training.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `lab_criteria_passed` | Passing criteria count. |
| `lab_criteria_total` | Total criteria count. |
| `lab_criterion_pass_rate` | Same scalar as rollout reward. |
| `lab_all_pass` | Whether all non-skipped criteria passed. |
| `lab_missing_deliverables` | Expected deliverables not found exactly under `/workspace/output`. |
| `lab_deliverable_errors` | Expected deliverables that failed sandbox parsing. |
| `lab_judge_calls` | Criterion judge calls made. |
| `lab_rollout_duration_seconds` | Approximate model-generation rollout duration. |
| `lab_sandbox_lifetime_seconds` | Approximate rollout sandbox lifetime. |
