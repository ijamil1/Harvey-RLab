# harvey-lab-classic

### Overview
- **Environment ID**: `harvey-lab-classic`
- **Short description**: Non-RLM Harvey LAB environment with provider-native tool calling and one Prime sandbox per rollout.
- **Tags**: legal, tools, sandbox, multi-turn, train, eval

### Datasets
- **Primary dataset**: `irfanjamil/Harvey-LAB`
- **Default split**: `train`
- **Document representation**: source documents are staged into the sandbox as text-backed files at their original paths under `/workspace/documents`.

### Task
- **Type**: multi-turn tool-use environment.
- **Model-facing tools**: `bash`, `read`, `write`, `edit`, `glob`, `grep`.
- **Output expectations**: exact `.docx`/`.xlsx` filenames directly under `/workspace/output`.
- **Rubric overview**: expected deliverables are parsed inside the sandbox with `parse-doc`, then each LAB criterion is judged with the same partial-credit contract used by `harvey_lab_rlm`.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run harvey-lab-classic
```

Configure model and sampling:

```bash
prime eval run harvey-lab-classic -m openai/gpt-4.1-mini -n 1 -r 1
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- `prime eval run` saves results automatically; do not add upload opt-out flags unless that is intentional.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"irfanjamil/Harvey-LAB"` | Hugging Face dataset repo. |
| `split` | str | `"train"` | Dataset split or slice expression. |
| `max_turns` | int | `200` | Maximum model turns. |
| `judge_model` | str | `"deepseek-v4-flash"` | DeepSeek model for criterion judging. |
| `judge_parallelism` | int | `6` | Concurrent criterion judge calls. |
| `sandbox_image` | str | `irfanjamil10/harvey-lab-rlm-sandbox:0.1.0` | Prime sandbox image with Office tooling and `parse-doc`. |

### Secrets

`PRIME_API_KEY` and `DEEPSEEK_API_KEY` may be exported in the shell, or placed in a local `.env` file at the workspace root or `environments/harvey_lab_classic/.env`. Exported environment variables take precedence over `.env` values.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Fraction of LAB criteria judged passing. |
| `lab_criteria_passed` | Passing criteria count. |
| `lab_criteria_total` | Total criteria count. |
| `lab_criterion_pass_rate` | Same scalar as reward before weighting. |
| `lab_missing_deliverables` | Expected deliverables not found exactly under `/workspace/output`. |
| `lab_deliverable_errors` | Expected deliverables that failed sandbox parsing. |
| `lab_judge_calls` | Criterion judge calls made. |
| `lab_tool_calls` | Total exposed tool calls. |
| `lab_sandbox_lifetime_seconds` | Approximate rollout sandbox lifetime. |
