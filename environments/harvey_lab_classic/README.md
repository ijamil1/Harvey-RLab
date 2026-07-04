# harvey-lab-classic

### Overview
- **Environment ID**: `harvey-lab-classic`
- **Short description**: Verifiers implementation of [Harvey LAB](https://github.com/harveyai/harvey-labs), an open-source benchmark for evaluating LLM agents' ability to perform legal work in realistic environments.
- **Tags**: legal, tools, sandbox, multi-turn, train, eval

### Datasets
- **Primary dataset**: [`irfanjamil/Harvey-LAB`](https://huggingface.co/datasets/irfanjamil/Harvey-LAB)
- **Training split**: `train`
- **Evaluation split**: `eval`
- **Dataset selection**: fixed by the environment and not configurable through environment arguments.
- **Document representation**: source documents are staged into the sandbox as text-backed files at their original paths under `/workspace/documents`.

### Environment Specifics
- **Type**: multi-turn tool-use environment for legal tasks.
- **Model-facing tools**: `bash`, `read`, `write`, `edit`, `glob`, `grep`.
- **Tool execution**: tool calls are executed in a Prime sandbox, with one sandbox per rollout.
- **Sandbox image**: fixed by the environment and not configurable through environment arguments.
- **Output expectations**: exact `.docx`/`.xlsx` filenames directly under `/workspace/output`.
- **Rubric overview**: reward follows the method used in the original Harvey LAB: expected deliverables are parsed inside the sandbox with `parse-doc`, then the fixed `deepseek-v4-flash` judge model evaluates the agent's outputs against the task's fixed set of criteria.

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
| `max_turns` | int | `200` | Maximum model turns. |
| `judge_parallelism` | int | `6` | Concurrent criterion judge calls. |

### Secrets

`PRIME_API_KEY` and `DEEPSEEK_API_KEY` may be exported in the shell, or placed in a `.env` file in the current working directory used to launch Python. Exported environment variables take precedence over `.env` values.

### User Costs

Eval costs come from three sources:

- Prime sandbox usage: one rollout-scoped sandbox is created for each rollout. The sandbox uses 1 CPU core, 2 GB RAM, and 5 GB disk, with a fixed 17 minute lifetime limit. Sandbox pricing information is available in the [Prime Intellect sandbox docs](https://docs.primeintellect.ai/sandboxes/overview).
- Inference for the model being evaluated. Lowering `max_turns` reduces the maximum number of model turns per rollout, which can limit inference cost and usually shortens rollout duration, reducing sandbox lifetime and sandbox cost.
- Inference for the fixed `deepseek-v4-flash` LLM-as-a-judge reward model. Each rollout is associated with a list of task criteria, and the judge evaluates each criterion individually. A rollout with N criteria therefore uses N judge calls, so judge inference cost scales with rollouts times criteria per rollout.

The cost per rollout is capped by the model turn limit, the sandbox lifetime limit, and the use of a relatively cheap fixed judge model. The environment code does not add any other model-inference calls.

### Reward

A rollout's reward is the fraction of task criteria judged passing. This scalar is the training signal used for RL training.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `lab_criteria_passed` | Passing criteria count. |
| `lab_criteria_total` | Total criteria count. |
| `lab_criterion_pass_rate` | Same scalar as rollout reward. |
| `lab_missing_deliverables` | Expected deliverables not found exactly under `/workspace/output`. |
| `lab_deliverable_errors` | Expected deliverables that failed sandbox parsing. |
| `lab_judge_calls` | Criterion judge calls made. |
| `lab_tool_calls` | Total exposed tool calls. |
| `lab_sandbox_lifetime_seconds` | Approximate rollout sandbox lifetime. |
