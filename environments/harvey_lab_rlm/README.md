# harvey-lab-rlm

A Verifiers-compatible Recursive Language Model environment for evaluating and
training models on [Harvey's Legal Agent Benchmark](https://huggingface.co/datasets/irfanjamil/Harvey-LAB).

## Environment contract

- The root model sees the LAB instructions and has one model-facing tool:
  `call_python_repl`.
- The persistent Python namespace contains native read-only `documents` and
  `skills` dictionaries, `instructions`, `expected_deliverables`, sandbox-local
  `read`/`write`/`bash`, host-routed `llm_batch`, and the `answer` completion
  signal.
- Sub-LLMs are stateless, receive no tools, and see only the task objective plus
  the context explicitly delegated by the root model.
- Deliverables must be written to `/workspace/output` using exact dataset
  filenames.
- Binary work products are parsed inside the sandbox and deleted with the
  sandbox. Only parsed text and structured errors remain in rollout state.
- Reward is the fraction of task criteria passed by the configured judge.

## Required configuration

Copy `.env.example` to `.env` and set:

```dotenv
PRIME_API_KEY=...
DEEPSEEK_API_KEY=...
```

Exported environment variables take precedence over `.env`.

The environment always uses the fixed image
`docker.io/irfanjamil/harvey-lab-rlm-sandbox:0.1.0`. That image must be
available to Prime's sandbox service. Build and push it from the environment
directory:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t docker.io/irfanjamil/harvey-lab-rlm-sandbox:0.1.0 \
  --push .
```

The image contains Pandoc, LibreOffice Writer/Calc, document fonts, `parse-doc`,
and the Python dependencies required by the bundled DOCX/XLSX skills.

## Tests

```bash
uv run pytest
```

The suite covers dataset validation, prompt contracts, worker-template
compatibility with Verifiers 0.1.14, sandbox-local helpers, exact deliverable
collection, bounded judge concurrency, and partial-credit scoring.

## Local smoke evaluation

Install and run from the Lab workspace root:

```bash
prime env install harvey-lab-rlm
prime eval run harvey-lab-rlm
```

The bare evaluation defaults to one example and one rollout. Unpublished or
locally modified environments save results locally for `prime eval tui`, but
Prime CLI 0.6.14 skips upload to the web Evaluations view.

## Private Hub release and platform-visible evaluation

After local tests and a smoke rollout pass:

```bash
prime env push --path ./environments/harvey_lab_rlm --visibility PRIVATE
prime eval run <owner>/harvey-lab-rlm -n 5 -r 1
```

Republish after every environment change before expecting automatic platform
upload. Prime CLI 0.6.14 also skips automatic upload for TOML-driven eval runs,
so use a direct Hub-slug command for platform acceptance.

## Environment arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `dataset_name` | `irfanjamil/Harvey-LAB` | Hugging Face dataset ID |
| `split` | `train` | Dataset split |
| `max_turns` | `200` | Maximum root-model turns |
| `sub_model` | root model | Model used by `llm_batch` |
| `judge_model` | `deepseek-v4-flash` | Criterion judge model |
| `judge_parallelism` | `6` | Maximum concurrent criterion calls |

## Rollout fields and metrics

Important state fields:

- `deliverables`: exact filename to parsed text.
- `missing_deliverables`: expected files not produced.
- `deliverable_errors`: exact filename to parser error.
- `criterion_results`: ordered judge verdicts and reasoning.

Main metrics:

- `lab_criterion_pass_rate`
- `lab_criteria_passed`
- `lab_criteria_total`
- `lab_all_pass`
- `lab_missing_deliverables`
- `lab_deliverable_errors`
- `lab_judge_calls`
