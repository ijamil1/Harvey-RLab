# AGENTS.md

This file is a handoff reference for future agents working on
`environments/harvey_lab_rlm`. Read the workspace-root `AGENTS.md` first, then
the parent `environments/AGENTS.md`, then this file.

## Project Summary

`harvey_lab_rlm` is a Verifiers-compatible Recursive Language Model environment
for Harvey's Legal Agent Benchmark. It evaluates root models on legal tasks that
require producing exact `.docx` and `.xlsx` deliverables inside a Prime sandbox.

The environment subclasses Verifiers' experimental `RLMEnv`, but replaces the
generic RLM Python worker namespace with a LAB-specific runtime:

- The root model only sees the model-facing `call_python_repl` tool.
- The persistent sandbox Python namespace exposes `instructions`, read-only
  `documents`, read-only `skills`, `expected_deliverables`, sandbox-local
  `read`, `write`, `bash`, host-routed `llm_batch`, and the mutable `answer`
  completion signal.
- Sub-LLMs are enabled through `llm_batch`, but they are stateless, tool-free,
  and receive only what the root model explicitly includes in each delegated
  prompt.
- Work products must be written under `/workspace/output` with filenames that
  exactly match the dataset's `deliverables` list.
- When `answer["ready"] = True`, the sandbox worker parses the expected output
  files and returns parsed text, missing file names, and parser errors to host
  rollout state.
- Reward is partial credit: the fraction of LAB criteria judged as passing.

## Important Files

- `harvey_lab_rlm/environment.py`: main environment class and
  `load_environment(...)`. This is where sandbox image, timeouts, prompt
  builder, worker customization, sub-LLM wrapping, state setup, cleanup, and
  key validation are wired together.
- `harvey_lab_rlm/dataset.py`: dataset loading and validation. It loads Harvey
  LAB Parquet files through Hugging Face APIs, normalizes JSON metadata for
  `datasets<5`, validates deliverable names, and stores LAB row fields in
  rollout `info`.
- `harvey_lab_rlm/prompts.py`: root and sub-LLM prompt contracts. Keep these
  synchronized with the actual runtime namespace.
- `harvey_lab_rlm/resources.py`: stages per-rollout `.lab/bootstrap.json`,
  copies packaged skill scripts, and creates the sandbox `output` directory.
- `harvey_lab_rlm/sandbox_runtime.py`: code copied into the sandbox and loaded
  by the customized worker. Defines read-only mappings, `read`, `write`,
  `bash`, path protections, exact deliverable collection, and `.docx`/`.xlsx`
  parsing through `parse-doc`.
- `harvey_lab_rlm/worker.py`: string-based patching of Verifiers'
  generated Python worker. This is intentionally guarded by anchor tests
  because upstream `verifiers` template drift can break it.
- `harvey_lab_rlm/rubric.py`: criterion scoring, missing-deliverable short
  circuiting, bounded judge concurrency, metrics, and final reward.
- `harvey_lab_rlm/judge.py`: DeepSeek-backed criterion judge with JSON parsing,
  retries, and a small protocol for test doubles.
- `harvey_lab_rlm/resources/skills/{docx,xlsx}/`: packaged manuals and scripts
  available to the root model inside the sandbox as `skills[...]` text and
  `/workspace/skills/<skill>/scripts/*`.
- `docker/Dockerfile` and `docker/parse_doc.py`: fixed Prime sandbox image
  contents, including LibreOffice, Pandoc, document libraries, and `parse-doc`.
- `tests/`: contract tests. Treat them as executable documentation.

Avoid spending time in generated or local artifacts unless specifically needed:
`.venv/`, `__pycache__/`, `.pytest_cache/`, and `dist/`.

## Runtime Flow

1. `load_environment(...)` loads `.env` if present, requires `PRIME_API_KEY` and
   `DEEPSEEK_API_KEY`, builds a lazy dataset, creates a `DeepSeekCriterionJudge`
   unless a test judge is injected, and returns `HarveyLabRLMEnv`.
2. The dataset builder loads `irfanjamil/Harvey-LAB` by default and stores the
   full LAB row in `info` while making the first user prompt equal to the task
   instructions.
3. `HarveyLabRLMEnv.setup_state(...)` normalizes the LAB row, initializes
   expected deliverable state, stages a temporary filesystem context, and lets
   `RLMEnv` upload that context into the sandbox.
4. The staged context contains `.lab/bootstrap.json`, `.lab/lab_runtime.py`,
   skill scripts, and an empty `output/` directory.
5. `customize_python_worker_script(...)` injects `load_runtime_namespace(...)`
   into Verifiers' generated Python worker, removes the generic `extra_data`
   exposure, deletes `/workspace/.lab` after bootstrapping, and collects
   deliverables only when `answer["ready"]` is true.
6. The root model runs Python through `call_python_repl`. It may write files,
   call skill scripts with `bash`, read generated files, and delegate narrow
   analysis to `llm_batch`.
7. When ready, the root model sets `answer["ready"] = True`. The worker parses
   exact expected output files into text for scoring.
8. `HarveyLabRubric` scores each criterion against only the deliverables it
   names, auto-fails criteria whose required deliverables are missing or
   unparsable, and writes metrics into rollout state.

## Commands

From the workspace root:

```bash
prime env install harvey-lab-rlm
prime eval run harvey-lab-rlm
```

From this environment directory or the workspace root:

```bash
uv run pytest
```

Build and push the fixed sandbox image from this environment directory when
Docker runtime contents change:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t irfanjamil10/harvey-lab-rlm-sandbox:0.1.0 \
  --push .
```

After tests and a local smoke rollout pass, publish with:

```bash
prime env push --path ./environments/harvey_lab_rlm --visibility PRIVATE
```

Per workspace guidance, use `prime eval run` as the canonical eval path and do
not add `--skip-upload` unless the user explicitly requests it.

## Configuration

Copy `.env.example` to `.env` if local shell variables are not already set:

```dotenv
PRIME_API_KEY=...
DEEPSEEK_API_KEY=...
```

Exported environment variables take precedence over `.env`.

Default environment arguments:

- `dataset_name="irfanjamil/Harvey-LAB"`
- `split="train"`
- `max_turns=200`
- `sub_model=None`, which means the root model is reused for sub-LLMs
- `judge_model="deepseek-v4-flash"`
- `judge_parallelism=6`

The sandbox image is fixed in code as
`irfanjamil10/harvey-lab-rlm-sandbox:0.1.0`. Keep code, README, tests, and image
tag expectations in sync if this changes.

## Contracts To Preserve

- Keep `load_environment(...) -> vf.Environment`; this is the Verifiers/Prime
  entrypoint.
- Keep API-key validation explicit in `load_environment()` with
  `vf.ensure_keys(...)`.
- Do not manually initialize or move this environment outside
  `environments/harvey_lab_rlm/`; use Prime CLI lifecycle commands.
- Preserve the root model surface: one model-facing tool,
  `call_python_repl`. `llm_batch` is a root REPL tool, not a model-facing tool.
- Preserve the distinction between prompt context and filesystem/runtime
  context. Source documents are native read-only Python mappings in the sandbox,
  not files under `/workspace/documents`.
- Preserve exact deliverable naming. Case changes, nested paths, symlinks, wrong
  extensions, and approximate filenames should not count.
- Preserve path protections in `sandbox_runtime.py`: helpers must reject paths
  outside `/workspace` and references to RLM control directories.
- Preserve `.lab` bootstrap deletion after runtime load. It prevents task
  bootstrap internals from remaining available during model work.
- Keep sub-LLMs stateless and tool-free. If changing delegation behavior, update
  both prompts and tests.
- Keep worker-template patching guarded. If upgrading `verifiers`, run the
  worker customization tests first; failures usually mean upstream worker
  anchors changed.
- Keep parsed deliverable text and structured errors in rollout state, but strip
  host/sandbox runtime paths during cleanup.
- Treat binary `.docx`/`.xlsx` products as sandbox-local and ephemeral. Scoring
  should use parsed text, not persisted binary artifacts.

## Test Map

- `tests/test_dataset.py`: row validation, JSON-field decoding, prompt/info
  construction, deliverable filename constraints, and criterion references.
- `tests/test_prompts.py`: root namespace documentation, obsolete prompt terms,
  sub-LLM statelessness, and delegated prompt wrapping.
- `tests/test_worker_customization.py`: worker injection behavior and pinned
  Verifiers template compatibility.
- `tests/test_sandbox_runtime.py`: read-only mappings, local helper behavior,
  exact deliverable collection, parser error reporting, and path protections.
- `tests/test_environment_contract.py`: RLM tool exposure, sandbox constants,
  setup/cleanup behavior, sub-LLM wrapping, ready-result state copying, and
  runtime-path stripping.
- `tests/test_rubric.py`: partial credit, per-criterion deliverable scoping,
  bounded judge concurrency, group scoring, and missing-deliverable auto-fail.

For narrow changes, run the matching test file plus any adjacent contract tests.
For changes to prompts, worker customization, runtime namespace, sandbox image,
or scoring, run the full suite with `uv run pytest`.

## Common Pitfalls

- Do not assume `state["completion"]` contains the final deliverable content.
  This environment scores parsed deliverables collected after `answer["ready"]`.
- Do not expose `query_llm`, `query_llm_batch`, `finish`, `extra_data`, or
  `/workspace/documents` in prompts; tests intentionally reject those older
  concepts.
- Do not use broad helper functions in the environment entrypoint. Keep reusable
  logic in focused modules, matching the existing style.
- Do not silently broaden supported deliverable types. Dataset validation,
  parser support, Docker dependencies, prompts, and tests all need to agree.
- Do not make judge infrastructure failures look like model failures. Missing
  deliverables auto-fail criteria, but judge API errors should propagate.
- Do not edit managed root/parent `AGENTS.md` files for project-specific notes.
  This environment-local file is the right place for this handoff.
