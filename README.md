# Harvey-RLab

Training and evaluation environments adapting [Harvey's Legal Agent Benchmark (LAB)](https://github.com/harveyai/harvey-labs) for [Verifiers](https://github.com/PrimeIntellect-ai/verifiers). Agents perform legal tasks using source documents, produce DOCX/XLSX deliverables, and receive feedback from an LLM judge against task-specific criteria.

## Environments

- **[Harvey LAB Classic](environments/harvey_lab_classic/README.md)** — A multi-turn environment following LAB's original tool-calling approach, with `bash`, `read`, `write`, `edit`, `glob`, and `grep` tools.
- **[Harvey LAB RLM](environments/harvey_lab_rlm/README.md)** — A Recursive Language Model variant with a persistent Python REPL, programmatic access to document context, and delegation to stateless submodels.

Both environments run agent tools in Prime Sandboxes and use the same [Harvey-LAB dataset](https://huggingface.co/datasets/irfanjamil/Harvey-LAB), with separate training and evaluation splits. Their READMEs cover setup, evaluation commands, configuration, scoring, and costs.

The original benchmark tasks, documents, and criteria come from [Harvey LAB](https://github.com/harveyai/harvey-labs).
