# Deterministic Inference Server

This repository provides a config-driven, OpenAI-compatible `vLLM` serving workflow designed for auditability and reproducibility.

Goal:
- A claimant can say: "using this server, I generated these tokens."
- A verifier can independently reproduce the exact same tokens from the same inputs and locked artifacts.

## Trust Model

Determinism depends on all mutable inputs being pinned and verified:
- model revision pin (`model.revision`)
- runtime image digest pin (`runtime.image` must be `...@sha256:...`)
- local snapshot manifest pin (`manifests/<profile>/<revision>.sha256`)
- reference prompt bundle hash pin (`manifests/reference_prompts/reference_prompts.sha256`)
- deterministic sampling params (`temperature=0`, fixed `seed`, `concurrency=1`)

The workflow is intentionally fail-closed:
- missing snapshot manifest stops `run` (`scripts/atomic/run_profile.sh`)
- sampling refuses weak modes (`scripts/core/sample_session.py`)
- profile loading rejects non-digest images and runtime bootstrap pip installs (`scripts/core/profile_config.py`)

## Repository Layout

- `configs/`: profile configs (model/runtime/vLLM flags).
- `manifests/`: pinned snapshot manifest files (`sha256  ./relative/path`).
- `artifacts/reference_prompts/`: reference prompt bundle.
- `scripts/atomic/run_profile.sh`: strict run orchestration (`start -> wait+verify -> smoke`).
- `scripts/core/serve.py`: start/stop/wait/hash/verify/lock tooling.
- `scripts/core/sample_session.py`: deterministic token sampling.
- `scripts/core/eval_determinism.py`: compare run logs.
- `scripts/workflow.sh`: top-level CLI.

## Prerequisites

- Linux host with Docker + Docker Compose plugin.
- NVIDIA GPUs for target model profiles.
- Python 3.10+.
- Access tokens for gated models if needed (`HUGGING_FACE_HUB_TOKEN` in `.env`).

Install Python deps:

```bash
python3 -m pip install -r requirements.txt
```

## Standard Strict Workflow

Use this for normal operation after manifests are already pinned and committed:

```bash
./scripts/workflow.sh run --config configs/qwen3-235b-a22b-instruct-2507.json --secrets-file .env
```

What it does:
1. starts server
2. waits for readiness and verifies full snapshot against pinned manifest
3. runs smoke test
4. optionally hashes snapshot (`--hash`)
5. optionally stops server (`--stop`)

Useful direct commands:

```bash
./scripts/workflow.sh start  --config <profile.json> --secrets-file .env
./scripts/workflow.sh wait   --config <profile.json> --secrets-file .env --verify-manifest
./scripts/workflow.sh smoke  --config <profile.json> --secrets-file .env
./scripts/workflow.sh hash   --config <profile.json> --output manifests/<profile>/<revision>.sha256
./scripts/workflow.sh verify --config <profile.json>
./scripts/workflow.sh stop   --config <profile.json> --secrets-file .env
```

## One-Time Manifest Bootstrap (Trusted Environment)

`run` will not auto-create trust roots. For a new profile/revision:

1. Pin model revision:

```bash
./scripts/workflow.sh lock-model --config configs/<profile>.json --write
```

2. Pin runtime image digest:

```bash
./scripts/workflow.sh lock-image --config configs/<profile>.json --write
```

3. Start server and wait until model files are downloaded.

```bash
./scripts/workflow.sh start --config configs/<profile>.json --secrets-file .env
```

4. Write manifest from local snapshot:

```bash
./scripts/workflow.sh hash --config configs/<profile>.json --output manifests/<profile>/<revision>.sha256
```

5. Verify:

```bash
./scripts/workflow.sh verify --config configs/<profile>.json
```

6. Commit config + manifest in source control.

## Deterministic Token Sampling

Sampling is strict mode only:
- `--config` is required
- full snapshot manifest verification is required
- prompt hash verification is required
- `--concurrency` must be `1`
- `--skip-reference-hash-check` is rejected

Run sampling:

```bash
./scripts/generate_tokens.sh \
  --config configs/qwen3-235b-a22b-instruct-2507.json \
  --n-prompts 100 \
  --max-tokens 200 \
  --run-log-output state/evals/logs/qwen_run_a.json \
  --output artifacts/samples/qwen_run_a.json
```

## Reproducibility Claim Verification

Given claimed run log + output bundle:

1. checkout same repo commit
2. run the same profile + sampling parameters to produce a fresh run log
3. compare logs

```bash
./scripts/workflow.sh eval-determinism \
  --run-a /path/to/claimed_run_log.json \
  --run-b /path/to/replayed_run_log.json \
  --output state/evals/reports/claim_check.json
```

Default evaluator thresholds are strict exact-match settings.

## Reference Prompt Bootstrap

Reference prompt generation is also strict:
- model/tokenizer revision must be pinned
- dataset revision must be pinned
- tokenizer identity must match target HF model

Example:

```bash
python3 scripts/core/bootstrap_reference_prompts.py \
  --config configs/qwen3-235b-a22b-instruct-2507.json \
  --dataset-revision <dataset_commit_or_tag> \
  --force
```

## Testing

Run test suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
