# Deterministic Inference Server

Deterministic Inference Server is a manifest/lock based workflow for reproducible LLM inference runs.
It provides:

- declarative manifests for inference intent,
- resolved lockfiles for pinned artifacts and runtime closure metadata,
- deterministic run execution with provenance bundles,
- pairwise bundle verification with determinism grading.

The reference behavior and schema live in `spec`.

## Repository Layout

- `deterministic_inference/`: core package (`schema`, `locking`, `execution`, `verification`, CLI)
- `configs/`: model manifests (supports inheritance via `x_base_manifest`)
- `manifests/`: lockfiles and manifest-related artifacts
- `artifacts/`: generated run/bundle outputs
- `tests/`: unit and workflow tests
- `spec`: normative project specification

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`

Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Use the CLI entrypoint:

```bash
python -m deterministic_inference.cli <command> ...
```

Typical workflow:

```bash
# 1) Create a new manifest template
python -m deterministic_inference.cli init \
  --output configs/my-model.json \
  --model-id org/model-name \
  --model-revision <revision>

# 2) (Optional) Bootstrap missing digests and write lock
python -m deterministic_inference.cli digest-bootstrap \
  --config configs/my-model.json \
  --in-place \
  --write-lock

# 3) Resolve lockfile (requires real or bootstrapped sha256 digests in manifest)
python -m deterministic_inference.cli lock --config configs/my-model.json

# 4) Build runtime closure metadata and optionally refresh lock digest
python -m deterministic_inference.cli build --config configs/my-model.json --update-lock

# 5) Start serving stack (OpenAI-compatible endpoint)
python -m deterministic_inference.cli serve \
  --config configs/my-model.json

# 6) Execute run and emit bundle + tokens + run log
python -m deterministic_inference.cli run --config configs/my-model.json

# 7) Compare two runs
python -m deterministic_inference.cli verify \
  --bundle-a runs/<run_a>/bundle.json \
  --bundle-b runs/<run_b>/bundle.json

# 8) Archive a run directory
python -m deterministic_inference.cli bundle \
  --run-dir runs/<run_id> \
  --output artifacts/bundles/<run_id>.tar.gz

# 9) Inspect manifest/lock/bundle metadata
python -m deterministic_inference.cli inspect --input runs/<run_id>/bundle.json
```

## CLI Commands

- `init`: create a new manifest template
- `lock`: resolve and write lockfile with artifact digests/runtime digest
- `build`: emit runtime closure metadata (`--update-lock` optionally rewrites lock)
- `serve`: start a vLLM docker-compose service from manifest settings
- `run`: execute inference and emit run bundle artifacts
- `verify`: compare two bundles and emit report + summary
- `bundle`: tar/gzip a run directory
- `inspect`: summarize manifest/lock/bundle metadata
- `digest-bootstrap`: populate missing digests (`sha256:unset`) for manifest bootstrap

## Manifest and Lock Model

- Manifest (`kind=vllm.deterministic_inference_manifest`):
  - declarative intent for hardware, runtime, model artifacts, inference requests, capture policy, and outputs.
- Lock (`kind=vllm.deterministic_inference_lock`):
  - resolved artifacts with digests, plus runtime closure digest and stable `lock_id`.

Stable IDs:

- `manifest_id = sha256(canonical_manifest)`
- `lock_id = sha256(canonical_lock_without_lock_id)`
- `run_id = sha256(manifest_id + lock_id + requests_digest + hardware_fingerprint_digest)`

## Determinism Behavior

- Pinned batching is enforced (`policy=fixed`, pinned batch size, deterministic ordering).
- Determinism grading:
  - `conformant`
  - `non_conformant_hardware`
  - `non_conformant_software`
  - `mismatch_outputs`
- Token output format is intentionally preserved:
  - `sequences[].prompt_token_ids`
  - `sequences[].output_token_ids`

## Artifact Digest Integrity Verification

`run` verifies artifact digests against the lockfile by default when local artifact paths are available.

- default: enabled
- disable flag: `--no-verify-artifact-digests`

## vLLM Image Selection and Verification

`serve` resolves the image in this order:

1. `--image` CLI override
2. `runtime.execution.vllm_image` in manifest
3. `VLLM_IMAGE` environment variable

Image references must be digest-pinned (`...@sha256:<64-hex>`).
Before starting the service, `serve` verifies the local image digest matches the pinned digest.
Use `--pull` to fetch the pinned image first.

Example:

```bash
python -m deterministic_inference.cli run \
  --config configs/my-model.json \
  --no-verify-artifact-digests
```

## Run Outputs

A run directory contains:

- `bundle.json`: provenance + run metadata + determinism grade
- `manifest.used.json`: exact manifest used
- `lock.used.json`: exact lockfile used
- `tokens.json`: token outputs (stable downstream format)
- `run_log.json`: detailed execution log, batch trace, determinism controls

`verify` outputs:

- `verify_report.json`: machine-readable comparison report
- `verify_summary.txt`: human-readable summary

## Configs in This Repository

The existing model manifests under `configs/` are spec-shaped and loadable.
Some include placeholder digests (for example `sha256:unset`) intended for migration/bootstrap workflows; those must be replaced with real digests before lock/run in strict mode.

### Shared Prompt Dataset

Repository configs now use a shared prompt source instead of hardcoded per-config prompts.

- source path (hardcoded): `/home/ubuntu/deterministic_inference_server/artifacts/reference_prompts/reference_prompts.json`
- prompt count control: `inference.n_prompts` (valid range: `1..100`)
- prompt request shape: `inference.request_template`

The shared prompt dataset is pinned into lockfiles as `inference.prompt_dataset` so `run` digest verification enforces prompt immutability.

## Testing

Run the test suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Roadmap and Status

### Migration Checklist

- [ ] Implement real logits capture + comparison
- [ ] Implement real activations capture + comparison
- [ ] Implement real vLLM trace capture + comparison
- [ ] Enforce `runtime.network_policy=offline_required` (disable outbound network + fail retrieval attempts)

### Current Known Gaps

- `capture.logits`, `capture.activations`, and `capture.engine_trace` are currently configuration-only placeholders and no-op in run/verify.
