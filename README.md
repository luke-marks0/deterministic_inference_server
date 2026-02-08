# Qwen3 235B A22B Instruct 2507: vLLM Reference Server (FP16)

This is a reproducible, audit-friendly reference setup for serving:

- `Qwen/Qwen3-235B-A22B-Instruct-2507`
- `vLLM` OpenAI-compatible server
- `fp16` weights (`--dtype float16`)

The setup prioritizes correctness and reproducibility over throughput.

## What this setup pins

- `vLLM` container image tag: `vllm/vllm-openai:v0.8.5.post1`
- Model repository: `Qwen/Qwen3-235B-A22B-Instruct-2507`
- Model revision: locked to an exact Hugging Face commit SHA via script
- Deterministic-oriented serving defaults:
  - fixed `--seed`
  - `--max-num-seqs 1` (single active sequence)
  - `--enforce-eager`

## Hardware expectation

`Qwen3-235B-A22B-Instruct-2507` in `fp16` is large. Plan for a multi-GPU server (commonly 8x80GB-class GPUs).

## Quick start

1. Copy env template and fill required values.

```bash
cd qwen3-vllm-reference
cp .env.example .env
```

2. Lock model revision to an exact SHA (writes into `.env`).

```bash
./scripts/lock_model_revision.sh
```

3. Start the server.

```bash
./scripts/start_server.sh
```

4. Wait until it is healthy.

```bash
./scripts/wait_ready.sh
```

5. Run a deterministic smoke test.

```bash
./scripts/smoke_test.sh
```

6. Record a file manifest of the downloaded snapshot for audit.

```bash
./scripts/hash_model_snapshot.sh
```

## One-session large token sampling

Generate a large sample sequentially (few hundred thousand tokens) in one run:

```bash
python3 scripts/sample_session.py \
  --target-tokens 300000 \
  --chunk-max-tokens 1024 \
  --temperature 0.7 \
  --top-p 0.95 \
  --seed 424242
```

Outputs are written to `artifacts/samples/*.jsonl`.

## Reproducibility notes

- Keep `docker`, `nvidia-driver`, `cuda`, and GPU hardware consistent across reruns.
- Keep `TENSOR_PARALLEL_SIZE` and `PIPELINE_PARALLEL_SIZE` fixed.
- Do not run concurrent requests during reference sampling.
- Use the same sampling params (`temperature`, `top_p`, `max_tokens`, `seed` sequence).
- Archive:
  - `.env`
  - `docker-compose.yml`
  - snapshot hash file from `artifacts/manifests/`
  - sampled output JSONL from `artifacts/samples/`

## Important caveat

vLLM online serving can still vary across different environments even with fixed seed.
This setup minimizes variance and makes the run auditable, but exact bitwise repeatability requires identical full stack (hardware, drivers, container image, model revision, and runtime settings).
