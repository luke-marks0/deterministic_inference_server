#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from profile_config import load_profile


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "qwen3-235b-a22b-instruct-2507"
DEFAULT_TARGET_TOKENS = 300000
DEFAULT_CHUNK_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 424242
DEFAULT_TIMEOUT_SECONDS = 600


def post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequentially sample many tokens from a vLLM OpenAI-compatible server."
    )
    parser.add_argument("--config", default="", help="Optional profile JSON path.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--chunk-max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--prompt-template",
        default=(
            "Reference sampling run. Produce coherent English prose. "
            "Chunk index: {chunk_index}. Continue naturally."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path. Default: artifacts/samples/session_<timestamp>.jsonl",
    )
    args = parser.parse_args()

    if args.config:
        profile = load_profile(args.config)
        default_base_url = f"http://127.0.0.1:{profile.runtime.host_port}"
        default_model = profile.model.served_name
        default_target_tokens = profile.sample_defaults.target_tokens
        default_chunk_max_tokens = profile.sample_defaults.chunk_max_tokens
        default_temperature = profile.sample_defaults.temperature
        default_top_p = profile.sample_defaults.top_p
        default_seed = profile.sample_defaults.seed
        default_timeout_seconds = profile.sample_defaults.timeout_seconds
    else:
        default_base_url = DEFAULT_BASE_URL
        default_model = DEFAULT_MODEL
        default_target_tokens = DEFAULT_TARGET_TOKENS
        default_chunk_max_tokens = DEFAULT_CHUNK_MAX_TOKENS
        default_temperature = DEFAULT_TEMPERATURE
        default_top_p = DEFAULT_TOP_P
        default_seed = DEFAULT_SEED
        default_timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    base_url = args.base_url or default_base_url
    model = args.model or default_model
    target_tokens = args.target_tokens if args.target_tokens is not None else default_target_tokens
    chunk_max_tokens = (
        args.chunk_max_tokens if args.chunk_max_tokens is not None else default_chunk_max_tokens
    )
    temperature = args.temperature if args.temperature is not None else default_temperature
    top_p = args.top_p if args.top_p is not None else default_top_p
    seed = args.seed if args.seed is not None else default_seed
    timeout_seconds = args.timeout_seconds if args.timeout_seconds is not None else default_timeout_seconds

    if target_tokens <= 0:
        raise ValueError("--target-tokens must be > 0")
    if chunk_max_tokens <= 0:
        raise ValueError("--chunk-max-tokens must be > 0")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or f"artifacts/samples/session_{timestamp}.jsonl"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    url = base_url.rstrip("/") + "/v1/completions"
    total_completion_tokens = 0
    total_chunks = 0
    started_at = time.time()

    print(f"Writing samples to {output_path}")
    with open(output_path, "w", encoding="utf-8") as out:
        while total_completion_tokens < target_tokens:
            chunk_index = total_chunks
            request_seed = seed + chunk_index
            remaining = target_tokens - total_completion_tokens
            request_max_tokens = min(chunk_max_tokens, remaining)

            payload = {
                "model": model,
                "prompt": args.prompt_template.format(chunk_index=chunk_index),
                "max_tokens": request_max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": request_seed,
            }

            try:
                response = post_json(url, payload, timeout_seconds)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                print(f"HTTP error {exc.code}: {detail}", file=sys.stderr)
                return 1
            except urllib.error.URLError as exc:
                print(f"Connection error: {exc}", file=sys.stderr)
                return 1

            choices = response.get("choices", [])
            if not choices:
                print("No choices returned in response.", file=sys.stderr)
                return 1

            text = choices[0].get("text", "")
            usage = response.get("usage", {})
            completion_tokens = usage.get("completion_tokens")
            if not isinstance(completion_tokens, int) or completion_tokens <= 0:
                print("Missing or invalid usage.completion_tokens.", file=sys.stderr)
                return 1

            record = {
                "chunk_index": chunk_index,
                "seed": request_seed,
                "completion_tokens": completion_tokens,
                "prompt": payload["prompt"],
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=True) + "\n")
            out.flush()

            total_completion_tokens += completion_tokens
            total_chunks += 1

            elapsed = time.time() - started_at
            print(
                f"chunk={chunk_index} tokens={completion_tokens} "
                f"total={total_completion_tokens}/{target_tokens} "
                f"elapsed_s={elapsed:.1f}"
            )

    elapsed = time.time() - started_at
    print(
        "Completed sampling session: "
        f"chunks={total_chunks} "
        f"completion_tokens={total_completion_tokens} "
        f"elapsed_s={elapsed:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
