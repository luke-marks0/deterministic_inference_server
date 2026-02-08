#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


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
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    parser.add_argument("--target-tokens", type=int, default=300000)
    parser.add_argument("--chunk-max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--timeout-seconds", type=int, default=600)
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

    if args.target_tokens <= 0:
        raise ValueError("--target-tokens must be > 0")
    if args.chunk_max_tokens <= 0:
        raise ValueError("--chunk-max-tokens must be > 0")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or f"artifacts/samples/session_{timestamp}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    url = args.base_url.rstrip("/") + "/v1/completions"
    total_completion_tokens = 0
    total_chunks = 0
    started_at = time.time()

    print(f"Writing samples to {output_path}")
    with open(output_path, "w", encoding="utf-8") as out:
        while total_completion_tokens < args.target_tokens:
            chunk_index = total_chunks
            request_seed = args.seed + chunk_index
            remaining = args.target_tokens - total_completion_tokens
            request_max_tokens = min(args.chunk_max_tokens, remaining)

            payload = {
                "model": args.model,
                "prompt": args.prompt_template.format(chunk_index=chunk_index),
                "max_tokens": request_max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": request_seed,
            }

            try:
                response = post_json(url, payload, args.timeout_seconds)
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
                f"total={total_completion_tokens}/{args.target_tokens} "
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
