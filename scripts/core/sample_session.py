#!/usr/bin/env python3
"""Generate Fireworks-format reference token bundles from a local OpenAI-compatible server.

This replaces placeholder prompt sampling with the same prompt set used by
inference-provider-leaderboard reference bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from profile_config import load_profile


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SERVED_MODEL = "qwen3-235b-a22b-instruct-2507"
DEFAULT_HF_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_N_PROMPTS = 100
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 42
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_CONCURRENCY = 8
DEFAULT_PROVIDER_LABEL = "fireworks"
DEFAULT_REFERENCE_BUNDLE_REL = Path("artifacts/reference_prompts/reference_prompts.json")
DEFAULT_REFERENCE_HASH_REL = Path("manifests/reference_prompts/reference_prompts.sha256")
_TOKEN_ID_RE = re.compile(r"^token_id:(-?\d+)$")


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_reference_bundle_path() -> Path:
    return _repo_root() / DEFAULT_REFERENCE_BUNDLE_REL


def _default_reference_hash_path() -> Path:
    return _repo_root() / DEFAULT_REFERENCE_HASH_REL


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_expected_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Hash lock file is empty: {path}")
    # Accept either "<sha256>" or "<sha256>  <filename>" formats.
    token = text.split()[0]
    if not re.fullmatch(r"[a-fA-F0-9]{64}", token):
        raise ValueError(f"Hash lock file does not start with sha256 digest: {path}")
    return token.lower()


def _verify_reference_bundle_hash(reference_bundle_path: Path, hash_path: Path) -> None:
    expected = _read_expected_hash(hash_path)
    actual = _sha256_file(reference_bundle_path)
    if actual.lower() != expected:
        raise ValueError(
            "Reference prompt bundle hash mismatch.\n"
            f"  bundle:   {reference_bundle_path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "Re-run scripts/core/bootstrap_reference_prompts.py to refresh prompts and lock."
        )


def _extract_generated_token_ids(response: dict, prompt_len: int) -> list[int]:
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("No choices returned in response.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Invalid response choice format.")

    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        # Fireworks style: logprobs.content[*].token_id
        content = logprobs.get("content")
        if isinstance(content, list) and content:
            token_ids = [row["token_id"] for row in content if isinstance(row, dict) and "token_id" in row]
            if len(token_ids) >= prompt_len:
                return [int(tok_id) for tok_id in token_ids[prompt_len:]]

        # OpenAI/vLLM style with return_tokens_as_token_ids:
        # logprobs.tokens == ["token_id:123", ...]
        tokens = logprobs.get("tokens")
        if isinstance(tokens, list) and tokens:
            maybe_ids: list[int] = []
            for token in tokens:
                if not isinstance(token, str):
                    maybe_ids = []
                    break
                match = _TOKEN_ID_RE.match(token.strip())
                if not match:
                    maybe_ids = []
                    break
                maybe_ids.append(int(match.group(1)))
            if maybe_ids and len(maybe_ids) >= prompt_len:
                return maybe_ids[prompt_len:]

    raise ValueError(
        "Could not extract token IDs from completion response. "
        "Expected Fireworks-style logprobs.content[].token_id or "
        "vLLM tokens with return_tokens_as_token_ids."
    )


def _load_reference_inputs(path: Path) -> tuple[str, list[list[dict[str, str]]], list[list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Reference bundle must be a JSON object: {path}")

    model_name = payload.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Missing or invalid 'model' in reference bundle: {path}")

    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"Missing or invalid 'conversations' in reference bundle: {path}")

    prompt_token_ids_list: list[list[int]] = []
    if isinstance(payload.get("prompt_token_ids"), list):
        for prompt_ids in payload["prompt_token_ids"]:
            if not isinstance(prompt_ids, list) or any(not isinstance(tok, int) for tok in prompt_ids):
                raise ValueError(f"Invalid prompt_token_ids entry in reference bundle: {path}")
            prompt_token_ids_list.append(prompt_ids)
    else:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list):
            raise ValueError(
                f"Missing prompt IDs in reference bundle: expected either "
                f"'prompt_token_ids' or 'sequences' at {path}"
            )
        for seq in sequences:
            if not isinstance(seq, dict):
                raise ValueError(f"Invalid sequence entry in reference bundle: {path}")
            prompt_token_ids = seq.get("prompt_token_ids")
            if not isinstance(prompt_token_ids, list) or any(not isinstance(tok, int) for tok in prompt_token_ids):
                raise ValueError(f"Invalid prompt_token_ids in reference bundle: {path}")
            prompt_token_ids_list.append(prompt_token_ids)

    if len(conversations) != len(prompt_token_ids_list):
        raise ValueError(
            f"Reference bundle has mismatched lengths: "
            f"conversations={len(conversations)} sequences={len(prompt_token_ids_list)}"
        )

    return model_name, conversations, prompt_token_ids_list


def _generate_one(
    *,
    url: str,
    timeout_seconds: int,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> tuple[list[int], int]:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "echo": True,
        "logprobs": 1,
        "top_logprobs": 1,
        "return_tokens_as_token_ids": True,
    }
    response = post_json(url, payload, timeout_seconds)
    output_token_ids = _extract_generated_token_ids(response, prompt_len=len(prompt_token_ids))

    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    completion_count = int(completion_tokens) if isinstance(completion_tokens, int) else len(output_token_ids)
    return output_token_ids, completion_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample reference tokens from a local OpenAI-compatible server using the same "
            "prompt inputs and output schema as inference-provider-leaderboard Fireworks bundles."
        )
    )
    parser.add_argument("--config", default="", help="Optional profile JSON path.")
    parser.add_argument("--base-url", default=None, help="Server base URL. Default: from config or localhost.")
    parser.add_argument(
        "--model",
        default=None,
        help="Served model name for local endpoint. Default: profile.model.served_name or qwen3 slug.",
    )
    parser.add_argument(
        "--hf-model",
        default=None,
        help="Hugging Face model name for output metadata and default reference bundle lookup.",
    )
    parser.add_argument(
        "--reference-bundle",
        default="",
        help=(
            "Path to local prompt bundle JSON. "
            "Default: artifacts/reference_prompts/reference_prompts.json"
        ),
    )
    parser.add_argument(
        "--reference-hash",
        default="",
        help="Path to SHA-256 lock file for the prompt bundle. Default: manifests/reference_prompts/reference_prompts.sha256",
    )
    parser.add_argument(
        "--skip-reference-hash-check",
        action="store_true",
        help="Skip prompt bundle hash verification.",
    )
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--provider-label",
        default=DEFAULT_PROVIDER_LABEL,
        help="Provider label written in output JSON (default: fireworks for compatibility).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. Default: artifacts/samples/reference_<hf_model_safe>_<timestamp>.json",
    )
    args = parser.parse_args()

    if args.config:
        profile = load_profile(args.config)
        default_base_url = f"http://127.0.0.1:{profile.runtime.host_port}"
        default_model = profile.model.served_name
        default_hf_model = profile.model.model_id
        default_timeout_seconds = profile.sample_defaults.timeout_seconds
    else:
        default_base_url = DEFAULT_BASE_URL
        default_model = DEFAULT_SERVED_MODEL
        default_hf_model = DEFAULT_HF_MODEL
        default_timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    base_url = args.base_url or default_base_url
    served_model = args.model or default_model
    hf_model = args.hf_model or default_hf_model
    timeout_seconds = args.timeout_seconds if args.timeout_seconds is not None else default_timeout_seconds
    if timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if args.n_prompts <= 0:
        raise ValueError("--n-prompts must be > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    reference_bundle_path = (
        Path(args.reference_bundle).expanduser().resolve()
        if args.reference_bundle
        else _default_reference_bundle_path()
    )
    reference_hash_path = (
        Path(args.reference_hash).expanduser().resolve()
        if args.reference_hash
        else _default_reference_hash_path()
    )

    if not reference_bundle_path.is_file():
        print(
            f"Reference bundle not found: {reference_bundle_path}\n"
            "Run scripts/core/bootstrap_reference_prompts.py once, or pass --reference-bundle.",
            file=sys.stderr,
        )
        return 1

    if not args.skip_reference_hash_check:
        if not reference_hash_path.is_file():
            print(
                f"Reference hash file not found: {reference_hash_path}\n"
                "Run scripts/core/bootstrap_reference_prompts.py once, or pass --reference-hash, "
                "or use --skip-reference-hash-check.",
                file=sys.stderr,
            )
            return 1
        try:
            _verify_reference_bundle_hash(reference_bundle_path, reference_hash_path)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    source_model_name, conversations, prompt_token_ids_list = _load_reference_inputs(reference_bundle_path)
    if source_model_name != hf_model:
        print(
            f"Warning: --hf-model={hf_model} differs from bundle model={source_model_name}; "
            "using --hf-model for output metadata. This is expected when using a shared "
            "reference prompt bundle across models.",
            file=sys.stderr,
        )

    total_available = min(len(conversations), len(prompt_token_ids_list))
    if args.n_prompts > total_available:
        print(
            f"--n-prompts={args.n_prompts} exceeds available prompts in bundle ({total_available}).",
            file=sys.stderr,
        )
        return 1

    conversations = conversations[: args.n_prompts]
    prompt_token_ids_list = prompt_token_ids_list[: args.n_prompts]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = hf_model.replace("/", "_")
    output_path = args.output or f"artifacts/samples/reference_{safe_model}_{timestamp}.json"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    url = base_url.rstrip("/") + "/v1/completions"
    print(f"Sampling {len(conversations)} prompts from {url} using served model '{served_model}'")
    print(f"Reference prompt source: {reference_bundle_path}")
    print(f"Writing output to {output_path}")

    started_at = time.time()
    sequence_results: list[dict[str, list[int]] | None] = [None] * len(prompt_token_ids_list)
    total_completion_tokens = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_map = {
            pool.submit(
                _generate_one,
                url=url,
                timeout_seconds=timeout_seconds,
                model=served_model,
                prompt_token_ids=prompt_ids,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            ): idx
            for idx, prompt_ids in enumerate(prompt_token_ids_list)
        }

        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                output_token_ids, completion_count = future.result()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                print(f"HTTP error on prompt {idx}: {exc.code}: {detail}", file=sys.stderr)
                return 1
            except urllib.error.URLError as exc:
                print(f"Connection error on prompt {idx}: {exc}", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"Failed to sample prompt {idx}: {exc}", file=sys.stderr)
                return 1

            sequence_results[idx] = {
                "prompt_token_ids": prompt_token_ids_list[idx],
                "output_token_ids": output_token_ids,
            }
            total_completion_tokens += completion_count

            elapsed = time.time() - started_at
            print(
                f"prompt={idx} output_tokens={len(output_token_ids)} "
                f"total_completion_tokens={total_completion_tokens} elapsed_s={elapsed:.1f}"
            )

    if any(seq is None for seq in sequence_results):
        print("Sampling did not produce all expected sequences.", file=sys.stderr)
        return 1
    sequences = [seq for seq in sequence_results if seq is not None]

    payload = {
        "model": hf_model,
        "provider": args.provider_label,
        "parameters": {
            "n_prompts": len(conversations),
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "conversations": conversations,
        "sequences": sequences,
    }

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)

    elapsed = time.time() - started_at
    print(
        "Completed reference sampling: "
        f"prompts={len(conversations)} "
        f"completion_tokens={total_completion_tokens} "
        f"elapsed_s={elapsed:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
