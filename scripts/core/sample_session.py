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

from profile_config import UNSET_LOCK_VALUES, load_profile


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
DEFAULT_CONCURRENCY = 1
DEFAULT_PROVIDER_LABEL = "fireworks"
DEFAULT_REFERENCE_BUNDLE_REL = Path("artifacts/reference_prompts/reference_prompts.json")
DEFAULT_REFERENCE_HASH_REL = Path("manifests/reference_prompts/reference_prompts.sha256")
DEFAULT_RUN_LOG_DIR_REL = Path("state/evals/logs")
DEFAULT_SNAPSHOT_MANIFEST_TEMPLATE = "manifests/{profile_id}/{revision}.sha256"
TOKENIZER_FILE_BASENAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
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


def _default_run_log_path(hf_model: str, timestamp: str, run_log_dir: str) -> Path:
    safe_name = hf_model.replace("/", "_")
    return (_repo_root() / Path(run_log_dir) / safe_name / f"run_{timestamp}.json").resolve()


def _token_ids_hash(token_ids: list[int]) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(token_ids, separators=(",", ":")).encode("utf-8"))
    return hasher.hexdigest()


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


def _load_reference_inputs(path: Path) -> tuple[str, list[list[dict[str, str]]], bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Reference bundle must be a JSON object: {path}")

    model_name = payload.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Missing or invalid 'model' in reference bundle: {path}")

    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"Missing or invalid 'conversations' in reference bundle: {path}")
    if any(not isinstance(conv, list) for conv in conversations):
        raise ValueError(f"Invalid conversation entry in reference bundle: {path}")

    has_pretokenized_prompts = False
    if isinstance(payload.get("prompt_token_ids"), list):
        has_pretokenized_prompts = True
    else:
        sequences = payload.get("sequences")
        if isinstance(sequences, list):
            has_pretokenized_prompts = any(
                isinstance(seq, dict) and isinstance(seq.get("prompt_token_ids"), list)
                for seq in sequences
            )

    return model_name, conversations, has_pretokenized_prompts


def _resolve_expected_manifest_path(profile) -> Path:
    template = profile.integrity.expected_snapshot_manifest.strip() or DEFAULT_SNAPSHOT_MANIFEST_TEMPLATE
    try:
        rendered = template.format(
            profile_id=profile.profile_id,
            revision=profile.model.revision,
            model_id=profile.model.model_id,
            model_id_slug=profile.model.model_id.replace("/", "--"),
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder '{exc.args[0]}' in manifest path template: {template}"
        ) from exc

    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = profile.root_dir / path
    return path


def _resolve_exact_snapshot_dir(
    *,
    root_dir: Path,
    hf_model: str,
    hf_cache_rel: str,
    revision: str | None,
) -> Path:
    if not revision or revision in UNSET_LOCK_VALUES:
        raise ValueError(
            "Model revision is not pinned; cannot verify tokenizer snapshot. "
            "Run ./scripts/workflow.sh lock-model --config <config> first."
        )

    cache_root = Path(hf_cache_rel)
    if not cache_root.is_absolute():
        cache_root = root_dir / cache_root

    snapshot_dir = (
        cache_root
        / "hub"
        / f"models--{hf_model.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not snapshot_dir.is_dir():
        raise ValueError(
            "Pinned snapshot directory not found for tokenizer verification:\n"
            f"  {snapshot_dir}\n"
            "Start the profile once and wait for model download completion first."
        )
    return snapshot_dir


def _load_manifest_entries(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Pinned snapshot manifest not found: {path}")

    entries: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid manifest format in {path}:{line_no}. "
                "Expected '<sha256>  ./relative/path'."
            )

        digest = parts[0].strip().lower()
        rel_path = parts[1].strip()
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Invalid digest in manifest {path}:{line_no}: {parts[0].strip()}")
        if not rel_path:
            raise ValueError(f"Invalid empty path in manifest {path}:{line_no}")
        if not rel_path.startswith("./"):
            rel_path = f"./{rel_path.lstrip('/')}"
        entries[rel_path] = digest

    if not entries:
        raise ValueError(
            "Pinned snapshot manifest is empty; cannot verify tokenizer pinning:\n"
            f"  {path}\n"
            "Regenerate it with ./scripts/workflow.sh hash --config <config> --output <manifest>."
        )
    return entries


def _is_tokenizer_related_path(rel_path: str) -> bool:
    name = Path(rel_path).name.lower()
    if name in TOKENIZER_FILE_BASENAMES:
        return True
    if "tokenizer" in name:
        return True
    if "chat_template" in name:
        return True
    return False


def _verify_tokenizer_files_against_manifest(
    *,
    profile,
    hf_model: str,
    model_revision: str | None,
) -> tuple[Path, Path, list[str]]:
    snapshot_dir = _resolve_exact_snapshot_dir(
        root_dir=profile.root_dir,
        hf_model=hf_model,
        hf_cache_rel=profile.runtime.paths.hf_cache,
        revision=model_revision,
    )
    manifest_path = _resolve_expected_manifest_path(profile)
    manifest_entries = _load_manifest_entries(manifest_path)

    tokenizer_rel_paths = sorted(
        rel_path
        for rel_path in (
            path.relative_to(snapshot_dir).as_posix()
            for path in snapshot_dir.rglob("*")
            if path.is_file()
        )
        if _is_tokenizer_related_path(rel_path)
    )
    if not tokenizer_rel_paths:
        raise ValueError(
            "No tokenizer files found in pinned snapshot directory:\n"
            f"  {snapshot_dir}"
        )

    for rel_path in tokenizer_rel_paths:
        manifest_key = f"./{rel_path}"
        if manifest_key not in manifest_entries:
            raise ValueError(
                "Tokenizer file missing from pinned snapshot manifest:\n"
                f"  manifest: {manifest_path}\n"
                f"  missing:  {manifest_key}"
            )

        actual = _sha256_file(snapshot_dir / rel_path).lower()
        expected = manifest_entries[manifest_key]
        if actual != expected:
            raise ValueError(
                "Tokenizer file digest mismatch against pinned manifest:\n"
                f"  manifest: {manifest_path}\n"
                f"  file:     {manifest_key}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

    return snapshot_dir, manifest_path, tokenizer_rel_paths


def _resolve_local_snapshot(
    *,
    root_dir: Path,
    hf_model: str,
    hf_cache_rel: str,
    revision: str | None,
) -> Path | None:
    cache_root = Path(hf_cache_rel)
    if not cache_root.is_absolute():
        cache_root = root_dir / cache_root

    model_cache_dir = (
        cache_root
        / "hub"
        / f"models--{hf_model.replace('/', '--')}"
        / "snapshots"
    )
    if not model_cache_dir.is_dir():
        return None

    if revision and revision not in UNSET_LOCK_VALUES:
        candidate = model_cache_dir / revision
        if candidate.is_dir():
            return candidate

    snapshots = sorted(
        (p for p in model_cache_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _load_target_tokenizer(
    *,
    hf_model: str,
    model_revision: str | None,
    root_dir: Path,
    hf_cache_rel: str,
    snapshot_dir: Path | None = None,
):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: transformers. Install project requirements first "
            "(e.g. pip install -r requirements.txt)."
        ) from exc

    local_snapshot = snapshot_dir
    if local_snapshot is None:
        local_snapshot = _resolve_local_snapshot(
            root_dir=root_dir,
            hf_model=hf_model,
            hf_cache_rel=hf_cache_rel,
            revision=model_revision,
        )
    if local_snapshot is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(local_snapshot),
                trust_remote_code=True,
                local_files_only=True,
            )
            return tokenizer, str(local_snapshot)
        except Exception as exc:
            if snapshot_dir is not None:
                raise RuntimeError(
                    "Failed to load tokenizer from verified pinned snapshot:\n"
                    f"  {snapshot_dir}\n"
                    f"error: {exc}"
                ) from exc
            print(
                "Warning: failed to load tokenizer from local snapshot "
                f"{local_snapshot}: {exc}. Falling back to Hugging Face source.",
                file=sys.stderr,
            )

    tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
    if model_revision and model_revision not in UNSET_LOCK_VALUES:
        tokenizer_kwargs["revision"] = model_revision

    tokenizer = AutoTokenizer.from_pretrained(hf_model, **tokenizer_kwargs)
    if "revision" in tokenizer_kwargs:
        source = f"{hf_model}@{tokenizer_kwargs['revision']}"
    else:
        source = hf_model
    return tokenizer, source


def _tokenize_conversations_for_model(
    conversations: list[list[dict[str, str]]],
    *,
    hf_model: str,
    model_revision: str | None,
    root_dir: Path,
    hf_cache_rel: str,
    snapshot_dir: Path | None = None,
) -> tuple[list[list[int]], str]:
    tokenizer, tokenizer_source = _load_target_tokenizer(
        hf_model=hf_model,
        model_revision=model_revision,
        root_dir=root_dir,
        hf_cache_rel=hf_cache_rel,
        snapshot_dir=snapshot_dir,
    )

    prompt_token_ids_list: list[list[int]] = []
    for idx, conversation in enumerate(conversations):
        if not isinstance(conversation, list):
            raise ValueError(f"Invalid conversation format at index {idx}")
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompt_token_ids_list.append([int(tok) for tok in token_ids])

    return prompt_token_ids_list, tokenizer_source


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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--disable-run-log",
        action="store_true",
        help="Disable per-run determinism log writing.",
    )
    parser.add_argument(
        "--run-log-dir",
        default=str(DEFAULT_RUN_LOG_DIR_REL),
        help="Directory used for default run log output.",
    )
    parser.add_argument(
        "--run-log-output",
        default="",
        help="Explicit run log JSON path. Overrides --run-log-dir.",
    )
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

    profile = None
    if args.config:
        profile = load_profile(args.config)
        default_base_url = f"http://127.0.0.1:{profile.runtime.host_port}"
        default_model = profile.model.served_name
        default_hf_model = profile.model.model_id
        default_model_revision = profile.model.revision
        default_hf_cache_rel = profile.runtime.paths.hf_cache
        profile_root_dir = profile.root_dir
        default_temperature = profile.sample_defaults.temperature
        default_top_p = profile.sample_defaults.top_p
        default_seed = profile.sample_defaults.seed
        default_timeout_seconds = profile.sample_defaults.timeout_seconds
    else:
        default_base_url = DEFAULT_BASE_URL
        default_model = DEFAULT_SERVED_MODEL
        default_hf_model = DEFAULT_HF_MODEL
        default_model_revision = None
        default_hf_cache_rel = "state/hf"
        profile_root_dir = _repo_root()
        default_temperature = DEFAULT_TEMPERATURE
        default_top_p = DEFAULT_TOP_P
        default_seed = DEFAULT_SEED
        default_timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    base_url = args.base_url or default_base_url
    served_model = args.model or default_model
    hf_model = args.hf_model or default_hf_model
    temperature = args.temperature if args.temperature is not None else default_temperature
    top_p = args.top_p if args.top_p is not None else default_top_p
    seed = args.seed if args.seed is not None else default_seed
    model_revision = default_model_revision
    timeout_seconds = args.timeout_seconds if args.timeout_seconds is not None else default_timeout_seconds
    if timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if args.n_prompts <= 0:
        raise ValueError("--n-prompts must be > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if temperature < 0.0:
        raise ValueError("--temperature must be >= 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("--top-p must satisfy 0 < top-p <= 1")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if args.concurrency > 1:
        print(
            "Warning: --concurrency > 1 can reduce reproducibility because requests "
            "may contend in the scheduler. Use --concurrency 1 for max determinism.",
            file=sys.stderr,
        )

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
    reference_bundle_sha256 = _sha256_file(reference_bundle_path) if reference_bundle_path.is_file() else ""

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

    source_model_name, conversations, has_pretokenized_prompts = _load_reference_inputs(reference_bundle_path)
    if source_model_name != hf_model:
        print(
            f"Warning: --hf-model={hf_model} differs from bundle model={source_model_name}; "
            "retokenizing prompts with --hf-model tokenizer before sampling.",
            file=sys.stderr,
        )
    if has_pretokenized_prompts:
        print(
            "Info: ignoring pretokenized prompt IDs from reference bundle; "
            "using conversation messages only.",
            file=sys.stderr,
        )

    total_available = len(conversations)
    if args.n_prompts > total_available:
        print(
            f"--n-prompts={args.n_prompts} exceeds available prompts in bundle ({total_available}).",
            file=sys.stderr,
        )
        return 1

    conversations = conversations[: args.n_prompts]
    tokenizer_snapshot_dir: Path | None = None
    if profile is not None:
        if hf_model != profile.model.model_id:
            print(
                "When --config is provided, --hf-model must match config.model.id "
                "so tokenizer pinning can be verified against the snapshot manifest.\n"
                f"  config model: {profile.model.model_id}\n"
                f"  --hf-model:   {hf_model}",
                file=sys.stderr,
            )
            return 1

        try:
            tokenizer_snapshot_dir, tokenizer_manifest_path, tokenizer_manifest_files = (
                _verify_tokenizer_files_against_manifest(
                    profile=profile,
                    hf_model=hf_model,
                    model_revision=model_revision,
                )
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(
            "Pinned tokenizer verification passed: "
            f"manifest={tokenizer_manifest_path} files={len(tokenizer_manifest_files)}"
        )
    else:
        print(
            "Warning: running without --config; tokenizer pinning cannot be verified against a snapshot manifest.",
            file=sys.stderr,
        )

    try:
        prompt_token_ids_list, tokenizer_source = _tokenize_conversations_for_model(
            conversations,
            hf_model=hf_model,
            model_revision=model_revision,
            root_dir=profile_root_dir,
            hf_cache_rel=default_hf_cache_rel,
            snapshot_dir=tokenizer_snapshot_dir,
        )
    except Exception as exc:
        print(f"Failed to tokenize conversations for {hf_model}: {exc}", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = hf_model.replace("/", "_")
    output_path = args.output or f"artifacts/samples/reference_{safe_model}_{timestamp}.json"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_log_path: Path | None
    if args.disable_run_log:
        run_log_path = None
    elif args.run_log_output:
        run_log_path = Path(args.run_log_output).expanduser().resolve()
    else:
        run_log_path = _default_run_log_path(hf_model, timestamp, args.run_log_dir)
    if run_log_path is not None:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)

    url = base_url.rstrip("/") + "/v1/completions"
    print(f"Sampling {len(conversations)} prompts from {url} using served model '{served_model}'")
    print(f"Reference prompt source: {reference_bundle_path}")
    print(f"Tokenizer source: {tokenizer_source}")
    print(f"Writing output to {output_path}")
    if run_log_path is not None:
        print(f"Writing run log to {run_log_path}")

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
                temperature=temperature,
                top_k=args.top_k,
                top_p=top_p,
                seed=seed,
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
            "seed": seed,
            "temperature": temperature,
            "top_k": args.top_k,
            "top_p": top_p,
            "tokenizer_source": tokenizer_source,
            "tokenizer_revision": model_revision if model_revision and model_revision not in UNSET_LOCK_VALUES else "",
            "reference_bundle_has_pretokenized_prompts": has_pretokenized_prompts,
        },
        "conversations": conversations,
        "sequences": sequences,
    }

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)

    elapsed = time.time() - started_at

    if run_log_path is not None:
        records = []
        for idx, seq in enumerate(sequences):
            prompt_ids = seq["prompt_token_ids"]
            output_ids = seq["output_token_ids"]
            records.append(
                {
                    "prompt_index": idx,
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_sha256": _token_ids_hash(prompt_ids),
                    "output_token_ids": output_ids,
                    "output_token_count": len(output_ids),
                    "output_token_sha256": _token_ids_hash(output_ids),
                }
            )

        run_log_payload = {
            "schema_version": 1,
            "run_type": "determinism_log",
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "profile_config": args.config or "",
            "source_reference_bundle": str(reference_bundle_path),
            "source_reference_bundle_sha256": reference_bundle_sha256,
            "source_reference_model": source_model_name,
            "source_reference_has_pretokenized_prompts": has_pretokenized_prompts,
            "tokenizer_source": tokenizer_source,
            "model": hf_model,
            "served_model": served_model,
            "base_url": base_url,
            "parameters": {
                "n_prompts": len(conversations),
                "max_tokens": args.max_tokens,
                "seed": seed,
                "temperature": temperature,
                "top_k": args.top_k,
                "top_p": top_p,
                "concurrency": args.concurrency,
                "timeout_seconds": timeout_seconds,
                "tokenizer_revision": model_revision if model_revision and model_revision not in UNSET_LOCK_VALUES else "",
            },
            "summary": {
                "prompt_count": len(conversations),
                "completion_tokens": total_completion_tokens,
                "elapsed_s": elapsed,
            },
            "records": records,
        }
        run_log_path.write_text(json.dumps(run_log_payload, indent=2), encoding="utf-8")

    print(
        "Completed reference sampling: "
        f"prompts={len(conversations)} "
        f"completion_tokens={total_completion_tokens} "
        f"elapsed_s={elapsed:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
