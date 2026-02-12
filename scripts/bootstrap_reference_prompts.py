#!/usr/bin/env python3
"""Build and lock reference prompt inputs from Hugging Face data.

This script is intended to be run occasionally (not per sampling run).
It reproduces token-difr prompt construction behavior from:
  - dataset: allenai/WildChat-1M (streaming)
  - tokenizer: Hugging Face model tokenizer
Then it writes a local prompt bundle and a SHA-256 lock file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from profile_config import load_profile


DEFAULT_HF_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_N_PROMPTS = 100
DEFAULT_MAX_CTX_LEN = 512
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_DATASET = "allenai/WildChat-1M"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_SCAN_LIMIT = 10000


def _enable_filelock_compat_shim() -> None:
    """Patch old filelock versions to accept newer kwargs used by HF libs.

    Ubuntu system Python commonly ships filelock<3.10, while newer
    huggingface_hub passes arguments like `mode=` into FileLock. This shim
    drops unsupported kwargs on old filelock versions so dataset streaming
    works without mutating the system environment.
    """
    try:
        import filelock
    except Exception:
        return

    base_cls = getattr(filelock, "BaseFileLock", None)
    if base_cls is None:
        return

    import inspect
    import threading

    try:
        init_sig = inspect.signature(base_cls.__init__)
    except (TypeError, ValueError):
        return

    if "mode" in init_sig.parameters:
        return

    original_init = base_cls.__init__

    def patched_init(self, lock_file, timeout=-1, *args, **kwargs):
        # Drop kwargs introduced in newer filelock APIs.
        kwargs.pop("mode", None)
        kwargs.pop("thread_local", None)
        kwargs.pop("is_singleton", None)
        kwargs.pop("blocking", None)
        kwargs.pop("permissions", None)
        # Old BaseFileLock accepts only (lock_file, timeout).
        original_init(self, lock_file, timeout)
        # Guard against partially-initialized instances in odd code paths.
        if not hasattr(self, "_thread_lock"):
            self._thread_lock = threading.Lock()

    base_cls.__init__ = patched_init  # type: ignore[assignment]

    original_del = getattr(base_cls, "__del__", None)
    if callable(original_del):
        def patched_del(self):
            try:
                original_del(self)
            except AttributeError as exc:
                if "_thread_lock" in str(exc):
                    return
                raise
        base_cls.__del__ = patched_del  # type: ignore[assignment]


def _safe_model_name(hf_model: str) -> str:
    return hf_model.replace("/", "_")


def _default_output_path(hf_model: str) -> Path:
    return Path("artifacts") / "reference_prompts" / f"{_safe_model_name(hf_model)}.json"


def _default_hash_path(hf_model: str) -> Path:
    return Path("manifests") / "reference_prompts" / f"{_safe_model_name(hf_model)}.sha256"


def _write_sha256(path: Path, digest: str, data_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{digest}  {data_path.name}\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_local_snapshot(
    *,
    root_dir: Path,
    hf_model: str,
    hf_cache_rel: str,
    revision: str | None,
) -> Path | None:
    model_cache_dir = (
        root_dir
        / hf_cache_rel
        / "hub"
        / f"models--{hf_model.replace('/', '--')}"
        / "snapshots"
    )
    if not model_cache_dir.is_dir():
        return None

    if revision:
        candidate = model_cache_dir / revision
        if candidate.is_dir():
            return candidate

    snapshots = sorted((p for p in model_cache_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and lock reference prompt inputs from Hugging Face data.",
    )
    parser.add_argument("--config", default="", help="Optional profile JSON path.")
    parser.add_argument("--hf-model", default=None, help="Hugging Face model name.")
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Optional tokenizer identifier/path. Defaults to --hf-model.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional tokenizer/model revision for AutoTokenizer.from_pretrained().",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=DEFAULT_N_PROMPTS,
        help="Number of prompts to construct (default: 100).",
    )
    parser.add_argument(
        "--max-ctx-len",
        type=int,
        default=DEFAULT_MAX_CTX_LEN,
        help="Maximum prompt context length in tokens (default: 512).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt prepended to each conversation.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name (default: allenai/WildChat-1M).",
    )
    parser.add_argument(
        "--dataset-split",
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split (default: train).",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional Hugging Face dataset revision.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=DEFAULT_SCAN_LIMIT,
        help="Maximum streamed dataset rows to scan (default: 10000).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output prompt bundle path. Default: artifacts/reference_prompts/<model>.json",
    )
    parser.add_argument(
        "--hash-output",
        default="",
        help="Hash lock file path. Default: manifests/reference_prompts/<model>.sha256",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output and hash files.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    profile = None
    if args.config:
        profile = load_profile(args.config)
        repo_root = profile.root_dir
        hf_model = args.hf_model or profile.model.model_id
        default_model_revision = profile.model.revision
        default_hf_cache_rel = profile.runtime.paths.hf_cache
    else:
        hf_model = args.hf_model or DEFAULT_HF_MODEL
        default_model_revision = None
        default_hf_cache_rel = "state/hf"
    tokenizer_name = args.tokenizer_name or hf_model
    model_revision = args.model_revision if args.model_revision is not None else default_model_revision

    if args.n_prompts <= 0:
        raise ValueError("--n-prompts must be > 0")
    if args.max_ctx_len <= 0:
        raise ValueError("--max-ctx-len must be > 0")
    if args.scan_limit <= 0:
        raise ValueError("--scan-limit must be > 0")

    output_path = Path(args.output).expanduser() if args.output else _default_output_path(hf_model)
    hash_path = Path(args.hash_output).expanduser() if args.hash_output else _default_hash_path(hf_model)

    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    if not hash_path.is_absolute():
        hash_path = Path.cwd() / hash_path

    if not args.force and (output_path.exists() or hash_path.exists()):
        raise SystemExit(
            "Output already exists. Re-run with --force to overwrite:\n"
            f"  {output_path}\n"
            f"  {hash_path}"
        )

    _enable_filelock_compat_shim()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it before bootstrapping prompts "
            "(for example: pip install datasets)."
        ) from exc
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install it before bootstrapping prompts "
            "(for example: pip install transformers)."
        ) from exc

    local_snapshot = _resolve_local_snapshot(
        root_dir=repo_root,
        hf_model=hf_model,
        hf_cache_rel=default_hf_cache_rel,
        revision=model_revision,
    )

    print(f"Loading tokenizer for {tokenizer_name}")
    if local_snapshot is not None:
        print(f"Using local tokenizer snapshot first: {local_snapshot}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(local_snapshot),
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer_source = str(local_snapshot)
        except Exception as local_exc:  # pragma: no cover
            print(f"Local snapshot tokenizer load failed, trying remote source: {local_exc}")
            try:
                tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
                if model_revision:
                    tokenizer_kwargs["revision"] = model_revision
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
                tokenizer_source = str(tokenizer_name)
            except Exception as remote_exc:  # pragma: no cover
                raise SystemExit(
                    f"Failed to load tokenizer from local snapshot '{local_snapshot}' and remote '{tokenizer_name}'.\n"
                    f"Local error: {local_exc}\n"
                    f"Remote error: {remote_exc}"
                ) from remote_exc
    else:
        try:
            tokenizer_kwargs = {"trust_remote_code": True}
            if model_revision:
                tokenizer_kwargs["revision"] = model_revision
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
            tokenizer_source = str(tokenizer_name)
        except Exception as remote_exc:  # pragma: no cover
            raise SystemExit(
                f"Failed to load tokenizer for {tokenizer_name}: {remote_exc}\n"
                "No local snapshot fallback found. If model files are already downloaded, pass "
                "--tokenizer-name <local-path>."
            ) from remote_exc

    print(
        f"Streaming dataset {args.dataset} split={args.dataset_split} "
        f"scan_limit={args.scan_limit}"
    )
    dataset_kwargs: dict[str, object] = {"split": args.dataset_split, "streaming": True}
    if args.dataset_revision:
        dataset_kwargs["revision"] = args.dataset_revision

    try:
        ds = load_dataset(args.dataset, **dataset_kwargs)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Failed to load dataset {args.dataset}: {exc}") from exc

    conversations: list[list[dict[str, str]]] = []
    prompt_token_ids: list[list[int]] = []
    unique_prompts: set[tuple[int, ...]] = set()

    for sample in islice(ds, args.scan_limit):
        if len(conversations) >= args.n_prompts:
            break

        language = str(sample.get("language", "")).lower()
        if language != "english":
            continue

        raw_conversation = sample.get("conversation")
        if not isinstance(raw_conversation, list):
            continue

        conversation = []
        valid_messages = True
        for msg in raw_conversation:
            if not isinstance(msg, dict):
                valid_messages = False
                break
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                valid_messages = False
                break
            conversation.append({"role": role, "content": content})
        if not valid_messages:
            continue

        while conversation and conversation[-1].get("role") == "assistant":
            conversation = conversation[:-1]

        if not conversation or conversation[-1].get("role") != "user":
            continue

        if any(not msg.get("content") or not msg["content"].strip() for msg in conversation):
            continue

        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)

        if len(token_ids) > args.max_ctx_len:
            continue

        prompt_key = tuple(token_ids)
        if prompt_key in unique_prompts:
            continue
        unique_prompts.add(prompt_key)
        conversations.append(conversation)

    if len(conversations) < args.n_prompts:
        raise SystemExit(
            f"Only constructed {len(conversations)} prompts (requested {args.n_prompts}). "
            "Increase --scan-limit or review dataset constraints."
        )

    system_prompt = args.system_prompt
    if system_prompt:
        conversations = [[{"role": "system", "content": system_prompt}] + conv for conv in conversations]

    for conv in conversations:
        rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompt_token_ids.append([int(tok) for tok in token_ids])

    prompt_bundle = {
        "model": hf_model,
        "parameters": {
            "n_prompts": args.n_prompts,
            "max_ctx_len": args.max_ctx_len,
            "system_prompt": system_prompt,
            "dataset": args.dataset,
            "dataset_split": args.dataset_split,
            "dataset_revision": args.dataset_revision or "",
            "scan_limit": args.scan_limit,
            "model_revision": model_revision or "",
            "tokenizer_name": str(tokenizer_name),
            "tokenizer_source": tokenizer_source,
        },
        "conversations": conversations,
        "prompt_token_ids": prompt_token_ids,
        "source": {
            "downloaded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": "huggingface_dataset_streaming",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prompt_bundle, indent=2) + "\n", encoding="utf-8")

    local_sha256 = _sha256_file(output_path)
    _write_sha256(hash_path, local_sha256, output_path)

    print(f"Wrote prompt bundle: {output_path}")
    print(f"Wrote hash lock:    {hash_path}")
    print(f"Prompts:            {len(prompt_bundle['conversations'])}")
    print(f"Local sha256:       {local_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
