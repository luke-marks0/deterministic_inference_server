from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_KIND = "vllm.deterministic_inference_manifest"
LOCK_KIND = "vllm.deterministic_inference_lock"
BUNDLE_KIND = "vllm.deterministic_run_bundle"
VERIFY_REPORT_KIND = "vllm.determinism_verify_report"

_TOKEN_ID_RE = re.compile(r"^token_id:(-?\d+)$")
_SHA256_DIGEST_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_PINNED_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Raised when a manifest violates the schema contract."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _normalize_string(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", value)


def canonicalize(value: Any) -> Any:
    """Apply canonicalization from Appendix A at the data-structure level."""

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(value.keys(), key=lambda raw: _normalize_string(str(raw))):
            if not isinstance(key, str):
                raise ValueError("Canonicalization requires string object keys.")
            normalized[_normalize_string(key)] = canonicalize(value[key])
        return normalized

    if isinstance(value, list):
        return [canonicalize(item) for item in value]

    if isinstance(value, str):
        return _normalize_string(value)

    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float values are not allowed in canonical payloads.")
        return value

    raise ValueError(f"Unsupported type for canonicalization: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    canonical = canonicalize(value)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_real_digest(value: Any, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_path}: expected non-empty sha256 digest string.")

    normalized = value.strip().lower()
    if normalized in {"sha256:unset", "unset"}:
        raise ValueError(f"{field_path}: digest is unset; provide a real sha256 digest.")

    if not _SHA256_DIGEST_RE.fullmatch(normalized):
        raise ValueError(
            f"{field_path}: expected sha256 digest in '<64-hex>' or 'sha256:<64-hex>' format."
        )

    if normalized.startswith("sha256:"):
        return normalized.split(":", 1)[1]
    return normalized


def _normalize_pinned_image_reference(value: Any, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_path}: expected non-empty image reference string.")

    normalized = value.strip()
    lowered = normalized.lower()
    if not _PINNED_IMAGE_RE.fullmatch(lowered):
        raise ValueError(
            f"{field_path}: expected digest-pinned image reference like "
            "'docker.io/repo/name@sha256:<64-hex>'."
        )
    return normalized


def _token_ids_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prompt_token_matrix_hash(prompt_token_ids_list: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(prompt_token_ids_list, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

