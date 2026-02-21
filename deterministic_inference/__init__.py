"""Deterministic inference workflow package."""

from __future__ import annotations

from .common import MANIFEST_KIND, ManifestValidationError, canonical_sha256
from .execution import execute_run
from .locking import build_lock_payload, compute_lock_id, load_lock, resolve_lock_path
from .schema import (
    compute_manifest_id,
    compute_requests_digest,
    create_manifest_template,
    load_manifest,
    resolve_inference_requests,
    validate_manifest,
)
from .verification import verify_bundles


def main(argv: list[str] | None = None) -> int:
    from .cli import main as _main

    return _main(argv)


__all__ = [
    "MANIFEST_KIND",
    "ManifestValidationError",
    "build_lock_payload",
    "canonical_sha256",
    "compute_lock_id",
    "compute_manifest_id",
    "compute_requests_digest",
    "create_manifest_template",
    "execute_run",
    "load_lock",
    "load_manifest",
    "main",
    "resolve_inference_requests",
    "resolve_lock_path",
    "validate_manifest",
    "verify_bundles",
]
