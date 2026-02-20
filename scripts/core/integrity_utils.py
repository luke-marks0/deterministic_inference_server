#!/usr/bin/env python3
"""Shared helpers for snapshot manifests and integrity checks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


MANIFEST_LINE_RE = re.compile(r"^([a-fA-F0-9]{64})  (.+)$")


@dataclass(frozen=True)
class ManifestDiff:
    missing_paths: list[str]
    extra_paths: list[str]
    changed_paths: list[str]

    @property
    def is_match(self) -> bool:
        return not self.missing_paths and not self.extra_paths and not self.changed_paths


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def snapshot_dir(
    *,
    root_dir: Path,
    hf_cache_rel: str,
    model_id: str,
    revision: str,
) -> Path:
    cache_root = Path(hf_cache_rel)
    if not cache_root.is_absolute():
        cache_root = root_dir / cache_root
    return (
        cache_root
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def resolve_manifest_template(
    *,
    template: str,
    root_dir: Path,
    profile_id: str,
    revision: str,
    model_id: str,
) -> Path:
    clean_template = template.strip()
    if not clean_template:
        raise ValueError("Manifest template/path cannot be empty.")

    try:
        rendered = clean_template.format(
            profile_id=profile_id,
            revision=revision,
            model_id=model_id,
            model_id_slug=model_id.replace("/", "--"),
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder '{exc.args[0]}' in manifest path template: {clean_template}"
        ) from exc

    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path


def snapshot_manifest_entries(snapshot_dir_path: Path) -> dict[str, str]:
    files = sorted(path for path in snapshot_dir_path.rglob("*") if path.is_file())
    entries: dict[str, str] = {}
    for path in files:
        rel = path.relative_to(snapshot_dir_path).as_posix()
        entries[f"./{rel}"] = sha256_file(path)
    return entries


def snapshot_manifest_lines(snapshot_dir_path: Path) -> list[str]:
    entries = snapshot_manifest_entries(snapshot_dir_path)
    return [f"{digest}  {rel_path}" for rel_path, digest in sorted(entries.items())]


def load_manifest_entries(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Expected manifest not found: {path}")

    entries: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        match = MANIFEST_LINE_RE.match(stripped)
        if match is None:
            raise ValueError(
                f"Invalid manifest format in {path}:{line_no}. "
                "Expected '<sha256>  ./relative/path'."
            )

        digest = match.group(1).lower()
        rel_path = match.group(2).strip()
        if not rel_path:
            raise ValueError(f"Invalid empty path in manifest {path}:{line_no}")
        if not rel_path.startswith("./"):
            rel_path = f"./{rel_path.lstrip('/')}"
        entries[rel_path] = digest

    return entries


def compare_manifest_entries(
    *,
    expected_entries: dict[str, str],
    actual_entries: dict[str, str],
) -> ManifestDiff:
    expected_paths = set(expected_entries.keys())
    actual_paths = set(actual_entries.keys())
    missing_paths = sorted(expected_paths - actual_paths)
    extra_paths = sorted(actual_paths - expected_paths)
    changed_paths = sorted(
        path for path in (expected_paths & actual_paths) if expected_entries[path] != actual_entries[path]
    )
    return ManifestDiff(
        missing_paths=missing_paths,
        extra_paths=extra_paths,
        changed_paths=changed_paths,
    )


def verify_snapshot_manifest(snapshot_dir_path: Path, expected_manifest_path: Path) -> ManifestDiff:
    expected_entries = load_manifest_entries(expected_manifest_path)
    actual_entries = snapshot_manifest_entries(snapshot_dir_path)
    return compare_manifest_entries(
        expected_entries=expected_entries,
        actual_entries=actual_entries,
    )
