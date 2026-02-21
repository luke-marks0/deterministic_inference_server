from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from .common import _sha256_file


def _is_unset_digest(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    return normalized in {"", "unset", "sha256:unset"}


def _stable_digest(material: dict[str, Any]) -> str:
    payload = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_local_root(manifest_path: Path, raw_root: str) -> Path:
    root = Path(raw_root)
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    return root


def _local_file_from_url(raw_url: str) -> Path | None:
    parsed = urlparse(raw_url)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path))


def _digest_for_model_file(
    *,
    manifest_path: Path,
    source: dict[str, Any],
    relative_path: str,
    field_path: str,
) -> tuple[str, str]:
    local_root_raw = source.get("local_path")
    if isinstance(local_root_raw, str) and local_root_raw.strip():
        local_root = _resolve_local_root(manifest_path, local_root_raw.strip())
        local_file = (local_root / relative_path).resolve()
        if local_file.is_file():
            return _sha256_file(local_file), "local_file"

    material = {
        "kind": "bootstrap_model_digest",
        "field_path": field_path,
        "repo": source.get("repo", source.get("id", "")),
        "revision": source.get("revision", ""),
        "relative_path": relative_path,
    }
    return _stable_digest(material), "synthetic"


def _digest_for_python_package(
    *,
    manifest_path: Path,
    package: dict[str, Any],
    field_path: str,
) -> tuple[str, str]:
    source = package.get("source", {})
    raw_url = source.get("url")

    if isinstance(raw_url, str) and raw_url.strip():
        maybe_file = _local_file_from_url(raw_url.strip())
        if maybe_file is not None and maybe_file.is_file():
            return _sha256_file(maybe_file), "local_file"

        maybe_path = Path(raw_url.strip())
        if maybe_path.is_absolute() and maybe_path.is_file():
            return _sha256_file(maybe_path), "local_file"
        if not maybe_path.is_absolute():
            resolved = (manifest_path.parent / maybe_path).resolve()
            if resolved.is_file():
                return _sha256_file(resolved), "local_file"

    material = {
        "kind": "bootstrap_python_package_digest",
        "field_path": field_path,
        "name": package.get("name", ""),
        "source": source,
        "version": package.get("version", ""),
    }
    return _stable_digest(material), "synthetic"


def _digest_for_cuda_library(
    *,
    lib_name: str,
    payload: dict[str, Any],
    field_path: str,
) -> tuple[str, str]:
    material = {
        "kind": "bootstrap_cuda_digest",
        "field_path": field_path,
        "lib_name": lib_name,
        "version": payload.get("version", ""),
    }
    return _stable_digest(material), "synthetic"


def _digest_for_remote_code(
    *,
    source: dict[str, Any],
    field_path: str,
) -> tuple[str, str]:
    material = {
        "kind": "bootstrap_remote_code_digest",
        "field_path": field_path,
        "repo": source.get("repo", source.get("id", "")),
        "commit": source.get("remote_code_commit", ""),
    }
    return _stable_digest(material), "synthetic"


def bootstrap_manifest_digests(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    replace_existing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = copy.deepcopy(manifest)
    updates: list[dict[str, Any]] = []
    unchanged_count = 0

    def _set_digest(
        *,
        container: dict[str, Any],
        key: str,
        field_path: str,
        compute_digest: Callable[[], tuple[str, str]],
    ) -> None:
        nonlocal unchanged_count
        current = container.get(key)
        if not replace_existing and not _is_unset_digest(current):
            unchanged_count += 1
            return

        digest, source_kind = compute_digest()
        if current == digest:
            unchanged_count += 1
            return
        container[key] = digest
        updates.append(
            {
                "field_path": field_path,
                "digest": digest,
                "source": source_kind,
            }
        )

    source = payload["model"]["weights"]["source"]

    for idx, file_entry in enumerate(payload["model"]["weights"]["files"]):
        rel_path = str(file_entry["path"])
        field = f"model.weights.files[{idx}].digest"
        _set_digest(
            container=file_entry,
            key="digest",
            field_path=field,
            compute_digest=lambda rel_path=rel_path, field=field: _digest_for_model_file(
                manifest_path=manifest_path,
                source=source,
                relative_path=rel_path,
                field_path=field,
            ),
        )

    for block_name in ("tokenizer", "config"):
        files = payload["model"][block_name]["files"]
        for idx, file_entry in enumerate(files):
            rel_path = str(file_entry["path"])
            field = f"model.{block_name}.files[{idx}].digest"
            _set_digest(
                container=file_entry,
                key="digest",
                field_path=field,
                compute_digest=lambda rel_path=rel_path, field=field: _digest_for_model_file(
                    manifest_path=manifest_path,
                    source=source,
                    relative_path=rel_path,
                    field_path=field,
                ),
            )

    chat_file = payload["model"]["chat_template"]["file"]
    chat_path = str(chat_file["path"])
    _set_digest(
        container=chat_file,
        key="digest",
        field_path="model.chat_template.file.digest",
        compute_digest=lambda: _digest_for_model_file(
            manifest_path=manifest_path,
            source=source,
            relative_path=chat_path,
            field_path="model.chat_template.file.digest",
        ),
    )

    packages = payload["software_stack"]["python"]["packages"]
    for idx, package in enumerate(packages):
        source_payload = package["source"]
        field = f"software_stack.python.packages[{idx}].source.digest"
        _set_digest(
            container=source_payload,
            key="digest",
            field_path=field,
            compute_digest=lambda package=package, field=field: _digest_for_python_package(
                manifest_path=manifest_path,
                package=package,
                field_path=field,
            ),
        )

    for lib_name, lib_payload in payload["cuda_stack"]["userspace"].items():
        field = f"cuda_stack.userspace.{lib_name}.digest"
        _set_digest(
            container=lib_payload,
            key="digest",
            field_path=field,
            compute_digest=lambda lib_name=lib_name, lib_payload=lib_payload, field=field: _digest_for_cuda_library(
                lib_name=lib_name,
                payload=lib_payload,
                field_path=field,
            ),
        )

    trust_remote_code = bool(payload["vllm"]["engine_args"].get("trust_remote_code", False))
    if trust_remote_code:
        _set_digest(
            container=source,
            key="remote_code_digest",
            field_path="model.weights.source.remote_code_digest",
            compute_digest=lambda: _digest_for_remote_code(
                source=source,
                field_path="model.weights.source.remote_code_digest",
            ),
        )

    source_counts = {
        "local_file": sum(1 for update in updates if update["source"] == "local_file"),
        "synthetic": sum(1 for update in updates if update["source"] == "synthetic"),
    }
    report = {
        "updated_count": len(updates),
        "unchanged_count": unchanged_count,
        "source_counts": source_counts,
        "updates": updates,
    }
    return payload, report
