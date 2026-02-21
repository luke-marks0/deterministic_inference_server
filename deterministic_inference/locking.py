from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .common import LOCK_KIND, SCHEMA_VERSION, _load_json_object, _normalize_real_digest, _repo_root, _sha256_file, canonical_sha256, utc_now_iso
from .schema import compute_manifest_id, compute_runtime_closure_digest

def resolve_lock_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    lockfile = manifest["artifacts"]["lockfile"]
    if not isinstance(lockfile, str):
        raise ValueError("manifest.artifacts.lockfile is embedded; no lockfile path is available.")
    lock_path = Path(lockfile)
    if not lock_path.is_absolute():
        lockfile_str = str(lockfile)
        if lockfile_str.startswith("./") or lockfile_str.startswith("../"):
            lock_path = (manifest_path.parent / lock_path).resolve()
        else:
            lock_path = (_repo_root() / lock_path).resolve()
    return lock_path


def _resolve_model_file_path(
    manifest_path: Path,
    model_source: dict[str, Any],
    relative_path: str,
) -> Path | None:
    local_root = model_source.get("local_path")
    if not isinstance(local_root, str) or not local_root.strip():
        return None
    base = Path(local_root)
    if not base.is_absolute():
        base = (manifest_path.parent / base).resolve()
    return (base / relative_path).resolve()


def _artifact_entry(
    *,
    name: str,
    source_path: str,
    digest: str,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "path": source_path,
        "digest": digest,
        "retrieval": retrieval,
    }


def _path_from_file_url(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path))


def _artifact_explicit_local_paths(
    artifact: dict[str, Any],
    *,
    manifest_path: Path,
) -> list[Path]:
    raw_artifact_path = artifact.get("path")
    candidates: list[Path] = []

    if isinstance(raw_artifact_path, str) and raw_artifact_path.strip():
        normalized_path = raw_artifact_path.strip()
        file_url_path = _path_from_file_url(normalized_path)
        if file_url_path is not None:
            candidates.append(file_url_path)
        else:
            as_path = Path(normalized_path)
            if as_path.is_absolute():
                candidates.append(as_path)

    retrieval = artifact.get("retrieval")
    if isinstance(retrieval, dict):
        local_root = retrieval.get("local_path")
        if (
            isinstance(local_root, str)
            and local_root.strip()
            and isinstance(raw_artifact_path, str)
            and raw_artifact_path.strip()
        ):
            base = Path(local_root.strip())
            if not base.is_absolute():
                base = (manifest_path.parent / base).resolve()
            candidates.append((base / raw_artifact_path.strip()).resolve())

        local_file = retrieval.get("local_file")
        if isinstance(local_file, str) and local_file.strip():
            file_path = Path(local_file.strip())
            if not file_path.is_absolute():
                file_path = (manifest_path.parent / file_path).resolve()
            candidates.append(file_path)

        retrieval_url = retrieval.get("url")
        if isinstance(retrieval_url, str) and retrieval_url.strip():
            file_url_path = _path_from_file_url(retrieval_url.strip())
            if file_url_path is not None:
                candidates.append(file_url_path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (manifest_path.parent / candidate).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def verify_lock_artifact_integrity(
    *,
    manifest_path: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("lock.artifacts: expected list for integrity verification.")

    checked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            failures.append(
                {
                    "index": index,
                    "name": f"artifact[{index}]",
                    "reason": "invalid_lock_artifact_entry",
                }
            )
            continue

        artifact_name = str(artifact.get("name", f"artifact[{index}]"))
        expected_digest = _normalize_real_digest(
            artifact.get("digest"),
            field_path=f"lock.artifacts[{index}].digest",
        )

        local_paths = _artifact_explicit_local_paths(artifact, manifest_path=manifest_path)
        if not local_paths:
            skipped.append(
                {
                    "index": index,
                    "name": artifact_name,
                    "reason": "no_explicit_local_path",
                }
            )
            continue

        local_file: Path | None = None
        for candidate in local_paths:
            if candidate.is_file():
                local_file = candidate
                break

        if local_file is None:
            failures.append(
                {
                    "index": index,
                    "name": artifact_name,
                    "reason": "missing_local_artifact",
                    "paths": [str(path) for path in local_paths],
                }
            )
            continue

        actual_digest = _sha256_file(local_file)
        if actual_digest != expected_digest:
            failures.append(
                {
                    "index": index,
                    "name": artifact_name,
                    "reason": "digest_mismatch",
                    "path": str(local_file),
                    "expected_digest": expected_digest,
                    "actual_digest": actual_digest,
                }
            )
            continue

        checked.append(
            {
                "index": index,
                "name": artifact_name,
                "path": str(local_file),
                "digest": actual_digest,
            }
        )

    report = {
        "enabled": True,
        "checked_count": len(checked),
        "skipped_count": len(skipped),
        "failure_count": len(failures),
        "checked": checked,
        "skipped": skipped,
    }

    if failures:
        preview = failures[:5]
        raise ValueError(
            "Artifact integrity check failed against lock digests.\n"
            f"failures={preview}"
        )

    return report


def _resolved_model_digest(
    *,
    manifest_path: Path,
    model_source: dict[str, Any],
    relative_path: str,
    declared_digest: Any,
    field_path: str,
) -> str:
    normalized_declared = _normalize_real_digest(declared_digest, field_path=field_path)
    resolved_file = _resolve_model_file_path(manifest_path, model_source, relative_path)
    if resolved_file is None:
        return normalized_declared

    if not resolved_file.is_file():
        raise FileNotFoundError(
            f"{field_path}: local_path is set but artifact file is missing at {resolved_file}."
        )

    actual_digest = _sha256_file(resolved_file)
    if normalized_declared != actual_digest:
        raise ValueError(
            f"{field_path}: digest mismatch for {relative_path}. "
            f"expected {normalized_declared}, actual {actual_digest}."
        )
    return actual_digest


def resolve_artifacts(manifest: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []

    weights = manifest["model"]["weights"]
    model_source = weights["source"]

    for idx, file_entry in enumerate(weights["files"]):
        rel_path = file_entry["path"]
        digest = _resolved_model_digest(
            manifest_path=manifest_path,
            model_source=model_source,
            relative_path=rel_path,
            declared_digest=file_entry.get("digest"),
            field_path=f"model.weights.files[{idx}].digest",
        )

        artifacts.append(
            _artifact_entry(
                name="model.weight",
                source_path=rel_path,
                digest=digest,
                retrieval=model_source,
            )
        )

    for block_name in ("tokenizer", "config"):
        block = manifest["model"][block_name]
        for idx, file_entry in enumerate(block["files"]):
            rel_path = str(file_entry["path"])
            digest = _resolved_model_digest(
                manifest_path=manifest_path,
                model_source=model_source,
                relative_path=rel_path,
                declared_digest=file_entry.get("digest"),
                field_path=f"model.{block_name}.files[{idx}].digest",
            )
            artifacts.append(
                _artifact_entry(
                    name=f"model.{block_name}",
                    source_path=rel_path,
                    digest=digest,
                    retrieval=model_source,
                )
            )

    chat_file = manifest["model"]["chat_template"]["file"]
    chat_path = str(chat_file["path"])
    chat_digest = _resolved_model_digest(
        manifest_path=manifest_path,
        model_source=model_source,
        relative_path=chat_path,
        declared_digest=chat_file.get("digest"),
        field_path="model.chat_template.file.digest",
    )
    artifacts.append(
        _artifact_entry(
            name="model.chat_template",
            source_path=chat_path,
            digest=chat_digest,
            retrieval=model_source,
        )
    )

    for idx, package in enumerate(manifest["software_stack"]["python"]["packages"]):
        source = package["source"]
        digest = _normalize_real_digest(
            source.get("digest"),
            field_path=f"software_stack.python.packages[{idx}].source.digest",
        )
        artifacts.append(
            _artifact_entry(
                name=f"python.package.{package['name']}",
                source_path=str(source.get("url", source.get("rev", ""))),
                digest=digest,
                retrieval=source,
            )
        )

    for lib_name, lib_payload in manifest["cuda_stack"]["userspace"].items():
        digest = _normalize_real_digest(
            lib_payload.get("digest"),
            field_path=f"cuda_stack.userspace.{lib_name}.digest",
        )
        artifacts.append(
            _artifact_entry(
                name=f"cuda.userspace.{lib_name}",
                source_path=lib_name,
                digest=digest,
                retrieval={"version": lib_payload.get("version", "")},
            )
        )

    trust_remote_code = bool(manifest["vllm"]["engine_args"].get("trust_remote_code", False))
    if trust_remote_code:
        remote_commit = str(model_source["remote_code_commit"])
        remote_digest = _normalize_real_digest(
            model_source.get("remote_code_digest"),
            field_path="model.weights.source.remote_code_digest",
        )
        artifacts.append(
            _artifact_entry(
                name="model.remote_code",
                source_path=remote_commit,
                digest=remote_digest,
                retrieval={
                    "type": "remote_code",
                    "repo": model_source.get("repo", model_source.get("id", "")),
                    "commit": remote_commit,
                },
            )
        )

    return artifacts


def compute_lock_id(lock_payload: dict[str, Any]) -> str:
    copy_payload = dict(lock_payload)
    copy_payload.pop("lock_id", None)
    return canonical_sha256(copy_payload)


def build_lock_payload(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    runtime_closure_digest: str | None = None,
) -> dict[str, Any]:
    resolved_artifacts = resolve_artifacts(manifest, manifest_path)
    manifest_id = compute_manifest_id(manifest)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": LOCK_KIND,
        "manifest_id": manifest_id,
        "generated_at": utc_now_iso(),
        "lock_algorithm": "sha256",
        "runtime_closure_digest": runtime_closure_digest or compute_runtime_closure_digest(manifest),
        "artifacts": resolved_artifacts,
        "compiled_extensions": manifest["software_stack"].get("compiled_extensions", []),
    }
    payload["lock_id"] = compute_lock_id(payload)
    return payload


def _validate_lock_payload(lock: dict[str, Any], *, source: str) -> dict[str, Any]:
    if lock.get("kind") != LOCK_KIND:
        raise ValueError(f"Unexpected lock kind in {source}: {lock.get('kind')!r}")
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected lock schema_version in {source}: {lock.get('schema_version')!r}"
        )

    manifest_id = _normalize_real_digest(lock.get("manifest_id"), field_path=f"{source}.manifest_id")
    runtime_digest = _normalize_real_digest(
        lock.get("runtime_closure_digest"),
        field_path=f"{source}.runtime_closure_digest",
    )

    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{source}.artifacts: expected non-empty list.")
    for idx, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"{source}.artifacts[{idx}]: expected object.")
        _normalize_real_digest(
            artifact.get("digest"),
            field_path=f"{source}.artifacts[{idx}].digest",
        )

    expected = compute_lock_id(lock)
    if lock.get("lock_id") != expected:
        raise ValueError(
            f"Lock checksum mismatch in {source}: expected lock_id {expected}, found {lock.get('lock_id')}."
        )

    normalized = dict(lock)
    normalized["manifest_id"] = manifest_id
    normalized["runtime_closure_digest"] = runtime_digest
    return normalized


def load_lock(lock_path: Path) -> dict[str, Any]:
    lock = _load_json_object(lock_path)
    return _validate_lock_payload(lock, source=str(lock_path))


def build_runtime_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "vllm.runtime_build",
        "generated_at": utc_now_iso(),
        "manifest_id": compute_manifest_id(manifest),
        "runtime_closure_digest": compute_runtime_closure_digest(manifest),
        "software_stack": manifest["software_stack"],
        "cuda_stack": manifest["cuda_stack"],
        "runtime": manifest["runtime"],
        "vllm": {
            "mode": manifest["vllm"]["mode"],
            "env": manifest["vllm"]["env"],
            "engine_args": manifest["vllm"]["engine_args"],
        },
    }
