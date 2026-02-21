from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Callable

import deterministic_inference as workflow


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _materialize_local_model_artifacts(path: Path, manifest: dict[str, Any]) -> None:
    model_dir = path.parent / "model-artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)

    file_contents = {
        "model.safetensors": "dummy-weights",
        "tokenizer.json": "{\"tokenizer\":\"dummy\"}",
        "tokenizer_config.json": "{\"config\":\"dummy\"}",
        "special_tokens_map.json": "{\"special\":\"dummy\"}",
        "config.json": "{\"model\":\"dummy\"}",
        "generation_config.json": "{\"generation\":\"dummy\"}",
        "chat_template.jinja": "{{ messages }}",
    }

    digests: dict[str, str] = {}
    for filename, content in file_contents.items():
        file_path = model_dir / filename
        file_path.write_text(content, encoding="utf-8")
        digests[filename] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    manifest["model"]["weights"]["source"]["local_path"] = str(model_dir.resolve())
    manifest["model"]["weights"]["files"][0]["digest"] = digests["model.safetensors"]

    for file_entry in manifest["model"]["tokenizer"]["files"]:
        file_entry["digest"] = digests[str(file_entry["path"])]
    for file_entry in manifest["model"]["config"]["files"]:
        file_entry["digest"] = digests[str(file_entry["path"])]
    manifest["model"]["chat_template"]["file"]["digest"] = digests["chat_template.jinja"]


def _materialize_non_model_digests(manifest: dict[str, Any]) -> None:
    for package in manifest["software_stack"]["python"]["packages"]:
        name = str(package["name"])
        package["source"]["digest"] = _sha256_text(f"python-package:{name}")

    for lib_name, lib_payload in manifest["cuda_stack"]["userspace"].items():
        lib_payload["digest"] = _sha256_text(f"cuda-lib:{lib_name}")


def make_manifest(
    path: Path,
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    manifest = workflow.create_manifest_template(
        model_id="example-org/example-model",
        model_revision="deadbeef",
        output_path=path,
    )

    manifest["runtime"]["execution"]["backend"] = "mock"
    manifest["runtime"]["execution"].pop("base_url", None)
    manifest["artifacts"]["lockfile"] = str((path.parent / "locks" / "manifest.lock.json").resolve())
    manifest["outputs"]["directory"] = str(path.parent / "runs" / "${run_id}")
    manifest["hardware"]["constraints"] = {
        "cpu_arch": platform.machine(),
    }
    manifest["runtime"]["execution"]["deterministic_failure_policy"] = "warn_only"

    _materialize_local_model_artifacts(path, manifest)
    _materialize_non_model_digests(manifest)

    if mutate is not None:
        mutate(manifest)

    workflow.validate_manifest(manifest)
    write_json(path, manifest)
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    return workflow.load_manifest(path)


def lock_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = workflow.load_manifest(manifest_path)
    lock_payload = workflow.build_lock_payload(manifest, manifest_path=manifest_path)
    lock_path = workflow.resolve_lock_path(manifest, manifest_path)
    write_json(lock_path, lock_payload)
    return lock_payload
