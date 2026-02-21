from __future__ import annotations

import copy
import platform
from pathlib import Path
from typing import Any

from .common import (
    MANIFEST_KIND,
    SCHEMA_VERSION,
    SHARED_PROMPT_DATASET_MAX_PROMPTS,
    SHARED_PROMPT_DATASET_PATH,
    ManifestValidationError,
    _load_json_object,
    _normalize_pinned_image_reference,
    _normalize_real_digest,
    canonical_sha256,
    canonicalize,
    utc_now_iso,
)

def _reject_unknown_keys(
    payload: dict[str, Any],
    allowed: set[str],
    *,
    path: str,
    errors: list[str],
) -> None:
    for key in payload.keys():
        if key in allowed or key.startswith("x_"):
            continue
        errors.append(f"{path}: unknown field '{key}'.")


def _require_keys(
    payload: dict[str, Any],
    required: set[str],
    *,
    path: str,
    errors: list[str],
) -> None:
    for key in sorted(required):
        if key not in payload:
            errors.append(f"{path}: missing required field '{key}'.")


def _expect_type(
    payload: dict[str, Any],
    key: str,
    expected_type: type,
    *,
    path: str,
    errors: list[str],
) -> Any:
    value = payload.get(key)
    if not isinstance(value, expected_type):
        errors.append(f"{path}.{key}: expected {expected_type.__name__}.")
        return None
    return value


def _validate_compare_block(compare: dict[str, Any], errors: list[str]) -> None:
    path = "determinism.compare"
    allowed = {"tokens", "logits", "activations", "engine_trace"}
    _reject_unknown_keys(compare, allowed, path=path, errors=errors)
    _require_keys(compare, allowed, path=path, errors=errors)

    for observable in sorted(allowed):
        value = compare.get(observable)
        if not isinstance(value, dict):
            errors.append(f"{path}.{observable}: expected object.")
            continue
        if "rule" not in value or not isinstance(value.get("rule"), str):
            errors.append(f"{path}.{observable}: missing string field 'rule'.")


def _validate_hardware(hardware: dict[str, Any], errors: list[str]) -> None:
    path = "hardware"
    allowed = {"constraints", "record"}
    _reject_unknown_keys(hardware, allowed, path=path, errors=errors)
    _require_keys(hardware, allowed, path=path, errors=errors)

    constraints = hardware.get("constraints")
    if isinstance(constraints, dict):
        c_allowed = {
            "accelerator_vendor",
            "gpu_arch_min_cc",
            "gpu_arch_max_cc",
            "allowed_gpu_models",
            "cpu_arch",
        }
        _reject_unknown_keys(constraints, c_allowed, path=f"{path}.constraints", errors=errors)
        models = constraints.get("allowed_gpu_models")
        if models is not None and (
            not isinstance(models, list) or any(not isinstance(item, str) for item in models)
        ):
            errors.append(f"{path}.constraints.allowed_gpu_models: expected list[str].")
    else:
        errors.append(f"{path}.constraints: expected object.")

    record = hardware.get("record")
    if isinstance(record, dict):
        r_allowed = {
            "capture_collect_env",
            "capture_nvidia_smi",
            "capture_driver_versions",
            "capture_pcie_topology",
        }
        _reject_unknown_keys(record, r_allowed, path=f"{path}.record", errors=errors)
        r_required = {
            "capture_collect_env",
            "capture_nvidia_smi",
            "capture_driver_versions",
        }
        _require_keys(record, r_required, path=f"{path}.record", errors=errors)
        for key in r_required:
            if key in record and not isinstance(record[key], bool):
                errors.append(f"{path}.record.{key}: expected bool.")
        if "capture_pcie_topology" in record and not isinstance(record["capture_pcie_topology"], bool):
            errors.append(f"{path}.record.capture_pcie_topology: expected bool.")
    else:
        errors.append(f"{path}.record: expected object.")


def _validate_artifacts(artifacts: dict[str, Any], errors: list[str]) -> None:
    path = "artifacts"
    allowed = {"store", "lockfile"}
    _reject_unknown_keys(artifacts, allowed, path=path, errors=errors)
    _require_keys(artifacts, allowed, path=path, errors=errors)

    lockfile = artifacts.get("lockfile")
    if isinstance(lockfile, str):
        if not lockfile.strip():
            errors.append(f"{path}.lockfile: expected non-empty string.")
    elif not isinstance(lockfile, dict):
        errors.append(f"{path}.lockfile: expected non-empty string path or embedded object.")

    store = artifacts.get("store")
    if not isinstance(store, dict):
        errors.append(f"{path}.store: expected object.")
        return

    s_allowed = {"type", "path", "registry"}
    _reject_unknown_keys(store, s_allowed, path=f"{path}.store", errors=errors)
    if not isinstance(store.get("type"), str):
        errors.append(f"{path}.store.type: expected string.")
        return

    store_type = store["type"]
    if store_type not in {"nix-store", "cas", "oci"}:
        errors.append(f"{path}.store.type: must be one of nix-store|cas|oci.")

    if store_type in {"nix-store", "cas"}:
        if not isinstance(store.get("path"), str) or not store["path"].strip():
            errors.append(f"{path}.store.path: required for store type {store_type}.")
    if store_type == "oci":
        if not isinstance(store.get("registry"), str) or not store["registry"].strip():
            errors.append(f"{path}.store.registry: required for store type oci.")


def _validate_software_stack(software: dict[str, Any], errors: list[str]) -> None:
    path = "software_stack"
    allowed = {"python", "nix", "compiled_extensions"}
    _reject_unknown_keys(software, allowed, path=path, errors=errors)

    python_block = software.get("python")
    if not isinstance(python_block, dict):
        errors.append(f"{path}.python: expected object.")
        return

    py_allowed = {"version", "distribution", "packages"}
    _reject_unknown_keys(python_block, py_allowed, path=f"{path}.python", errors=errors)
    _require_keys(python_block, {"version", "packages"}, path=f"{path}.python", errors=errors)

    if "version" in python_block and not isinstance(python_block["version"], str):
        errors.append(f"{path}.python.version: expected string.")

    packages = python_block.get("packages")
    if not isinstance(packages, list) or not packages:
        errors.append(f"{path}.python.packages: expected non-empty list.")
    else:
        for idx, package in enumerate(packages):
            ppath = f"{path}.python.packages[{idx}]"
            if not isinstance(package, dict):
                errors.append(f"{ppath}: expected object.")
                continue
            p_allowed = {"name", "source", "version"}
            _reject_unknown_keys(package, p_allowed, path=ppath, errors=errors)
            _require_keys(package, {"name", "source"}, path=ppath, errors=errors)
            if "name" in package and not isinstance(package["name"], str):
                errors.append(f"{ppath}.name: expected string.")
            source = package.get("source")
            if not isinstance(source, dict):
                errors.append(f"{ppath}.source: expected object.")
                continue
            s_allowed = {"type", "digest", "rev", "url"}
            _reject_unknown_keys(source, s_allowed, path=f"{ppath}.source", errors=errors)
            if "type" not in source or not isinstance(source.get("type"), str):
                errors.append(f"{ppath}.source.type: expected string.")
            digest = source.get("digest")
            if not isinstance(digest, str) or not digest.strip():
                errors.append(f"{ppath}.source.digest: expected non-empty digest string.")

    compiled_extensions = software.get("compiled_extensions")
    if compiled_extensions is not None and (
        not isinstance(compiled_extensions, list)
        or any(not isinstance(item, dict) for item in compiled_extensions)
    ):
        errors.append(f"{path}.compiled_extensions: expected list[object] when provided.")


def _validate_cuda_stack(cuda_stack: dict[str, Any], errors: list[str]) -> None:
    path = "cuda_stack"
    allowed = {"userspace", "env", "driver_policy"}
    _reject_unknown_keys(cuda_stack, allowed, path=path, errors=errors)
    _require_keys(cuda_stack, {"userspace", "env"}, path=path, errors=errors)

    userspace = cuda_stack.get("userspace")
    if not isinstance(userspace, dict) or not userspace:
        errors.append(f"{path}.userspace: expected non-empty object.")
    else:
        for lib_name, lib_payload in userspace.items():
            lpath = f"{path}.userspace.{lib_name}"
            if not isinstance(lib_payload, dict):
                errors.append(f"{lpath}: expected object.")
                continue
            allowed_keys = {"version", "digest"}
            _reject_unknown_keys(lib_payload, allowed_keys, path=lpath, errors=errors)
            if "digest" not in lib_payload or not isinstance(lib_payload.get("digest"), str):
                errors.append(f"{lpath}.digest: expected string.")

    env = cuda_stack.get("env")
    if not isinstance(env, dict):
        errors.append(f"{path}.env: expected object.")
    elif any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        errors.append(f"{path}.env: expected map[string]string.")


def _validate_runtime(runtime: dict[str, Any], errors: list[str]) -> None:
    path = "runtime"
    allowed = {"env", "locale", "threads", "network_policy", "execution"}
    _reject_unknown_keys(runtime, allowed, path=path, errors=errors)
    _require_keys(runtime, {"env", "locale", "threads", "execution"}, path=path, errors=errors)

    env = runtime.get("env")
    if not isinstance(env, dict):
        errors.append(f"{path}.env: expected object.")
    else:
        if "PYTHONHASHSEED" not in env:
            errors.append(f"{path}.env: missing PYTHONHASHSEED.")
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                errors.append(f"{path}.env: expected map[string]string.")
                break

    locale = runtime.get("locale")
    if not isinstance(locale, dict):
        errors.append(f"{path}.locale: expected object.")
    else:
        _require_keys(locale, {"lang", "tz"}, path=f"{path}.locale", errors=errors)
        if "lang" in locale and not isinstance(locale["lang"], str):
            errors.append(f"{path}.locale.lang: expected string.")
        if "tz" in locale and not isinstance(locale["tz"], str):
            errors.append(f"{path}.locale.tz: expected string.")

    threads = runtime.get("threads")
    if not isinstance(threads, dict):
        errors.append(f"{path}.threads: expected object.")
    else:
        _require_keys(threads, {"omp_num_threads", "mkl_num_threads"}, path=f"{path}.threads", errors=errors)
        for key in ("omp_num_threads", "mkl_num_threads"):
            if key in threads and (not isinstance(threads[key], int) or threads[key] <= 0):
                errors.append(f"{path}.threads.{key}: expected int > 0.")

    execution = runtime.get("execution")
    if not isinstance(execution, dict):
        errors.append(f"{path}.execution: expected object.")
    else:
        e_allowed = {
            "backend",
            "base_url",
            "timeout_seconds",
            "deterministic_failure_policy",
            "vllm_image",
        }
        _reject_unknown_keys(execution, e_allowed, path=f"{path}.execution", errors=errors)
        _require_keys(
            execution,
            {"backend", "timeout_seconds", "deterministic_failure_policy"},
            path=f"{path}.execution",
            errors=errors,
        )
        backend = execution.get("backend")
        if not isinstance(backend, str) or backend not in {"mock", "openai_compatible"}:
            errors.append(f"{path}.execution.backend: must be 'mock' or 'openai_compatible'.")
        timeout_seconds = execution.get("timeout_seconds")
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            errors.append(f"{path}.execution.timeout_seconds: expected int > 0.")
        policy = execution.get("deterministic_failure_policy")
        if policy not in {"error", "warn_only", "off"}:
            errors.append(
                f"{path}.execution.deterministic_failure_policy: must be one of error|warn_only|off."
            )
        vllm_image = execution.get("vllm_image")
        if vllm_image is not None:
            try:
                _normalize_pinned_image_reference(
                    vllm_image,
                    field_path=f"{path}.execution.vllm_image",
                )
            except ValueError as exc:
                errors.append(str(exc))
        if backend == "openai_compatible" and (
            not isinstance(execution.get("base_url"), str) or not execution["base_url"].strip()
        ):
            errors.append(f"{path}.execution.base_url: required for openai_compatible backend.")


def _validate_vllm(vllm: dict[str, Any], errors: list[str]) -> None:
    path = "vllm"
    allowed = {"mode", "reproducibility", "env", "engine_args"}
    _reject_unknown_keys(vllm, allowed, path=path, errors=errors)
    _require_keys(vllm, allowed, path=path, errors=errors)

    if "mode" in vllm and vllm["mode"] not in {"offline", "server"}:
        errors.append(f"{path}.mode: must be 'offline' or 'server'.")

    for field in ("reproducibility", "env", "engine_args"):
        if field in vllm and not isinstance(vllm[field], dict):
            errors.append(f"{path}.{field}: expected object.")

    engine_args = vllm.get("engine_args")
    if isinstance(engine_args, dict):
        trust_remote_code = engine_args.get("trust_remote_code")
        if trust_remote_code is not None and not isinstance(trust_remote_code, bool):
            errors.append(f"{path}.engine_args.trust_remote_code: expected bool when provided.")


def _validate_model(model: dict[str, Any], errors: list[str]) -> None:
    path = "model"
    allowed = {"weights", "tokenizer", "config", "chat_template", "adapters"}
    _reject_unknown_keys(model, allowed, path=path, errors=errors)
    _require_keys(model, {"weights", "tokenizer", "config", "chat_template"}, path=path, errors=errors)

    def _validate_files_block(block_name: str, block_payload: dict[str, Any]) -> None:
        bpath = f"{path}.{block_name}"
        if block_name == "weights":
            _reject_unknown_keys(block_payload, {"source", "files"}, path=bpath, errors=errors)
            _require_keys(block_payload, {"source", "files"}, path=bpath, errors=errors)
            source = block_payload.get("source")
            if not isinstance(source, dict):
                errors.append(f"{bpath}.source: expected object.")
            else:
                source_allowed = {
                    "type",
                    "repo",
                    "id",
                    "revision",
                    "local_path",
                    "remote_code_commit",
                    "remote_code_digest",
                }
                _reject_unknown_keys(source, source_allowed, path=f"{bpath}.source", errors=errors)
                if "type" in source and not isinstance(source["type"], str):
                    errors.append(f"{bpath}.source.type: expected string.")
                if "repo" in source and not isinstance(source["repo"], str):
                    errors.append(f"{bpath}.source.repo: expected string.")
                if "id" in source and not isinstance(source["id"], str):
                    errors.append(f"{bpath}.source.id: expected string.")
                if "revision" in source and not isinstance(source["revision"], str):
                    errors.append(f"{bpath}.source.revision: expected string.")
                if "local_path" in source and not isinstance(source["local_path"], str):
                    errors.append(f"{bpath}.source.local_path: expected string.")
                if "remote_code_commit" in source and not isinstance(source["remote_code_commit"], str):
                    errors.append(f"{bpath}.source.remote_code_commit: expected string.")
                if "remote_code_digest" in source and not isinstance(source["remote_code_digest"], str):
                    errors.append(f"{bpath}.source.remote_code_digest: expected string.")
        else:
            _reject_unknown_keys(block_payload, {"files"}, path=bpath, errors=errors)
            _require_keys(block_payload, {"files"}, path=bpath, errors=errors)

        files = block_payload.get("files")
        if not isinstance(files, list) or not files:
            errors.append(f"{bpath}.files: expected non-empty list.")
            return
        for idx, file_entry in enumerate(files):
            fpath = f"{bpath}.files[{idx}]"
            if not isinstance(file_entry, dict):
                errors.append(f"{fpath}: expected object.")
                continue
            _reject_unknown_keys(file_entry, {"path", "digest"}, path=fpath, errors=errors)
            _require_keys(file_entry, {"path", "digest"}, path=fpath, errors=errors)
            if "path" in file_entry and not isinstance(file_entry["path"], str):
                errors.append(f"{fpath}.path: expected string.")
            if "digest" in file_entry and not isinstance(file_entry["digest"], str):
                errors.append(f"{fpath}.digest: expected string.")

    for block_name in ("weights", "tokenizer", "config"):
        block_payload = model.get(block_name)
        if isinstance(block_payload, dict):
            _validate_files_block(block_name, block_payload)
        else:
            errors.append(f"{path}.{block_name}: expected object.")

    chat_template = model.get("chat_template")
    if isinstance(chat_template, dict):
        _reject_unknown_keys(chat_template, {"file"}, path=f"{path}.chat_template", errors=errors)
        _require_keys(chat_template, {"file"}, path=f"{path}.chat_template", errors=errors)
        file_payload = chat_template.get("file")
        if not isinstance(file_payload, dict):
            errors.append(f"{path}.chat_template.file: expected object.")
        else:
            _reject_unknown_keys(file_payload, {"path", "digest"}, path=f"{path}.chat_template.file", errors=errors)
            _require_keys(file_payload, {"path", "digest"}, path=f"{path}.chat_template.file", errors=errors)
            if "path" in file_payload and not isinstance(file_payload["path"], str):
                errors.append(f"{path}.chat_template.file.path: expected string.")
            if "digest" in file_payload and not isinstance(file_payload["digest"], str):
                errors.append(f"{path}.chat_template.file.digest: expected string.")
    else:
        errors.append(f"{path}.chat_template: expected object.")


def _validate_request_sampling(
    sampling: Any,
    *,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(sampling, dict):
        errors.append(f"{path}.sampling: expected object.")
        return

    s_allowed = {"temperature", "top_p", "top_k", "max_tokens", "seed"}
    _reject_unknown_keys(sampling, s_allowed, path=f"{path}.sampling", errors=errors)
    _require_keys(sampling, s_allowed, path=f"{path}.sampling", errors=errors)
    if "temperature" in sampling and not isinstance(sampling["temperature"], (int, float)):
        errors.append(f"{path}.sampling.temperature: expected number.")
    if "top_p" in sampling and not isinstance(sampling["top_p"], (int, float)):
        errors.append(f"{path}.sampling.top_p: expected number.")
    top_k = sampling.get("top_k")
    if not (top_k is None or isinstance(top_k, int)):
        errors.append(f"{path}.sampling.top_k: expected int|null.")
    if "max_tokens" in sampling and (
        not isinstance(sampling["max_tokens"], int) or sampling["max_tokens"] <= 0
    ):
        errors.append(f"{path}.sampling.max_tokens: expected int > 0.")
    if "seed" in sampling and not isinstance(sampling["seed"], int):
        errors.append(f"{path}.sampling.seed: expected int.")


def _validate_request_stop(
    stop: Any,
    *,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(stop, dict):
        errors.append(f"{path}.stop: expected object.")
        return

    stop_allowed = {"sequences", "token_ids"}
    _reject_unknown_keys(stop, stop_allowed, path=f"{path}.stop", errors=errors)
    _require_keys(stop, stop_allowed, path=f"{path}.stop", errors=errors)
    sequences = stop.get("sequences")
    token_ids = stop.get("token_ids")
    if not isinstance(sequences, list) or any(not isinstance(item, str) for item in sequences):
        errors.append(f"{path}.stop.sequences: expected list[str].")
    if not isinstance(token_ids, list) or any(not isinstance(item, int) for item in token_ids):
        errors.append(f"{path}.stop.token_ids: expected list[int].")


def _validate_inference(inference: dict[str, Any], errors: list[str]) -> None:
    path = "inference"
    allowed = {"requests", "batching", "n_prompts", "request_template"}
    _reject_unknown_keys(inference, allowed, path=path, errors=errors)
    _require_keys(inference, {"batching"}, path=path, errors=errors)

    requests = inference.get("requests")
    request_template = inference.get("request_template")
    n_prompts = inference.get("n_prompts")

    uses_explicit_requests = isinstance(requests, list)
    uses_shared_prompt_dataset = requests is None and (
        "n_prompts" in inference or "request_template" in inference
    )

    if uses_explicit_requests and uses_shared_prompt_dataset:
        errors.append(
            f"{path}: choose either explicit requests or n_prompts+request_template, not both."
        )
    elif uses_explicit_requests:
        if not requests:
            errors.append(f"{path}.requests: expected non-empty list.")
        else:
            request_ids: set[str] = set()
            for idx, request in enumerate(requests):
                rpath = f"{path}.requests[{idx}]"
                if not isinstance(request, dict):
                    errors.append(f"{rpath}: expected object.")
                    continue
                allowed_keys = {
                    "id",
                    "kind",
                    "prompt",
                    "messages",
                    "prompt_token_ids",
                    "sampling",
                    "stop",
                }
                _reject_unknown_keys(request, allowed_keys, path=rpath, errors=errors)
                _require_keys(request, {"id", "kind", "sampling", "stop"}, path=rpath, errors=errors)

                req_id = request.get("id")
                if isinstance(req_id, str):
                    if req_id in request_ids:
                        errors.append(f"{rpath}.id: duplicate request id '{req_id}'.")
                    request_ids.add(req_id)
                else:
                    errors.append(f"{rpath}.id: expected string.")

                if "kind" in request and request["kind"] not in {"completion", "chat"}:
                    errors.append(f"{rpath}.kind: must be 'completion' or 'chat'.")

                has_prompt = isinstance(request.get("prompt"), str)
                has_messages = isinstance(request.get("messages"), list)
                has_prompt_ids = isinstance(request.get("prompt_token_ids"), list)
                if sum([has_prompt, has_messages, has_prompt_ids]) == 0:
                    errors.append(
                        f"{rpath}: one of prompt|messages|prompt_token_ids is required."
                    )

                if has_messages:
                    messages = request["messages"]
                    if any(
                        not isinstance(msg, dict)
                        or not isinstance(msg.get("role"), str)
                        or not isinstance(msg.get("content"), str)
                        for msg in messages
                    ):
                        errors.append(f"{rpath}.messages: expected list of role/content objects.")

                if has_prompt_ids:
                    prompt_ids = request["prompt_token_ids"]
                    if any(not isinstance(tok, int) for tok in prompt_ids):
                        errors.append(f"{rpath}.prompt_token_ids: expected list[int].")

                _validate_request_sampling(request.get("sampling"), path=rpath, errors=errors)
                _validate_request_stop(request.get("stop"), path=rpath, errors=errors)
    else:
        if not isinstance(n_prompts, int):
            errors.append(f"{path}.n_prompts: required int when requests are not provided.")
        elif n_prompts < 1 or n_prompts > SHARED_PROMPT_DATASET_MAX_PROMPTS:
            errors.append(
                f"{path}.n_prompts: expected int in range 1..{SHARED_PROMPT_DATASET_MAX_PROMPTS}."
            )

        tpath = f"{path}.request_template"
        if not isinstance(request_template, dict):
            errors.append(f"{tpath}: required object when requests are not provided.")
        else:
            t_allowed = {"kind", "sampling", "stop", "id_prefix"}
            _reject_unknown_keys(request_template, t_allowed, path=tpath, errors=errors)
            _require_keys(request_template, {"kind", "sampling", "stop"}, path=tpath, errors=errors)
            kind = request_template.get("kind")
            if kind not in {"completion", "chat"}:
                errors.append(f"{tpath}.kind: must be 'completion' or 'chat'.")
            id_prefix = request_template.get("id_prefix")
            if id_prefix is not None and (not isinstance(id_prefix, str) or not id_prefix.strip()):
                errors.append(f"{tpath}.id_prefix: expected non-empty string when provided.")

            _validate_request_sampling(request_template.get("sampling"), path=tpath, errors=errors)
            _validate_request_stop(request_template.get("stop"), path=tpath, errors=errors)

    batching = inference.get("batching")
    if not isinstance(batching, dict):
        errors.append(f"{path}.batching: expected object.")
    else:
        b_allowed = {
            "policy",
            "schedule",
            "max_num_seqs",
            "max_num_batched_tokens",
            "fixed_request_order",
            "concurrency",
        }
        _reject_unknown_keys(batching, b_allowed, path=f"{path}.batching", errors=errors)
        _require_keys(batching, b_allowed, path=f"{path}.batching", errors=errors)

        if batching.get("policy") != "fixed":
            errors.append(f"{path}.batching.policy: only 'fixed' is currently supported.")

        schedule = batching.get("schedule")
        if schedule not in {"offline_deterministic", "batch_invariant", "replay"}:
            errors.append(
                f"{path}.batching.schedule: must be one of offline_deterministic|batch_invariant|replay."
            )

        for key in ("max_num_seqs", "max_num_batched_tokens", "concurrency"):
            value = batching.get(key)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{path}.batching.{key}: expected int > 0.")

        if "fixed_request_order" in batching and not isinstance(batching["fixed_request_order"], bool):
            errors.append(f"{path}.batching.fixed_request_order: expected bool.")

        if schedule == "replay":
            if not isinstance(requests, list):
                errors.append(
                    f"{path}.batching.schedule: replay requires explicit requests with x_arrival_ms."
                )
            else:
                for idx, request in enumerate(requests):
                    arrival_ms = request.get("x_arrival_ms") if isinstance(request, dict) else None
                    if not isinstance(arrival_ms, int) or arrival_ms < 0:
                        errors.append(
                            f"{path}.requests[{idx}].x_arrival_ms: required int >= 0 when batching.schedule='replay'."
                        )


def _validate_capture(capture: dict[str, Any], errors: list[str]) -> None:
    path = "capture"
    allowed = {"tokens", "logits", "activations", "engine_trace"}
    _reject_unknown_keys(capture, allowed, path=path, errors=errors)
    _require_keys(capture, allowed, path=path, errors=errors)

    if capture.get("tokens") is not True:
        errors.append("capture.tokens: must be true (tokens are always captured).")

    logits = capture.get("logits")
    if isinstance(logits, dict):
        _reject_unknown_keys(
            logits,
            {"enabled", "scope", "dtype", "capture_prefill", "capture_decode"},
            path="capture.logits",
            errors=errors,
        )
        _require_keys(logits, {"enabled", "scope", "dtype", "capture_prefill", "capture_decode"}, path="capture.logits", errors=errors)
    else:
        errors.append("capture.logits: expected object.")

    activations = capture.get("activations")
    if isinstance(activations, dict):
        _reject_unknown_keys(activations, {"enabled", "hooks"}, path="capture.activations", errors=errors)
        _require_keys(activations, {"enabled", "hooks"}, path="capture.activations", errors=errors)
        hooks = activations.get("hooks")
        if not isinstance(hooks, list):
            errors.append("capture.activations.hooks: expected list.")
    else:
        errors.append("capture.activations: expected object.")

    engine_trace = capture.get("engine_trace")
    if isinstance(engine_trace, dict):
        _reject_unknown_keys(engine_trace, {"enabled", "events"}, path="capture.engine_trace", errors=errors)
        _require_keys(engine_trace, {"enabled", "events"}, path="capture.engine_trace", errors=errors)
        events = engine_trace.get("events")
        if not isinstance(events, list) or any(not isinstance(item, str) for item in events):
            errors.append("capture.engine_trace.events: expected list[str].")
    else:
        errors.append("capture.engine_trace: expected object.")


def _validate_outputs(outputs: dict[str, Any], errors: list[str]) -> None:
    path = "outputs"
    allowed = {"directory", "golden", "bundle_policy"}
    _reject_unknown_keys(outputs, allowed, path=path, errors=errors)
    _require_keys(outputs, allowed, path=path, errors=errors)

    if "directory" in outputs and (
        not isinstance(outputs["directory"], str) or not outputs["directory"].strip()
    ):
        errors.append("outputs.directory: expected non-empty string.")

    golden = outputs.get("golden")
    if not isinstance(golden, dict):
        errors.append("outputs.golden: expected object.")
    else:
        _reject_unknown_keys(golden, {"enabled", "path"}, path="outputs.golden", errors=errors)
        _require_keys(golden, {"enabled"}, path="outputs.golden", errors=errors)
        if "enabled" in golden and not isinstance(golden["enabled"], bool):
            errors.append("outputs.golden.enabled: expected bool.")

    bundle_policy = outputs.get("bundle_policy")
    if not isinstance(bundle_policy, dict):
        errors.append("outputs.bundle_policy: expected object.")
    else:
        _reject_unknown_keys(
            bundle_policy,
            {"include", "retention_days"},
            path="outputs.bundle_policy",
            errors=errors,
        )
        _require_keys(bundle_policy, {"include", "retention_days"}, path="outputs.bundle_policy", errors=errors)
        include = bundle_policy.get("include")
        if not isinstance(include, list) or any(not isinstance(item, str) for item in include):
            errors.append("outputs.bundle_policy.include: expected list[str].")
        if "retention_days" in bundle_policy and (
            not isinstance(bundle_policy["retention_days"], int)
            or bundle_policy["retention_days"] < 0
        ):
            errors.append("outputs.bundle_policy.retention_days: expected int >= 0.")


def validate_manifest(manifest: dict[str, Any]) -> None:
    errors: list[str] = []

    top_required = {
        "schema_version",
        "kind",
        "metadata",
        "determinism",
        "hardware",
        "artifacts",
        "software_stack",
        "cuda_stack",
        "runtime",
        "vllm",
        "model",
        "inference",
        "capture",
        "outputs",
    }
    _reject_unknown_keys(manifest, top_required, path="manifest", errors=errors)
    _require_keys(manifest, top_required, path="manifest", errors=errors)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"manifest.schema_version: expected {SCHEMA_VERSION}, found {manifest.get('schema_version')!r}."
        )

    if manifest.get("kind") != MANIFEST_KIND:
        errors.append(
            f"manifest.kind: expected '{MANIFEST_KIND}', found {manifest.get('kind')!r}."
        )

    metadata = manifest.get("metadata")
    if isinstance(metadata, dict):
        allowed = {"name", "description", "created_at", "owners"}
        _reject_unknown_keys(metadata, allowed, path="metadata", errors=errors)
        _require_keys(metadata, {"name", "description", "created_at", "owners"}, path="metadata", errors=errors)
        if "name" in metadata and not isinstance(metadata["name"], str):
            errors.append("metadata.name: expected string.")
        if "description" in metadata and not isinstance(metadata["description"], str):
            errors.append("metadata.description: expected string.")
        if "created_at" in metadata and not isinstance(metadata["created_at"], str):
            errors.append("metadata.created_at: expected RFC3339 string.")
        owners = metadata.get("owners")
        if not isinstance(owners, list) or any(not isinstance(owner, str) for owner in owners):
            errors.append("metadata.owners: expected list[str].")
    else:
        errors.append("metadata: expected object.")

    determinism = manifest.get("determinism")
    if isinstance(determinism, dict):
        allowed = {"strict_hardware", "compare"}
        _reject_unknown_keys(determinism, allowed, path="determinism", errors=errors)
        _require_keys(determinism, allowed, path="determinism", errors=errors)
        if "strict_hardware" in determinism and not isinstance(determinism["strict_hardware"], bool):
            errors.append("determinism.strict_hardware: expected bool.")
        compare = determinism.get("compare")
        if isinstance(compare, dict):
            _validate_compare_block(compare, errors)
        else:
            errors.append("determinism.compare: expected object.")
    else:
        errors.append("determinism: expected object.")

    hardware = manifest.get("hardware")
    if isinstance(hardware, dict):
        _validate_hardware(hardware, errors)
    else:
        errors.append("hardware: expected object.")

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        _validate_artifacts(artifacts, errors)
    else:
        errors.append("artifacts: expected object.")

    software = manifest.get("software_stack")
    if isinstance(software, dict):
        _validate_software_stack(software, errors)
    else:
        errors.append("software_stack: expected object.")

    cuda_stack = manifest.get("cuda_stack")
    if isinstance(cuda_stack, dict):
        _validate_cuda_stack(cuda_stack, errors)
    else:
        errors.append("cuda_stack: expected object.")

    runtime = manifest.get("runtime")
    if isinstance(runtime, dict):
        _validate_runtime(runtime, errors)
    else:
        errors.append("runtime: expected object.")

    vllm = manifest.get("vllm")
    if isinstance(vllm, dict):
        _validate_vllm(vllm, errors)
    else:
        errors.append("vllm: expected object.")

    model = manifest.get("model")
    if isinstance(model, dict):
        _validate_model(model, errors)
    else:
        errors.append("model: expected object.")

    inference = manifest.get("inference")
    if isinstance(inference, dict):
        _validate_inference(inference, errors)
    else:
        errors.append("inference: expected object.")

    capture = manifest.get("capture")
    if isinstance(capture, dict):
        _validate_capture(capture, errors)
    else:
        errors.append("capture: expected object.")

    outputs = manifest.get("outputs")
    if isinstance(outputs, dict):
        _validate_outputs(outputs, errors)
    else:
        errors.append("outputs: expected object.")

    try:
        trust_remote_code = bool(
            manifest.get("vllm", {})
            .get("engine_args", {})
            .get("trust_remote_code", False)
        )
        if trust_remote_code:
            source = manifest.get("model", {}).get("weights", {}).get("source", {})
            remote_commit = source.get("remote_code_commit")
            if not isinstance(remote_commit, str) or not remote_commit.strip():
                errors.append(
                    "model.weights.source.remote_code_commit: required when vllm.engine_args.trust_remote_code=true."
                )
            remote_digest = source.get("remote_code_digest")
            if remote_digest is None:
                errors.append(
                    "model.weights.source.remote_code_digest: required when vllm.engine_args.trust_remote_code=true."
                )
            else:
                try:
                    _normalize_real_digest(
                        remote_digest,
                        field_path="model.weights.source.remote_code_digest",
                    )
                except ValueError as exc:
                    errors.append(str(exc))
    except AttributeError:
        errors.append("vllm.engine_args: expected object.")

    if errors:
        raise ManifestValidationError("\n".join(errors))


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_manifest_with_base(
    manifest_path: Path,
    *,
    visited: set[Path],
) -> dict[str, Any]:
    resolved = manifest_path.resolve()
    if resolved in visited:
        chain = " -> ".join(str(path) for path in sorted(visited))
        raise ValueError(f"Cyclic x_base_manifest reference detected while loading {resolved}: {chain}")
    visited.add(resolved)

    payload = _load_json_object(resolved)
    base_ref = payload.get("x_base_manifest")
    if isinstance(base_ref, str) and base_ref.strip():
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (resolved.parent / base_path).resolve()
        base_payload = _load_manifest_with_base(base_path, visited=visited)
        overlay = dict(payload)
        overlay.pop("x_base_manifest", None)
        payload = _deep_merge_dict(base_payload, overlay)

    visited.remove(resolved)
    return payload


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_manifest_with_base(manifest_path, visited=set())
    validate_manifest(manifest)
    return manifest


def _manifest_for_id(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = canonicalize(manifest)
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        if "lockfile" in artifacts:
            artifacts["lockfile"] = {"normalized": True}
    return payload


def compute_manifest_id(manifest: dict[str, Any]) -> str:
    return canonical_sha256(_manifest_for_id(manifest))


def _load_shared_prompt_dataset_conversations(
    *,
    prompt_dataset_path: Path | None = None,
) -> list[list[dict[str, str]]]:
    source_path = SHARED_PROMPT_DATASET_PATH if prompt_dataset_path is None else prompt_dataset_path
    if not source_path.is_file():
        raise FileNotFoundError(
            "Shared prompt dataset is missing. "
            f"Expected: {source_path}"
        )

    payload = _load_json_object(source_path)
    raw_conversations = payload.get("conversations")
    if not isinstance(raw_conversations, list) or not raw_conversations:
        raise ValueError(
            "Shared prompt dataset must define a non-empty 'conversations' list. "
            f"Source: {source_path}"
        )

    conversations: list[list[dict[str, str]]] = []
    for conv_index, raw_conversation in enumerate(raw_conversations):
        if not isinstance(raw_conversation, list) or not raw_conversation:
            raise ValueError(
                "Shared prompt dataset conversation must be a non-empty message list. "
                f"Source: {source_path}, index={conv_index}"
            )
        normalized: list[dict[str, str]] = []
        for msg_index, message in enumerate(raw_conversation):
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(
                    "Shared prompt dataset message must contain string role/content. "
                    f"Source: {source_path}, conversation={conv_index}, message={msg_index}"
                )
            normalized.append(
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
            )
        conversations.append(normalized)

    return conversations


def uses_shared_prompt_dataset(manifest: dict[str, Any]) -> bool:
    inference = manifest.get("inference")
    if not isinstance(inference, dict):
        return False
    return not isinstance(inference.get("requests"), list)


def resolve_inference_requests(
    manifest: dict[str, Any],
    *,
    prompt_dataset_path: Path | None = None,
) -> list[dict[str, Any]]:
    inference = manifest["inference"]
    requests = inference.get("requests")
    if isinstance(requests, list):
        return copy.deepcopy(requests)

    n_prompts = int(inference["n_prompts"])
    if n_prompts < 1 or n_prompts > SHARED_PROMPT_DATASET_MAX_PROMPTS:
        raise ValueError(
            f"inference.n_prompts must be in range 1..{SHARED_PROMPT_DATASET_MAX_PROMPTS}."
        )

    template = inference["request_template"]
    if not isinstance(template, dict):
        raise ValueError("inference.request_template must be an object.")

    conversations = _load_shared_prompt_dataset_conversations(
        prompt_dataset_path=prompt_dataset_path,
    )
    if n_prompts > len(conversations):
        raise ValueError(
            "inference.n_prompts exceeds available conversations in shared prompt dataset "
            f"(requested={n_prompts}, available={len(conversations)})."
        )

    id_prefix = str(template.get("id_prefix", "req-"))
    sampling = copy.deepcopy(template["sampling"])
    stop = copy.deepcopy(template["stop"])
    kind = str(template["kind"])

    resolved: list[dict[str, Any]] = []
    for index, conversation in enumerate(conversations[:n_prompts], start=1):
        resolved.append(
            {
                "id": f"{id_prefix}{index:04d}",
                "kind": kind,
                "messages": copy.deepcopy(conversation),
                "sampling": copy.deepcopy(sampling),
                "stop": copy.deepcopy(stop),
            }
        )
    return resolved


def compute_requests_digest(
    manifest: dict[str, Any],
    *,
    prompt_dataset_path: Path | None = None,
) -> str:
    return canonical_sha256(
        resolve_inference_requests(manifest, prompt_dataset_path=prompt_dataset_path)
    )


def compute_runtime_closure_digest(manifest: dict[str, Any]) -> str:
    closure = {
        "software_stack": manifest["software_stack"],
        "cuda_stack": manifest["cuda_stack"],
        "runtime": manifest["runtime"],
        "vllm": {
            "mode": manifest["vllm"]["mode"],
            "env": manifest["vllm"]["env"],
            "engine_args": manifest["vllm"]["engine_args"],
        },
    }
    return canonical_sha256(closure)


def create_manifest_template(
    *,
    model_id: str,
    model_revision: str,
    output_path: Path,
) -> dict[str, Any]:
    manifest_name = output_path.stem
    lock_path = f"manifests/{manifest_name}/manifest.lock.json"

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "metadata": {
            "name": manifest_name,
            "description": "Spec-aligned deterministic inference manifest.",
            "created_at": utc_now_iso(),
            "owners": ["ml-platform@example.com"],
        },
        "determinism": {
            "strict_hardware": False,
            "compare": {
                "tokens": {"rule": "exact"},
                "logits": {"rule": "exact"},
                "activations": {"rule": "absrel", "atol": 0.0, "rtol": 0.0},
                "engine_trace": {"rule": "exact"},
            },
        },
        "hardware": {
            "constraints": {
                "accelerator_vendor": "nvidia",
                "gpu_arch_min_cc": "8.0",
                "allowed_gpu_models": [],
            },
            "record": {
                "capture_collect_env": True,
                "capture_nvidia_smi": True,
                "capture_driver_versions": True,
                "capture_pcie_topology": False,
            },
        },
        "artifacts": {
            "store": {"type": "cas", "path": "state/store"},
            "lockfile": lock_path,
        },
        "software_stack": {
            "python": {
                "version": platform.python_version(),
                "packages": [
                    {"name": "vllm", "source": {"type": "wheel", "digest": "sha256:unset"}},
                    {"name": "torch", "source": {"type": "wheel", "digest": "sha256:unset"}},
                ],
            },
            "compiled_extensions": [],
        },
        "cuda_stack": {
            "userspace": {
                "cuda_toolkit": {"version": "12.x", "digest": "sha256:unset"},
                "cublas": {"version": "12.x", "digest": "sha256:unset"},
                "cudnn": {"version": "9.x", "digest": "sha256:unset"},
                "nccl": {"version": "2.x", "digest": "sha256:unset"},
            },
            "env": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            "driver_policy": "pinned_or_recorded",
        },
        "runtime": {
            "env": {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            "locale": {"lang": "C.UTF-8", "tz": "UTC"},
            "threads": {"omp_num_threads": 1, "mkl_num_threads": 1},
            "network_policy": "online_allowed",
            "execution": {
                "backend": "mock",
                "timeout_seconds": 120,
                "deterministic_failure_policy": "warn_only",
                "vllm_image": "docker.io/vllm/vllm-openai@sha256:c48cf118e1e6e39d7790e174d6014f7af5d06f79c2d29d984d11cbe2e8d414e7",
            },
        },
        "vllm": {
            "mode": "offline",
            "reproducibility": {
                "enable_batch_invariance": True,
                "enable_v1_multiprocessing": False,
            },
            "env": {
                "VLLM_BATCH_INVARIANT": "1",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            },
            "engine_args": {
                "model": model_id,
                "trust_remote_code": False,
                "dtype": "float16",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "max_model_len": 32768,
            },
        },
        "model": {
            "weights": {
                "source": {
                    "type": "huggingface",
                    "repo": model_id,
                    "revision": model_revision,
                },
                "files": [
                    {"path": "model.safetensors", "digest": "sha256:unset"},
                ],
            },
            "tokenizer": {
                "files": [
                    {"path": "tokenizer.json", "digest": "sha256:unset"},
                    {"path": "tokenizer_config.json", "digest": "sha256:unset"},
                    {"path": "special_tokens_map.json", "digest": "sha256:unset"},
                ],
            },
            "config": {
                "files": [
                    {"path": "config.json", "digest": "sha256:unset"},
                    {"path": "generation_config.json", "digest": "sha256:unset"},
                ],
            },
            "chat_template": {
                "file": {"path": "chat_template.jinja", "digest": "sha256:unset"}
            },
        },
        "inference": {
            "requests": [
                {
                    "id": "req-0001",
                    "kind": "completion",
                    "prompt": "The future of AI is",
                    "sampling": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": None,
                        "max_tokens": 32,
                        "seed": 424242,
                    },
                    "stop": {"sequences": [], "token_ids": []},
                }
            ],
            "batching": {
                "policy": "fixed",
                "schedule": "offline_deterministic",
                "max_num_seqs": 1,
                "max_num_batched_tokens": 4096,
                "fixed_request_order": True,
                "concurrency": 1,
            },
        },
        "capture": {
            "tokens": True,
            "logits": {
                "enabled": False,
                "scope": "chosen_only",
                "dtype": "float16",
                "capture_prefill": False,
                "capture_decode": False,
            },
            "activations": {
                "enabled": False,
                "hooks": [],
            },
            "engine_trace": {
                "enabled": False,
                "events": [],
            },
        },
        "outputs": {
            "directory": "runs/${run_id}",
            "golden": {"enabled": False},
            "bundle_policy": {"include": ["manifest", "lock", "tokens", "run_log"], "retention_days": 30},
        },
    }
