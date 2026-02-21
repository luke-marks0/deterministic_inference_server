from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import BUNDLE_KIND, SCHEMA_VERSION, _prompt_token_matrix_hash, _repo_root, _token_ids_hash, _write_json, canonical_sha256, utc_now_iso
from .locking import resolve_lock_path, verify_lock_artifact_integrity
from .schema import (
    compute_manifest_id,
    compute_requests_digest,
    compute_runtime_closure_digest,
    resolve_inference_requests,
    uses_shared_prompt_dataset,
)

_TOKEN_ID_RE = re.compile(r"^token_id:(-?\d+)$")

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

@dataclass(frozen=True)
class HardwareConformance:
    conformant: bool
    reasons: list[str]


def probe_hardware(record_config: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "captured_at": utc_now_iso(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "accelerator_vendor": "none",
        "gpu_models": [],
        "gpu_compute_capabilities": [],
        "driver_versions": [],
    }

    if record_config.get("capture_collect_env", False):
        env_subset = {}
        for key in sorted((
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES",
            "LANG",
            "TZ",
        )):
            if key in os.environ:
                env_subset[key] = os.environ[key]
        info["env"] = env_subset

    if not shutil.which("nvidia-smi"):
        return info

    if record_config.get("capture_nvidia_smi", False) or record_config.get("capture_driver_versions", False):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            info["accelerator_vendor"] = "nvidia"
            for line in result.stdout.splitlines():
                parts = [item.strip() for item in line.split(",")]
                if len(parts) < 3:
                    continue
                model_name, compute_capability, driver_version = parts[0], parts[1], parts[2]
                info["gpu_models"].append(model_name)
                info["gpu_compute_capabilities"].append(compute_capability)
                info["driver_versions"].append(driver_version)

    if record_config.get("capture_pcie_topology", False):
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            capture_output=True,
            text=True,
        )
        if topo.returncode == 0:
            info["pcie_topology"] = topo.stdout.strip()

    return info


def _parse_cc(value: str) -> tuple[int, int] | None:
    value = value.strip()
    if not value:
        return None
    parts = value.split(".")
    if len(parts) != 2:
        return None
    if not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1])


def _cc_is_less(lhs: str, rhs: str) -> bool:
    lhs_cc = _parse_cc(lhs)
    rhs_cc = _parse_cc(rhs)
    if lhs_cc is None or rhs_cc is None:
        return False
    return lhs_cc < rhs_cc


def evaluate_hardware_conformance(
    constraints: dict[str, Any],
    fingerprint: dict[str, Any],
) -> HardwareConformance:
    reasons: list[str] = []

    expected_vendor = constraints.get("accelerator_vendor")
    if isinstance(expected_vendor, str):
        actual_vendor = str(fingerprint.get("accelerator_vendor", "none")).lower()
        if expected_vendor.lower() != actual_vendor:
            reasons.append(
                f"accelerator_vendor mismatch (expected={expected_vendor}, actual={actual_vendor})"
            )

    allowed_models = constraints.get("allowed_gpu_models")
    if isinstance(allowed_models, list) and allowed_models:
        actual_models = [str(model) for model in fingerprint.get("gpu_models", [])]
        if not actual_models:
            reasons.append("GPU model list missing while allowed_gpu_models was declared")
        elif any(model not in allowed_models for model in actual_models):
            reasons.append(
                f"gpu_models mismatch (allowed={allowed_models}, actual={actual_models})"
            )

    min_cc = constraints.get("gpu_arch_min_cc")
    if isinstance(min_cc, str) and min_cc.strip():
        actual_caps = [str(cap) for cap in fingerprint.get("gpu_compute_capabilities", []) if str(cap).strip()]
        if not actual_caps:
            reasons.append("missing gpu_compute_capabilities while gpu_arch_min_cc was declared")
        elif any(_cc_is_less(actual, min_cc) for actual in actual_caps):
            reasons.append(
                f"gpu_arch_min_cc violation (expected >= {min_cc}, actual={actual_caps})"
            )

    max_cc = constraints.get("gpu_arch_max_cc")
    if isinstance(max_cc, str) and max_cc.strip():
        actual_caps = [str(cap) for cap in fingerprint.get("gpu_compute_capabilities", []) if str(cap).strip()]
        if actual_caps and any(_cc_is_less(max_cc, actual) for actual in actual_caps):
            reasons.append(
                f"gpu_arch_max_cc violation (expected <= {max_cc}, actual={actual_caps})"
            )

    expected_cpu_arch = constraints.get("cpu_arch")
    if isinstance(expected_cpu_arch, str) and expected_cpu_arch.strip():
        actual_cpu_arch = str(fingerprint.get("platform", {}).get("machine", ""))
        if actual_cpu_arch != expected_cpu_arch:
            reasons.append(
                f"cpu_arch mismatch (expected={expected_cpu_arch}, actual={actual_cpu_arch})"
            )

    return HardwareConformance(conformant=(len(reasons) == 0), reasons=reasons)


def _apply_runtime_environment(manifest: dict[str, Any]) -> dict[str, str]:
    runtime_env = manifest["runtime"]["env"]
    cuda_env = manifest["cuda_stack"]["env"]
    vllm_env = manifest["vllm"]["env"]
    thread_settings = manifest["runtime"]["threads"]
    locale_settings = manifest["runtime"]["locale"]

    applied: dict[str, str] = {}
    for block in (runtime_env, cuda_env, vllm_env):
        for key, value in block.items():
            normalized_key = str(key)
            normalized_value = str(value)
            os.environ[normalized_key] = normalized_value
            applied[normalized_key] = normalized_value

    thread_env = {
        "OMP_NUM_THREADS": str(thread_settings["omp_num_threads"]),
        "MKL_NUM_THREADS": str(thread_settings["mkl_num_threads"]),
    }
    for key, value in thread_env.items():
        os.environ[key] = value
        applied[key] = value

    locale_env = {
        "LANG": str(locale_settings["lang"]),
        "LC_ALL": str(locale_settings["lang"]),
        "TZ": str(locale_settings["tz"]),
    }
    for key, value in locale_env.items():
        os.environ[key] = value
        applied[key] = value

    if hasattr(time, "tzset"):
        time.tzset()

    return applied


def _apply_runtime_determinism_controls(
    manifest: dict[str, Any],
    *,
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    seed = int(requests[0]["sampling"]["seed"])
    policy = str(manifest["runtime"]["execution"]["deterministic_failure_policy"])
    warnings: list[str] = []

    random.seed(seed)

    numpy_status: dict[str, Any] = {"available": False, "seeded": False}
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed % (2**32))
        numpy_status = {
            "available": True,
            "seeded": True,
            "version": str(getattr(np, "__version__", "")),
        }
    except Exception as exc:  # pragma: no cover - depends on environment
        numpy_status["error"] = str(exc)

    torch_status: dict[str, Any] = {"available": False, "seeded": False}
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if bool(torch.cuda.is_available()):
            torch.cuda.manual_seed_all(seed)

        deterministic_enabled = False
        if policy == "off":
            try:
                torch.use_deterministic_algorithms(False)
            except Exception as exc:
                warnings.append(f"Unable to disable deterministic algorithms in torch: {exc}")
        elif policy == "error":
            torch.use_deterministic_algorithms(True)
            deterministic_enabled = True
        else:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
                warnings.append(
                    "torch.use_deterministic_algorithms(warn_only=True) unavailable; upgraded to strict mode."
                )
            deterministic_enabled = True

        if hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = bool(deterministic_enabled)
            except Exception as exc:
                warnings.append(f"Unable to configure cuDNN determinism flags: {exc}")

        torch_status = {
            "available": True,
            "seeded": True,
            "version": str(getattr(torch, "__version__", "")),
            "cuda_version": str(getattr(torch.version, "cuda", "")),
            "deterministic_algorithms": bool(deterministic_enabled),
        }
    except Exception as exc:  # pragma: no cover - depends on environment
        message = f"Torch controls unavailable: {exc}"
        torch_status["error"] = str(exc)
        if policy == "error":
            raise RuntimeError(
                "runtime.execution.deterministic_failure_policy='error' but torch controls could not be applied.\n"
                f"{message}"
            ) from exc
        warnings.append(message)

    return {
        "seed": seed,
        "deterministic_failure_policy": policy,
        "python_random_seeded": True,
        "numpy": numpy_status,
        "torch": torch_status,
        "warnings": warnings,
    }


def _probe_software_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
    }

    try:
        import vllm  # type: ignore

        metadata["vllm"] = {
            "available": True,
            "version": str(getattr(vllm, "__version__", "")),
        }
    except Exception as exc:  # pragma: no cover - depends on environment
        metadata["vllm"] = {"available": False, "error": str(exc)}

    try:
        import torch  # type: ignore

        cudnn_version = ""
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "version"):
            try:
                raw_version = torch.backends.cudnn.version()
                cudnn_version = "" if raw_version is None else str(raw_version)
            except Exception:
                cudnn_version = ""

        metadata["torch"] = {
            "available": True,
            "version": str(getattr(torch, "__version__", "")),
            "cuda_version": str(getattr(torch.version, "cuda", "")),
            "cudnn_version": cudnn_version,
        }
    except Exception as exc:  # pragma: no cover - depends on environment
        metadata["torch"] = {"available": False, "error": str(exc)}

    return metadata


def _request_messages(request: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(request.get("messages"), list):
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in request["messages"]
        ]
    if isinstance(request.get("prompt"), str):
        return [{"role": "user", "content": str(request["prompt"])}]
    if isinstance(request.get("prompt_token_ids"), list):
        return [{"role": "user", "content": "<pretokenized_prompt>"}]
    raise ValueError("Request is missing prompt/messages/prompt_token_ids.")


def _request_prompt_token_ids(request: dict[str, Any]) -> list[int]:
    if isinstance(request.get("prompt_token_ids"), list):
        return [int(token) for token in request["prompt_token_ids"]]

    if isinstance(request.get("prompt"), str):
        return [int(byte) for byte in request["prompt"].encode("utf-8")]

    if isinstance(request.get("messages"), list):
        rendered = "\n".join(
            f"{message['role']}:{message['content']}" for message in request["messages"]
        )
        return [int(byte) for byte in rendered.encode("utf-8")]

    raise ValueError("Request is missing prompt/messages/prompt_token_ids.")


def _extract_generated_token_ids(response: dict[str, Any], prompt_len: int) -> list[int]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No choices returned in completion response.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Invalid completion choice payload.")

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError("Completion response did not include logprobs payload.")

    content = logprobs.get("content")
    if isinstance(content, list):
        token_ids = [
            int(row["token_id"])
            for row in content
            if isinstance(row, dict) and isinstance(row.get("token_id"), int)
        ]
        if len(token_ids) >= prompt_len:
            return token_ids[prompt_len:]

    tokens = logprobs.get("tokens")
    if isinstance(tokens, list):
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
        "Unable to parse token ids from response. "
        "Expected logprobs.content[*].token_id or logprobs.tokens token_id:<int> strings."
    )


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Completion response must be a JSON object.")
    return decoded


def _mock_generate_tokens(prompt_token_ids: list[int], *, seed: int, max_tokens: int) -> list[int]:
    base = (sum(prompt_token_ids) + seed) % 100_000
    return [(base + index) % 100_000 for index in range(max_tokens)]


def _openai_generate_tokens(
    *,
    request: dict[str, Any],
    prompt_token_ids: list[int],
    manifest: dict[str, Any],
) -> tuple[list[int], int]:
    sampling = request["sampling"]
    execution = manifest["runtime"]["execution"]
    base_url = str(execution["base_url"]).rstrip("/")
    timeout_seconds = int(execution["timeout_seconds"])
    model_name = str(manifest["vllm"]["engine_args"].get("model", ""))

    payload = {
        "model": model_name,
        "prompt": prompt_token_ids,
        "max_tokens": int(sampling["max_tokens"]),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "top_k": sampling["top_k"],
        "seed": int(sampling["seed"]),
        "echo": True,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
    }

    response = _post_json(f"{base_url}/v1/completions", payload, timeout_seconds)
    output_ids = _extract_generated_token_ids(response, prompt_len=len(prompt_token_ids))

    usage = response.get("usage")
    completion_tokens = None
    if isinstance(usage, dict):
        raw_completion = usage.get("completion_tokens")
        if isinstance(raw_completion, int):
            completion_tokens = raw_completion

    if completion_tokens is None:
        completion_tokens = len(output_ids)

    return output_ids, completion_tokens


def _generate_tokens_for_request(
    *,
    request: dict[str, Any],
    prompt_token_ids: list[int],
    manifest: dict[str, Any],
) -> tuple[list[int], int]:
    backend = str(manifest["runtime"]["execution"]["backend"])
    sampling = request["sampling"]

    if backend == "mock":
        output_ids = _mock_generate_tokens(
            prompt_token_ids,
            seed=int(sampling["seed"]),
            max_tokens=int(sampling["max_tokens"]),
        )
        return output_ids, len(output_ids)

    if backend == "openai_compatible":
        return _openai_generate_tokens(
            request=request,
            prompt_token_ids=prompt_token_ids,
            manifest=manifest,
        )

    raise ValueError(f"Unsupported runtime.execution.backend: {backend}")


def _grade_from_run_conformance(
    *,
    hardware_conformant: bool,
    runtime_digest_matches_lock: bool,
) -> str:
    if not hardware_conformant:
        return "non_conformant_hardware"
    if not runtime_digest_matches_lock:
        return "non_conformant_software"
    return "conformant"


def _build_capture_status(manifest: dict[str, Any]) -> dict[str, Any]:
    capture = manifest["capture"]

    def _status_for(enabled: bool) -> str:
        if enabled:
            return "configured_noop"
        return "disabled"

    return {
        "tokens": {"enabled": True, "status": "captured"},
        "logits": {
            "enabled": bool(capture["logits"]["enabled"]),
            "status": _status_for(bool(capture["logits"]["enabled"])),
            "note": "TODO: logits capture intentionally not implemented yet.",
        },
        "activations": {
            "enabled": bool(capture["activations"]["enabled"]),
            "status": _status_for(bool(capture["activations"]["enabled"])),
            "note": "TODO: activation capture intentionally not implemented yet.",
        },
        "engine_trace": {
            "enabled": bool(capture["engine_trace"]["enabled"]),
            "status": _status_for(bool(capture["engine_trace"]["enabled"])),
            "note": "TODO: vLLM engine trace capture intentionally not implemented yet.",
        },
    }


def _render_run_directory(template: str, *, run_id: str, manifest_path: Path) -> Path:
    rendered = template.replace("${run_id}", run_id)
    rendered = rendered.replace("${manifest_name}", manifest_path.stem)
    run_dir = Path(rendered)
    if not run_dir.is_absolute():
        run_dir = (_repo_root() / run_dir).resolve()
    return run_dir


def _common_sampling_values(requests: list[dict[str, Any]]) -> dict[str, Any]:
    first = requests[0]["sampling"]
    reference = {
        "max_tokens": int(first["max_tokens"]),
        "seed": int(first["seed"]),
        "temperature": float(first["temperature"]),
        "top_k": first["top_k"],
        "top_p": float(first["top_p"]),
    }
    for request in requests[1:]:
        sampling = request["sampling"]
        candidate = {
            "max_tokens": int(sampling["max_tokens"]),
            "seed": int(sampling["seed"]),
            "temperature": float(sampling["temperature"]),
            "top_k": sampling["top_k"],
            "top_p": float(sampling["top_p"]),
        }
        if candidate != reference:
            raise ValueError(
                "All requests must currently share identical sampling values to preserve "
                "the established output token format."
            )
    return reference


def _decode_token_ids(token_ids: list[int]) -> str:
    return " ".join(f"token_id:{token}" for token in token_ids)


def _shared_prompt_dataset_path_from_lock(lock: dict[str, Any]) -> Path:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("lock.artifacts: expected list.")

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("name") != "inference.prompt_dataset":
            continue

        path_value = artifact.get("path")
        if isinstance(path_value, str) and path_value.strip():
            return Path(path_value)

        retrieval = artifact.get("retrieval")
        if isinstance(retrieval, dict):
            local_file = retrieval.get("local_file")
            if isinstance(local_file, str) and local_file.strip():
                return Path(local_file)
        break

    raise ValueError(
        "Lockfile is missing required artifact 'inference.prompt_dataset' for shared prompt mode."
    )


def execute_run(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    lock: dict[str, Any],
    verify_artifact_digests: bool = True,
    token_output_override: Path | None = None,
    run_log_override: Path | None = None,
    run_dir_override: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    if verify_artifact_digests:
        artifact_integrity = verify_lock_artifact_integrity(
            manifest_path=manifest_path,
            lock=lock,
        )
    else:
        artifact_integrity = {
            "enabled": False,
            "checked_count": 0,
            "skipped_count": 0,
            "failure_count": 0,
            "checked": [],
            "skipped": [{"reason": "disabled_by_flag"}],
        }

    prompt_dataset_path: Path | None = None
    if uses_shared_prompt_dataset(manifest):
        prompt_dataset_path = _shared_prompt_dataset_path_from_lock(lock)

    requests = resolve_inference_requests(
        manifest,
        prompt_dataset_path=prompt_dataset_path,
    )

    applied_environment = _apply_runtime_environment(manifest)
    determinism_controls = _apply_runtime_determinism_controls(
        manifest,
        requests=requests,
    )
    software_metadata = _probe_software_metadata()

    runtime_digest = compute_runtime_closure_digest(manifest)
    runtime_digest_matches_lock = runtime_digest == str(lock["runtime_closure_digest"])

    hardware_fingerprint = probe_hardware(manifest["hardware"]["record"])
    hardware_conformance = evaluate_hardware_conformance(
        manifest["hardware"]["constraints"],
        hardware_fingerprint,
    )

    strict_hardware = bool(manifest["determinism"]["strict_hardware"])
    if strict_hardware and not hardware_conformance.conformant:
        joined = "\n".join(hardware_conformance.reasons)
        raise RuntimeError(
            "Hardware constraints were not satisfied and strict_hardware=true.\n"
            f"{joined}"
        )

    batching = manifest["inference"]["batching"]
    schedule = str(batching["schedule"])
    mode = str(manifest["vllm"]["mode"])

    if batching["policy"] != "fixed":
        raise ValueError("Only fixed batching policy is supported.")

    if not bool(batching.get("fixed_request_order", False)):
        raise ValueError(
            "Fixed batching requires batching.fixed_request_order=true to preserve deterministic ordering."
        )

    if schedule == "offline_deterministic" and mode != "offline":
        raise ValueError("batch.schedule=offline_deterministic requires vllm.mode=offline.")

    if schedule == "batch_invariant":
        enabled = bool(manifest["vllm"]["reproducibility"].get("enable_batch_invariance", False))
        if not enabled:
            raise ValueError(
                "batch.schedule=batch_invariant requires vllm.reproducibility.enable_batch_invariance=true."
            )

    ordered_requests = list(requests)
    if schedule == "replay":
        replay_ready = []
        for idx, request in enumerate(requests):
            arrival_ms = request.get("x_arrival_ms")
            if not isinstance(arrival_ms, int) or arrival_ms < 0:
                raise ValueError(
                    "batch.schedule=replay requires every request to declare integer x_arrival_ms >= 0."
                )
            replay_ready.append((arrival_ms, idx, request))
        replay_ready.sort(key=lambda item: (item[0], item[1]))
        ordered_requests = [item[2] for item in replay_ready]

    batch_size = int(batching["max_num_seqs"])
    if len(ordered_requests) % batch_size != 0:
        raise ValueError(
            "With fixed batching policy, request count must be an exact multiple of "
            f"batch.max_num_seqs (count={len(ordered_requests)}, batch={batch_size})."
        )

    if int(batching["concurrency"]) != 1:
        raise ValueError(
            "Only concurrency=1 is currently supported to keep deterministic run ordering explicit."
        )

    sequences: list[dict[str, list[int]]] = []
    conversations: list[list[dict[str, str]]] = []
    decoded_outputs: list[str] = []
    total_completion_tokens = 0
    batch_steps: list[dict[str, Any]] = []
    max_batched_tokens = int(batching["max_num_batched_tokens"])
    progress = None
    if _tqdm is not None:
        progress = _tqdm(
            total=len(ordered_requests),
            desc="Generating",
            unit="req",
            disable=not sys.stderr.isatty(),
        )

    try:
        for batch_start in range(0, len(ordered_requests), batch_size):
            batch = ordered_requests[batch_start : batch_start + batch_size]
            if len(batch) != batch_size:
                raise ValueError("Fixed batching policy requires exact batch size for every step.")

            request_ids: list[str] = []
            prompt_token_counts: list[int] = []
            max_tokens_per_request: list[int] = []
            replay_arrival_ms: list[int] = []
            batch_token_budget = 0

            for request in batch:
                prompt_ids = _request_prompt_token_ids(request)
                request_max_tokens = int(request["sampling"]["max_tokens"])
                batch_token_budget += len(prompt_ids) + request_max_tokens
                request_ids.append(str(request["id"]))
                prompt_token_counts.append(len(prompt_ids))
                max_tokens_per_request.append(request_max_tokens)
                if schedule == "replay":
                    replay_arrival_ms.append(int(request["x_arrival_ms"]))

            if batch_token_budget > max_batched_tokens:
                raise ValueError(
                    "Batch token budget exceeded max_num_batched_tokens "
                    f"(observed={batch_token_budget}, limit={max_batched_tokens})."
                )

            batch_steps.append(
                {
                    "step_index": len(batch_steps),
                    "batch_size": len(batch),
                    "request_ids": request_ids,
                    "prompt_token_counts": prompt_token_counts,
                    "max_tokens_per_request": max_tokens_per_request,
                    "batch_token_budget": batch_token_budget,
                    "replay_arrival_ms": replay_arrival_ms,
                }
            )

            for request in batch:
                prompt_token_ids = _request_prompt_token_ids(request)
                output_token_ids, completion_count = _generate_tokens_for_request(
                    request=request,
                    prompt_token_ids=prompt_token_ids,
                    manifest=manifest,
                )
                sequences.append(
                    {
                        "prompt_token_ids": prompt_token_ids,
                        "output_token_ids": output_token_ids,
                    }
                )
                conversations.append(_request_messages(request))
                decoded_outputs.append(_decode_token_ids(output_token_ids))
                total_completion_tokens += int(completion_count)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    prompt_matrix_hash = _prompt_token_matrix_hash(
        [sequence["prompt_token_ids"] for sequence in sequences]
    )

    manifest_id = compute_manifest_id(manifest)
    lock_id = str(lock["lock_id"])
    requests_digest = compute_requests_digest(
        manifest,
        prompt_dataset_path=prompt_dataset_path,
    )
    hardware_fingerprint_digest = canonical_sha256(hardware_fingerprint)
    run_id = hashlib.sha256(
        f"{manifest_id}{lock_id}{requests_digest}{hardware_fingerprint_digest}".encode("utf-8")
    ).hexdigest()

    run_dir = run_dir_override or _render_run_directory(
        manifest["outputs"]["directory"],
        run_id=run_id,
        manifest_path=manifest_path,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    sampling = _common_sampling_values(requests)
    model_source = manifest["model"]["weights"]["source"]
    model_name = str(model_source.get("repo", model_source.get("id", "unknown-model")))
    provider_label = str(manifest["metadata"]["name"])
    tokenizer_digest = canonical_sha256(manifest["model"]["tokenizer"]["files"])

    token_output_payload = {
        "model": model_name,
        "provider": provider_label,
        "parameters": {
            "n_prompts": len(sequences),
            "max_tokens": sampling["max_tokens"],
            "seed": sampling["seed"],
            "temperature": sampling["temperature"],
            "top_k": sampling["top_k"],
            "top_p": sampling["top_p"],
            "prompt_token_ids_sha256": prompt_matrix_hash,
        },
        "tokenizer": {
            "digest": tokenizer_digest,
            "revision": str(model_source.get("revision", "")),
            "files": manifest["model"]["tokenizer"]["files"],
        },
        "conversations": conversations,
        "sequences": sequences,
        "decoded_output_text": decoded_outputs,
    }

    token_output_path = token_output_override or (run_dir / "tokens.json")
    _write_json(token_output_path, token_output_payload)

    lock_manifest_ref = "<embedded_lock>"
    if lock_path is not None:
        lock_manifest_ref = str(lock_path)
    elif isinstance(manifest["artifacts"]["lockfile"], str):
        lock_manifest_ref = str(resolve_lock_path(manifest, manifest_path))

    records = []
    for prompt_index, sequence in enumerate(sequences):
        prompt_ids = sequence["prompt_token_ids"]
        output_ids = sequence["output_token_ids"]
        records.append(
            {
                "prompt_index": prompt_index,
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
        "created_at_utc": utc_now_iso(),
        "profile_config": str(manifest_path),
        "generation_lock_manifest": lock_manifest_ref,
        "generation_lock_manifest_sha256": str(lock_id),
        "source_reference_bundle": "",
        "source_reference_bundle_sha256": "",
        "source_reference_model": model_name,
        "source_reference_prompt_token_ids_sha256": prompt_matrix_hash,
        "source_reference_has_pretokenized_prompts": any(
            isinstance(request.get("prompt_token_ids"), list) for request in requests
        ),
        "tokenizer_source": model_name,
        "model": model_name,
        "served_model": str(manifest["vllm"]["engine_args"].get("model", model_name)),
        "base_url": str(manifest["runtime"]["execution"].get("base_url", "")),
        "parameters": {
            "n_prompts": len(sequences),
            "max_tokens": sampling["max_tokens"],
            "seed": sampling["seed"],
            "temperature": sampling["temperature"],
            "top_k": sampling["top_k"],
            "top_p": sampling["top_p"],
            "concurrency": int(batching["concurrency"]),
            "timeout_seconds": int(manifest["runtime"]["execution"]["timeout_seconds"]),
            "tokenizer_revision": str(model_source.get("revision", "")),
            "prompt_token_ids_sha256": prompt_matrix_hash,
            "generation_lock_manifest_sha256": str(lock_id),
        },
        "summary": {
            "prompt_count": len(sequences),
            "completion_tokens": total_completion_tokens,
            "elapsed_s": 0.0,
        },
        "runtime_environment": applied_environment,
        "determinism_controls": determinism_controls,
        "artifact_integrity": artifact_integrity,
        "batch_trace": {
            "policy": str(batching["policy"]),
            "schedule": schedule,
            "max_num_seqs": batch_size,
            "max_num_batched_tokens": max_batched_tokens,
            "steps": batch_steps,
        },
        "records": records,
    }

    run_log_path = run_log_override or (run_dir / "run_log.json")
    _write_json(run_log_path, run_log_payload)

    capture_status = _build_capture_status(manifest)
    grade = _grade_from_run_conformance(
        hardware_conformant=hardware_conformance.conformant,
        runtime_digest_matches_lock=runtime_digest_matches_lock,
    )

    bundle_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "manifest_id": manifest_id,
        "lock_id": lock_id,
        "requests_digest": requests_digest,
        "hardware_fingerprint": hardware_fingerprint,
        "hardware_conformant": hardware_conformance.conformant,
        "hardware_non_conformance_reasons": hardware_conformance.reasons,
        "runtime_closure_digest": runtime_digest,
        "runtime_closure_matches_lock": runtime_digest_matches_lock,
        "determinism_grade": grade,
        "runtime_environment": applied_environment,
        "determinism_controls": determinism_controls,
        "software_metadata": software_metadata,
        "artifact_integrity": artifact_integrity,
        "batch_trace": {
            "policy": str(batching["policy"]),
            "schedule": schedule,
            "max_num_seqs": batch_size,
            "max_num_batched_tokens": max_batched_tokens,
            "steps": batch_steps,
        },
        "capture": capture_status,
        "paths": {
            "manifest": "manifest.used.json",
            "lock": "lock.used.json",
            "tokens": "tokens.json",
            "run_log": "run_log.json",
        },
        "notes": {
            "logits": "configured via capture.logits but intentionally no-op",
            "activations": "configured via capture.activations but intentionally no-op",
            "engine_trace": "configured via capture.engine_trace but intentionally no-op",
        },
    }

    _write_json(run_dir / "manifest.used.json", manifest)
    _write_json(run_dir / "lock.used.json", lock)
    _write_json(run_dir / "bundle.json", bundle_payload)

    return {
        "run_dir": run_dir,
        "bundle_path": run_dir / "bundle.json",
        "token_output_path": token_output_path,
        "run_log_path": run_log_path,
        "grade": grade,
        "run_id": run_id,
    }
