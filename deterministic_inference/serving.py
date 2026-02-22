from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .common import _normalize_pinned_image_reference, _repo_root


@dataclass(frozen=True)
class ServePlan:
    base_url: str
    compose_file: Path
    env: dict[str, str]
    image: str
    model_id: str
    revision: str
    served_model_name: str


@dataclass(frozen=True)
class ActiveServeContainer:
    container_id: str
    name: str
    image: str
    status: str
    ports: str


def _required_non_empty_string(value: Any, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_path}: expected non-empty string.")
    return value.strip()


def _safe_container_name(raw_name: str) -> str:
    normalized = []
    for char in raw_name:
        if char.isalnum() or char in {"-", "_", "."}:
            normalized.append(char)
        else:
            normalized.append("-")
    cleaned = "".join(normalized).strip("-")
    if not cleaned:
        cleaned = "vllm_server"
    return cleaned


def _base_url_port(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("runtime.execution.base_url must use http or https.")
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    return 80


def build_serve_plan(
    manifest: dict[str, Any],
    *,
    image: str | None = None,
    container_port: int = 8000,
) -> ServePlan:
    runtime_execution = manifest["runtime"]["execution"]
    backend = _required_non_empty_string(
        runtime_execution.get("backend"),
        field_path="runtime.execution.backend",
    )
    if backend != "openai_compatible":
        raise ValueError(
            "serve command requires runtime.execution.backend='openai_compatible'."
        )

    mode = _required_non_empty_string(
        manifest["vllm"].get("mode"),
        field_path="vllm.mode",
    )
    if mode != "server":
        raise ValueError("serve command requires vllm.mode='server'.")

    base_url = _required_non_empty_string(
        runtime_execution.get("base_url"),
        field_path="runtime.execution.base_url",
    )
    model_source = manifest["model"]["weights"]["source"]
    model_id = _required_non_empty_string(
        model_source.get("repo", model_source.get("id")),
        field_path="model.weights.source.repo",
    )
    revision = _required_non_empty_string(
        model_source.get("revision"),
        field_path="model.weights.source.revision",
    )
    served_model_name = _required_non_empty_string(
        manifest["vllm"]["engine_args"].get("model"),
        field_path="vllm.engine_args.model",
    )

    execution_image = runtime_execution.get("vllm_image")
    image_candidate = ""
    if isinstance(image, str) and image.strip():
        image_candidate = image.strip()
    elif isinstance(execution_image, str) and execution_image.strip():
        image_candidate = execution_image.strip()
    else:
        image_candidate = str(os.environ.get("VLLM_IMAGE", "")).strip()

    if not image_candidate:
        raise ValueError(
            "Missing vLLM image. Set runtime.execution.vllm_image in config, or pass --image, "
            "or set VLLM_IMAGE."
        )

    normalized_image = _normalize_pinned_image_reference(
        image_candidate,
        field_path="runtime.execution.vllm_image",
    )

    host_port = _base_url_port(base_url)
    container_name = _safe_container_name(
        _required_non_empty_string(manifest["metadata"].get("name"), field_path="metadata.name")
    )
    runtime_env = manifest["runtime"]["env"]
    cuda_env = manifest["cuda_stack"]["env"]
    vllm_env = manifest["vllm"]["env"]
    thread_settings = manifest["runtime"]["threads"]
    locale_settings = manifest["runtime"]["locale"]
    engine_args = manifest["vllm"]["engine_args"]
    batching = manifest["inference"]["batching"]

    env = dict(os.environ)
    env.update(
        {
            "VLLM_IMAGE": normalized_image,
            "MODEL_ID": model_id,
            "MODEL_REVISION": revision,
            "SERVED_MODEL_NAME": served_model_name,
            "VLLM_PORT": str(host_port),
            "VLLM_CONTAINER_PORT": str(container_port),
            "VLLM_HOST": "0.0.0.0",
            "CONTAINER_NAME": container_name,
            "VLLM_DTYPE": str(engine_args.get("dtype", "auto")),
            "VLLM_TENSOR_PARALLEL_SIZE": str(engine_args.get("tensor_parallel_size", 1)),
            "VLLM_PIPELINE_PARALLEL_SIZE": str(engine_args.get("pipeline_parallel_size", 1)),
            "VLLM_MAX_MODEL_LEN": str(engine_args.get("max_model_len", 32768)),
            "VLLM_MAX_NUM_SEQS": str(batching["max_num_seqs"]),
            "VLLM_MAX_NUM_BATCHED_TOKENS": str(batching["max_num_batched_tokens"]),
        }
    )
    for block in (runtime_env, cuda_env, vllm_env):
        for key, value in block.items():
            env[str(key)] = str(value)
    env["OMP_NUM_THREADS"] = str(thread_settings["omp_num_threads"])
    env["MKL_NUM_THREADS"] = str(thread_settings["mkl_num_threads"])
    env["LANG"] = str(locale_settings["lang"])
    env["LC_ALL"] = str(locale_settings["lang"])
    env["TZ"] = str(locale_settings["tz"])
    if (
        env.get("VLLM_BATCH_INVARIANT", "") == "1"
        and str(engine_args.get("dtype", "")).strip().lower() == "float16"
    ):
        # vLLM batch_invariant currently fails on fp16 log-softmax paths.
        env["VLLM_BATCH_INVARIANT"] = "0"
    if env.get("VLLM_BATCH_INVARIANT", "") == "1":
        attention_backend = str(env.get("VLLM_ATTENTION_BACKEND", "")).strip()
        if not attention_backend:
            env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    hf_home = env.get("HF_HOME", "").strip()
    if not hf_home:
        env["HF_HOME"] = "/data/hf"

    return ServePlan(
        base_url=base_url,
        compose_file=(_repo_root() / "docker-compose.yml").resolve(),
        env=env,
        image=normalized_image,
        model_id=model_id,
        revision=revision,
        served_model_name=served_model_name,
    )


def _image_digest(image_reference: str) -> str:
    _, digest = image_reference.rsplit("@", 1)
    return digest.lower()


def _verify_local_image_digest(*, image_reference: str, env: dict[str, str]) -> None:
    expected_digest = _image_digest(image_reference)

    result = subprocess.run(
        ["docker", "image", "inspect", image_reference, "--format", "{{json .RepoDigests}}"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            "Unable to inspect local image for digest verification. "
            f"image={image_reference} stderr={stderr!r}"
        )

    try:
        repo_digests = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to parse docker image inspect output for {image_reference}."
        ) from exc

    if not isinstance(repo_digests, list):
        raise RuntimeError(
            f"Unexpected docker image inspect format for {image_reference}: {repo_digests!r}"
        )

    actual_digests = {
        str(entry).split("@", 1)[1].lower()
        for entry in repo_digests
        if isinstance(entry, str) and "@" in entry
    }
    if expected_digest not in actual_digests:
        raise RuntimeError(
            "Local image digest does not match required manifest image digest.\n"
            f"image={image_reference}\n"
            f"expected={expected_digest}\n"
            f"actual={sorted(actual_digests)}"
        )


def run_serve_plan(plan: ServePlan, *, pull: bool = False) -> None:
    if pull:
        subprocess.run(
            ["docker", "pull", plan.image],
            check=True,
            env=plan.env,
        )

    _verify_local_image_digest(image_reference=plan.image, env=plan.env)

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(plan.compose_file),
            "up",
            "-d",
            "vllm",
        ],
        check=True,
        env=plan.env,
    )


def list_active_serve_containers(*, env: dict[str, str] | None = None) -> list[ActiveServeContainer]:
    effective_env = os.environ if env is None else env
    result = subprocess.run(
        [
            "docker",
            "ps",
            "--filter",
            "label=com.docker.compose.service=vllm",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=effective_env,
    )
    containers: list[ActiveServeContainer] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        containers.append(
            ActiveServeContainer(
                container_id=parts[0],
                name=parts[1],
                image=parts[2],
                status=parts[3],
                ports=parts[4],
            )
        )
    return containers


def kill_serve_containers(
    *,
    names: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    targets = [name.strip() for name in (names or []) if name.strip()]
    if not targets:
        targets = [container.name for container in list_active_serve_containers(env=env)]
    if not targets:
        return []

    effective_env = os.environ if env is None else env
    subprocess.run(
        ["docker", "rm", "-f", *targets],
        check=True,
        env=effective_env,
    )
    return targets


def wait_for_openai_server(
    *,
    base_url: str,
    timeout_seconds: int = 180,
    poll_interval_seconds: float = 2.0,
) -> bool:
    deadline = time.time() + float(timeout_seconds)
    url = f"{base_url.rstrip('/')}/v1/models"

    while time.time() < deadline:
        try:
            request = urllib.request.Request(url=url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval_seconds)

    return False
