from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .common import _repo_root, _write_json


DEFAULT_MODAL_APP_NAME = "deterministic-inference-vllm"
DEFAULT_MODAL_CLASS_NAME = "VllmServer"
DEFAULT_MODAL_GPU_TYPE = "H100"
DEFAULT_MODAL_MIN_CONTAINERS = 0
DEFAULT_MODAL_SCALEDOWN_WINDOW_SECONDS = 600

_STATE_FILE = (_repo_root() / "state" / "modal" / "servers.json").resolve()

_GPU_MODEL_HINTS = (
    "B300",
    "B200",
    "H200",
    "H100",
    "A100-80GB",
    "A100-40GB",
    "A100",
    "L40S",
    "L4",
    "A10",
    "T4",
)


@dataclass(frozen=True)
class ModalServePlan:
    app_name: str
    class_name: str
    config_name: str
    gpu: str
    min_containers: int | None
    scaledown_window_seconds: int | None
    app_file: Path


def _required_string(value: Any, *, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_path}: expected non-empty string.")
    return value.strip()


def _modal_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    runtime_execution = manifest.get("runtime", {}).get("execution", {})
    value = runtime_execution.get("x_modal")
    if isinstance(value, dict):
        return value
    return {}


def _first_gpu_model_hint(models: list[str]) -> str | None:
    for model in models:
        upper = model.upper()
        for hint in _GPU_MODEL_HINTS:
            if hint in upper:
                return hint
    return None


def infer_modal_gpu(manifest: dict[str, Any], *, override: str | None = None) -> str:
    if isinstance(override, str) and override.strip():
        return override.strip()

    settings = _modal_settings(manifest)
    gpu_type = str(settings.get("gpu_type", "")).strip()

    allowed_models = manifest.get("hardware", {}).get("constraints", {}).get("allowed_gpu_models", [])
    if isinstance(allowed_models, list):
        hints = [str(item) for item in allowed_models if isinstance(item, str)]
        chosen = _first_gpu_model_hint(hints)
        if chosen:
            gpu_type = chosen

    if not gpu_type:
        gpu_type = DEFAULT_MODAL_GPU_TYPE

    tensor_parallel = manifest.get("vllm", {}).get("engine_args", {}).get("tensor_parallel_size", 1)
    try:
        gpu_count = int(tensor_parallel)
    except (TypeError, ValueError):
        gpu_count = 1
    if gpu_count < 1:
        gpu_count = 1

    if gpu_count == 1:
        return gpu_type
    return f"{gpu_type}:{gpu_count}"


def _resolve_int(value: Any, *, default: int | None, field_path: str) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field_path}: expected int >= 0.")
        return value
    raise ValueError(f"{field_path}: expected int >= 0 when provided.")


def _modal_app_file() -> Path:
    return (_repo_root() / "deterministic_inference" / "modal_app.py").resolve()


def build_modal_serve_plan(
    manifest: dict[str, Any],
    *,
    app_name: str | None = None,
    class_name: str | None = None,
    gpu: str | None = None,
    min_containers: int | None = None,
    scaledown_window_seconds: int | None = None,
) -> ModalServePlan:
    config_name = _required_string(manifest.get("metadata", {}).get("name"), field_path="metadata.name")
    settings = _modal_settings(manifest)

    resolved_app_name = str(app_name or settings.get("app_name") or DEFAULT_MODAL_APP_NAME).strip()
    if not resolved_app_name:
        resolved_app_name = DEFAULT_MODAL_APP_NAME

    resolved_class_name = str(class_name or settings.get("class_name") or DEFAULT_MODAL_CLASS_NAME).strip()
    if not resolved_class_name:
        resolved_class_name = DEFAULT_MODAL_CLASS_NAME

    resolved_min = _resolve_int(
        min_containers if min_containers is not None else settings.get("min_containers"),
        default=DEFAULT_MODAL_MIN_CONTAINERS,
        field_path="runtime.execution.x_modal.min_containers",
    )
    resolved_scaledown = _resolve_int(
        scaledown_window_seconds
        if scaledown_window_seconds is not None
        else settings.get("scaledown_window_seconds"),
        default=DEFAULT_MODAL_SCALEDOWN_WINDOW_SECONDS,
        field_path="runtime.execution.x_modal.scaledown_window_seconds",
    )

    return ModalServePlan(
        app_name=resolved_app_name,
        class_name=resolved_class_name,
        config_name=config_name,
        gpu=infer_modal_gpu(manifest, override=gpu),
        min_containers=resolved_min,
        scaledown_window_seconds=resolved_scaledown,
        app_file=_modal_app_file(),
    )


def deploy_modal_app(plan: ModalServePlan) -> None:
    if not plan.app_file.is_file():
        raise FileNotFoundError(f"Missing Modal app file: {plan.app_file}")

    subprocess.run(
        [
            "modal",
            "deploy",
            str(plan.app_file),
            "--name",
            plan.app_name,
        ],
        check=True,
    )


def _append_query(url: str, *, params: dict[str, str]) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    encoded = urlencode(query)
    return urlunparse(parsed._replace(query=encoded))


def _import_modal() -> Any:
    try:
        import modal  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Modal SDK is required for Modal workflows. Install it with 'pip install modal' and run 'modal token new'."
        ) from exc
    return modal


def lookup_modal_endpoint(
    plan: ModalServePlan,
    *,
    apply_autoscaler: bool,
) -> str:
    modal = _import_modal()

    cls_obj = modal.Cls.from_name(plan.app_name, plan.class_name)
    with_options_kwargs: dict[str, Any] = {"gpu": plan.gpu}
    if plan.scaledown_window_seconds is not None:
        with_options_kwargs["scaledown_window"] = int(plan.scaledown_window_seconds)
    cls_obj = cls_obj.with_options(**with_options_kwargs)

    instance = cls_obj(config_name=plan.config_name)

    if apply_autoscaler and plan.min_containers is not None:
        update_kwargs: dict[str, Any] = {"min_containers": int(plan.min_containers)}
        if plan.scaledown_window_seconds is not None:
            update_kwargs["scaledown_window"] = int(plan.scaledown_window_seconds)
        instance.update_autoscaler(**update_kwargs)

    endpoint = instance.serve.get_web_url()
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise RuntimeError(
            "Modal endpoint URL lookup failed. Ensure the app is deployed and the class method is exposed with @modal.web_server."
        )

    return _append_query(endpoint.strip(), params={"config_name": plan.config_name})


def scale_down_modal_endpoint(plan: ModalServePlan) -> None:
    modal = _import_modal()

    cls_obj = modal.Cls.from_name(plan.app_name, plan.class_name)
    cls_obj = cls_obj.with_options(gpu=plan.gpu)
    instance = cls_obj(config_name=plan.config_name)
    instance.update_autoscaler(min_containers=0)


def _read_state() -> dict[str, Any]:
    if not _STATE_FILE.is_file():
        return {"servers": {}}
    payload = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"servers": {}}
    servers = payload.get("servers")
    if not isinstance(servers, dict):
        return {"servers": {}}
    return {"servers": servers}


def list_modal_servers() -> list[dict[str, Any]]:
    state = _read_state()
    servers = state["servers"]
    rows: list[dict[str, Any]] = []
    for key in sorted(servers.keys()):
        entry = servers[key]
        if not isinstance(entry, dict):
            continue
        rows.append(dict(entry))
    return rows


def upsert_modal_server(plan: ModalServePlan, *, url: str) -> None:
    state = _read_state()
    servers = dict(state["servers"])
    servers[plan.config_name] = {
        "config_name": plan.config_name,
        "app_name": plan.app_name,
        "class_name": plan.class_name,
        "gpu": plan.gpu,
        "min_containers": plan.min_containers,
        "scaledown_window_seconds": plan.scaledown_window_seconds,
        "url": url,
    }
    _write_json(_STATE_FILE, {"servers": servers})


def remove_modal_server(*, config_name: str) -> None:
    state = _read_state()
    servers = dict(state["servers"])
    servers.pop(config_name, None)
    _write_json(_STATE_FILE, {"servers": servers})


def plan_from_state(entry: dict[str, Any]) -> ModalServePlan:
    config_name = _required_string(entry.get("config_name"), field_path="state.config_name")
    app_name = _required_string(entry.get("app_name"), field_path="state.app_name")
    class_name = _required_string(entry.get("class_name"), field_path="state.class_name")
    gpu = _required_string(entry.get("gpu"), field_path="state.gpu")

    min_containers = entry.get("min_containers")
    scaledown_window_seconds = entry.get("scaledown_window_seconds")

    if min_containers is not None and not isinstance(min_containers, int):
        raise ValueError("state.min_containers: expected int when provided.")
    if scaledown_window_seconds is not None and not isinstance(scaledown_window_seconds, int):
        raise ValueError("state.scaledown_window_seconds: expected int when provided.")

    return ModalServePlan(
        app_name=app_name,
        class_name=class_name,
        config_name=config_name,
        gpu=gpu,
        min_containers=min_containers,
        scaledown_window_seconds=scaledown_window_seconds,
        app_file=_modal_app_file(),
    )
