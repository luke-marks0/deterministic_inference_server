#!/usr/bin/env python3
"""Shared profile loading/validation for config-driven model serving."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UNSET_LOCK_VALUES = {"", "UNSET", "UNSET_RUN_LOCK_SCRIPT"}


@dataclass(frozen=True)
class SampleDefaults:
    target_tokens: int
    chunk_max_tokens: int
    temperature: float
    top_p: float
    seed: int
    timeout_seconds: int


@dataclass(frozen=True)
class SmokeTestConfig:
    prompt: str
    max_tokens: int
    temperature: float
    seed: int


@dataclass(frozen=True)
class RuntimePaths:
    hf_cache: str
    artifacts: str


@dataclass(frozen=True)
class RuntimeConfig:
    image: str
    gpus: str
    ipc_mode: str
    restart: str
    host_port: int
    container_port: int
    api_host: str
    container_name: str
    compose_project_name: str
    required_secret_env: list[str]
    paths: RuntimePaths
    environment: dict[str, str]
    bootstrap_pip_packages: list[str]
    memlock: int
    stack: int


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    locked_at_utc: str
    served_name: str


@dataclass(frozen=True)
class IntegrityConfig:
    expected_snapshot_manifest: str
    enforce_on_wait: bool


@dataclass(frozen=True)
class ServeProfile:
    root_dir: Path
    config_path: Path
    schema_version: int
    profile_id: str
    description: str
    runtime: RuntimeConfig
    model: ModelConfig
    integrity: IntegrityConfig
    vllm_flags: list[str]
    smoke_test: SmokeTestConfig
    sample_defaults: SampleDefaults

    @property
    def generated_dir(self) -> Path:
        return self.root_dir / "state" / "generated" / self.profile_id

    @property
    def compose_file(self) -> Path:
        return self.generated_dir / "docker-compose.yml"

    @property
    def lock_file(self) -> Path:
        return self.generated_dir / "resolved_profile.json"

    @property
    def model_cache_path(self) -> str:
        return f"models--{self.model.model_id.replace('/', '--')}"

    def is_model_revision_locked(self) -> bool:
        return self.model.revision not in UNSET_LOCK_VALUES

    def to_resolved_dict(self) -> dict[str, Any]:
        command = build_vllm_command(self)
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "description": self.description,
            "config_path": str(self.config_path),
            "runtime": {
                "image": self.runtime.image,
                "gpus": self.runtime.gpus,
                "ipc_mode": self.runtime.ipc_mode,
                "restart": self.runtime.restart,
                "host_port": self.runtime.host_port,
                "container_port": self.runtime.container_port,
                "api_host": self.runtime.api_host,
                "container_name": self.runtime.container_name,
                "compose_project_name": self.runtime.compose_project_name,
                "required_secret_env": self.runtime.required_secret_env,
                "paths": {
                    "hf_cache": self.runtime.paths.hf_cache,
                    "artifacts": self.runtime.paths.artifacts,
                },
                "environment": self.runtime.environment,
                "bootstrap_pip_packages": self.runtime.bootstrap_pip_packages,
                "memlock": self.runtime.memlock,
                "stack": self.runtime.stack,
            },
            "model": {
                "id": self.model.model_id,
                "revision": self.model.revision,
                "locked_at_utc": self.model.locked_at_utc,
                "served_name": self.model.served_name,
            },
            "integrity": {
                "expected_snapshot_manifest": self.integrity.expected_snapshot_manifest,
                "enforce_on_wait": self.integrity.enforce_on_wait,
            },
            "vllm_command": command,
            "smoke_test": {
                "prompt": self.smoke_test.prompt,
                "max_tokens": self.smoke_test.max_tokens,
                "temperature": self.smoke_test.temperature,
                "seed": self.smoke_test.seed,
            },
            "sampling_defaults": {
                "target_tokens": self.sample_defaults.target_tokens,
                "chunk_max_tokens": self.sample_defaults.chunk_max_tokens,
                "temperature": self.sample_defaults.temperature,
                "top_p": self.sample_defaults.top_p,
                "seed": self.sample_defaults.seed,
                "timeout_seconds": self.sample_defaults.timeout_seconds,
            },
        }


def _expect_table(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid object value for '{key}'.")
    return value


def _expect_str(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid string value for '{key}'.")
    return value.strip()


def _expect_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Missing or invalid integer value for '{key}'.")
    return value


def _expect_number(parent: dict[str, Any], key: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing or invalid numeric value for '{key}'.")
    return float(value)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_profile(config_path: str | Path) -> ServeProfile:
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Config file not found: {config}")

    try:
        with config.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config at {config}: {exc}") from exc

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be set to 1.")

    profile_table = _expect_table(raw, "profile")
    profile_id = profile_table.get("id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        profile_id = _slugify(config.stem)
    else:
        profile_id = _slugify(profile_id)

    description = str(profile_table.get("description", "")).strip()

    runtime_table = _expect_table(raw, "runtime")
    runtime_paths_table = _expect_table(runtime_table, "paths")
    runtime_env_table = _expect_table(runtime_table, "environment")

    required_secret_env = runtime_table.get("required_secret_env", [])
    if not isinstance(required_secret_env, list) or any(
        not isinstance(item, str) or not item for item in required_secret_env
    ):
        raise ValueError("runtime.required_secret_env must be a list of non-empty strings.")

    runtime_environment: dict[str, str] = {}
    for key, value in runtime_env_table.items():
        if not isinstance(value, (str, int, float)):
            raise ValueError(
                f"runtime.environment.{key} must be a string, int, or float scalar."
            )
        runtime_environment[str(key)] = str(value)

    bootstrap_pip_packages = runtime_table.get("bootstrap_pip_packages", [])
    if not isinstance(bootstrap_pip_packages, list) or any(
        not isinstance(item, str) or not item.strip() for item in bootstrap_pip_packages
    ):
        raise ValueError("runtime.bootstrap_pip_packages must be a list of non-empty strings.")

    runtime = RuntimeConfig(
        image=_expect_str(runtime_table, "image"),
        gpus=str(runtime_table.get("gpus", "all")),
        ipc_mode=str(runtime_table.get("ipc_mode", "host")),
        restart=str(runtime_table.get("restart", "unless-stopped")),
        host_port=_expect_int(runtime_table, "host_port"),
        container_port=_expect_int(runtime_table, "container_port"),
        api_host=str(runtime_table.get("api_host", "0.0.0.0")),
        container_name=_expect_str(runtime_table, "container_name"),
        compose_project_name=_expect_str(runtime_table, "compose_project_name"),
        required_secret_env=required_secret_env,
        paths=RuntimePaths(
            hf_cache=_expect_str(runtime_paths_table, "hf_cache"),
            artifacts=_expect_str(runtime_paths_table, "artifacts"),
        ),
        environment=runtime_environment,
        bootstrap_pip_packages=[item.strip() for item in bootstrap_pip_packages],
        memlock=int(runtime_table.get("memlock", -1)),
        stack=int(runtime_table.get("stack", 67_108_864)),
    )

    model_table = _expect_table(raw, "model")
    model = ModelConfig(
        model_id=_expect_str(model_table, "id"),
        revision=_expect_str(model_table, "revision"),
        locked_at_utc=str(model_table.get("locked_at_utc", "UNSET")),
        served_name=_expect_str(model_table, "served_name"),
    )

    integrity_table = raw.get("integrity", {})
    if not isinstance(integrity_table, dict):
        raise ValueError("integrity must be an object if provided.")
    integrity = IntegrityConfig(
        expected_snapshot_manifest=str(
            integrity_table.get("expected_snapshot_manifest", "")
        ).strip(),
        enforce_on_wait=bool(integrity_table.get("enforce_on_wait", False)),
    )

    vllm_table = _expect_table(raw, "vllm")
    vllm_flags = vllm_table.get("flags")
    if not isinstance(vllm_flags, list) or any(
        not isinstance(flag, str) or not flag.strip() for flag in vllm_flags
    ):
        raise ValueError("vllm.flags must be a list of non-empty strings.")

    smoke_table = _expect_table(raw, "smoke_test")
    smoke_test = SmokeTestConfig(
        prompt=_expect_str(smoke_table, "prompt"),
        max_tokens=_expect_int(smoke_table, "max_tokens"),
        temperature=_expect_number(smoke_table, "temperature"),
        seed=_expect_int(smoke_table, "seed"),
    )

    sample_table = _expect_table(raw, "sampling_defaults")
    sample_defaults = SampleDefaults(
        target_tokens=_expect_int(sample_table, "target_tokens"),
        chunk_max_tokens=_expect_int(sample_table, "chunk_max_tokens"),
        temperature=_expect_number(sample_table, "temperature"),
        top_p=_expect_number(sample_table, "top_p"),
        seed=_expect_int(sample_table, "seed"),
        timeout_seconds=_expect_int(sample_table, "timeout_seconds"),
    )

    # profile_config.py now lives in scripts/core/, so repo root is two levels up.
    root_dir = Path(__file__).resolve().parents[2]

    return ServeProfile(
        root_dir=root_dir,
        config_path=config,
        schema_version=schema_version,
        profile_id=profile_id,
        description=description,
        runtime=runtime,
        model=model,
        integrity=integrity,
        vllm_flags=vllm_flags,
        smoke_test=smoke_test,
        sample_defaults=sample_defaults,
    )


def build_vllm_command(profile: ServeProfile) -> list[str]:
    command = [
        "--model",
        profile.model.model_id,
        "--revision",
        profile.model.revision,
        "--served-model-name",
        profile.model.served_name,
        "--host",
        profile.runtime.api_host,
        "--port",
        str(profile.runtime.container_port),
    ]
    command.extend(profile.vllm_flags)
    return command


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render_compose_yaml(profile: ServeProfile) -> str:
    command = build_vllm_command(profile)
    hf_cache_host = (profile.root_dir / profile.runtime.paths.hf_cache).resolve()
    artifacts_host = (profile.root_dir / profile.runtime.paths.artifacts).resolve()

    env_lines = []
    for key, value in profile.runtime.environment.items():
        env_lines.append(f"      {key}: {_yaml_string(value)}")
    for secret_key in profile.runtime.required_secret_env:
        env_lines.append(f"      {secret_key}: {_yaml_string('${' + secret_key + '}')}")

    command_lines = [f"      - {_yaml_string(item)}" for item in command]
    environment_block = ["    environment:", *env_lines] if env_lines else ["    environment: {}"]

    entrypoint_lines: list[str] = []
    if profile.runtime.bootstrap_pip_packages:
        pip_packages = " ".join(shlex.quote(pkg) for pkg in profile.runtime.bootstrap_pip_packages)
        serve_cmd = " ".join(shlex.quote(arg) for arg in command)
        bootstrap_command = (
            f"python3 -m pip install --no-cache-dir {pip_packages} && "
            f"python3 -m vllm.entrypoints.openai.api_server {serve_cmd}"
        )
        entrypoint_lines = [
            "    entrypoint:",
            f"      - {_yaml_string('/bin/bash')}",
            f"      - {_yaml_string('-lc')}",
            "    command:",
            f"      - {_yaml_string(bootstrap_command)}",
        ]
    else:
        entrypoint_lines = ["    command:", *command_lines]

    return "\n".join(
        [
            "services:",
            "  vllm:",
            f"    image: {_yaml_string(profile.runtime.image)}",
            f"    container_name: {_yaml_string(profile.runtime.container_name)}",
            f"    gpus: {_yaml_string(profile.runtime.gpus)}",
            f"    ipc: {_yaml_string(profile.runtime.ipc_mode)}",
            f"    restart: {_yaml_string(profile.runtime.restart)}",
            "    ulimits:",
            f"      memlock: {profile.runtime.memlock}",
            f"      stack: {profile.runtime.stack}",
            "    ports:",
            f"      - {_yaml_string(f'{profile.runtime.host_port}:{profile.runtime.container_port}')}",
            "    volumes:",
            f"      - {_yaml_string(f'{hf_cache_host}:/data/hf')}",
            f"      - {_yaml_string(f'{artifacts_host}:/artifacts')}",
            *environment_block,
            *entrypoint_lines,
            "",
        ]
    )


def write_rendered_files(profile: ServeProfile) -> tuple[Path, Path]:
    profile.generated_dir.mkdir(parents=True, exist_ok=True)

    compose_text = render_compose_yaml(profile)
    profile.compose_file.write_text(compose_text, encoding="utf-8")

    resolved = profile.to_resolved_dict()
    profile.lock_file.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")

    return profile.compose_file, profile.lock_file
