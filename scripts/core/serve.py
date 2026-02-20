#!/usr/bin/env python3
"""Config-driven model serving workflow for reproducible vLLM runs."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from integrity_utils import (
    compare_manifest_entries,
    load_manifest_entries,
    resolve_manifest_template,
    snapshot_dir as resolve_snapshot_dir,
    snapshot_manifest_lines,
    snapshot_manifest_entries,
)
from profile_config import (
    UNSET_LOCK_VALUES,
    ServeProfile,
    is_image_digest_pinned,
    load_profile,
    write_rendered_files,
)


DEFAULT_CONFIG = "configs/qwen3-235b-a22b-instruct-2507.json"


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _container_state(container_name: str) -> str | None:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    state = result.stdout.strip()
    return state if state else None


def _print_container_logs(container_name: str, *, tail: int = 120) -> None:
    result = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown docker error"
        print(
            f"Could not read logs for container '{container_name}': {detail}",
            file=sys.stderr,
        )
        return
    print(
        f"Recent logs from container '{container_name}' (tail={tail}):",
        file=sys.stderr,
    )
    print(result.stdout.rstrip(), file=sys.stderr)


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _build_docker_env(
    profile: ServeProfile,
    secrets_file: Path | None,
    *,
    enforce_required: bool = True,
) -> dict[str, str]:
    env = os.environ.copy()
    if secrets_file is not None and secrets_file.is_file():
        env.update(_load_env_file(secrets_file))

    if enforce_required:
        missing = [
            key
            for key in profile.runtime.required_secret_env
            if not env.get(key)
        ]
        if missing:
            joined = ", ".join(missing)
            raise SystemExit(
                f"Missing required secret env var(s): {joined}. "
                "Set them in your shell or pass --secrets-file."
            )
    return env


def _query_latest_model_sha(model_id: str) -> tuple[str, str]:
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to query Hugging Face API: {exc}") from exc

    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha:
        raise SystemExit("Could not read model sha from Hugging Face API response.")

    locked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return sha, locked_at


def _upsert_model_lock(config_path: Path, revision: str, locked_at: str) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"Invalid config shape at {config_path}; expected top-level object.")

    model = raw.get("model")
    if not isinstance(model, dict):
        model = {}
        raw["model"] = model

    model["revision"] = revision
    model["locked_at_utc"] = locked_at
    config_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _upsert_runtime_image(config_path: Path, image_ref: str) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"Invalid config shape at {config_path}; expected top-level object.")

    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        raw["runtime"] = runtime

    runtime["image"] = image_ref
    config_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_revision_locked(profile: ServeProfile) -> None:
    if profile.model.revision in UNSET_LOCK_VALUES:
        raise SystemExit(
            "Model revision is not pinned. Run:\n"
            f"  ./scripts/workflow.sh lock-model --config {profile.config_path}"
        )


def _prepare_dirs(profile: ServeProfile) -> None:
    hf_cache = profile.root_dir / profile.runtime.paths.hf_cache
    artifacts = profile.root_dir / profile.runtime.paths.artifacts
    (artifacts / "manifests").mkdir(parents=True, exist_ok=True)
    (artifacts / "samples").mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    profile.generated_dir.mkdir(parents=True, exist_ok=True)


def _snapshot_dir(profile: ServeProfile) -> Path:
    return resolve_snapshot_dir(
        root_dir=profile.root_dir,
        hf_cache_rel=profile.runtime.paths.hf_cache,
        model_id=profile.model.model_id,
        revision=profile.model.revision,
    )


def _resolve_manifest_template(profile: ServeProfile, template: str) -> Path:
    try:
        return resolve_manifest_template(
            template=template,
            root_dir=profile.root_dir,
            profile_id=profile.profile_id,
            revision=profile.model.revision,
            model_id=profile.model.model_id,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _verify_snapshot_manifest(profile: ServeProfile, expected_template_or_path: str) -> int:
    _ensure_revision_locked(profile)
    snapshot_dir = _snapshot_dir(profile)
    if not snapshot_dir.is_dir():
        print(f"Snapshot directory not found: {snapshot_dir}", file=sys.stderr)
        print("Start the server once and wait for download completion first.", file=sys.stderr)
        return 1

    expected_path = _resolve_manifest_template(profile, expected_template_or_path)
    if not expected_path.is_file():
        print(f"Expected manifest not found: {expected_path}", file=sys.stderr)
        return 1

    try:
        expected_entries = load_manifest_entries(expected_path)
        actual_entries = snapshot_manifest_entries(snapshot_dir)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    diff = compare_manifest_entries(
        expected_entries=expected_entries,
        actual_entries=actual_entries,
    )
    missing_paths = diff.missing_paths
    extra_paths = diff.extra_paths
    changed_paths = diff.changed_paths

    if diff.is_match:
        print(f"Snapshot manifest verification passed: {expected_path}")
        print(f"Verified files: {len(actual_entries)}")
        return 0

    print(f"Snapshot manifest verification FAILED for profile '{profile.profile_id}'.", file=sys.stderr)
    print(f"Expected manifest: {expected_path}", file=sys.stderr)
    print(f"Missing files: {len(missing_paths)}", file=sys.stderr)
    print(f"Unexpected files: {len(extra_paths)}", file=sys.stderr)
    print(f"Digest mismatches: {len(changed_paths)}", file=sys.stderr)

    max_examples = 20
    if missing_paths:
        print("Missing file examples:", file=sys.stderr)
        for path in missing_paths[:max_examples]:
            print(f"  {path}", file=sys.stderr)
    if extra_paths:
        print("Unexpected file examples:", file=sys.stderr)
        for path in extra_paths[:max_examples]:
            print(f"  {path}", file=sys.stderr)
    if changed_paths:
        print("Digest mismatch examples:", file=sys.stderr)
        for path in changed_paths[:max_examples]:
            print(
                f"  {path}\n"
                f"    expected: {expected_entries[path]}\n"
                f"    actual:   {actual_entries[path]}",
                file=sys.stderr,
            )
    return 1


def cmd_render(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    compose_path, lock_path = write_rendered_files(profile)
    print(f"Rendered compose file: {compose_path}")
    print(f"Resolved profile lock: {lock_path}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    print(json.dumps(profile.to_resolved_dict(), indent=2, sort_keys=True))
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    _ensure_revision_locked(profile)
    if not is_image_digest_pinned(profile.runtime.image):
        raise SystemExit(
            "runtime.image is not pinned to an immutable digest. "
            "Run ./scripts/workflow.sh lock-image --config <config>."
        )
    _prepare_dirs(profile)
    compose_path, lock_path = write_rendered_files(profile)
    env = _build_docker_env(profile, args.secrets_file)

    cmd = [
        "docker",
        "compose",
        "-p",
        profile.runtime.compose_project_name,
        "-f",
        str(compose_path),
    ]
    if args.secrets_file is not None and args.secrets_file.is_file():
        cmd.extend(["--env-file", str(args.secrets_file)])
    cmd.extend(["up", "-d"])

    _run(cmd, env=env)
    print(f"Server started for profile '{profile.profile_id}'.")
    print(f"Compose file: {compose_path}")
    print(f"Resolved lock: {lock_path}")
    print("Use ./scripts/workflow.sh wait --config <profile.json> to wait for readiness.")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    compose_path, _ = write_rendered_files(profile)
    env = _build_docker_env(profile, args.secrets_file, enforce_required=False)

    cmd = [
        "docker",
        "compose",
        "-p",
        profile.runtime.compose_project_name,
        "-f",
        str(compose_path),
    ]
    if args.secrets_file is not None and args.secrets_file.is_file():
        cmd.extend(["--env-file", str(args.secrets_file)])
    cmd.append("down")

    _run(cmd, env=env)
    print(f"Server stopped for profile '{profile.profile_id}'.")
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    url = f"http://127.0.0.1:{profile.runtime.host_port}/v1/models"
    start_epoch = time.time()
    print(f"Waiting for {url} (timeout: {args.timeout_seconds}s)")
    attempt = 0
    last_status_line = ""
    next_status_at = 60

    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    should_verify = args.verify_manifest or profile.integrity.enforce_on_wait
                    if should_verify:
                        expected_manifest = args.expected_manifest or profile.integrity.expected_snapshot_manifest
                        if not expected_manifest:
                            print(
                                "Manifest verification requested but no expected manifest is configured.",
                                file=sys.stderr,
                            )
                            return 1
                        print("Server is ready. Running snapshot manifest verification...")
                        verify_rc = _verify_snapshot_manifest(profile, expected_manifest)
                        if verify_rc != 0:
                            return verify_rc
                    print("Server is ready.")
                    return 0
                last_status_line = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            last_status_line = f"HTTPError {exc.code}: {exc.reason}"
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            exc = sys.exc_info()[1]
            if exc is not None:
                last_status_line = f"{type(exc).__name__}: {exc}"

        container_state = _container_state(profile.runtime.container_name)
        if container_state is not None and container_state != "running":
            print(
                f"Container '{profile.runtime.container_name}' is '{container_state}' while waiting for readiness.",
                file=sys.stderr,
            )
            _print_container_logs(profile.runtime.container_name)
            return 1

        elapsed = int(time.time() - start_epoch)
        if elapsed >= next_status_at:
            status_suffix = f" last_error='{last_status_line}'" if last_status_line else ""
            print(f"Still waiting: elapsed={elapsed}s attempts={attempt}{status_suffix}")
            next_status_at += 60
        if elapsed >= args.timeout_seconds:
            print(f"Timed out after {elapsed}s waiting for server readiness.", file=sys.stderr)
            _print_container_logs(profile.runtime.container_name)
            return 1
        time.sleep(10)


def _post_json(url: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def cmd_smoke(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    url = f"http://127.0.0.1:{profile.runtime.host_port}/v1/completions"
    payload = {
        "model": profile.model.served_name,
        "prompt": profile.smoke_test.prompt,
        "max_tokens": profile.smoke_test.max_tokens,
        "temperature": profile.smoke_test.temperature,
        "seed": profile.smoke_test.seed,
    }

    try:
        response = _post_json(url, payload, timeout=120)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return 1

    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        print("Smoke test failed: response contained no choices.", file=sys.stderr)
        return 1

    text = str(choices[0].get("text", "")).strip()
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens", "unknown") if isinstance(usage, dict) else "unknown"
    print(f"Smoke test response: {text}")
    print(f"Completion tokens: {completion_tokens}")
    return 0


def cmd_hash(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    _ensure_revision_locked(profile)
    snapshot_dir = _snapshot_dir(profile)
    if not snapshot_dir.is_dir():
        print(f"Snapshot directory not found: {snapshot_dir}", file=sys.stderr)
        print("Start the server once and wait for download completion first.", file=sys.stderr)
        return 1

    if args.output:
        output_file = _resolve_manifest_template(profile, args.output)
    else:
        manifests_dir = profile.root_dir / profile.runtime.paths.artifacts / "manifests"
        output_file = manifests_dir / f"{profile.profile_id}_model_snapshot_{profile.model.revision}.sha256"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for line in _snapshot_manifest_lines(snapshot_dir):
            handle.write(f"{line}\n")

    print(f"Wrote snapshot manifest: {output_file}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    expected_manifest = args.expected_manifest or profile.integrity.expected_snapshot_manifest
    if not expected_manifest:
        print(
            "No expected manifest configured. Set integrity.expected_snapshot_manifest in profile "
            "or pass --expected-manifest.",
            file=sys.stderr,
        )
        return 1
    return _verify_snapshot_manifest(profile, expected_manifest)


def cmd_lock_model(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    sha, locked_at = _query_latest_model_sha(profile.model.model_id)
    print(f"Resolved model SHA for {profile.model.model_id}: {sha}")
    print(f"Locked at: {locked_at}")

    if args.write:
        _upsert_model_lock(profile.config_path, sha, locked_at)
        print(f"Updated {profile.config_path}")
    else:
        print("Dry run only. Pass --write to update config.")
    return 0


def _first_repo_digest(image: str) -> str:
    result = subprocess.run(
        ["docker", "manifest", "inspect", image, "--verbose"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if isinstance(payload, dict):
        candidates = [payload]
    elif isinstance(payload, list):
        candidates = [item for item in payload if isinstance(item, dict)]
    else:
        raise SystemExit(f"Unexpected manifest payload type for image: {image}")
    if not candidates:
        raise SystemExit(f"No manifest entries found for image: {image}")

    chosen = None
    for candidate in candidates:
        descriptor = candidate.get("Descriptor")
        if not isinstance(descriptor, dict):
            continue
        platform = descriptor.get("platform")
        if isinstance(platform, dict) and platform.get("os") == "linux" and platform.get("architecture") == "amd64":
            chosen = candidate
            break
    if chosen is None:
        chosen = candidates[0]

    descriptor = chosen.get("Descriptor")
    if not isinstance(descriptor, dict):
        raise SystemExit(f"Manifest entry missing descriptor for image: {image}")
    digest = descriptor.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise SystemExit(f"Could not resolve sha256 digest for image: {image}")

    ref = chosen.get("Ref")
    if not isinstance(ref, str) or not ref:
        ref = image
    repo = ref.split("@", 1)[0]
    last_slash = repo.rfind("/")
    last_colon = repo.rfind(":")
    if last_colon > last_slash:
        repo = repo[:last_colon]

    return f"{repo}@{digest}"


def cmd_lock_image(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    digest_ref = _first_repo_digest(profile.runtime.image)
    print(f"Resolved image digest: {digest_ref}")

    if args.write:
        _upsert_runtime_image(profile.config_path, digest_ref)
        print(f"Updated {profile.config_path}")
    else:
        print("Dry run only. Pass --write to update config.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Config-driven reproducible vLLM serving.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to profile JSON config (default: {DEFAULT_CONFIG})",
    )
    common.add_argument(
        "--secrets-file",
        default=".env",
        type=Path,
        help="Optional env file containing secret values (default: .env)",
    )

    render = subparsers.add_parser(
        "render",
        parents=[common],
        help="Render compose and resolved lock files.",
    )
    render.set_defaults(func=cmd_render)

    show = subparsers.add_parser("show", parents=[common], help="Print resolved profile JSON.")
    show.set_defaults(func=cmd_show)

    start = subparsers.add_parser("start", parents=[common], help="Start server for profile.")
    start.set_defaults(func=cmd_start)

    stop = subparsers.add_parser("stop", parents=[common], help="Stop server for profile.")
    stop.set_defaults(func=cmd_stop)

    wait = subparsers.add_parser("wait", parents=[common], help="Wait for readiness endpoint.")
    wait.add_argument("--timeout-seconds", type=int, default=7200)
    wait.add_argument(
        "--verify-manifest",
        action="store_true",
        help="After readiness, verify local snapshot against expected manifest.",
    )
    wait.add_argument(
        "--expected-manifest",
        default="",
        help=(
            "Override expected manifest path/template. Supports {profile_id}, {revision}, "
            "{model_id}, {model_id_slug}."
        ),
    )
    wait.set_defaults(func=cmd_wait)

    smoke = subparsers.add_parser("smoke", parents=[common], help="Run smoke completion request.")
    smoke.set_defaults(func=cmd_smoke)

    model_hash = subparsers.add_parser("hash", parents=[common], help="Hash model snapshot files.")
    model_hash.add_argument(
        "--output",
        default="",
        help=(
            "Output path/template for manifest. Supports {profile_id}, {revision}, "
            "{model_id}, {model_id_slug}."
        ),
    )
    model_hash.set_defaults(func=cmd_hash)

    verify = subparsers.add_parser(
        "verify",
        parents=[common],
        help="Strictly verify local model snapshot against expected manifest.",
    )
    verify.add_argument(
        "--expected-manifest",
        default="",
        help=(
            "Expected manifest path/template. Supports {profile_id}, {revision}, "
            "{model_id}, {model_id_slug}."
        ),
    )
    verify.set_defaults(func=cmd_verify)

    lock_model = subparsers.add_parser(
        "lock-model",
        parents=[common],
        help="Resolve latest model SHA from Hugging Face.",
    )
    lock_model.add_argument("--write", action="store_true", help="Update config in-place.")
    lock_model.set_defaults(func=cmd_lock_model)

    lock_image = subparsers.add_parser(
        "lock-image",
        parents=[common],
        help="Resolve image tag to immutable digest.",
    )
    lock_image.add_argument("--write", action="store_true", help="Update config in-place.")
    lock_image.set_defaults(func=cmd_lock_image)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
