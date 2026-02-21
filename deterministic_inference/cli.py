from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import bootstrap_manifest_digests
from .common import _normalize_real_digest, _repo_root, _utc_now, _write_json
from .execution import execute_run
from .locking import _validate_lock_payload, build_lock_payload, build_runtime_payload, load_lock, resolve_lock_path
from .schema import compute_manifest_id, create_manifest_template, load_manifest, validate_manifest
from .serving import build_serve_plan, run_serve_plan, wait_for_openai_server
from .verification import create_bundle_archive, inspect_payload, verify_bundles

def _parse_manifest_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    return path


def _parse_output_path(raw: str) -> Path:
    return _parse_manifest_path(raw)


def _most_recent_bundle_paths(*, count: int = 2) -> list[Path]:
    runs_root = (_repo_root() / "runs").resolve()
    if not runs_root.is_dir():
        return []

    bundles = sorted(
        runs_root.glob("*/*/bundle.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return bundles[:count]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spec-first deterministic vLLM workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a manifest template.")
    init_parser.add_argument("--output", required=True, help="Output manifest path.")
    init_parser.add_argument("--model-id", default="org/model-name")
    init_parser.add_argument("--model-revision", default="UNSET")

    lock_parser = subparsers.add_parser("lock", help="Resolve artifacts into a lockfile.")
    lock_parser.add_argument("--config", required=True, help="Manifest path.")
    lock_parser.add_argument("--output", default="", help="Override lock output path.")

    build_parser = subparsers.add_parser("build", help="Compute runtime closure digest.")
    build_parser.add_argument("--config", required=True, help="Manifest path.")
    build_parser.add_argument("--output", default="", help="Output build metadata path.")
    build_parser.add_argument(
        "--update-lock",
        action="store_true",
        help="Also refresh lockfile runtime_closure_digest.",
    )

    run_parser = subparsers.add_parser("run", help="Execute deterministic run and emit bundle.")
    run_parser.add_argument("--config", required=True, help="Manifest path.")
    run_parser.add_argument("--run-dir", default="", help="Override run directory.")
    run_parser.add_argument("--token-output", default="", help="Optional token output override path.")
    run_parser.add_argument("--run-log-output", default="", help="Optional run log output override path.")
    run_parser.add_argument(
        "--no-verify-artifact-digests",
        action="store_true",
        help="Disable pre-run artifact digest verification against the lockfile.",
    )

    serve_parser = subparsers.add_parser("serve", help="Start a vLLM server using docker compose.")
    serve_parser.add_argument("--config", required=True, help="Manifest path.")
    serve_parser.add_argument("--image", default="", help="vLLM image (or set VLLM_IMAGE).")
    serve_parser.add_argument(
        "--container-port",
        type=int,
        default=8000,
        help="Container port for vLLM service.",
    )
    serve_parser.add_argument("--pull", action="store_true", help="Pull image before starting.")
    serve_parser.add_argument("--dry-run", action="store_true", help="Print plan without starting server.")
    serve_parser.add_argument("--no-wait", action="store_true", help="Do not wait for /v1/models readiness.")
    serve_parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        default=180,
        help="Maximum wait time for server readiness.",
    )

    bootstrap_parser = subparsers.add_parser(
        "digest-bootstrap",
        help="Populate missing manifest digests for lock/run bootstrap.",
    )
    bootstrap_parser.add_argument("--config", required=True, help="Input manifest path.")
    bootstrap_parser.add_argument("--output", default="", help="Optional output manifest path.")
    bootstrap_parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write updates back to the input manifest.",
    )
    bootstrap_parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace existing digest values in addition to missing/unset digests.",
    )
    bootstrap_parser.add_argument(
        "--write-lock",
        action="store_true",
        help="Also write a lockfile after bootstrapping digests.",
    )

    verify_parser = subparsers.add_parser("verify", help="Verify two run bundles.")
    verify_parser.add_argument("--bundle-a", default="", help="Bundle A path (file or run dir).")
    verify_parser.add_argument("--bundle-b", default="", help="Bundle B path (file or run dir).")
    verify_parser.add_argument("--output-dir", default="", help="Output directory for verify report.")

    bundle_parser = subparsers.add_parser("bundle", help="Archive a run directory.")
    bundle_parser.add_argument("--run-dir", required=True, help="Run directory containing bundle.json.")
    bundle_parser.add_argument("--output", required=True, help="Output .tar.gz path.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect manifest/lock/bundle payload.")
    inspect_parser.add_argument("--input", required=True, help="Input JSON path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        output_path = _parse_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = create_manifest_template(
            model_id=str(args.model_id),
            model_revision=str(args.model_revision),
            output_path=output_path,
        )
        validate_manifest(manifest)
        _write_json(output_path, manifest)
        print(f"Wrote manifest template: {output_path}")
        return 0

    if args.command == "lock":
        manifest_path = _parse_manifest_path(args.config)
        manifest = load_manifest(manifest_path)

        if args.output:
            output_path = _parse_output_path(args.output)
        else:
            lock_ref = manifest["artifacts"]["lockfile"]
            if isinstance(lock_ref, str):
                output_path = resolve_lock_path(manifest, manifest_path)
            else:
                output_path = (
                    _repo_root() / "state" / "locks" / f"{manifest_path.stem}.embedded.lock.json"
                ).resolve()
        lock_payload = build_lock_payload(manifest, manifest_path=manifest_path)
        _write_json(output_path, lock_payload)
        print(f"Wrote lockfile: {output_path}")
        print(f"manifest_id: {lock_payload['manifest_id']}")
        print(f"lock_id: {lock_payload['lock_id']}")
        return 0

    if args.command == "build":
        manifest_path = _parse_manifest_path(args.config)
        manifest = load_manifest(manifest_path)
        runtime_payload = build_runtime_payload(manifest)

        if args.output:
            output_path = _parse_output_path(args.output)
        else:
            output_path = (_repo_root() / "state" / "build" / f"{manifest_path.stem}.build.json").resolve()

        _write_json(output_path, runtime_payload)
        print(f"Wrote build metadata: {output_path}")
        print(f"runtime_closure_digest: {runtime_payload['runtime_closure_digest']}")

        if args.update_lock:
            lock_ref = manifest["artifacts"]["lockfile"]
            if isinstance(lock_ref, str):
                lock_path = resolve_lock_path(manifest, manifest_path)
                lock_payload = build_lock_payload(
                    manifest,
                    manifest_path=manifest_path,
                    runtime_closure_digest=runtime_payload["runtime_closure_digest"],
                )
                _write_json(lock_path, lock_payload)
                print(f"Updated lockfile: {lock_path}")
            else:
                print(
                    "Skipped --update-lock because manifest.artifacts.lockfile is embedded. "
                    "Use --output with lock command to materialize a lock file."
                )

        return 0

    if args.command == "run":
        manifest_path = _parse_manifest_path(args.config)
        manifest = load_manifest(manifest_path)
        lock_path_for_run: Path | None = None
        lock_ref = manifest["artifacts"]["lockfile"]
        if isinstance(lock_ref, str):
            lock_path_for_run = resolve_lock_path(manifest, manifest_path)
            if not lock_path_for_run.is_file():
                raise SystemExit(
                    "Missing lockfile for manifest. Run lock first:\n"
                    f"  python -m deterministic_inference.cli lock --config {manifest_path}"
                )
            lock = load_lock(lock_path_for_run)
        else:
            lock = _validate_lock_payload(
                lock_ref,
                source=f"{manifest_path}:embedded_lock",
            )

        expected_manifest_id = compute_manifest_id(manifest)
        lock_manifest_id = _normalize_real_digest(lock["manifest_id"], field_path="lock.manifest_id")
        if lock_manifest_id != expected_manifest_id:
            raise SystemExit(
                "Lockfile does not match manifest_id. Regenerate with lock command.\n"
                f"  expected manifest_id: {expected_manifest_id}\n"
                f"  lock manifest_id:     {lock_manifest_id}"
            )

        run_dir_override = _parse_output_path(args.run_dir) if args.run_dir else None
        token_output_override = _parse_output_path(args.token_output) if args.token_output else None
        run_log_override = _parse_output_path(args.run_log_output) if args.run_log_output else None

        result = execute_run(
            manifest,
            manifest_path=manifest_path,
            lock=lock,
            verify_artifact_digests=not args.no_verify_artifact_digests,
            token_output_override=token_output_override,
            run_log_override=run_log_override,
            run_dir_override=run_dir_override,
            lock_path=lock_path_for_run,
        )
        print(f"Run directory: {result['run_dir']}")
        print(f"Bundle: {result['bundle_path']}")
        print(f"Token output: {result['token_output_path']}")
        print(f"Run log: {result['run_log_path']}")
        print(f"Determinism grade: {result['grade']}")
        return 0

    if args.command == "serve":
        manifest_path = _parse_manifest_path(args.config)
        manifest = load_manifest(manifest_path)
        image_override = str(args.image).strip() or None

        if args.container_port <= 0:
            raise SystemExit("--container-port must be > 0.")
        if args.wait_timeout_seconds <= 0:
            raise SystemExit("--wait-timeout-seconds must be > 0.")

        plan = build_serve_plan(
            manifest,
            image=image_override,
            container_port=int(args.container_port),
        )

        if args.dry_run:
            print("Serve plan:")
            print(f"  compose_file: {plan.compose_file}")
            print(f"  base_url: {plan.base_url}")
            print(f"  image: {plan.image}")
            print(f"  model_id: {plan.model_id}")
            print(f"  revision: {plan.revision}")
            print(f"  served_model_name: {plan.served_model_name}")
            print(f"  host_port: {plan.env['VLLM_PORT']}")
            print(f"  container_port: {plan.env['VLLM_CONTAINER_PORT']}")
            return 0

        run_serve_plan(plan, pull=bool(args.pull))
        print(f"Started vLLM service for model '{plan.model_id}' at {plan.base_url}")

        if not bool(args.no_wait):
            ready = wait_for_openai_server(
                base_url=plan.base_url,
                timeout_seconds=int(args.wait_timeout_seconds),
            )
            if not ready:
                print(
                    "Server did not become ready within timeout. "
                    f"Check container logs and endpoint: {plan.base_url}/v1/models"
                )
                return 1
            print(f"Server is ready: {plan.base_url}/v1/models")

        return 0

    if args.command == "digest-bootstrap":
        if args.in_place and args.output:
            raise SystemExit("Use either --in-place or --output, not both.")

        manifest_path = _parse_manifest_path(args.config)
        manifest = load_manifest(manifest_path)
        updated_manifest, report = bootstrap_manifest_digests(
            manifest,
            manifest_path=manifest_path,
            replace_existing=bool(args.replace_existing),
        )

        if args.in_place:
            output_path = manifest_path
        elif args.output:
            output_path = _parse_output_path(args.output)
        else:
            output_path = manifest_path.with_name(
                f"{manifest_path.stem}.bootstrapped{manifest_path.suffix or '.json'}"
            )

        validate_manifest(updated_manifest)
        _write_json(output_path, updated_manifest)

        print(f"Wrote bootstrapped manifest: {output_path}")
        print(f"updated_count: {report['updated_count']}")
        print(f"unchanged_count: {report['unchanged_count']}")
        print(f"updated_from_local_files: {report['source_counts']['local_file']}")
        print(f"updated_synthetic: {report['source_counts']['synthetic']}")

        if args.write_lock:
            lock_ref = updated_manifest["artifacts"]["lockfile"]
            if isinstance(lock_ref, str):
                lock_path = resolve_lock_path(updated_manifest, output_path)
            else:
                lock_path = (
                    _repo_root() / "state" / "locks" / f"{output_path.stem}.embedded.lock.json"
                ).resolve()
            lock_payload = build_lock_payload(updated_manifest, manifest_path=output_path)
            _write_json(lock_path, lock_payload)
            print(f"Wrote lockfile: {lock_path}")
            print(f"manifest_id: {lock_payload['manifest_id']}")
            print(f"lock_id: {lock_payload['lock_id']}")

        return 0

    if args.command == "verify":
        bundle_a_raw = str(args.bundle_a).strip()
        bundle_b_raw = str(args.bundle_b).strip()

        if bundle_a_raw and bundle_b_raw:
            bundle_a = _parse_output_path(bundle_a_raw)
            bundle_b = _parse_output_path(bundle_b_raw)
        elif not bundle_a_raw and not bundle_b_raw:
            recent = _most_recent_bundle_paths(count=2)
            if len(recent) < 2:
                raise SystemExit(
                    "Unable to auto-select bundles: need at least two run bundles under "
                    f"{(_repo_root() / 'runs').resolve()}.\n"
                    "Provide --bundle-a and --bundle-b explicitly."
                )
            bundle_a, bundle_b = recent[0], recent[1]
            print(f"Auto-selected bundle-a: {bundle_a}")
            print(f"Auto-selected bundle-b: {bundle_b}")
        else:
            raise SystemExit(
                "Provide both --bundle-a and --bundle-b, or omit both to auto-select the two most recent runs."
            )

        if args.output_dir:
            output_dir = _parse_output_path(args.output_dir)
        else:
            timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
            output_dir = (_repo_root() / "state" / "verify" / timestamp).resolve()

        result = verify_bundles(bundle_a, bundle_b, output_dir=output_dir)
        print(f"verify_report: {result['report_path']}")
        print(f"verify_summary: {result['summary_path']}")
        print(f"determinism_grade: {result['grade']}")
        return 0 if result["grade"] == "conformant" else 1

    if args.command == "bundle":
        run_dir = _parse_output_path(args.run_dir)
        output = _parse_output_path(args.output)
        archive_path = create_bundle_archive(run_dir=run_dir, output_path=output)
        print(f"Wrote archive: {archive_path}")
        return 0

    if args.command == "inspect":
        payload_path = _parse_output_path(args.input)
        result = inspect_payload(payload_path)
        print(json.dumps(result, indent=2))
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
