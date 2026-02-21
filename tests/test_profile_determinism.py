from __future__ import annotations

import json
import platform
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import deterministic_inference as workflow

from test_helpers import load_manifest, lock_manifest, make_manifest


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _apply_real_artifact_digests(tmp_path: Path, manifest: dict) -> None:
    model_dir = tmp_path / "model-files"
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
        path = model_dir / filename
        path.write_text(content, encoding="utf-8")
        digests[filename] = _sha256_text(content)

    source = manifest["model"]["weights"]["source"]
    source["local_path"] = str(model_dir.resolve())
    manifest["model"]["weights"]["files"][0]["digest"] = digests["model.safetensors"]
    for file_entry in manifest["model"]["tokenizer"]["files"]:
        file_entry["digest"] = digests[str(file_entry["path"])]
    for file_entry in manifest["model"]["config"]["files"]:
        file_entry["digest"] = digests[str(file_entry["path"])]
    manifest["model"]["chat_template"]["file"]["digest"] = digests["chat_template.jinja"]

    for package in manifest["software_stack"]["python"]["packages"]:
        package["source"]["digest"] = _sha256_text(f"python-package:{package['name']}")

    for lib_name, payload in manifest["cuda_stack"]["userspace"].items():
        payload["digest"] = _sha256_text(f"cuda-lib:{lib_name}")


class TestLockBuildAndCli(unittest.TestCase):
    def test_lock_and_build_update_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            make_manifest(manifest_path)

            lock_rc = workflow.main(["lock", "--config", str(manifest_path)])
            self.assertEqual(lock_rc, 0)

            build_output = tmp_path / "build.json"
            build_rc = workflow.main(
                [
                    "build",
                    "--config",
                    str(manifest_path),
                    "--output",
                    str(build_output),
                    "--update-lock",
                ]
            )
            self.assertEqual(build_rc, 0)
            self.assertTrue(build_output.is_file())

            build_payload = json.loads(build_output.read_text(encoding="utf-8"))
            manifest = load_manifest(manifest_path)
            lock_path = workflow.resolve_lock_path(manifest, manifest_path)
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))

            self.assertEqual(
                lock_payload["runtime_closure_digest"],
                build_payload["runtime_closure_digest"],
            )

    def test_cli_init_lock_run_inspect_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "cli-manifest.json"

            init_rc = workflow.main(
                [
                    "init",
                    "--output",
                    str(manifest_path),
                    "--model-id",
                    "example/model",
                    "--model-revision",
                    "rev1",
                ]
            )
            self.assertEqual(init_rc, 0)

            # Keep tests self-contained without external server.
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime"]["execution"]["backend"] = "mock"
            manifest["runtime"]["execution"].pop("base_url", None)
            manifest["runtime"]["execution"]["deterministic_failure_policy"] = "warn_only"
            manifest["hardware"]["constraints"] = {"cpu_arch": platform.machine()}
            manifest["artifacts"]["lockfile"] = str((tmp_path / "locks" / "cli-manifest.lock.json").resolve())
            _apply_real_artifact_digests(tmp_path, manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            lock_rc = workflow.main(["lock", "--config", str(manifest_path)])
            self.assertEqual(lock_rc, 0)

            run_dir = tmp_path / "run-1"
            run_rc = workflow.main(
                [
                    "run",
                    "--config",
                    str(manifest_path),
                    "--run-dir",
                    str(run_dir),
                ]
            )
            self.assertEqual(run_rc, 0)
            self.assertTrue((run_dir / "bundle.json").is_file())

            inspect_manifest_rc = workflow.main(["inspect", "--input", str(manifest_path)])
            inspect_bundle_rc = workflow.main(["inspect", "--input", str(run_dir / "bundle.json")])
            self.assertEqual(inspect_manifest_rc, 0)
            self.assertEqual(inspect_bundle_rc, 0)

            archive_path = tmp_path / "run-1.tar.gz"
            bundle_rc = workflow.main(
                [
                    "bundle",
                    "--run-dir",
                    str(run_dir),
                    "--output",
                    str(archive_path),
                ]
            )
            self.assertEqual(bundle_rc, 0)
            self.assertTrue(archive_path.is_file())

    def test_verify_returns_non_zero_for_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)
            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))

            run_a = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=tmp_path / "run-a",
            )
            run_b = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=tmp_path / "run-b",
            )

            tokens_b_path = Path(run_b["token_output_path"])
            tokens_b = json.loads(tokens_b_path.read_text(encoding="utf-8"))
            tokens_b["sequences"][0]["output_token_ids"][0] += 1
            tokens_b_path.write_text(json.dumps(tokens_b, indent=2) + "\n", encoding="utf-8")

            verify_rc = workflow.main(
                [
                    "verify",
                    "--bundle-a",
                    str(run_a["bundle_path"]),
                    "--bundle-b",
                    str(run_b["bundle_path"]),
                    "--output-dir",
                    str(tmp_path / "verify"),
                ]
            )
            self.assertEqual(verify_rc, 1)

    def test_run_with_embedded_lock_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            make_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            lock_payload = workflow.build_lock_payload(manifest, manifest_path=manifest_path)
            manifest["artifacts"]["lockfile"] = lock_payload
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            run_dir = tmp_path / "run-embedded"
            run_rc = workflow.main(
                [
                    "run",
                    "--config",
                    str(manifest_path),
                    "--run-dir",
                    str(run_dir),
                ]
            )
            self.assertEqual(run_rc, 0)
            bundle = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle["determinism_grade"], "conformant")
            run_log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
            self.assertEqual(run_log["generation_lock_manifest"], "<embedded_lock>")

    def test_cli_run_flag_disables_digest_integrity_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            local_model_dir = Path(manifest["model"]["weights"]["source"]["local_path"])
            (local_model_dir / "tokenizer.json").write_text(
                "{\"tokenizer\":\"tampered\"}",
                encoding="utf-8",
            )

            run_dir = tmp_path / "run-no-integrity"
            run_rc = workflow.main(
                [
                    "run",
                    "--config",
                    str(manifest_path),
                    "--run-dir",
                    str(run_dir),
                    "--no-verify-artifact-digests",
                ]
            )
            self.assertEqual(run_rc, 0)
            bundle = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle["artifact_integrity"]["enabled"], False)

    def test_verify_auto_selects_two_most_recent_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))

            run_a_dir = tmp_path / "runs" / "auto" / "run-a"
            run_b_dir = tmp_path / "runs" / "auto" / "run-b"
            run_a = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=run_a_dir,
            )
            run_b = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=run_b_dir,
            )

            report_dir = tmp_path / "verify-auto"
            with mock.patch("deterministic_inference.cli._repo_root", return_value=tmp_path):
                verify_rc = workflow.main(
                    [
                        "verify",
                        "--output-dir",
                        str(report_dir),
                    ]
                )

            self.assertEqual(verify_rc, 0)
            report = json.loads((report_dir / "verify_report.json").read_text(encoding="utf-8"))
            selected = {
                str(report["bundle_a"]["path"]),
                str(report["bundle_b"]["path"]),
            }
            expected = {
                str(run_a["bundle_path"]),
                str(run_b["bundle_path"]),
            }
            self.assertEqual(selected, expected)


if __name__ == "__main__":
    unittest.main()
