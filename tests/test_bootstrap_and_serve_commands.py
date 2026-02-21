from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import deterministic_inference as workflow
from deterministic_inference.serving import wait_for_openai_server


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _materialize_model_files(manifest_path: Path, manifest: dict) -> Path:
    model_dir = manifest_path.parent / "model-artifacts"
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
    for name, content in file_contents.items():
        (model_dir / name).write_text(content, encoding="utf-8")
    manifest["model"]["weights"]["source"]["local_path"] = str(model_dir.resolve())
    return model_dir


def _build_server_manifest(path: Path) -> dict:
    manifest = workflow.create_manifest_template(
        model_id="openai/gpt-oss-20b",
        model_revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        output_path=path,
    )
    manifest["runtime"]["execution"]["backend"] = "openai_compatible"
    manifest["runtime"]["execution"]["base_url"] = "http://127.0.0.1:8123"
    manifest["vllm"]["mode"] = "server"
    manifest["vllm"]["engine_args"]["model"] = "openai/gpt-oss-20b"
    manifest["model"]["weights"]["source"]["repo"] = "openai/gpt-oss-20b"
    manifest["artifacts"]["lockfile"] = str((path.parent / "locks" / "manifest.lock.json").resolve())
    _materialize_model_files(path, manifest)
    _write_json(path, manifest)
    return manifest


class TestBootstrapAndServeCommands(unittest.TestCase):
    def test_digest_bootstrap_populates_unset_digests_and_writes_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            _build_server_manifest(manifest_path)

            rc = workflow.main(
                [
                    "digest-bootstrap",
                    "--config",
                    str(manifest_path),
                    "--in-place",
                    "--write-lock",
                ]
            )
            self.assertEqual(rc, 0)

            manifest = workflow.load_manifest(manifest_path)
            digest_fields = []
            digest_fields.extend(manifest["model"]["weights"]["files"])
            digest_fields.extend(manifest["model"]["tokenizer"]["files"])
            digest_fields.extend(manifest["model"]["config"]["files"])
            digest_fields.append(manifest["model"]["chat_template"]["file"])
            for package in manifest["software_stack"]["python"]["packages"]:
                digest_fields.append(package["source"])
            for payload in manifest["cuda_stack"]["userspace"].values():
                digest_fields.append(payload)

            for field in digest_fields:
                digest = str(field.get("digest", "")).strip().lower()
                self.assertNotEqual(digest, "")
                self.assertNotEqual(digest, "sha256:unset")

            model_file = Path(manifest["model"]["weights"]["source"]["local_path"]) / "model.safetensors"
            expected = hashlib.sha256(model_file.read_bytes()).hexdigest()
            self.assertEqual(manifest["model"]["weights"]["files"][0]["digest"], expected)

            lock_path = workflow.resolve_lock_path(manifest, manifest_path)
            lock = workflow.load_lock(lock_path)
            self.assertEqual(lock["manifest_id"], workflow.compute_manifest_id(manifest))

    def test_serve_dry_run_and_command_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            _build_server_manifest(manifest_path)
            pinned_image = "docker.io/vllm/vllm-openai@sha256:" + ("1" * 64)

            dry_run_rc = workflow.main(
                [
                    "serve",
                    "--config",
                    str(manifest_path),
                    "--image",
                    pinned_image,
                    "--dry-run",
                ]
            )
            self.assertEqual(dry_run_rc, 0)

            with mock.patch("deterministic_inference.serving.subprocess.run") as run_mock:
                def _fake_run(cmd, **kwargs):
                    if cmd[:3] == ["docker", "image", "inspect"]:
                        return mock.Mock(
                            returncode=0,
                            stdout=json.dumps([pinned_image]),
                            stderr="",
                        )
                    return mock.Mock(returncode=0, stdout="", stderr="")

                run_mock.side_effect = _fake_run
                with mock.patch("deterministic_inference.cli.wait_for_openai_server", return_value=True):
                    run_rc = workflow.main(
                        [
                            "serve",
                            "--config",
                            str(manifest_path),
                            "--image",
                            pinned_image,
                            "--pull",
                        ]
                    )

            self.assertEqual(run_rc, 0)
            self.assertGreaterEqual(run_mock.call_count, 3)
            first_cmd = run_mock.call_args_list[0].args[0]
            second_cmd = run_mock.call_args_list[1].args[0]
            third_cmd = run_mock.call_args_list[2].args[0]
            self.assertEqual(first_cmd[:2], ["docker", "pull"])
            self.assertEqual(second_cmd[:3], ["docker", "image", "inspect"])
            self.assertEqual(third_cmd[:3], ["docker", "compose", "-f"])

    def test_serve_uses_manifest_image_when_flag_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            manifest = _build_server_manifest(manifest_path)
            pinned_image = "docker.io/vllm/vllm-openai@sha256:" + ("2" * 64)
            manifest["runtime"]["execution"]["vllm_image"] = pinned_image
            _write_json(manifest_path, manifest)

            dry_run_rc = workflow.main(
                [
                    "serve",
                    "--config",
                    str(manifest_path),
                    "--dry-run",
                ]
            )
            self.assertEqual(dry_run_rc, 0)


class _FakeHTTPResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


class TestServeReadinessWait(unittest.TestCase):
    def test_wait_retries_after_connection_reset(self) -> None:
        side_effects = [
            ConnectionResetError("connection reset"),
            _FakeHTTPResponse(status=200),
        ]

        with mock.patch(
            "deterministic_inference.serving.urllib.request.urlopen",
            side_effect=side_effects,
        ):
            with mock.patch("deterministic_inference.serving.time.sleep"):
                ready = wait_for_openai_server(
                    base_url="http://127.0.0.1:8123",
                    timeout_seconds=5,
                    poll_interval_seconds=0.01,
                )

        self.assertTrue(ready)

    def test_wait_returns_false_on_persistent_errors(self) -> None:
        with mock.patch(
            "deterministic_inference.serving.urllib.request.urlopen",
            side_effect=ConnectionResetError("connection reset"),
        ):
            with mock.patch("deterministic_inference.serving.time.sleep"):
                ready = wait_for_openai_server(
                    base_url="http://127.0.0.1:8123",
                    timeout_seconds=0.01,
                    poll_interval_seconds=0.01,
                )

        self.assertFalse(ready)


if __name__ == "__main__":
    unittest.main()
