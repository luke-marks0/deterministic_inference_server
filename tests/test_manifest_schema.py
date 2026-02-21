from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import deterministic_inference as workflow

from test_helpers import make_manifest


class TestManifestSchema(unittest.TestCase):
    def test_all_repo_configs_load(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config_paths = sorted((repo_root / "configs").glob("*.json"))
        self.assertGreater(len(config_paths), 0)
        for path in config_paths:
            with self.subTest(config=path.name):
                workflow.load_manifest(path)

    def test_unknown_top_level_field_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = workflow.create_manifest_template(
                model_id="org/model",
                model_revision="rev",
                output_path=manifest_path,
            )
            manifest["unknown"] = True
            with self.assertRaises(workflow.ManifestValidationError):
                workflow.validate_manifest(manifest)

    def test_x_extension_field_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = workflow.create_manifest_template(
                model_id="org/model",
                model_revision="rev",
                output_path=manifest_path,
            )
            manifest["x_extension"] = {"note": "allowed"}
            workflow.validate_manifest(manifest)

    def test_capture_tokens_must_be_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            manifest["capture"]["tokens"] = False
            with self.assertRaises(workflow.ManifestValidationError):
                workflow.validate_manifest(manifest)

    def test_batch_policy_must_be_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            manifest["inference"]["batching"]["policy"] = "dynamic"
            with self.assertRaises(workflow.ManifestValidationError):
                workflow.validate_manifest(manifest)

    def test_runtime_execution_requires_deterministic_failure_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            manifest["runtime"]["execution"].pop("deterministic_failure_policy", None)
            with self.assertRaises(workflow.ManifestValidationError):
                workflow.validate_manifest(manifest)

    def test_runtime_execution_vllm_image_must_be_digest_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            manifest["runtime"]["execution"]["vllm_image"] = "docker.io/vllm/vllm-openai:latest"
            with self.assertRaises(workflow.ManifestValidationError):
                workflow.validate_manifest(manifest)

    def test_trust_remote_code_requires_commit_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            manifest["vllm"]["engine_args"]["trust_remote_code"] = True
            manifest["model"]["weights"]["source"].pop("remote_code_commit", None)
            manifest["model"]["weights"]["source"].pop("remote_code_digest", None)
            with self.assertRaises(workflow.ManifestValidationError):
                workflow.validate_manifest(manifest)

    def test_embedded_lock_object_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            make_manifest(manifest_path)
            manifest = workflow.load_manifest(manifest_path)
            lock_payload = workflow.build_lock_payload(manifest, manifest_path=manifest_path)
            manifest["artifacts"]["lockfile"] = lock_payload
            workflow.validate_manifest(manifest)

    def test_shared_prompt_dataset_mode_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            template_request = manifest["inference"]["requests"][0]
            batching = manifest["inference"]["batching"]
            manifest["inference"] = {
                "n_prompts": 5,
                "request_template": {
                    "kind": template_request["kind"],
                    "sampling": template_request["sampling"],
                    "stop": template_request["stop"],
                },
                "batching": batching,
            }
            workflow.validate_manifest(manifest)

    def test_shared_prompt_dataset_mode_rejects_invalid_n_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            template_request = manifest["inference"]["requests"][0]
            batching = manifest["inference"]["batching"]
            manifest["inference"] = {
                "n_prompts": 101,
                "request_template": {
                    "kind": template_request["kind"],
                    "sampling": template_request["sampling"],
                    "stop": template_request["stop"],
                },
                "batching": batching,
            }
            with self.assertRaises(workflow.ManifestValidationError):
                workflow.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
