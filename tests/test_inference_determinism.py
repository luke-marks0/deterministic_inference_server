from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import deterministic_inference as workflow

from test_helpers import lock_manifest, make_manifest


class TestRunOutputsAndTokenFormat(unittest.TestCase):
    def test_run_preserves_output_token_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            def mutate(manifest: dict) -> None:
                request = manifest["inference"]["requests"][0]
                request["id"] = "req-0001"
                request["prompt"] = "alpha"

                second = json.loads(json.dumps(request))
                second["id"] = "req-0002"
                second["prompt"] = "beta"

                manifest["inference"]["requests"] = [request, second]
                manifest["inference"]["batching"]["max_num_seqs"] = 2
                manifest["inference"]["batching"]["max_num_batched_tokens"] = 2048

            make_manifest(manifest_path, mutate=mutate)
            lock_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))
            result = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=tmp_path / "run-a",
            )

            token_payload = json.loads(Path(result["token_output_path"]).read_text(encoding="utf-8"))
            self.assertIn("sequences", token_payload)
            self.assertEqual(len(token_payload["sequences"]), 2)

            sequence = token_payload["sequences"][0]
            self.assertEqual(sorted(sequence.keys()), ["output_token_ids", "prompt_token_ids"])
            self.assertTrue(all(isinstance(token, int) for token in sequence["prompt_token_ids"]))
            self.assertTrue(all(isinstance(token, int) for token in sequence["output_token_ids"]))

            prompt_ids = sequence["prompt_token_ids"]
            seed = token_payload["parameters"]["seed"]
            max_tokens = token_payload["parameters"]["max_tokens"]
            base = (sum(prompt_ids) + seed) % 100_000
            expected = [(base + idx) % 100_000 for idx in range(max_tokens)]
            self.assertEqual(sequence["output_token_ids"], expected)

            run_log_payload = json.loads(Path(result["run_log_path"]).read_text(encoding="utf-8"))
            self.assertIn("records", run_log_payload)
            self.assertEqual(run_log_payload["records"][0]["output_token_ids"], expected)
            self.assertIn("tokenizer", token_payload)
            self.assertIn("decoded_output_text", token_payload)
            self.assertEqual(len(token_payload["decoded_output_text"]), 2)
            self.assertIn("determinism_controls", run_log_payload)
            self.assertIn("runtime_environment", run_log_payload)
            self.assertIn("LANG", run_log_payload["runtime_environment"])
            self.assertIn("TZ", run_log_payload["runtime_environment"])

    def test_capture_options_are_noop_but_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            def mutate(manifest: dict) -> None:
                manifest["capture"]["logits"]["enabled"] = True
                manifest["capture"]["activations"]["enabled"] = True
                manifest["capture"]["engine_trace"]["enabled"] = True
                manifest["capture"]["activations"]["hooks"] = [
                    {
                        "name": "layer.residual_post",
                        "layers": "all",
                        "steps": "all",
                        "dtype": "float16",
                    }
                ]
                manifest["capture"]["engine_trace"]["events"] = ["batch_composition"]

            make_manifest(manifest_path, mutate=mutate)
            lock_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))
            result = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=tmp_path / "run-b",
            )

            bundle_payload = json.loads(Path(result["bundle_path"]).read_text(encoding="utf-8"))
            self.assertEqual(bundle_payload["capture"]["logits"]["status"], "configured_noop")
            self.assertEqual(bundle_payload["capture"]["activations"]["status"], "configured_noop")
            self.assertEqual(bundle_payload["capture"]["engine_trace"]["status"], "configured_noop")

    def test_strict_hardware_rejects_non_conformant_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            def mutate(manifest: dict) -> None:
                manifest["determinism"]["strict_hardware"] = True
                manifest["hardware"]["constraints"] = {"cpu_arch": "definitely-not-this-cpu-arch"}

            make_manifest(manifest_path, mutate=mutate)
            lock_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))
            with self.assertRaises(RuntimeError):
                workflow.execute_run(
                    manifest,
                    manifest_path=manifest_path,
                    lock=lock,
                    run_dir_override=tmp_path / "run-c",
                )

    def test_batch_token_budget_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            def mutate(manifest: dict) -> None:
                request = manifest["inference"]["requests"][0]
                request["prompt"] = "a" * 400
                second = json.loads(json.dumps(request))
                second["id"] = "req-0002"
                second["prompt"] = "b" * 400
                manifest["inference"]["requests"] = [request, second]
                manifest["inference"]["batching"]["max_num_seqs"] = 2
                manifest["inference"]["batching"]["max_num_batched_tokens"] = 64

            make_manifest(manifest_path, mutate=mutate)
            lock_manifest(manifest_path)
            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))

            with self.assertRaises(ValueError):
                workflow.execute_run(
                    manifest,
                    manifest_path=manifest_path,
                    lock=lock,
                    run_dir_override=tmp_path / "run-budget",
                )

    def test_replay_schedule_uses_arrival_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            def mutate(manifest: dict) -> None:
                request = manifest["inference"]["requests"][0]
                request["prompt"] = "first"
                request["x_arrival_ms"] = 200

                second = json.loads(json.dumps(request))
                second["id"] = "req-0002"
                second["prompt"] = "second"
                second["x_arrival_ms"] = 100

                manifest["inference"]["requests"] = [request, second]
                manifest["inference"]["batching"]["max_num_seqs"] = 1
                manifest["inference"]["batching"]["schedule"] = "replay"

            make_manifest(manifest_path, mutate=mutate)
            lock_manifest(manifest_path)
            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))
            result = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=tmp_path / "run-replay",
            )

            bundle_payload = json.loads(Path(result["bundle_path"]).read_text(encoding="utf-8"))
            first_step = bundle_payload["batch_trace"]["steps"][0]
            self.assertEqual(first_step["request_ids"][0], "req-0002")

    def test_default_digest_integrity_check_rejects_modified_local_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))

            local_model_dir = Path(manifest["model"]["weights"]["source"]["local_path"])
            (local_model_dir / "tokenizer.json").write_text(
                "{\"tokenizer\":\"tampered\"}",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                workflow.execute_run(
                    manifest,
                    manifest_path=manifest_path,
                    lock=lock,
                    run_dir_override=tmp_path / "run-integrity-fail",
                )

    def test_digest_integrity_check_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))

            local_model_dir = Path(manifest["model"]["weights"]["source"]["local_path"])
            (local_model_dir / "tokenizer.json").write_text(
                "{\"tokenizer\":\"tampered\"}",
                encoding="utf-8",
            )

            result = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                verify_artifact_digests=False,
                run_dir_override=tmp_path / "run-integrity-disabled",
            )

            bundle_payload = json.loads(Path(result["bundle_path"]).read_text(encoding="utf-8"))
            self.assertEqual(bundle_payload["artifact_integrity"]["enabled"], False)

    def test_shared_prompt_dataset_mode_uses_n_prompts_and_pins_prompt_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            def mutate(manifest: dict) -> None:
                template_request = manifest["inference"]["requests"][0]
                manifest["inference"] = {
                    "n_prompts": 2,
                    "request_template": {
                        "kind": template_request["kind"],
                        "sampling": template_request["sampling"],
                        "stop": template_request["stop"],
                    },
                    "batching": manifest["inference"]["batching"],
                }

            make_manifest(manifest_path, mutate=mutate)
            lock_payload = lock_manifest(manifest_path)
            self.assertIn(
                "inference.prompt_dataset",
                [artifact["name"] for artifact in lock_payload["artifacts"]],
            )

            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))
            result = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=tmp_path / "run-shared-prompts",
            )

            token_payload = json.loads(Path(result["token_output_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(token_payload["sequences"]), 2)
            self.assertEqual(token_payload["parameters"]["n_prompts"], 2)


if __name__ == "__main__":
    unittest.main()
