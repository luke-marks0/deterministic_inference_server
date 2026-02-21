from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import deterministic_inference as workflow

from test_helpers import lock_manifest, make_manifest


class TestRunIdAndConfigInheritance(unittest.TestCase):
    def test_run_id_matches_spec_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)

            manifest = workflow.load_manifest(manifest_path)
            lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))
            result = workflow.execute_run(
                manifest,
                manifest_path=manifest_path,
                lock=lock,
                run_dir_override=root / "run",
            )

            bundle = json.loads(Path(result["bundle_path"]).read_text(encoding="utf-8"))
            manifest_id = workflow.compute_manifest_id(manifest)
            lock_id = lock["lock_id"]
            requests_digest = workflow.compute_requests_digest(manifest)
            fingerprint_digest = workflow.canonical_sha256(bundle["hardware_fingerprint"])
            expected_run_id = hashlib.sha256(
                f"{manifest_id}{lock_id}{requests_digest}{fingerprint_digest}".encode("utf-8")
            ).hexdigest()
            self.assertEqual(bundle["run_id"], expected_run_id)

    def test_model_configs_resolve_from_standard_base(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        manifest = workflow.load_manifest(repo_root / "configs" / "qwen3-8b.json")

        self.assertEqual(manifest["kind"], workflow.MANIFEST_KIND)
        self.assertEqual(manifest["runtime"]["execution"]["backend"], "openai_compatible")
        self.assertEqual(manifest["model"]["weights"]["source"]["repo"], "Qwen/Qwen3-8B")
        self.assertEqual(manifest["inference"]["batching"]["policy"], "fixed")
        self.assertEqual(manifest["inference"]["n_prompts"], 100)
        self.assertNotIn("requests", manifest["inference"])


if __name__ == "__main__":
    unittest.main()
