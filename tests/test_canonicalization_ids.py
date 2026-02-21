from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import deterministic_inference as workflow

from test_helpers import make_manifest


class TestCanonicalizationAndIds(unittest.TestCase):
    def test_manifest_id_stable_across_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)

            reordered = {
                "kind": manifest["kind"],
                "schema_version": manifest["schema_version"],
                "metadata": manifest["metadata"],
                "determinism": manifest["determinism"],
                "hardware": manifest["hardware"],
                "artifacts": manifest["artifacts"],
                "software_stack": manifest["software_stack"],
                "cuda_stack": manifest["cuda_stack"],
                "runtime": manifest["runtime"],
                "vllm": manifest["vllm"],
                "model": manifest["model"],
                "inference": manifest["inference"],
                "capture": manifest["capture"],
                "outputs": manifest["outputs"],
            }

            self.assertEqual(
                workflow.compute_manifest_id(manifest),
                workflow.compute_manifest_id(reordered),
            )

    def test_manifest_id_changes_on_semantic_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            manifest_b = copy.deepcopy(manifest)
            manifest_b["inference"]["requests"][0]["sampling"]["seed"] += 1

            self.assertNotEqual(
                workflow.compute_manifest_id(manifest),
                workflow.compute_manifest_id(manifest_b),
            )

    def test_newline_normalization_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.json"
            manifest_a = make_manifest(path_a)
            manifest_b = copy.deepcopy(manifest_a)

            manifest_a["metadata"]["description"] = "line1\r\nline2"
            manifest_b["metadata"]["description"] = "line1\nline2"

            self.assertEqual(
                workflow.compute_manifest_id(manifest_a),
                workflow.compute_manifest_id(manifest_b),
            )

    def test_lock_id_is_computed_without_lock_id_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            make_manifest(manifest_path)
            manifest = workflow.load_manifest(manifest_path)
            lock_payload = workflow.build_lock_payload(manifest, manifest_path=manifest_path)

            computed = workflow.compute_lock_id(lock_payload)
            self.assertEqual(lock_payload["lock_id"], computed)

    def test_lock_build_rejects_unset_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            manifest["software_stack"]["python"]["packages"][0]["source"]["digest"] = "sha256:unset"
            with self.assertRaises(ValueError):
                workflow.build_lock_payload(manifest, manifest_path=manifest_path)

    def test_lock_build_rejects_mismatched_local_tokenizer_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = make_manifest(manifest_path)
            manifest["model"]["tokenizer"]["files"][0]["digest"] = "0" * 64
            with self.assertRaises(ValueError):
                workflow.build_lock_payload(manifest, manifest_path=manifest_path)


if __name__ == "__main__":
    unittest.main()
