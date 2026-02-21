from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import deterministic_inference as workflow

from test_helpers import lock_manifest, make_manifest


class TestVerifyDeterminismGrading(unittest.TestCase):
    def _execute_bundle(
        self,
        *,
        root: Path,
        manifest_path: Path,
        run_name: str,
    ) -> Path:
        manifest = workflow.load_manifest(manifest_path)
        lock = workflow.load_lock(workflow.resolve_lock_path(manifest, manifest_path))
        result = workflow.execute_run(
            manifest,
            manifest_path=manifest_path,
            lock=lock,
            run_dir_override=root / run_name,
        )
        return Path(result["bundle_path"])

    def test_conformant_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)

            bundle_a = self._execute_bundle(root=root, manifest_path=manifest_path, run_name="run-a")
            bundle_b = self._execute_bundle(root=root, manifest_path=manifest_path, run_name="run-b")

            out = workflow.verify_bundles(
                bundle_a,
                bundle_b,
                output_dir=root / "verify-a-b",
            )
            self.assertEqual(out["grade"], "conformant")

    def test_mismatch_outputs_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)

            bundle_a = self._execute_bundle(root=root, manifest_path=manifest_path, run_name="run-a")
            bundle_b = self._execute_bundle(root=root, manifest_path=manifest_path, run_name="run-b")

            tokens_path = bundle_b.parent / "tokens.json"
            tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
            tokens["sequences"][0]["output_token_ids"][0] += 999
            tokens_path.write_text(json.dumps(tokens, indent=2) + "\n", encoding="utf-8")

            out = workflow.verify_bundles(
                bundle_a,
                bundle_b,
                output_dir=root / "verify-mismatch",
            )
            self.assertEqual(out["grade"], "mismatch_outputs")

    def test_non_conformant_software_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_a_path = root / "manifest-a.json"
            make_manifest(manifest_a_path)
            lock_manifest(manifest_a_path)
            bundle_a = self._execute_bundle(root=root, manifest_path=manifest_a_path, run_name="run-a")

            def mutate(manifest: dict) -> None:
                manifest["runtime"]["env"]["PYTHONHASHSEED"] = "123"

            manifest_b_path = root / "manifest-b.json"
            make_manifest(manifest_b_path, mutate=mutate)
            lock_manifest(manifest_b_path)
            bundle_b = self._execute_bundle(root=root, manifest_path=manifest_b_path, run_name="run-b")

            out = workflow.verify_bundles(
                bundle_a,
                bundle_b,
                output_dir=root / "verify-software",
            )
            self.assertEqual(out["grade"], "non_conformant_software")

    def test_non_conformant_hardware_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_a_path = root / "manifest-a.json"
            make_manifest(manifest_a_path)
            lock_manifest(manifest_a_path)
            bundle_a = self._execute_bundle(root=root, manifest_path=manifest_a_path, run_name="run-a")

            def mutate(manifest: dict) -> None:
                manifest["hardware"]["constraints"] = {"cpu_arch": "intentionally-mismatched-arch"}
                manifest["determinism"]["strict_hardware"] = False

            manifest_b_path = root / "manifest-b.json"
            make_manifest(manifest_b_path, mutate=mutate)
            lock_manifest(manifest_b_path)
            bundle_b = self._execute_bundle(root=root, manifest_path=manifest_b_path, run_name="run-b")

            out = workflow.verify_bundles(
                bundle_a,
                bundle_b,
                output_dir=root / "verify-hardware",
            )
            self.assertEqual(out["grade"], "non_conformant_hardware")

    def test_non_conformant_software_when_token_rules_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_a_path = root / "manifest-a.json"
            make_manifest(manifest_a_path)
            lock_manifest(manifest_a_path)
            bundle_a = self._execute_bundle(root=root, manifest_path=manifest_a_path, run_name="run-a")

            def mutate(manifest: dict) -> None:
                manifest["determinism"]["compare"]["tokens"]["rule"] = "exact"

            manifest_b_path = root / "manifest-b.json"
            make_manifest(manifest_b_path, mutate=mutate)
            lock_manifest(manifest_b_path)
            bundle_b = self._execute_bundle(root=root, manifest_path=manifest_b_path, run_name="run-b")

            manifest_used_b = json.loads((bundle_b.parent / "manifest.used.json").read_text(encoding="utf-8"))
            manifest_used_b["determinism"]["compare"]["tokens"]["rule"] = "prefix_match"
            (bundle_b.parent / "manifest.used.json").write_text(
                json.dumps(manifest_used_b, indent=2) + "\n",
                encoding="utf-8",
            )

            out = workflow.verify_bundles(
                bundle_a,
                bundle_b,
                output_dir=root / "verify-token-rule-mismatch",
            )
            self.assertEqual(out["grade"], "non_conformant_software")

    def test_verify_rejects_unsupported_token_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            make_manifest(manifest_path)
            lock_manifest(manifest_path)
            bundle_a = self._execute_bundle(root=root, manifest_path=manifest_path, run_name="run-a")
            bundle_b = self._execute_bundle(root=root, manifest_path=manifest_path, run_name="run-b")

            manifest_used = json.loads((bundle_a.parent / "manifest.used.json").read_text(encoding="utf-8"))
            manifest_used["determinism"]["compare"]["tokens"]["rule"] = "top_k_match"
            (bundle_a.parent / "manifest.used.json").write_text(
                json.dumps(manifest_used, indent=2) + "\n",
                encoding="utf-8",
            )
            (bundle_b.parent / "manifest.used.json").write_text(
                json.dumps(manifest_used, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                workflow.verify_bundles(
                    bundle_a,
                    bundle_b,
                    output_dir=root / "verify-unsupported-token-rule",
                )


if __name__ == "__main__":
    unittest.main()
