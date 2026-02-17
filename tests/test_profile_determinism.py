import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts" / "core"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import profile_config  # noqa: E402


class ProfileDeterminismTests(unittest.TestCase):
    def test_profiles_include_deterministic_controls(self) -> None:
        config_paths = [
            ROOT_DIR / "configs" / "qwen3-235b-a22b-instruct-2507.json",
            ROOT_DIR / "configs" / "kimi-k2-thinking.json",
        ]

        for config_path in config_paths:
            with self.subTest(config=str(config_path)):
                profile = profile_config.load_profile(config_path)
                self.assertEqual(profile.smoke_test.temperature, 0.0)
                self.assertEqual(profile.sample_defaults.temperature, 0.0)
                self.assertEqual(profile.smoke_test.seed, profile.sample_defaults.seed)
                self.assertTrue(any(flag.startswith("--seed=") for flag in profile.vllm_flags))
                self.assertIn("--max-num-seqs=1", profile.vllm_flags)
                self.assertIn("--enforce-eager", profile.vllm_flags)

    def test_render_compose_yaml_is_stable(self) -> None:
        profile = profile_config.load_profile(ROOT_DIR / "configs" / "kimi-k2-thinking.json")
        rendered_one = profile_config.render_compose_yaml(profile)
        rendered_two = profile_config.render_compose_yaml(profile)
        self.assertEqual(rendered_one, rendered_two)

    def test_config_inheritance_merges_base_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_path = tmp / "base.json"
            child_path = tmp / "child.json"

            base_payload = {
                "schema_version": 1,
                "profile": {"id": "base-profile"},
                "runtime": {
                    "image": "vllm/vllm-openai:v0.8.5.post1",
                    "gpus": "all",
                    "ipc_mode": "host",
                    "restart": "unless-stopped",
                    "host_port": 8000,
                    "container_port": 8000,
                    "api_host": "0.0.0.0",
                    "container_name": "base_container",
                    "compose_project_name": "base_project",
                    "required_secret_env": [],
                    "memlock": -1,
                    "stack": 67108864,
                    "paths": {"hf_cache": "state/hf", "artifacts": "artifacts"},
                    "environment": {"HF_HOME": "/data/hf"},
                    "bootstrap_pip_packages": [],
                },
                "model": {
                    "id": "org/base-model",
                    "revision": "UNSET_RUN_LOCK_SCRIPT",
                    "locked_at_utc": "UNSET",
                    "served_name": "base-model",
                },
                "integrity": {"expected_snapshot_manifest": "manifests/{profile_id}/{revision}.sha256"},
                "vllm": {"flags": ["--seed=424242"]},
                "smoke_test": {
                    "prompt": "ok",
                    "max_tokens": 8,
                    "temperature": 0.0,
                    "seed": 424242,
                },
                "sampling_defaults": {
                    "target_tokens": 20000,
                    "chunk_max_tokens": 1024,
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "seed": 424242,
                    "timeout_seconds": 600,
                },
            }
            child_payload = {
                "base_config": "base.json",
                "profile": {"id": "child-profile"},
                "runtime": {
                    "host_port": 8100,
                    "container_name": "child_container",
                    "compose_project_name": "child_project",
                },
                "model": {
                    "id": "org/child-model",
                    "revision": "abc123",
                    "locked_at_utc": "2026-02-17T00:00:00Z",
                    "served_name": "child-model",
                },
                "vllm": {"flags": ["--dtype=float16"]},
            }

            base_path.write_text(json.dumps(base_payload), encoding="utf-8")
            child_path.write_text(json.dumps(child_payload), encoding="utf-8")

            profile = profile_config.load_profile(child_path)
            self.assertEqual(profile.profile_id, "child-profile")
            self.assertEqual(profile.runtime.host_port, 8100)
            self.assertEqual(profile.runtime.container_port, 8000)
            self.assertEqual(profile.runtime.container_name, "child_container")
            self.assertEqual(profile.model.model_id, "org/child-model")
            self.assertEqual(profile.vllm_flags, ["--seed=424242", "--dtype=float16"])

    def test_config_inheritance_cycle_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            a_path = tmp / "a.json"
            b_path = tmp / "b.json"

            a_path.write_text(json.dumps({"base_config": "b.json"}), encoding="utf-8")
            b_path.write_text(json.dumps({"base_config": "a.json"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inheritance cycle"):
                profile_config.load_profile(a_path)


if __name__ == "__main__":
    unittest.main()
