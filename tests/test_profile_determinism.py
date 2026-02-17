import sys
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


if __name__ == "__main__":
    unittest.main()
