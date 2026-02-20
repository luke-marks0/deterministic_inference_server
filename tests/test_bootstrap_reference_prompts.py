import sys
import unittest
from types import SimpleNamespace
from unittest import mock


from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts" / "core"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bootstrap_reference_prompts  # noqa: E402


class BootstrapReferencePromptsTests(unittest.TestCase):
    def test_requires_dataset_revision_pin(self) -> None:
        fake_profile = SimpleNamespace(
            root_dir=ROOT_DIR,
            model=SimpleNamespace(
                model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
                revision="ac9c66cc9b46af7306746a9250f23d47083d689e",
            ),
            runtime=SimpleNamespace(paths=SimpleNamespace(hf_cache="state/hf")),
        )
        argv = [
            "bootstrap_reference_prompts.py",
            "--config",
            "configs/qwen3-235b-a22b-instruct-2507.json",
        ]
        with mock.patch.object(bootstrap_reference_prompts, "load_profile", return_value=fake_profile):
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "Dataset revision must be pinned"):
                    bootstrap_reference_prompts.main()

    def test_requires_model_revision_pin(self) -> None:
        fake_profile = SimpleNamespace(
            root_dir=ROOT_DIR,
            model=SimpleNamespace(
                model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
                revision="UNSET",
            ),
            runtime=SimpleNamespace(paths=SimpleNamespace(hf_cache="state/hf")),
        )
        argv = [
            "bootstrap_reference_prompts.py",
            "--config",
            "configs/qwen3-235b-a22b-instruct-2507.json",
            "--dataset-revision",
            "abc123",
        ]
        with mock.patch.object(bootstrap_reference_prompts, "load_profile", return_value=fake_profile):
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "Tokenizer/model revision must be pinned"):
                    bootstrap_reference_prompts.main()

    def test_rejects_tokenizer_name_override(self) -> None:
        fake_profile = SimpleNamespace(
            root_dir=ROOT_DIR,
            model=SimpleNamespace(
                model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
                revision="ac9c66cc9b46af7306746a9250f23d47083d689e",
            ),
            runtime=SimpleNamespace(paths=SimpleNamespace(hf_cache="state/hf")),
        )
        argv = [
            "bootstrap_reference_prompts.py",
            "--config",
            "configs/qwen3-235b-a22b-instruct-2507.json",
            "--dataset-revision",
            "abc123",
            "--tokenizer-name",
            "custom/tokenizer",
        ]
        with mock.patch.object(bootstrap_reference_prompts, "load_profile", return_value=fake_profile):
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "tokenizer identity to match"):
                    bootstrap_reference_prompts.main()


if __name__ == "__main__":
    unittest.main()
