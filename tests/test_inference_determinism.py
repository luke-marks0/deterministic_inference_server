import hashlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts" / "core"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sample_session  # noqa: E402
import serve  # noqa: E402


def _fake_completion_response(payload: dict) -> dict:
    prompt_token_ids = [int(tok) for tok in payload["prompt"]]
    seed = int(payload["seed"])
    max_tokens = int(payload["max_tokens"])

    base = (sum(prompt_token_ids) + seed) % 100_000
    output_token_ids = [(base + idx) % 100_000 for idx in range(max_tokens)]
    echoed = prompt_token_ids + output_token_ids

    return {
        "choices": [
            {
                "logprobs": {
                    "tokens": [f"token_id:{tok}" for tok in echoed],
                },
                "text": "ok",
            }
        ],
        "usage": {"completion_tokens": max_tokens},
    }


class InferenceDeterminismTests(unittest.TestCase):
    def test_generate_one_is_deterministic_for_same_seed(self) -> None:
        args = {
            "url": "http://localhost:8000/v1/completions",
            "timeout_seconds": 30,
            "model": "qwen3-235b-a22b-instruct-2507",
            "prompt_token_ids": [101, 102, 103],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_k": 50,
            "top_p": 0.95,
            "seed": 424242,
        }

        with mock.patch.object(sample_session, "post_json", side_effect=lambda _, payload, __: _fake_completion_response(payload)):
            first = sample_session._generate_one(**args)
            second = sample_session._generate_one(**args)

        self.assertEqual(first, second)

    def test_generate_one_changes_when_seed_changes(self) -> None:
        base_args = {
            "url": "http://localhost:8000/v1/completions",
            "timeout_seconds": 30,
            "model": "qwen3-235b-a22b-instruct-2507",
            "prompt_token_ids": [201, 202, 203],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_k": 50,
            "top_p": 0.95,
        }

        with mock.patch.object(sample_session, "post_json", side_effect=lambda _, payload, __: _fake_completion_response(payload)):
            output_a, _ = sample_session._generate_one(seed=111, **base_args)
            output_b, _ = sample_session._generate_one(seed=222, **base_args)

        self.assertNotEqual(output_a, output_b)

    def test_generate_one_does_not_send_top_logprobs(self) -> None:
        captured_payloads: list[dict[str, object]] = []

        def fake_post_json(_: str, payload: dict[str, object], __: int) -> dict[str, object]:
            captured_payloads.append(payload)
            return _fake_completion_response(payload)

        with mock.patch.object(sample_session, "post_json", side_effect=fake_post_json):
            sample_session._generate_one(
                url="http://localhost:8000/v1/completions",
                timeout_seconds=30,
                model="qwen3-235b-a22b-instruct-2507",
                prompt_token_ids=[101, 102, 103],
                max_tokens=8,
                temperature=0.0,
                top_k=50,
                top_p=0.95,
                seed=424242,
            )

        self.assertEqual(len(captured_payloads), 1)
        self.assertNotIn("top_logprobs", captured_payloads[0])

    def test_sample_session_main_uses_profile_sampling_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reference_bundle_path = tmp / "reference.json"
            reference_hash_path = tmp / "reference.sha256"
            output_path = tmp / "output.json"
            reference_bundle_path.write_text("{}", encoding="utf-8")
            reference_hash_path.write_text(f"{'0' * 64}\n", encoding="utf-8")

            captured_kwargs: list[dict[str, object]] = []

            fake_profile = SimpleNamespace(
                root_dir=tmp,
                runtime=SimpleNamespace(
                    host_port=8123,
                    paths=SimpleNamespace(hf_cache="state/hf"),
                ),
                model=SimpleNamespace(
                    served_name="kimi-k2-thinking",
                    model_id="moonshotai/Kimi-K2-Thinking",
                    revision="deadbeef",
                ),
                sample_defaults=SimpleNamespace(
                    temperature=0.0,
                    top_p=0.95,
                    seed=424242,
                    timeout_seconds=777,
                ),
            )

            def fake_generate_one(**kwargs: object) -> tuple[list[int], int]:
                captured_kwargs.append(dict(kwargs))
                return [11, 22, 33], 3

            argv = [
                "sample_session.py",
                "--config",
                "configs/kimi-k2-thinking.json",
                "--reference-bundle",
                str(reference_bundle_path),
                "--reference-hash",
                str(reference_hash_path),
                "--n-prompts",
                "1",
                "--output",
                str(output_path),
                "--disable-run-log",
            ]

            with mock.patch.object(sample_session, "load_profile", return_value=fake_profile):
                with mock.patch.object(sample_session, "_verify_reference_bundle_hash"):
                    with mock.patch.object(
                        sample_session,
                        "_load_reference_inputs",
                        return_value=(
                            "moonshotai/Kimi-K2-Thinking",
                            [[{"role": "user", "content": "test"}]],
                            False,
                        ),
                    ):
                        with mock.patch.object(sample_session, "_sha256_file", return_value="f" * 64):
                            with mock.patch.object(
                                sample_session,
                                "_verify_full_snapshot_against_manifest",
                                return_value=(tmp / "snapshot", tmp / "manifest.sha256", 123),
                            ):
                                with mock.patch.object(
                                    sample_session,
                                    "_tokenize_conversations_for_model",
                                    return_value=([[1, 2, 3]], "moonshotai/Kimi-K2-Thinking@deadbeef"),
                                ):
                                    with mock.patch.object(
                                        sample_session,
                                        "_generate_one",
                                        side_effect=fake_generate_one,
                                    ):
                                        with mock.patch.object(sys, "argv", argv):
                                            with redirect_stdout(io.StringIO()):
                                                rc = sample_session.main()

            self.assertEqual(rc, 0)
            self.assertEqual(len(captured_kwargs), 1)
            kwargs = captured_kwargs[0]
            self.assertEqual(kwargs["url"], "http://127.0.0.1:8123/v1/completions")
            self.assertEqual(kwargs["model"], "kimi-k2-thinking")
            self.assertEqual(kwargs["temperature"], 0.0)
            self.assertEqual(kwargs["top_k"], 50)
            self.assertEqual(kwargs["top_p"], 0.95)
            self.assertEqual(kwargs["seed"], 424242)
            self.assertEqual(kwargs["timeout_seconds"], 777)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["parameters"]["temperature"], 0.0)
            self.assertEqual(payload["parameters"]["top_k"], 50)
            self.assertEqual(payload["parameters"]["top_p"], 0.95)
            self.assertEqual(payload["parameters"]["seed"], 424242)

    def test_sample_session_main_retokenizes_prompts_for_target_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reference_bundle_path = tmp / "reference.json"
            reference_hash_path = tmp / "reference.sha256"
            output_path = tmp / "output.json"
            reference_bundle_path.write_text("{}", encoding="utf-8")
            reference_hash_path.write_text(f"{'0' * 64}\n", encoding="utf-8")

            captured_kwargs: list[dict[str, object]] = []

            fake_profile = SimpleNamespace(
                root_dir=tmp,
                runtime=SimpleNamespace(
                    host_port=8123,
                    paths=SimpleNamespace(hf_cache="state/hf"),
                ),
                model=SimpleNamespace(
                    served_name="gpt-oss-20b",
                    model_id="openai/gpt-oss-20b",
                    revision="6cee5e81ee83917806bbde320786a8fb61efebee",
                ),
                sample_defaults=SimpleNamespace(
                    temperature=0.0,
                    top_p=0.95,
                    seed=424242,
                    timeout_seconds=777,
                ),
            )

            def fake_generate_one(**kwargs: object) -> tuple[list[int], int]:
                captured_kwargs.append(dict(kwargs))
                return [11, 22, 33], 3

            argv = [
                "sample_session.py",
                "--config",
                "configs/gpt-oss-20b.json",
                "--reference-bundle",
                str(reference_bundle_path),
                "--reference-hash",
                str(reference_hash_path),
                "--n-prompts",
                "1",
                "--output",
                str(output_path),
                "--disable-run-log",
            ]

            with mock.patch.object(sample_session, "load_profile", return_value=fake_profile):
                with mock.patch.object(sample_session, "_verify_reference_bundle_hash"):
                    with mock.patch.object(
                        sample_session,
                        "_load_reference_inputs",
                        return_value=(
                            "Qwen/Qwen3-235B-A22B-Instruct-2507",
                            [[{"role": "user", "content": "test"}]],
                            True,
                        ),
                    ):
                        with mock.patch.object(sample_session, "_sha256_file", return_value="f" * 64):
                            with mock.patch.object(
                                sample_session,
                                "_verify_full_snapshot_against_manifest",
                                return_value=(tmp / "snapshot", tmp / "manifest.sha256", 123),
                            ):
                                with mock.patch.object(
                                    sample_session,
                                    "_tokenize_conversations_for_model",
                                    return_value=([[1, 2, 3, 4]], "openai/gpt-oss-20b@6cee5e8"),
                                ):
                                    with mock.patch.object(
                                        sample_session,
                                        "_generate_one",
                                        side_effect=fake_generate_one,
                                    ):
                                        with mock.patch.object(sys, "argv", argv):
                                            with redirect_stdout(io.StringIO()):
                                                rc = sample_session.main()

            self.assertEqual(rc, 0)
            self.assertEqual(len(captured_kwargs), 1)
            self.assertEqual(captured_kwargs[0]["prompt_token_ids"], [1, 2, 3, 4])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["sequences"][0]["prompt_token_ids"], [1, 2, 3, 4])

    def test_sample_session_main_rejects_concurrency_above_one(self) -> None:
        fake_profile = SimpleNamespace(
            runtime=SimpleNamespace(host_port=8123),
            model=SimpleNamespace(
                served_name="gpt-oss-20b",
                model_id="openai/gpt-oss-20b",
                revision="6cee5e81ee83917806bbde320786a8fb61efebee",
            ),
            sample_defaults=SimpleNamespace(
                temperature=0.0,
                top_p=0.95,
                seed=424242,
                timeout_seconds=777,
            ),
        )
        argv = [
            "sample_session.py",
            "--config",
            "configs/gpt-oss-20b.json",
            "--concurrency",
            "2",
        ]
        with mock.patch.object(sample_session, "load_profile", return_value=fake_profile):
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "exactly 1"):
                    sample_session.main()

    def test_sample_session_main_rejects_skip_hash_flag(self) -> None:
        fake_profile = SimpleNamespace(
            runtime=SimpleNamespace(host_port=8123),
            model=SimpleNamespace(
                served_name="gpt-oss-20b",
                model_id="openai/gpt-oss-20b",
                revision="6cee5e81ee83917806bbde320786a8fb61efebee",
            ),
            sample_defaults=SimpleNamespace(
                temperature=0.0,
                top_p=0.95,
                seed=424242,
                timeout_seconds=777,
            ),
        )
        argv = [
            "sample_session.py",
            "--config",
            "configs/gpt-oss-20b.json",
            "--skip-reference-hash-check",
        ]
        with mock.patch.object(sample_session, "load_profile", return_value=fake_profile):
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "disallowed"):
                    sample_session.main()

    def test_smoke_request_payload_is_repeatable(self) -> None:
        config_path = ROOT_DIR / "configs" / "qwen3-235b-a22b-instruct-2507.json"
        args = Namespace(config=str(config_path), secrets_file=Path(".env"))
        captured: list[dict[str, object]] = []

        def fake_post_json(url: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
            payload_copy = json.loads(json.dumps(payload, sort_keys=True))
            captured.append({"url": url, "payload": payload_copy, "timeout": timeout})
            digest = hashlib.sha256(json.dumps(payload_copy, sort_keys=True).encode("utf-8")).hexdigest()[:10]
            return {"choices": [{"text": f"deterministic-{digest}"}], "usage": {"completion_tokens": 1}}

        with mock.patch.object(serve, "_post_json", side_effect=fake_post_json):
            with redirect_stdout(io.StringIO()):
                rc_one = serve.cmd_smoke(args)
                rc_two = serve.cmd_smoke(args)

        self.assertEqual(rc_one, 0)
        self.assertEqual(rc_two, 0)
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["url"], captured[1]["url"])
        self.assertEqual(captured[0]["timeout"], captured[1]["timeout"])
        self.assertEqual(captured[0]["payload"], captured[1]["payload"])

        payload = captured[0]["payload"]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 424242)


if __name__ == "__main__":
    unittest.main()
