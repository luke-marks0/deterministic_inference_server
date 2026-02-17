import hashlib
import io
import json
import sys
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
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
