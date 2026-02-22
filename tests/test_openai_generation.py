from __future__ import annotations

import unittest
from unittest import mock

from deterministic_inference import execution


class TestOpenAIGeneration(unittest.TestCase):
    def _manifest(self) -> dict:
        return {
            "runtime": {
                "execution": {
                    "base_url": "http://127.0.0.1:8000",
                    "timeout_seconds": 10,
                }
            },
            "vllm": {
                "engine_args": {
                    "model": "unit-test-model",
                }
            },
        }

    def test_openai_generation_sends_text_prompt_and_splits_usage_counts(self) -> None:
        request = {
            "id": "req-0001",
            "kind": "completion",
            "prompt": "hello world",
            "sampling": {
                "max_tokens": 4,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 5,
                "seed": 7,
            },
            "stop": {"sequences": [], "token_ids": []},
        }
        response = {
            "choices": [
                {
                    "logprobs": {
                        "tokens": [
                            "token_id:101",
                            "token_id:102",
                            "token_id:201",
                            "token_id:202",
                            "token_id:203",
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
            },
        }

        with mock.patch("deterministic_inference.execution._post_json", return_value=response) as post_json:
            prompt_ids, output_ids, completion_count = execution._openai_generate_tokens(  # type: ignore[attr-defined]
                request=request,
                manifest=self._manifest(),
            )

        self.assertEqual(prompt_ids, [101, 102])
        self.assertEqual(output_ids, [201, 202, 203])
        self.assertEqual(completion_count, 3)
        payload = post_json.call_args.args[1]
        self.assertEqual(payload["prompt"], "hello world")

    def test_openai_generation_uses_explicit_prompt_token_ids_when_present(self) -> None:
        request = {
            "id": "req-0001",
            "kind": "completion",
            "prompt_token_ids": [11, 12, 13],
            "sampling": {
                "max_tokens": 3,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 5,
                "seed": 7,
            },
            "stop": {"sequences": [], "token_ids": []},
        }
        response = {
            "choices": [
                {
                    "logprobs": {
                        "tokens": [
                            "token_id:11",
                            "token_id:12",
                            "token_id:13",
                            "token_id:20",
                            "token_id:21",
                        ]
                    }
                }
            ],
            "usage": {
                "completion_tokens": 2,
            },
        }

        with mock.patch("deterministic_inference.execution._post_json", return_value=response) as post_json:
            prompt_ids, output_ids, completion_count = execution._openai_generate_tokens(  # type: ignore[attr-defined]
                request=request,
                manifest=self._manifest(),
            )

        self.assertEqual(prompt_ids, [11, 12, 13])
        self.assertEqual(output_ids, [20, 21])
        self.assertEqual(completion_count, 2)
        payload = post_json.call_args.args[1]
        self.assertEqual(payload["prompt"], [11, 12, 13])


if __name__ == "__main__":
    unittest.main()
