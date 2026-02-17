import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts" / "core"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_determinism  # noqa: E402


def _write_run_log(path: Path, records: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": 1,
        "run_type": "determinism_log",
        "created_at_utc": "2026-02-17T00:00:00Z",
        "model": "test/model",
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DeterminismEvaluationTests(unittest.TestCase):
    def test_identical_logs_pass_strict_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_a = tmp / "run_a.json"
            run_b = tmp / "run_b.json"
            records = [
                {"prompt_index": 0, "prompt_token_ids": [1, 2], "output_token_ids": [10, 11, 12]},
                {"prompt_index": 1, "prompt_token_ids": [3, 4], "output_token_ids": [20, 21]},
            ]
            _write_run_log(run_a, records)
            _write_run_log(run_b, records)

            settings = eval_determinism.EvalSettings(
                exact_match_threshold=1.0,
                position_match_threshold=1.0,
                max_mean_length_delta=0.0,
                max_total_token_delta=0,
            )
            report = eval_determinism.evaluate_runs(run_a, run_b, settings)

            self.assertTrue(report["summary"]["all_passed"])
            exact = next(item for item in report["evaluations"] if item["name"] == "exact_output_match_rate")
            self.assertEqual(exact["score"], 1.0)

    def test_prompt_mismatch_fails_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_a = tmp / "run_a.json"
            run_b = tmp / "run_b.json"
            _write_run_log(run_a, [{"prompt_index": 0, "prompt_token_ids": [1, 2], "output_token_ids": [10]}])
            _write_run_log(run_b, [{"prompt_index": 0, "prompt_token_ids": [1, 999], "output_token_ids": [10]}])

            settings = eval_determinism.EvalSettings(
                exact_match_threshold=1.0,
                position_match_threshold=1.0,
                max_mean_length_delta=0.0,
                max_total_token_delta=0,
            )
            report = eval_determinism.evaluate_runs(run_a, run_b, settings)

            self.assertFalse(report["summary"]["all_passed"])
            alignment = next(item for item in report["evaluations"] if item["name"] == "prompt_alignment")
            self.assertFalse(alignment["passed"])

    def test_supports_sample_output_json_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_a = tmp / "run_a.json"
            run_b = tmp / "run_b.json"
            payload = {
                "model": "test/model",
                "sequences": [
                    {"prompt_token_ids": [1, 2], "output_token_ids": [10, 11]},
                ],
            }
            run_a.write_text(json.dumps(payload), encoding="utf-8")
            run_b.write_text(json.dumps(payload), encoding="utf-8")

            settings = eval_determinism.EvalSettings(
                exact_match_threshold=1.0,
                position_match_threshold=1.0,
                max_mean_length_delta=0.0,
                max_total_token_delta=0,
            )
            report = eval_determinism.evaluate_runs(run_a, run_b, settings)
            self.assertTrue(report["summary"]["all_passed"])


if __name__ == "__main__":
    unittest.main()
