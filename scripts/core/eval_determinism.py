#!/usr/bin/env python3
"""Evaluate determinism by comparing token-generation logs across runs.

This module intentionally treats evaluation as a pure function of two run logs.
To extend the suite, add a new function with the @register_evaluator decorator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_LOG_DIR = Path("state/evals/logs")
DEFAULT_REPORT_DIR = Path("state/evals/reports")


@dataclass(frozen=True)
class SequenceRecord:
    prompt_index: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]


@dataclass(frozen=True)
class RunLog:
    source_path: Path
    run_id: str
    model: str
    records: list[SequenceRecord]


@dataclass(frozen=True)
class Alignment:
    aligned_pairs: list[tuple[SequenceRecord, SequenceRecord]]
    prompt_mismatch_indices: list[int]
    extra_records_a: int
    extra_records_b: int


@dataclass(frozen=True)
class EvalSettings:
    exact_match_threshold: float
    position_match_threshold: float
    max_mean_length_delta: float
    max_total_token_delta: int


@dataclass(frozen=True)
class EvalResult:
    name: str
    score: float
    passed: bool
    details: dict[str, Any]


EvaluatorFn = Callable[[RunLog, RunLog, Alignment, EvalSettings], EvalResult]
EVALUATORS: dict[str, EvaluatorFn] = {}


def register_evaluator(name: str) -> Callable[[EvaluatorFn], EvaluatorFn]:
    def decorator(func: EvaluatorFn) -> EvaluatorFn:
        if name in EVALUATORS:
            raise ValueError(f"Duplicate evaluator name: {name}")
        EVALUATORS[name] = func
        return func

    return decorator


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Log must be a JSON object: {path}")
    return raw


def _normalize_run_log(path: Path) -> RunLog:
    payload = _load_json(path)
    model = str(payload.get("model", ""))
    run_id = str(payload.get("created_at_utc", "")) or path.stem

    records: list[SequenceRecord] = []
    if isinstance(payload.get("records"), list):
        for idx, item in enumerate(payload["records"]):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid record at index {idx} in {path}")
            prompt_index = int(item.get("prompt_index", idx))
            prompt_ids = item.get("prompt_token_ids")
            output_ids = item.get("output_token_ids")
            if not isinstance(prompt_ids, list) or any(not isinstance(tok, int) for tok in prompt_ids):
                raise ValueError(f"Invalid prompt_token_ids at index {idx} in {path}")
            if not isinstance(output_ids, list) or any(not isinstance(tok, int) for tok in output_ids):
                raise ValueError(f"Invalid output_token_ids at index {idx} in {path}")
            records.append(
                SequenceRecord(
                    prompt_index=prompt_index,
                    prompt_token_ids=[int(tok) for tok in prompt_ids],
                    output_token_ids=[int(tok) for tok in output_ids],
                )
            )
    elif isinstance(payload.get("sequences"), list):
        # Backward-compatible fallback: sample output JSON.
        for idx, item in enumerate(payload["sequences"]):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid sequence at index {idx} in {path}")
            prompt_ids = item.get("prompt_token_ids")
            output_ids = item.get("output_token_ids")
            if not isinstance(prompt_ids, list) or any(not isinstance(tok, int) for tok in prompt_ids):
                raise ValueError(f"Invalid prompt_token_ids at index {idx} in {path}")
            if not isinstance(output_ids, list) or any(not isinstance(tok, int) for tok in output_ids):
                raise ValueError(f"Invalid output_token_ids at index {idx} in {path}")
            records.append(
                SequenceRecord(
                    prompt_index=idx,
                    prompt_token_ids=[int(tok) for tok in prompt_ids],
                    output_token_ids=[int(tok) for tok in output_ids],
                )
            )
    else:
        raise ValueError(
            f"Unsupported log format for {path}. Expected 'records' (run log) or 'sequences' (sample output)."
        )

    records.sort(key=lambda item: item.prompt_index)
    return RunLog(source_path=path, run_id=run_id, model=model, records=records)


def _align_records(run_a: RunLog, run_b: RunLog) -> Alignment:
    aligned: list[tuple[SequenceRecord, SequenceRecord]] = []
    mismatches: list[int] = []
    shared = min(len(run_a.records), len(run_b.records))

    for idx in range(shared):
        rec_a = run_a.records[idx]
        rec_b = run_b.records[idx]
        if rec_a.prompt_token_ids != rec_b.prompt_token_ids:
            mismatches.append(idx)
            continue
        aligned.append((rec_a, rec_b))

    return Alignment(
        aligned_pairs=aligned,
        prompt_mismatch_indices=mismatches,
        extra_records_a=max(0, len(run_a.records) - shared),
        extra_records_b=max(0, len(run_b.records) - shared),
    )


@register_evaluator("prompt_alignment")
def evaluate_prompt_alignment(run_a: RunLog, run_b: RunLog, alignment: Alignment, _: EvalSettings) -> EvalResult:
    total_prompts = max(len(run_a.records), len(run_b.records))
    mismatches = (
        len(alignment.prompt_mismatch_indices)
        + alignment.extra_records_a
        + alignment.extra_records_b
    )
    score = 1.0 if total_prompts == 0 else max(0.0, 1.0 - (mismatches / total_prompts))
    return EvalResult(
        name="prompt_alignment",
        score=score,
        passed=(mismatches == 0),
        details={
            "total_prompts_a": len(run_a.records),
            "total_prompts_b": len(run_b.records),
            "mismatched_indices": alignment.prompt_mismatch_indices,
            "extra_records_a": alignment.extra_records_a,
            "extra_records_b": alignment.extra_records_b,
        },
    )


@register_evaluator("exact_output_match_rate")
def evaluate_exact_output_match_rate(
    _: RunLog, __: RunLog, alignment: Alignment, settings: EvalSettings
) -> EvalResult:
    total = len(alignment.aligned_pairs)
    if total == 0:
        score = 0.0
        matched = 0
    else:
        matched = sum(1 for rec_a, rec_b in alignment.aligned_pairs if rec_a.output_token_ids == rec_b.output_token_ids)
        score = matched / total
    return EvalResult(
        name="exact_output_match_rate",
        score=score,
        passed=(total > 0 and score >= settings.exact_match_threshold),
        details={
            "aligned_prompts": total,
            "exact_matches": matched,
            "threshold": settings.exact_match_threshold,
        },
    )


@register_evaluator("position_token_match_rate")
def evaluate_position_token_match_rate(
    _: RunLog, __: RunLog, alignment: Alignment, settings: EvalSettings
) -> EvalResult:
    per_prompt_scores: list[float] = []
    for rec_a, rec_b in alignment.aligned_pairs:
        max_len = max(len(rec_a.output_token_ids), len(rec_b.output_token_ids))
        if max_len == 0:
            per_prompt_scores.append(1.0)
            continue
        matches = sum(
            1 for tok_a, tok_b in zip(rec_a.output_token_ids, rec_b.output_token_ids) if tok_a == tok_b
        )
        per_prompt_scores.append(matches / max_len)

    score = sum(per_prompt_scores) / len(per_prompt_scores) if per_prompt_scores else 0.0
    return EvalResult(
        name="position_token_match_rate",
        score=score,
        passed=(len(per_prompt_scores) > 0 and score >= settings.position_match_threshold),
        details={
            "aligned_prompts": len(per_prompt_scores),
            "threshold": settings.position_match_threshold,
        },
    )


@register_evaluator("mean_output_length_delta")
def evaluate_mean_output_length_delta(
    run_a: RunLog, run_b: RunLog, alignment: Alignment, settings: EvalSettings
) -> EvalResult:
    deltas = [abs(len(rec_a.output_token_ids) - len(rec_b.output_token_ids)) for rec_a, rec_b in alignment.aligned_pairs]
    mean_delta = (sum(deltas) / len(deltas)) if deltas else float("inf")
    score = 0.0 if mean_delta == float("inf") else 1.0 / (1.0 + mean_delta)
    return EvalResult(
        name="mean_output_length_delta",
        score=score,
        passed=(mean_delta != float("inf") and mean_delta <= settings.max_mean_length_delta),
        details={
            "aligned_prompts": len(alignment.aligned_pairs),
            "mean_length_delta": mean_delta,
            "threshold": settings.max_mean_length_delta,
            "total_output_tokens_a": sum(len(rec.output_token_ids) for rec in run_a.records),
            "total_output_tokens_b": sum(len(rec.output_token_ids) for rec in run_b.records),
        },
    )


@register_evaluator("total_output_token_delta")
def evaluate_total_output_token_delta(
    run_a: RunLog, run_b: RunLog, _: Alignment, settings: EvalSettings
) -> EvalResult:
    total_a = sum(len(rec.output_token_ids) for rec in run_a.records)
    total_b = sum(len(rec.output_token_ids) for rec in run_b.records)
    delta = abs(total_a - total_b)
    score = 1.0 / (1.0 + delta)
    return EvalResult(
        name="total_output_token_delta",
        score=score,
        passed=(delta <= settings.max_total_token_delta),
        details={
            "total_output_tokens_a": total_a,
            "total_output_tokens_b": total_b,
            "abs_delta": delta,
            "threshold": settings.max_total_token_delta,
        },
    )


def evaluate_runs(run_a_path: Path, run_b_path: Path, settings: EvalSettings) -> dict[str, Any]:
    run_a = _normalize_run_log(run_a_path)
    run_b = _normalize_run_log(run_b_path)
    alignment = _align_records(run_a, run_b)
    evaluations = [
        evaluator(run_a, run_b, alignment, settings)
        for evaluator in EVALUATORS.values()
    ]
    all_passed = all(result.passed for result in evaluations)

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "run_a": {
            "path": str(run_a.source_path),
            "run_id": run_a.run_id,
            "model": run_a.model,
            "prompt_count": len(run_a.records),
        },
        "run_b": {
            "path": str(run_b.source_path),
            "run_id": run_b.run_id,
            "model": run_b.model,
            "prompt_count": len(run_b.records),
        },
        "evaluations": [
            {
                "name": result.name,
                "score": result.score,
                "passed": result.passed,
                "details": result.details,
            }
            for result in evaluations
        ],
        "summary": {
            "all_passed": all_passed,
            "suite_size": len(evaluations),
            "passing_evaluations": sum(1 for result in evaluations if result.passed),
        },
    }


def _discover_latest_pair(log_dir: Path) -> tuple[Path, Path]:
    if not log_dir.is_absolute():
        log_dir = (Path(__file__).resolve().parents[2] / log_dir).resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    candidates = sorted(
        [path for path in log_dir.rglob("*.json") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
    )
    if len(candidates) < 2:
        raise FileNotFoundError(f"Need at least two log files under {log_dir} to evaluate.")
    return candidates[-2], candidates[-1]


def _default_report_path(run_a: Path, run_b: Path) -> Path:
    report_dir = (Path(__file__).resolve().parents[2] / DEFAULT_REPORT_DIR).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return report_dir / f"determinism_eval_{run_a.stem}_vs_{run_b.stem}_{ts}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate determinism by comparing two run logs.")
    parser.add_argument("--run-a", default="", help="Path to first run log JSON.")
    parser.add_argument("--run-b", default="", help="Path to second run log JSON.")
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory to auto-discover latest two logs when --run-a/--run-b are omitted.",
    )
    parser.add_argument("--output", default="", help="Optional output report JSON path.")
    parser.add_argument("--exact-match-threshold", type=float, default=1.0)
    parser.add_argument("--position-match-threshold", type=float, default=1.0)
    parser.add_argument("--max-mean-length-delta", type=float, default=0.0)
    parser.add_argument("--max-total-token-delta", type=int, default=0)
    parser.add_argument(
        "--no-fail-on-eval",
        action="store_true",
        help="Always exit 0 even if evaluations fail.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.run_a or args.run_b:
        if not (args.run_a and args.run_b):
            raise SystemExit("Provide both --run-a and --run-b, or provide neither to auto-discover latest logs.")
        run_a = Path(args.run_a).expanduser().resolve()
        run_b = Path(args.run_b).expanduser().resolve()
    else:
        run_a, run_b = _discover_latest_pair(Path(args.log_dir).expanduser())

    settings = EvalSettings(
        exact_match_threshold=args.exact_match_threshold,
        position_match_threshold=args.position_match_threshold,
        max_mean_length_delta=args.max_mean_length_delta,
        max_total_token_delta=args.max_total_token_delta,
    )
    report = evaluate_runs(run_a, run_b, settings)

    report_path = Path(args.output).expanduser().resolve() if args.output else _default_report_path(run_a, run_b)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Run A: {run_a}")
    print(f"Run B: {run_b}")
    print(f"Report: {report_path}")
    print(f"All evaluations passed: {report['summary']['all_passed']}")

    if args.no_fail_on_eval:
        return 0
    return 0 if report["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
