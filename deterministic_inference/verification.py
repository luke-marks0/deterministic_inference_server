from __future__ import annotations

import textwrap
import tarfile
from pathlib import Path
from typing import Any

from .common import BUNDLE_KIND, LOCK_KIND, SCHEMA_VERSION, VERIFY_REPORT_KIND, _load_json_object, _write_json, utc_now_iso
from .locking import _validate_lock_payload
from .schema import MANIFEST_KIND, compute_manifest_id, load_manifest, resolve_inference_requests, validate_manifest

def _resolve_bundle_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "bundle.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing bundle.json under run directory: {path}")
        return candidate
    return path


def load_bundle(path: Path) -> tuple[Path, dict[str, Any]]:
    bundle_path = _resolve_bundle_path(path)
    bundle = _load_json_object(bundle_path)
    if bundle.get("kind") != BUNDLE_KIND:
        raise ValueError(f"Unexpected bundle kind in {bundle_path}: {bundle.get('kind')!r}")
    return bundle_path, bundle


def _load_sequences_for_bundle(bundle_path: Path, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    token_path = bundle_path.parent / str(bundle["paths"]["tokens"])
    payload = _load_json_object(token_path)
    sequences = payload.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError(f"Invalid token payload at {token_path}: missing sequences list.")
    for idx, sequence in enumerate(sequences):
        if not isinstance(sequence, dict):
            raise ValueError(f"Invalid sequence at index {idx} in {token_path}.")
        prompt = sequence.get("prompt_token_ids")
        output = sequence.get("output_token_ids")
        if not isinstance(prompt, list) or any(not isinstance(tok, int) for tok in prompt):
            raise ValueError(f"Invalid prompt_token_ids for sequence {idx} in {token_path}.")
        if not isinstance(output, list) or any(not isinstance(tok, int) for tok in output):
            raise ValueError(f"Invalid output_token_ids for sequence {idx} in {token_path}.")
    return sequences


def _compare_token_sequences(
    sequences_a: list[dict[str, Any]],
    sequences_b: list[dict[str, Any]],
    *,
    rule: str,
) -> dict[str, Any]:
    if rule != "exact":
        raise ValueError(
            "Only determinism.compare.tokens.rule='exact' is currently implemented. "
            f"Found: {rule!r}."
        )

    max_pairs = min(len(sequences_a), len(sequences_b))
    mismatches = 0
    first_divergence: dict[str, Any] | None = None
    max_abs_diff = 0
    max_rel_diff = 0.0

    for seq_idx in range(max_pairs):
        prompt_a = sequences_a[seq_idx]["prompt_token_ids"]
        prompt_b = sequences_b[seq_idx]["prompt_token_ids"]
        if prompt_a != prompt_b:
            mismatches += 1
            if first_divergence is None:
                first_divergence = {
                    "request_id": seq_idx,
                    "token_step": None,
                    "field": "prompt_token_ids",
                    "detail": "Prompt token ids differ.",
                }
            continue

        output_a = sequences_a[seq_idx]["output_token_ids"]
        output_b = sequences_b[seq_idx]["output_token_ids"]
        max_step = max(len(output_a), len(output_b))
        sequence_mismatch = False
        for step in range(max_step):
            tok_a = output_a[step] if step < len(output_a) else None
            tok_b = output_b[step] if step < len(output_b) else None
            if tok_a == tok_b:
                continue

            sequence_mismatch = True
            abs_diff = abs(int(tok_a or 0) - int(tok_b or 0))
            max_abs_diff = max(max_abs_diff, abs_diff)
            denominator = abs(int(tok_b or 0))
            rel_diff = float(abs_diff) if denominator == 0 else abs_diff / denominator
            max_rel_diff = max(max_rel_diff, rel_diff)

            if first_divergence is None:
                first_divergence = {
                    "request_id": seq_idx,
                    "token_step": step,
                    "field": "output_token_ids",
                    "token_a": tok_a,
                    "token_b": tok_b,
                }
        if sequence_mismatch:
            mismatches += 1

    mismatches += abs(len(sequences_a) - len(sequences_b))

    exact_match = mismatches == 0 and len(sequences_a) == len(sequences_b)
    return {
        "exact_match": exact_match,
        "mismatched_sequences": mismatches,
        "first_divergence": first_divergence,
        "counts": {
            "sequence_count_a": len(sequences_a),
            "sequence_count_b": len(sequences_b),
        },
        "numeric_diff_stats": {
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "ulp_stats": {
                "supported": False,
                "note": "ULP stats are not computed for token id comparisons.",
            },
        },
    }


def _capture_compare_status(bundle_a: dict[str, Any], bundle_b: dict[str, Any], key: str) -> dict[str, Any]:
    enabled_a = bool(bundle_a.get("capture", {}).get(key, {}).get("enabled", False))
    enabled_b = bool(bundle_b.get("capture", {}).get(key, {}).get("enabled", False))
    if enabled_a or enabled_b:
        return {
            "status": "skipped_not_implemented",
            "note": "Configured in manifest but intentionally a no-op for now.",
        }
    return {"status": "disabled"}


def _load_manifest_for_bundle(bundle_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    manifest_relative = str(bundle.get("paths", {}).get("manifest", "manifest.used.json"))
    manifest_path = bundle_path.parent / manifest_relative
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest payload referenced by bundle is missing: {manifest_path}")
    manifest_payload = _load_json_object(manifest_path)
    validate_manifest(manifest_payload)
    return manifest_payload


def _token_compare_rule(manifest: dict[str, Any]) -> str:
    compare = manifest.get("determinism", {}).get("compare", {})
    token_compare = compare.get("tokens", {})
    rule = token_compare.get("rule")
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError("Manifest determinism.compare.tokens.rule must be a non-empty string.")
    return rule


def _compare_batch_trace(bundle_a: dict[str, Any], bundle_b: dict[str, Any]) -> dict[str, Any]:
    trace_a = bundle_a.get("batch_trace", {})
    trace_b = bundle_b.get("batch_trace", {})
    steps_a = trace_a.get("steps")
    steps_b = trace_b.get("steps")
    if not isinstance(steps_a, list) or not isinstance(steps_b, list):
        return {
            "status": "missing",
            "exact_match": False,
            "first_divergence": "batch_trace.steps missing in one or both bundles.",
        }

    if steps_a == steps_b:
        return {
            "status": "compared",
            "exact_match": True,
            "first_divergence": None,
        }

    first_diff: dict[str, Any] | None = None
    max_len = max(len(steps_a), len(steps_b))
    for idx in range(max_len):
        step_a = steps_a[idx] if idx < len(steps_a) else None
        step_b = steps_b[idx] if idx < len(steps_b) else None
        if step_a == step_b:
            continue
        first_diff = {
            "step_index": idx,
            "step_a": step_a,
            "step_b": step_b,
        }
        break

    return {
        "status": "compared",
        "exact_match": False,
        "first_divergence": first_diff,
    }


def verify_bundles(
    bundle_a_path: Path,
    bundle_b_path: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    resolved_a_path, bundle_a = load_bundle(bundle_a_path)
    resolved_b_path, bundle_b = load_bundle(bundle_b_path)
    manifest_a = _load_manifest_for_bundle(resolved_a_path, bundle_a)
    manifest_b = _load_manifest_for_bundle(resolved_b_path, bundle_b)
    token_rule_a = _token_compare_rule(manifest_a)
    token_rule_b = _token_compare_rule(manifest_b)
    token_rule_match = token_rule_a == token_rule_b

    sequences_a = _load_sequences_for_bundle(resolved_a_path, bundle_a)
    sequences_b = _load_sequences_for_bundle(resolved_b_path, bundle_b)

    if token_rule_match:
        tokens_report = _compare_token_sequences(
            sequences_a,
            sequences_b,
            rule=token_rule_a,
        )
    else:
        tokens_report = {
            "exact_match": False,
            "mismatched_sequences": abs(len(sequences_a) - len(sequences_b)),
            "first_divergence": {
                "field": "determinism.compare.tokens.rule",
                "rule_a": token_rule_a,
                "rule_b": token_rule_b,
            },
            "counts": {
                "sequence_count_a": len(sequences_a),
                "sequence_count_b": len(sequences_b),
            },
            "numeric_diff_stats": {
                "max_abs_diff": 0,
                "max_rel_diff": 0.0,
                "ulp_stats": {
                    "supported": False,
                    "note": "Token comparison was not executed because token compare rules differ.",
                },
            },
        }
    tokens_report["rule"] = token_rule_a if token_rule_match else None

    environment_diffs = {
        "runtime_closure_digest_match": bundle_a.get("runtime_closure_digest")
        == bundle_b.get("runtime_closure_digest"),
        "manifest_id_match": bundle_a.get("manifest_id") == bundle_b.get("manifest_id"),
        "lock_id_match": bundle_a.get("lock_id") == bundle_b.get("lock_id"),
        "hardware_fingerprint_match": bundle_a.get("hardware_fingerprint")
        == bundle_b.get("hardware_fingerprint"),
        "token_compare_rule_match": token_rule_match,
    }
    batch_trace_report = _compare_batch_trace(bundle_a, bundle_b)

    if not bundle_a.get("hardware_conformant", False) or not bundle_b.get(
        "hardware_conformant", False
    ):
        grade = "non_conformant_hardware"
    elif not (
        environment_diffs["runtime_closure_digest_match"]
        and environment_diffs["manifest_id_match"]
        and environment_diffs["lock_id_match"]
        and environment_diffs["token_compare_rule_match"]
    ):
        grade = "non_conformant_software"
    elif not tokens_report["exact_match"]:
        grade = "mismatch_outputs"
    else:
        grade = "conformant"

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": VERIFY_REPORT_KIND,
        "created_at": utc_now_iso(),
        "bundle_a": {
            "path": str(resolved_a_path),
            "run_id": bundle_a.get("run_id"),
        },
        "bundle_b": {
            "path": str(resolved_b_path),
            "run_id": bundle_b.get("run_id"),
        },
        "comparisons": {
            "tokens": tokens_report,
            "batch_trace": batch_trace_report,
            "logits": _capture_compare_status(bundle_a, bundle_b, "logits"),
            "activations": _capture_compare_status(bundle_a, bundle_b, "activations"),
            "engine_trace": _capture_compare_status(bundle_a, bundle_b, "engine_trace"),
        },
        "environment_diffs": environment_diffs,
        "determinism_grade": grade,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "verify_report.json"
    summary_path = output_dir / "verify_summary.txt"

    _write_json(report_path, report)

    summary = textwrap.dedent(
        f"""\
        verify_report: {report_path}
        bundle_a: {resolved_a_path}
        bundle_b: {resolved_b_path}
        determinism_grade: {grade}
        tokens_exact_match: {tokens_report['exact_match']}
        first_divergence: {tokens_report['first_divergence']}
        runtime_closure_digest_match: {environment_diffs['runtime_closure_digest_match']}
        manifest_id_match: {environment_diffs['manifest_id_match']}
        lock_id_match: {environment_diffs['lock_id_match']}
        token_compare_rule_match: {environment_diffs['token_compare_rule_match']}
        batch_trace_exact_match: {batch_trace_report['exact_match']}
        """
    ).strip() + "\n"
    summary_path.write_text(summary, encoding="utf-8")

    return {
        "report_path": report_path,
        "summary_path": summary_path,
        "grade": grade,
    }


def create_bundle_archive(*, run_dir: Path, output_path: Path) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    bundle_path = run_dir / "bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing bundle.json in run directory: {run_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    return output_path


def inspect_payload(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)

    # Allow inspecting inherited manifests that rely on x_base_manifest.
    if isinstance(payload.get("x_base_manifest"), str):
        manifest = load_manifest(path)
        lock_ref = manifest["artifacts"]["lockfile"]
        request_count = len(resolve_inference_requests(manifest))
        return {
            "type": "manifest",
            "schema_version": manifest["schema_version"],
            "manifest_id": compute_manifest_id(manifest),
            "lockfile": "<embedded>" if isinstance(lock_ref, dict) else lock_ref,
            "request_count": request_count,
            "batching": manifest["inference"]["batching"],
        }

    kind = payload.get("kind")
    if kind == MANIFEST_KIND:
        validate_manifest(payload)
        lock_ref = payload["artifacts"]["lockfile"]
        request_count = len(resolve_inference_requests(payload))
        return {
            "type": "manifest",
            "schema_version": payload["schema_version"],
            "manifest_id": compute_manifest_id(payload),
            "lockfile": "<embedded>" if isinstance(lock_ref, dict) else lock_ref,
            "request_count": request_count,
            "batching": payload["inference"]["batching"],
        }

    if kind == LOCK_KIND:
        normalized = _validate_lock_payload(payload, source=str(path))
        return {
            "type": "lock",
            "schema_version": payload["schema_version"],
            "manifest_id": normalized["manifest_id"],
            "lock_id": payload["lock_id"],
            "artifact_count": len(payload.get("artifacts", [])),
            "runtime_closure_digest": normalized.get("runtime_closure_digest", ""),
        }

    if kind == BUNDLE_KIND:
        return {
            "type": "bundle",
            "schema_version": payload["schema_version"],
            "run_id": payload.get("run_id"),
            "manifest_id": payload.get("manifest_id"),
            "lock_id": payload.get("lock_id"),
            "determinism_grade": payload.get("determinism_grade"),
        }

    raise ValueError(f"Unsupported payload kind for inspect: {kind!r}")
