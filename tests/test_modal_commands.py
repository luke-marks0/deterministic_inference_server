from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import deterministic_inference as workflow

from test_helpers import lock_manifest, make_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class TestModalCommands(unittest.TestCase):
    def test_serve_modal_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            manifest = workflow.create_manifest_template(
                model_id="openai/gpt-oss-20b",
                model_revision="6cee5e81ee83917806bbde320786a8fb61efebee",
                output_path=manifest_path,
            )
            manifest["runtime"]["execution"]["backend"] = "openai_compatible"
            manifest["runtime"]["execution"]["base_url"] = "http://127.0.0.1:8123"
            manifest["vllm"]["mode"] = "server"
            _write_json(manifest_path, manifest)

            rc = workflow.main(
                [
                    "serve-modal",
                    "--config",
                    str(manifest_path),
                    "--dry-run",
                ]
            )

        self.assertEqual(rc, 0)

    def test_serve_modal_start_calls_deploy_lookup_and_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            manifest = workflow.create_manifest_template(
                model_id="openai/gpt-oss-20b",
                model_revision="6cee5e81ee83917806bbde320786a8fb61efebee",
                output_path=manifest_path,
            )
            manifest["runtime"]["execution"]["backend"] = "openai_compatible"
            manifest["runtime"]["execution"]["base_url"] = "http://127.0.0.1:8123"
            manifest["vllm"]["mode"] = "server"
            _write_json(manifest_path, manifest)

            with (
                mock.patch("deterministic_inference.cli.deploy_modal_app") as deploy_mock,
                mock.patch(
                    "deterministic_inference.cli.lookup_modal_endpoint",
                    return_value="https://example.modal.run?config_name=manifest",
                ) as lookup_mock,
                mock.patch("deterministic_inference.cli.upsert_modal_server") as upsert_mock,
                mock.patch("deterministic_inference.cli.wait_for_openai_server", return_value=True) as wait_mock,
            ):
                rc = workflow.main(
                    [
                        "serve-modal",
                        "--config",
                        str(manifest_path),
                    ]
                )

        self.assertEqual(rc, 0)
        deploy_mock.assert_called_once()
        lookup_mock.assert_called_once()
        upsert_mock.assert_called_once()
        wait_mock.assert_called_once()

    def test_run_use_modal_passes_base_url_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"

            manifest = make_manifest(manifest_path)
            manifest["runtime"]["execution"]["backend"] = "openai_compatible"
            manifest["runtime"]["execution"]["base_url"] = "http://127.0.0.1:8123"
            manifest["vllm"]["mode"] = "server"
            _write_json(manifest_path, manifest)
            lock_manifest(manifest_path)

            with (
                mock.patch(
                    "deterministic_inference.cli.lookup_modal_endpoint",
                    return_value="https://example.modal.run?config_name=manifest",
                ),
                mock.patch("deterministic_inference.cli.list_modal_servers", return_value=[]),
                mock.patch("deterministic_inference.cli.execute_run") as execute_run_mock,
            ):
                execute_run_mock.return_value = {
                    "run_dir": tmp_path / "run",
                    "bundle_path": tmp_path / "run" / "bundle.json",
                    "token_output_path": tmp_path / "run" / "tokens.json",
                    "run_log_path": tmp_path / "run" / "run_log.json",
                    "grade": "conformant",
                }

                rc = workflow.main(
                    [
                        "run",
                        "--config",
                        str(manifest_path),
                        "--use-modal",
                    ]
                )

        self.assertEqual(rc, 0)
        self.assertEqual(execute_run_mock.call_count, 1)
        kwargs = execute_run_mock.call_args.kwargs
        self.assertEqual(
            kwargs["base_url_override"],
            "https://example.modal.run?config_name=manifest",
        )


if __name__ == "__main__":
    unittest.main()
