import os
import subprocess
import time
from pathlib import Path

import modal

from deterministic_inference.schema import load_manifest
from deterministic_inference.serving import build_serve_plan, wait_for_openai_server

APP_NAME = os.environ.get("DETERMINISTIC_MODAL_APP_NAME", "deterministic-inference-vllm")
DEFAULT_VLLM_IMAGE = os.environ.get(
    "DETERMINISTIC_MODAL_VLLM_IMAGE",
    "docker.io/vllm/vllm-openai@sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0",
)
HF_VOLUME_NAME = os.environ.get("DETERMINISTIC_MODAL_HF_VOLUME", "deterministic-hf-cache")
HF_SECRET_NAME = os.environ.get("DETERMINISTIC_MODAL_HF_SECRET", "huggingface-secret")

REMOTE_ROOT = Path("/opt/deterministic")
REMOTE_CONFIGS = REMOTE_ROOT / "configs"
REMOTE_LAUNCH_SCRIPT = REMOTE_ROOT / "launch-vllm.sh"

SERVE_PORT = 8000
STARTUP_TIMEOUT_SECONDS = 1200


image = (
    modal.Image.from_registry(DEFAULT_VLLM_IMAGE, add_python="3.11")
    .entrypoint([])
    .add_local_python_source("deterministic_inference")
    .add_local_dir(str((Path(__file__).resolve().parents[1] / "configs")), str(REMOTE_CONFIGS))
    .add_local_file(str((Path(__file__).resolve().parents[1] / "scripts" / "launch-vllm.sh")), str(REMOTE_LAUNCH_SCRIPT))
)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/data/hf": modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    min_containers=0,
    scaledown_window=600,
    timeout=3600,
)
@modal.concurrent(max_inputs=256)
class VllmServer:
    config_name: str = modal.parameter()

    def _config_path(self) -> Path:
        config_path = REMOTE_CONFIGS / f"{self.config_name}.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Unable to find config '{self.config_name}' at {config_path}. "
                "The config must exist under the repository configs/ directory."
            )
        return config_path

    @modal.enter()
    def start(self) -> None:
        manifest = load_manifest(self._config_path())
        plan = build_serve_plan(manifest, container_port=SERVE_PORT)

        env = dict(plan.env)
        env["VLLM_HOST"] = "0.0.0.0"
        env["VLLM_PORT"] = str(SERVE_PORT)
        env["VLLM_CONTAINER_PORT"] = str(SERVE_PORT)
        env.setdefault("HF_HOME", "/data/hf")

        self._process = subprocess.Popen(
            ["/bin/sh", str(REMOTE_LAUNCH_SCRIPT)],
            env=env,
        )

        ready = wait_for_openai_server(
            base_url=f"http://127.0.0.1:{SERVE_PORT}",
            timeout_seconds=STARTUP_TIMEOUT_SECONDS,
            poll_interval_seconds=2.0,
        )
        if not ready:
            self._process.terminate()
            raise RuntimeError(
                "vLLM server did not become ready before timeout. "
                f"config={self.config_name}"
            )

    @modal.web_server(port=SERVE_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
    def serve(self) -> None:
        # The server is started in @modal.enter and kept alive by the subprocess.
        return

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is not None:
            return

        process.terminate()
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.5)

        process.kill()


@app.cls(
    image=image,
    gpu="H100:8",
    volumes={"/data/hf": modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    min_containers=0,
    scaledown_window=600,
    timeout=3600,
)
@modal.concurrent(max_inputs=256)
class VllmServerTP8:
    config_name: str = modal.parameter()

    def _config_path(self) -> Path:
        config_path = REMOTE_CONFIGS / f"{self.config_name}.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Unable to find config '{self.config_name}' at {config_path}. "
                "The config must exist under the repository configs/ directory."
            )
        return config_path

    @modal.enter()
    def start(self) -> None:
        manifest = load_manifest(self._config_path())
        plan = build_serve_plan(manifest, container_port=SERVE_PORT)

        env = dict(plan.env)
        env["VLLM_HOST"] = "0.0.0.0"
        env["VLLM_PORT"] = str(SERVE_PORT)
        env["VLLM_CONTAINER_PORT"] = str(SERVE_PORT)
        env.setdefault("HF_HOME", "/data/hf")

        self._process = subprocess.Popen(
            ["/bin/sh", str(REMOTE_LAUNCH_SCRIPT)],
            env=env,
        )

        ready = wait_for_openai_server(
            base_url=f"http://127.0.0.1:{SERVE_PORT}",
            timeout_seconds=STARTUP_TIMEOUT_SECONDS,
            poll_interval_seconds=2.0,
        )
        if not ready:
            self._process.terminate()
            raise RuntimeError(
                "vLLM server did not become ready before timeout. "
                f"config={self.config_name}"
            )

    @modal.web_server(port=SERVE_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
    def serve(self) -> None:
        # The server is started in @modal.enter and kept alive by the subprocess.
        return

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is not None:
            return

        process.terminate()
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.5)

        process.kill()
