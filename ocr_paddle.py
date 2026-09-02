"""Main-process client for the isolated PaddleOCR-VL worker."""

import base64
import json
import logging
import subprocess
import sys
import threading
import time
import uuid
import queue
from pathlib import Path

log = logging.getLogger("LiveTranslate.PaddleOCR")


class PaddleOCRClientError(RuntimeError):
    pass


class PaddleOCRWorkerError(PaddleOCRClientError):
    pass


class PaddleOCRClient:
    def __init__(self, python_path, model_path, device="gpu:0", cache_dir=None):
        self.python_path = str(python_path)
        self.model_path = str(model_path)
        self.device = str(device or "gpu:0")
        self.cache_dir = str(cache_dir or Path(__file__).parent / "paddlex_cache")
        self._process = None
        self._lock = threading.RLock()
        self._status = "created"
        self._stdout_queue = queue.Queue()

    @property
    def status(self):
        if self._process is not None and self._process.poll() is not None:
            self._status = "exited"
        return self._status

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    def start(self, timeout=240.0):
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            script = Path(__file__).with_name("ocr_paddle_worker.py")
            command = [
                self.python_path,
                "-u",
                str(script),
                "--model-path",
                self.model_path,
                "--device",
                self.device,
                "--cache-dir",
                self.cache_dir,
            ]
            log.info("Starting PaddleOCR worker: %s", command)
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._status = "loading"
            self._stdout_queue = queue.Queue()
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._drain_stderr, daemon=True).start()
            response = self._readline_until(timeout)
            if not response.get("ok") or response.get("type") != "ready":
                self._status = "failed"
                raise PaddleOCRWorkerError(self._error_message(response))
            self._status = "ready"
            log.info("PaddleOCR worker ready: pid=%s, %s", self.pid, response.get("payload"))

    def recognize(self, image_bytes, timeout=180.0):
        with self._lock:
            if self.status != "ready":
                self.start()
            msg_id = uuid.uuid4().hex
            request = {
                "id": msg_id,
                "type": "recognize",
                "payload": {"image_b64": base64.b64encode(image_bytes).decode("ascii")},
            }
            try:
                self._process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._status = "exited"
                raise PaddleOCRClientError(f"PaddleOCR worker pipe closed: {exc}") from exc
            response = self._readline_until(timeout, expected_id=msg_id)
            if not response.get("ok"):
                raise PaddleOCRWorkerError(self._error_message(response))
            return response.get("payload") or {}

    def shutdown(self):
        with self._lock:
            process = self._process
            if process is None:
                return
            try:
                if process.poll() is None and process.stdin:
                    process.stdin.write(json.dumps({"id": uuid.uuid4().hex, "type": "shutdown"}) + "\n")
                    process.stdin.flush()
                    process.wait(timeout=5)
            except Exception:
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            self._process = None
            self._status = "stopped"
            log.info("PaddleOCR worker stopped")

    def _readline_until(self, timeout, expected_id=None):
        if self._process is None:
            raise PaddleOCRClientError("PaddleOCR worker is not running")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise PaddleOCRClientError(
                    f"PaddleOCR worker exited with code {self._process.returncode}"
                )
            try:
                result = self._stdout_queue.get(
                    timeout=max(0.01, min(0.25, deadline - time.monotonic()))
                )
            except queue.Empty:
                continue
            if result is None:
                raise PaddleOCRClientError("PaddleOCR worker closed stdout")
            if isinstance(result, Exception):
                raise PaddleOCRClientError(str(result))
            if expected_id is None or result.get("id") == expected_id:
                return result
            raise PaddleOCRClientError(
                "PaddleOCR worker response id mismatch: "
                f"expected={expected_id}, got={result.get('id')}"
            )
        raise PaddleOCRClientError(f"PaddleOCR worker timed out after {timeout:g}s")

    def _read_stdout(self):
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                try:
                    self._stdout_queue.put(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.warning("Invalid PaddleOCR worker response: %r", line[:300])
                    self._stdout_queue.put(exc)
        except Exception as exc:
            self._stdout_queue.put(exc)
        finally:
            self._stdout_queue.put(None)

    def _drain_stderr(self):
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            line = line.rstrip()
            if line:
                log.debug("PaddleOCR: %s", line)

    @staticmethod
    def _error_message(response):
        error = response.get("error") or {}
        return error.get("message") or response.get("type") or "PaddleOCR worker error"
