"""Qwen3-ASR client running in its dedicated Conda environment."""

import base64
import collections
import json
import logging
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid

import numpy as np

from asr_client import ASRClientError, ASRWorkerError, ASRWorkerExited, ASRWorkerTimeout


log = logging.getLogger("LiveTranslate.ASR.Qwen3")


_CONTEXT_FILLERS = {
    "ああ", "あー", "ええ", "えー", "うん", "はい", "へえ", "へー",
    "おお", "おー", "まあ", "そう", "そうか", "なるほど", "えっと",
    "嗯", "啊", "哦", "好的", "对", "是的", "okay", "ok", "uh", "um",
    "yeah", "yes", "right",
}


def _context_key(text):
    return "".join(ch.casefold() for ch in str(text or "") if ch.isalnum())


def _is_useful_context(text):
    key = _context_key(text)
    return len(key) > 1 and key not in _CONTEXT_FILLERS


def _normalize_hotwords(text):
    """Normalize a user-entered keyword list while preserving multi-word names."""
    parts = re.split(r"[\r\n,，、;；|]+", str(text or ""))
    output = []
    seen = set()
    for part in parts:
        value = " ".join(part.split())
        key = value.casefold()
        if value and key not in seen:
            output.append(value)
            seen.add(key)
    return " ".join(output)


def _hotword_keys(text):
    """Return comparison keys for the comma-separated ASR keyword list."""
    return {
        _context_key(value)
        for value in re.split(r"[\r\n,，、;；|]+", str(text or ""))
        if _context_key(value)
    }


def _config_int(config, key, default, minimum=0):
    """Read an integer setting without treating a valid zero as missing."""
    value = config.get(key)
    if value is None or value == "":
        value = default
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


class Qwen3ASRClient:
    """ASRClient-compatible proxy for a Qwen worker launched by Conda Python."""

    def __init__(self, config: dict, ready_timeout=300.0, request_timeout=60.0, shutdown_timeout=10.0):
        self.config = dict(config)
        self.ready_timeout = ready_timeout
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self._process = None
        self._responses = queue.Queue()
        self._lock = threading.RLock()
        self._status = "created"
        self._language = None
        self._context_max_chars = _config_int(
            self.config, "qwen_context_max_chars", 320
        )
        self._static_context_max_chars = _config_int(
            self.config, "qwen_static_context_max_chars", 1536
        )
        self._static_context = _normalize_hotwords(
            self.config.get("qwen_hotwords") or ""
        )
        self._static_context_keys = _hotword_keys(
            self.config.get("qwen_hotwords") or ""
        )
        self._static_context_sequence = _context_key(self._static_context)
        self._context = collections.deque(
            maxlen=_config_int(self.config, "qwen_context_turns", 3)
        )

    @property
    def status(self):
        if self._process is not None and self._process.poll() is not None:
            if self._status not in ("failed", "stopping", "stopped"):
                self._status = "exited"
        return self._status

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    def start(self):
        with self._lock:
            if self._process is not None:
                return
            python = Path(str(self.config.get("qwen_python") or ""))
            project = Path(str(self.config.get("qwen_project") or ""))
            model = Path(str(self.config.get("qwen_model_path") or ""))
            worker = Path(__file__).with_name("asr_qwen3_worker.py")
            for label, path, check in (
                ("Qwen3-ASR Python", python, Path.is_file),
                ("Qwen3-ASR source directory", project, Path.is_dir),
                ("Qwen3-ASR model directory", model, Path.is_dir),
            ):
                if not check(path):
                    raise FileNotFoundError(f"{label} was not found: {path or '(not configured)'}")

            args = [
                str(python), "-u", str(worker),
                "--project-path", str(project),
                "--model-path", str(model),
                "--device", str(self.config.get("device") or "cuda:0"),
                "--dtype", str(self.config.get("qwen_dtype") or "auto"),
                "--max-new-tokens", str(self.config.get("qwen_max_new_tokens") or 128),
            ]
            self._process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            self._status = "starting"
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
            log.info("Qwen3-ASR worker started: pid=%s", self.pid)

    def wait_ready(self, timeout=None):
        self._status = "loading"
        response = self._recv_response(self.ready_timeout if timeout is None else timeout, None)
        if not response.get("ok"):
            self._status = "failed"
            raise ASRWorkerError(response.get("error") or {})
        if response.get("type") != "ready":
            self._status = "failed"
            raise ASRClientError(f"Unexpected Qwen3-ASR startup response: {response.get('type')}")
        self._status = "ready"
        return response.get("payload")

    def transcribe(self, audio, word_timestamps=False, **kwargs):
        del word_timestamps
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        use_static_context = bool(kwargs.pop("use_static_context", True))
        extra_context = str(kwargs.pop("context", "") or "").strip()
        draft = str(kwargs.pop("draft", "") or "").strip()
        rolling_context = self._build_context(use_static_context)
        context = "\n".join(
            value for value in (rolling_context, extra_context, draft) if value
        )
        payload = {
            "audio_b64": base64.b64encode(samples.tobytes()).decode("ascii"),
            "context": context,
        }
        response = self._request("transcribe", payload, self.request_timeout)
        result = response.get("payload")
        if isinstance(result, dict):
            metrics = result.get("metrics") or {}
            log.debug(
                "Qwen ASR result: audio=%.2fs context=%d chars decode=%.0fms output=%d chars",
                float(metrics.get("audio_seconds") or 0),
                int(metrics.get("context_chars") or 0),
                float(metrics.get("decode_ms") or 0),
                len(str(result.get("text") or "")),
            )
        return result

    def refine(self, audio, draft, word_timestamps=False, **kwargs):
        """Re-decode audio with the first transcript as a constrained hint."""
        return self.transcribe(
            audio,
            word_timestamps=word_timestamps,
            draft=f"First-pass transcript; correct only against the audio:\n{draft}",
            **kwargs,
        )

    def set_language(self, language):
        self._request("set_language", {"language": language}, min(10.0, self.request_timeout))
        normalized = str(language or "auto").strip().lower()
        with self._lock:
            if self._language is not None and normalized != self._language:
                self._context.clear()
            self._language = normalized

    def set_context_turns(self, turns):
        turns = max(0, int(turns))
        with self._lock:
            previous = list(self._context)
            self._context = collections.deque(previous[-turns:] if turns else (), maxlen=turns)

    def commit_context(self, text):
        text = str(text or "").strip()
        if not _is_useful_context(text):
            log.debug("Qwen context: ignored low-information text: %r", text)
            return
        key = _context_key(text)
        with self._lock:
            if self._context.maxlen and (
                not self._context or _context_key(self._context[-1]) != key
            ):
                self._context.append(text)

    def set_static_context(self, text):
        value = _normalize_hotwords(text)
        with self._lock:
            self._static_context = value
            self._static_context_keys = _hotword_keys(text)
            self._static_context_sequence = _context_key(value)
        log.info("Qwen ASR keywords updated: %d chars", len(value))

    def is_static_context_echo(self, text) -> bool:
        """Whether a result is only a keyword hint, possibly repeated."""
        key = _context_key(text)
        if not key:
            return False
        with self._lock:
            keys = tuple(self._static_context_keys)
            sequence = self._static_context_sequence
        # A failed decode can spill the full vocabulary prompt instead of one
        # word. This is much longer than any normal isolated proper name.
        if len(key) >= 80 and sequence.startswith(key):
            return True
        for hotword_key in keys:
            if key == hotword_key:
                return True
            # Qwen can repeat a context term on short or near-silent audio.
            if len(key) > len(hotword_key) and key == hotword_key * (
                len(key) // len(hotword_key)
            ):
                return True
        return False

    def clear_context(self):
        with self._lock:
            self._context.clear()

    def _build_context(self, use_static_context=True):
        with self._lock:
            static_context = self._static_context if use_static_context else ""
            rolling_context = "\n".join(self._context)
        if (
            self._static_context_max_chars
            and len(static_context) > self._static_context_max_chars
        ):
            static_context = static_context[: self._static_context_max_chars].rstrip()
        if (
            self._context_max_chars
            and len(rolling_context) > self._context_max_chars
        ):
            rolling_context = rolling_context[-self._context_max_chars :]
        values = []
        if static_context:
            # Qwen3-ASR receives this as a system message.  Make the intended
            # role explicit so a vocabulary list is not treated as text to emit.
            values.append(
                "Vocabulary hints for recognizing the audio only. "
                "Transcribe only words supported by the audio; never output a "
                "hint merely because it appears here:\n"
                f"{static_context}"
            )
        if rolling_context:
            values.append(f"Previous transcript context:\n{rolling_context}")
        log.debug(
            "Qwen ASR request context: keywords=%s rolling_turns=%d chars=%d",
            bool(static_context), len(self._context), sum(len(value) for value in values),
        )
        return "\n\n".join(values)

    def set_input_padding(self, pad_seconds):
        del pad_seconds  # Qwen consumes the exact VAD segment; no Whisper-style padding.

    def shutdown(self):
        with self._lock:
            process = self._process
            if process is None:
                return
            self._status = "stopping"
            if process.poll() is None:
                try:
                    self._send({"id": uuid.uuid4().hex, "type": "shutdown", "payload": {}})
                except (ASRWorkerExited, BrokenPipeError, OSError):
                    pass
        try:
            process.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            log.warning("Qwen3-ASR worker did not exit; terminating pid=%s", process.pid)
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.shutdown_timeout)
        finally:
            with self._lock:
                self._close_handles()
                self._status = "stopped"

    def terminate(self):
        with self._lock:
            process = self._process
            self._status = "failed"
            if process is not None and process.poll() is None:
                process.terminate()
        if process is not None:
            try:
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
        with self._lock:
            self._close_handles()

    def _request(self, request_type, payload, timeout):
        with self._lock:
            self._ensure_ready()
            msg_id = uuid.uuid4().hex
            self._send({"id": msg_id, "type": request_type, "payload": payload})
            previous_status = self._status
            if request_type == "transcribe":
                self._status = "busy"
        try:
            response = self._recv_response(timeout, msg_id)
        finally:
            with self._lock:
                if self._status == "busy":
                    self._status = previous_status
        if not response.get("ok"):
            raise ASRWorkerError(response.get("error") or {})
        return response

    def _send(self, message):
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            self._status = "exited"
            raise ASRWorkerExited("Qwen3-ASR worker has exited")
        try:
            # Keep stdin ASCII-only for the same reason as worker stdout: the
            # LiveTranslate runtime and Conda Python may use different Windows
            # locale encodings. JSON escapes preserve Japanese/Korean exactly.
            process.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._status = "exited"
            raise ASRWorkerExited(f"Qwen3-ASR worker pipe closed: {exc}") from exc

    def _recv_response(self, timeout, expected_id):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._status = "failed"
                self.terminate()
                raise ASRWorkerTimeout(f"Qwen3-ASR worker response timed out after {timeout:g}s")
            try:
                response = self._responses.get(timeout=min(0.2, remaining))
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    self._status = "exited"
                    raise ASRWorkerExited(f"Qwen3-ASR worker exited with code {self._process.returncode}")
                continue
            if expected_id is None or response.get("id") == expected_id:
                return response
            raise ASRClientError(f"Qwen3-ASR response id mismatch: expected={expected_id}, got={response.get('id')}")

    def _read_stdout(self):
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Unexpected Qwen3-ASR worker stdout: %s", line.rstrip())

    def _read_stderr(self):
        if self._process is None or self._process.stderr is None:
            return
        for line in self._process.stderr:
            if line.strip():
                log.info("Qwen3-ASR: %s", line.rstrip())

    def _ensure_ready(self):
        if self._process is None:
            raise ASRClientError("Qwen3-ASR worker has not been started")
        if self.status != "ready":
            raise ASRClientError(f"Qwen3-ASR worker is not ready: {self.status}")

    def _close_handles(self):
        if self._process is not None:
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self._process = None
