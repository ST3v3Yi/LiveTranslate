"""Dedicated-process Qwen3-ASR worker for LiveTranslate."""

import argparse
import base64
import contextlib
import json
import logging
import sys
import time
import traceback

import numpy as np


LANGUAGE_MAP = {
    "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fa": "Persian", "fi": "Finnish", "fil": "Filipino",
    "fr": "French", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "mk": "Macedonian", "ms": "Malay", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sv": "Swedish",
    "th": "Thai", "tr": "Turkish", "vi": "Vietnamese", "yue": "Cantonese", "zh": "Chinese",
}
LANGUAGE_CODE_MAP = {name.lower(): code for code, name in LANGUAGE_MAP.items()}


def emit(message):
    # Keep the wire protocol ASCII-only. Windows pipe encodings otherwise turn
    # Japanese/Korean text into replacement characters when the parent process
    # and the Conda worker use different locale encodings.
    sys.stdout.write(json.dumps(message, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def emit_error(msg_id, exc, recoverable):
    emit({"id": msg_id, "ok": False, "type": "error", "error": {
        "message": str(exc), "traceback": traceback.format_exc(), "recoverable": recoverable,
    }})


def map_language(language):
    value = str(language or "").strip().lower()
    return None if not value or value == "auto" else LANGUAGE_MAP.get(value)


def language_code(language):
    """Convert Qwen's canonical language names back to LiveTranslate codes."""
    names = [part.strip() for part in str(language or "").split(",") if part.strip()]
    if not names:
        return "unknown"
    return ",".join(LANGUAGE_CODE_MAP.get(name.lower(), name.lower()) for name in names)


def load_model(args):
    sys.path.insert(0, args.project_path)
    logging.getLogger().setLevel(logging.WARNING)
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from qwen_asr import Qwen3ASRModel

        device = "cuda:0" if args.device == "cuda" else args.device
        dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
        return Qwen3ASRModel.from_pretrained(
            args.model_path,
            device_map=device,
            dtype=dtype,
            max_inference_batch_size=1,
            max_new_tokens=args.max_new_tokens,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    try:
        model = load_model(args)
    except BaseException as exc:
        emit_error(None, exc, False)
        return 1

    language = None
    emit({"id": None, "ok": True, "type": "ready", "payload": {"engine_type": "qwen3-asr", "device": args.device}})
    for line in sys.stdin:
        msg_id = None
        try:
            message = json.loads(line)
            msg_id = message.get("id")
            request_type = message.get("type")
            payload = message.get("payload") or {}
            if request_type == "shutdown":
                emit({"id": msg_id, "ok": True, "type": "shutdown"})
                return 0
            if request_type == "set_language":
                language = payload.get("language")
                emit({"id": msg_id, "ok": True, "type": "ack"})
                continue
            if request_type != "transcribe":
                raise ValueError(f"Unknown Qwen3-ASR worker command: {request_type}")
            raw = base64.b64decode(payload["audio_b64"])
            if len(raw) % 4:
                raise ValueError("Qwen3-ASR audio payload is not float32 PCM")
            audio = np.frombuffer(raw, dtype=np.float32).copy()
            start = time.perf_counter()
            context = str(payload.get("context") or "")
            with contextlib.redirect_stdout(sys.stderr):
                result = model.transcribe(
                    (audio, 16000),
                    context=context,
                    language=map_language(language),
                )
            decode_ms = (time.perf_counter() - start) * 1000
            item = result[0] if result else None
            output = None if item is None or not item.text else {
                "text": item.text,
                "language": language_code(item.language),
                "language_name": item.language or "unknown",
                "metrics": {
                    "audio_seconds": len(audio) / 16000,
                    "context_chars": len(context),
                    "decode_ms": decode_ms,
                },
            }
            emit({"id": msg_id, "ok": True, "type": "result", "payload": output})
        except BaseException as exc:
            emit_error(msg_id, exc, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
