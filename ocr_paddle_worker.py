"""Dedicated PaddleOCR-VL worker used by the screenshot translation feature.

The OCR environment is intentionally isolated from LiveTranslate's PyTorch
environment.  Communication is JSON-lines over stdin/stdout; images are sent
as PNG bytes encoded with base64.
"""

import argparse
import base64
import contextlib
import html
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import traceback


def emit(message):
    sys.stdout.write(json.dumps(message, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def emit_error(msg_id, exc, recoverable=True):
    emit(
        {
            "id": msg_id,
            "ok": False,
            "type": "error",
            "error": {
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "recoverable": recoverable,
            },
        }
    )


def _json_data(result):
    """Return a JSON-compatible PaddleX result payload."""
    value = getattr(result, "json", None)
    if value is not None:
        try:
            return value.get("res", value)
        except AttributeError:
            return value
    if isinstance(result, dict):
        return result
    return {}


def _as_box(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        # PaddleOCR returns text polygons as four ``[x, y]`` points.  The
        # previous condition accidentally expected at least eight outer
        # elements, so these valid polygons were rejected and the caller fell
        # back to one large layout block for an entire table.
        if isinstance(value[0], (list, tuple)):
            xs = [float(p[0]) for p in value]
            ys = [float(p[1]) for p in value]
            return [min(xs), min(ys), max(xs), max(ys)]
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except (TypeError, ValueError, IndexError):
        return None


def _collect_regions(node, output):
    """Extract text boxes from both OCR and layout-parsing result shapes."""
    if isinstance(node, dict):
        texts = node.get("rec_texts")
        boxes = node.get("rec_boxes") or node.get("rec_polys")
        has_line_regions = False
        if isinstance(texts, list) and isinstance(boxes, list):
            for text, box in zip(texts, boxes):
                box = _as_box(box)
                text = str(text or "").strip()
                if text and box:
                    output.append({"text": text, "bbox": box, "source": "line"})
                    has_line_regions = True

        text = node.get("block_content", node.get("text", node.get("content")))
        box = node.get("block_bbox", node.get("bbox", node.get("coordinate")))
        # A layout block's content often combines every cell in a table but
        # has only one bounding box.  Prefer the per-line OCR entries above;
        # retain block content only as a fallback for result shapes that do
        # not provide line boxes at all.
        if not has_line_regions and isinstance(text, str) and text.strip() and box is not None:
            box = _as_box(box)
            if box:
                output.append({"text": text.strip(), "bbox": box, "source": "block"})

        for value in node.values():
            if isinstance(value, (dict, list, tuple)):
                _collect_regions(value, output)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _collect_regions(value, output)


def _deduplicate_regions(regions):
    result = []
    seen = set()
    for item in regions:
        text = item["text"]
        bbox = tuple(round(float(v), 1) for v in item["bbox"])
        key = (text, bbox)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {"text": text, "bbox": list(bbox), "source": item.get("source", "block")}
        )

    # If a generic layout block surrounds independently located OCR lines,
    # keep the lines. This eliminates duplicate/concatenated table contents
    # without discarding the fallback path for images where only a block is
    # returned.
    line_regions = [item for item in result if item["source"] == "line"]
    if line_regions:
        def contains(outer, inner):
            ox1, oy1, ox2, oy2 = outer["bbox"]
            ix1, iy1, ix2, iy2 = inner["bbox"]
            return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

        result = [
            item
            for item in result
            if item["source"] == "line"
            or not any(contains(item, line) for line in line_regions)
        ]
    result.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return [{"text": item["text"], "bbox": item["bbox"]} for item in result]


def _table_rows(text):
    """Read simple HTML table output emitted by PaddleOCR-VL.

    PaddleOCR-VL represents many tables as a single ``<table>`` block.  This
    keeps the semantic cell boundaries, but its layout result has only one
    bounding box.  Preserve the cells here so they can be translated and
    rendered individually.
    """
    if not isinstance(text, str) or "<table" not in text.lower():
        return []
    rows = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr\s*>", text, flags=re.I | re.S):
        cells = []
        for attrs, cell_html in re.findall(
            r"<t[dh]\b([^>]*)>(.*?)</t[dh]\s*>", row_html, flags=re.I | re.S
        ):
            plain = re.sub(r"<[^>]+>", " ", cell_html)
            plain = re.sub(r"\s+", " ", html.unescape(plain)).strip()
            if not plain:
                plain = " "
            span_match = re.search(r"\bcolspan\s*=\s*['\"]?(\d+)", attrs, flags=re.I)
            span = max(1, int(span_match.group(1))) if span_match else 1
            cells.append((plain, span))
        if cells:
            rows.append(cells)
    return rows


def _group_positions(indices):
    """Turn consecutive projection hits into one pixel coordinate each."""
    if len(indices) == 0:
        return []
    groups = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value > previous + 1:
            groups.append((start + previous) // 2)
            start = value
        previous = value
    groups.append((start + previous) // 2)
    return groups


def _table_grid_boundaries(image, bbox, row_count, col_count, cv2, np):
    """Find grid lines inside a table box; fall back to an even grid.

    The projection threshold is deliberately high: text strokes are sparse,
    whereas a table rule spans most of the table's height or width.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return [], []
    crop = image[y1 : y2 + 1, x1 : x2 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = gray < 220
    vertical = _group_positions(np.flatnonzero(dark.mean(axis=0) >= 0.70))
    horizontal = _group_positions(np.flatnonzero(dark.mean(axis=1) >= 0.70))
    vertical = [x1 + value for value in vertical]
    horizontal = [y1 + value for value in horizontal]

    # The OCR layout box is normally already aligned with the outside rules.
    # Use a predictable equally-spaced grid if the visible rules are missing
    # or more complex than this simple cell renderer can reliably interpret.
    if len(vertical) != col_count + 1:
        vertical = [round(x1 + (x2 - x1) * index / col_count) for index in range(col_count + 1)]
    if len(horizontal) != row_count + 1:
        horizontal = [round(y1 + (y2 - y1) * index / row_count) for index in range(row_count + 1)]
    return vertical, horizontal


def _expand_html_tables(regions, image, cv2, np):
    """Replace one HTML-table region with independently positioned cells."""
    expanded = []
    for region in regions:
        rows = _table_rows(region.get("text"))
        if not rows:
            expanded.append(region)
            continue
        row_count = len(rows)
        col_count = max(sum(span for _text, span in row) for row in rows)
        if row_count < 1 or col_count < 1:
            expanded.append(region)
            continue
        xs, ys = _table_grid_boundaries(
            image, region["bbox"], row_count, col_count, cv2, np
        )
        if len(xs) != col_count + 1 or len(ys) != row_count + 1:
            expanded.append(region)
            continue
        for row_index, row in enumerate(rows):
            column = 0
            for text, span in row:
                x1, x2 = xs[column], xs[min(col_count, column + span)]
                y1, y2 = ys[row_index], ys[row_index + 1]
                # Retain a few pixels of the original table rules. The image
                # compositor uses this marker to avoid spilling a text card
                # over an adjacent cell.
                inset = 3
                expanded.append(
                    {
                        "text": text,
                        "bbox": [x1 + inset, y1 + inset, x2 - inset, y2 - inset],
                        "kind": "table_cell",
                    }
                )
                column += span
    return expanded


def _collect_plain_ocr_regions(node, min_score=0.55):
    """Extract high-confidence line boxes from the PP-OCR fallback output."""
    regions = []
    if not isinstance(node, dict):
        return regions
    texts = node.get("rec_texts")
    boxes = node.get("rec_boxes") or node.get("rec_polys")
    scores = node.get("rec_scores") or []
    if not isinstance(texts, list) or not isinstance(boxes, list):
        return regions
    for index, (text, box) in enumerate(zip(texts, boxes)):
        text = str(text or "").strip()
        box = _as_box(box)
        try:
            score = float(scores[index]) if index < len(scores) else 1.0
        except (TypeError, ValueError):
            score = 0.0
        if text and box and score >= min_score:
            regions.append(
                {"text": text, "bbox": box, "source": "plain_ocr", "score": score}
            )
    return regions


def _intersection_area(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    return max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )


def _merge_plain_ocr_regions(vl_regions, plain_regions):
    """Fuse document-VL blocks with precise ordinary OCR line boxes.

    Document layout is stronger for paragraphs and tables. PP-OCR is stronger
    at decorative raster text and compact UI controls.  When several PP-OCR
    lines land inside one VL block, that block is a parent/merged layout item
    (for example an entire navigation bar), so omit it instead of translating
    the same text twice in an unusable large card.
    """
    if not plain_regions:
        return vl_regions

    # A VL table cell has reconstructed grid boundaries.  It must remain the
    # rendering source for that cell, even when PP-OCR recognizes the same
    # characters, otherwise the normal text-card padding can cover table
    # rules and lose the table layout again.
    table_cells = [item for item in vl_regions if item.get("kind") == "table_cell"]
    if table_cells:
        plain_regions = [
            plain
            for plain in plain_regions
            if not any(
                _intersection_area(cell["bbox"], plain["bbox"])
                / max(
                    1.0,
                    (plain["bbox"][2] - plain["bbox"][0])
                    * (plain["bbox"][3] - plain["bbox"][1]),
                )
                >= 0.80
                for cell in table_cells
            )
        ]

    merged = []
    for region in vl_regions:
        bbox = region.get("bbox") or []
        if len(bbox) < 4:
            continue
        covered = []
        for plain in plain_regions:
            plain_box = plain["bbox"]
            area = max(1.0, (plain_box[2] - plain_box[0]) * (plain_box[3] - plain_box[1]))
            if _intersection_area(bbox, plain_box) / area >= 0.80:
                covered.append(plain)
        if len(covered) >= 2:
            continue
        if len(covered) == 1:
            plain = covered[0]
            vl_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            overlap = _intersection_area(bbox, plain["bbox"])
            # A nearly identical VL/PP-OCR box is a duplicate; prefer the
            # ordinary OCR box because its recognition is tuned for UI text.
            if overlap / vl_area >= 0.70:
                continue
        merged.append(region)

    merged.extend(plain_regions)
    merged.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return [
        {key: value for key, value in item.items() if key not in {"source", "score"}}
        for item in merged
    ]


def _looks_like_table(image, cv2, np):
    """Return True for a visible ruled grid with at least 2x2 cells.

    PP-OCR is the default fast path.  A table needs the document-VL pipeline
    because its HTML output preserves the row/column structure.  Detecting
    long horizontal and vertical rules before OCR lets us avoid loading VL for
    normal UI, subtitles, and illustrations.
    """
    height, width = image.shape[:2]
    if height < 50 or width < 80:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray < 220
    horizontal = _group_positions(np.flatnonzero(dark.mean(axis=1) >= 0.35))
    vertical = _group_positions(np.flatnonzero(dark.mean(axis=0) >= 0.35))
    return len(horizontal) >= 3 and len(vertical) >= 3


def _needs_vl_pipeline(image, plain_regions, cv2, np):
    """Choose the slower layout-aware OCR only when PP-OCR needs help."""
    if _looks_like_table(image, cv2, np):
        return True, "table"
    if not plain_regions:
        return True, "no_plain_text"
    return False, "plain_ocr"


def _configure_windows_cuda_dlls():
    """Make pip-installed NVIDIA DLLs visible before importing Paddle.

    Recent Windows wheels install CUDA/cuDNN under ``site-packages/nvidia``.
    Paddle's native loader does not always discover those directories from the
    Conda environment PATH, so the worker must register them explicitly before
    importing ``paddleocr``/``paddle``.
    """
    if os.name != "nt":
        return
    candidates = []
    env_root = Path(sys.prefix)
    candidates.extend([env_root, env_root / "Library" / "bin"])
    nvidia_root = env_root / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        candidates.extend(path for path in nvidia_root.glob("*/bin") if path.is_dir())
    paddle_libs = env_root / "Lib" / "site-packages" / "paddle" / "libs"
    if paddle_libs.is_dir():
        candidates.append(paddle_libs)

    path_items = [str(path) for path in candidates if path.is_dir()]
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(path_items + [existing_path])
    for path in path_items:
        try:
            os.add_dll_directory(path)
        except (AttributeError, OSError):
            pass


def load_pipeline(args):
    # PaddleX writes model metadata/cache during import and initialization.
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", args.cache_dir)
    # The required layout model is cached alongside LiveTranslate. Avoid a
    # connectivity probe on every application launch and keep OCR local-first.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.makedirs(args.cache_dir, exist_ok=True)
    _configure_windows_cuda_dlls()
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        from paddleocr import PaddleOCR

        # PP-OCR handles the default path: compact UI controls, captions and
        # lettering printed into images.  It is much lighter than the VL
        # model and produces the exact line boxes needed for replacement.
        plain_ocr = PaddleOCR(
            lang="japan",
            ocr_version="PP-OCRv6",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=args.device,
        )
    return plain_ocr, cv2, np


def load_vl_pipeline(args):
    """Lazy-load the layout model only for tables or OCR fallback cases."""
    with contextlib.redirect_stdout(sys.stderr):
        from paddleocr import PaddleOCRVL

        return PaddleOCRVL(
            pipeline_version="v1.6",
            vl_rec_model_dir=args.model_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_ocr_for_image_block=True,
            device=args.device,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    try:
        plain_ocr, cv2, np = load_pipeline(args)
    except BaseException as exc:
        emit_error(None, exc, recoverable=False)
        return 1

    vl_pipeline = None
    emit({"id": None, "ok": True, "type": "ready", "payload": {"device": args.device}})
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
            if request_type != "recognize":
                raise ValueError(f"Unknown PaddleOCR worker command: {request_type}")

            raw = base64.b64decode(payload["image_b64"])
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("PaddleOCR could not decode the screenshot")
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                plain_results = list(
                    plain_ocr.predict(
                        image,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                )
            plain_regions = []
            for result in plain_results:
                plain_regions.extend(_collect_plain_ocr_regions(_json_data(result)))
            use_vl, route = _needs_vl_pipeline(image, plain_regions, cv2, np)
            regions = plain_regions
            if use_vl:
                if vl_pipeline is None:
                    vl_pipeline = load_vl_pipeline(args)
                with contextlib.redirect_stdout(sys.stderr):
                    results = list(
                        vl_pipeline.predict(
                            image,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_layout_detection=True,
                            use_ocr_for_image_block=True,
                        )
                    )
                vl_regions = []
                for result in results:
                    _collect_regions(_json_data(result), vl_regions)
                vl_regions = _deduplicate_regions(vl_regions)
                vl_regions = _expand_html_tables(vl_regions, image, cv2, np)
                regions = _merge_plain_ocr_regions(vl_regions, plain_regions)
            height, width = image.shape[:2]
            emit(
                {
                    "id": msg_id,
                    "ok": True,
                    "type": "result",
                    "payload": {
                        "width": int(width),
                        "height": int(height),
                        "regions": regions,
                        "ocr_ms": (time.perf_counter() - started) * 1000,
                        "engine": "pp-ocrv6+vl" if use_vl else "pp-ocrv6",
                        "route": route,
                    },
                }
            )
        except BaseException as exc:
            emit_error(msg_id, exc, recoverable=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
