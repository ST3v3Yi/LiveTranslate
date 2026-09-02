"""Window and image compositor for screenshot translation results."""

import logging

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger("LiveTranslate.ScreenshotTranslation")


def _average_border_color(image: QImage, rect: QRect):
    points = []
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    for x in range(max(0, x1), min(image.width(), x2 + 1), max(1, rect.width() // 16)):
        if y1 > 0:
            points.append(image.pixelColor(x, y1 - 1))
        if y2 + 1 < image.height():
            points.append(image.pixelColor(x, y2 + 1))
    for y in range(max(0, y1), min(image.height(), y2 + 1), max(1, rect.height() // 8)):
        if x1 > 0:
            points.append(image.pixelColor(x1 - 1, y))
        if x2 + 1 < image.width():
            points.append(image.pixelColor(x2 + 1, y))
    if not points:
        return QColor(25, 25, 30)
    return QColor(
        sum(c.red() for c in points) // len(points),
        sum(c.green() for c in points) // len(points),
        sum(c.blue() for c in points) // len(points),
    )


_WRAP_FLAGS = int(Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere)


def _text_bounds(font, rect, text):
    return QFontMetrics(font).boundingRect(rect, _WRAP_FLAGS, text)


def _fit_font(text, rect, family="Microsoft YaHei", max_size=36, min_size=5):
    for size in range(max_size, min_size - 1, -1):
        font = QFont(family, size)
        bounds = _text_bounds(font, rect, text)
        if bounds.height() <= rect.height() and bounds.width() <= rect.width():
            return font
    return QFont(family, min_size)


def _fits_at_readable_size(text, rect, family="Microsoft YaHei", font_size=12):
    """Keep compact UI labels in place, but move paragraphs into a text card."""
    text_rect = rect.adjusted(8, 4, -8, -4)
    if text_rect.width() < 24 or text_rect.height() < 12:
        return False
    return _text_bounds(QFont(family, font_size), text_rect, text).height() <= text_rect.height()


def _long_text_card(rect, image_rect, text, family="Microsoft YaHei"):
    """Lay long translated text out as a compact caption near its source."""
    margin, padding_x, padding_y = 8, 10, 7
    available_width = max(1, image_rect.width() - margin * 2)
    preferred_width = max(rect.width(), min(available_width, max(220, rect.width() * 2)))
    card_width = min(available_width, preferred_width)
    probe = QRect(0, 0, card_width - padding_x * 2, image_rect.height())
    preferred_font = QFont(family, 16)
    available_height = max(1, image_rect.height() - margin * 2)
    card_height = _text_bounds(preferred_font, probe, text).height() + padding_y * 2
    card_height = min(available_height, max(34, card_height))

    x = rect.center().x() - card_width // 2
    x = max(image_rect.left() + margin, min(x, image_rect.right() - margin - card_width + 1))

    # Captions belong below their source when possible. Fall back above it;
    # on very small crops center the card rather than creating a tall box over
    # a narrow OCR line.
    below_y = rect.bottom() + 7
    above_y = rect.top() - 7 - card_height
    if below_y + card_height <= image_rect.bottom() - margin + 1:
        y = below_y
    elif above_y >= image_rect.top() + margin:
        y = above_y
    else:
        y = max(
            image_rect.top() + margin,
            min(rect.center().y() - card_height // 2, image_rect.bottom() - margin - card_height + 1),
        )
    return QRect(x, y, card_width, card_height).intersected(image_rect)


def _rect_overlaps(rect, placed, gap=3):
    """Return whether a candidate card collides with an already placed card."""
    probe = rect.adjusted(-gap, -gap, gap, gap)
    return any(probe.intersects(other) for other in placed)


def _clamp_rect(rect, bounds):
    """Keep a card fully inside the screenshot without changing its size."""
    width = min(rect.width(), bounds.width())
    height = min(rect.height(), bounds.height())
    x = max(bounds.left(), min(rect.left(), bounds.right() - width + 1))
    y = max(bounds.top(), min(rect.top(), bounds.bottom() - height + 1))
    return QRect(x, y, max(1, width), max(1, height))


def _reference_font_size(source_rect, is_table_cell=False):
    """Estimate the source glyph size from the OCR line/cell height."""
    # OCR boxes include a little leading around glyphs. Keeping roughly 72%
    # of the box height gives translated text a visual scale close to the
    # original while still leaving room for outlines and card padding.
    ratio = 0.68 if is_table_cell else 0.72
    return max(7, min(36, round(source_rect.height() * ratio)))


def _card_size(text, source_rect, image_rect, family, font_size):
    """Choose a readable card size while reserving room for wrapped text."""
    padding_x, padding_y = 10, 7
    available_width = max(1, image_rect.width() - 16)
    width = min(
        available_width,
        max(source_rect.width(), min(520, max(220, source_rect.width() * 2))),
    )
    available_height = max(1, image_rect.height() - 16)
    # Measure with the source-relative font first. For unusually long
    # paragraphs, probe progressively smaller sizes so the card height remains
    # bounded by the screenshot.
    height = available_height
    probe_sizes = [font_size] + [size for size in (24, 20, 16, 14, 12, 10, 8, 7) if size < font_size]
    for size in probe_sizes:
        probe = QRect(0, 0, max(1, width - padding_x * 2), available_height)
        measured = _text_bounds(QFont(family, size), probe, text).height()
        desired = measured + padding_y * 2
        if desired <= available_height:
            height = max(34, desired)
            break
    return width, min(available_height, max(24, height))


def _layout_translation_regions(image_rect, regions):
    """Assign non-overlapping cards and fitted fonts before painting them."""
    entries = []
    for region in regions:
        translated = str(region.get("translation") or "").strip()
        bbox = region.get("bbox") or []
        if not translated or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue
        source = QRect(
            x1,
            y1,
            max(1, x2 - x1),
            max(1, y2 - y1),
        ).intersected(image_rect)
        if not source.isEmpty():
            entries.append((region, translated, source))

    # Fixed table cells get reserved first. They must remain inside their grid
    # boundaries, while normal text cards can move into nearby free space.
    entries.sort(key=lambda item: (item[0].get("kind") != "table_cell", item[2].top(), item[2].left()))
    placed = []
    result = []
    for region, translated, source in entries:
        is_table_cell = region.get("kind") == "table_cell"
        source_font_size = _reference_font_size(source, is_table_cell)
        if is_table_cell:
            card = source.adjusted(1, 1, -1, -1).intersected(image_rect)
            candidates = [card]
        else:
            base = source.adjusted(-6, -5, 6, 5).intersected(image_rect)
            card_width, card_height = _card_size(
                translated, base, image_rect, "Microsoft YaHei", source_font_size
            )
            if _fits_at_readable_size(
                translated, base, font_size=source_font_size
            ):
                card_width, card_height = base.width(), base.height()
            x = source.center().x() - card_width // 2
            below = QRect(x, source.bottom() + 7, card_width, card_height)
            above = QRect(x, source.top() - 7 - card_height, card_width, card_height)
            right = QRect(source.right() + 7, source.center().y() - card_height // 2, card_width, card_height)
            left = QRect(source.left() - 7 - card_width, source.center().y() - card_height // 2, card_width, card_height)
            # Keep short labels in their source area. Long translations get
            # nearby expanded space first, so they remain readable instead of
            # being squeezed into a one-line OCR box.
            if _fits_at_readable_size(
                translated, base, font_size=source_font_size
            ):
                candidates = [base, below, above, right, left]
            else:
                candidates = [below, above, right, left, base]
            candidates = [_clamp_rect(candidate, image_rect) for candidate in candidates]

        card = next(
            (candidate for candidate in candidates if not _rect_overlaps(candidate, placed)),
            None,
        )
        if card is None and not is_table_cell:
            # Find the nearest free slot when neighboring OCR boxes leave no
            # room around the source. This guarantees that translated cards do
            # not cover one another even in dense UI screenshots.
            width, height = candidates[0].width(), candidates[0].height()
            for y in range(image_rect.top(), image_rect.bottom() - height + 2, 8):
                for x in range(image_rect.left(), image_rect.right() - width + 2, 8):
                    candidate = QRect(x, y, width, height)
                    if not _rect_overlaps(candidate, placed):
                        card = candidate
                        break
                if card is not None:
                    break
        if card is None and not is_table_cell:
            # If the screenshot is densely packed, progressively reduce the
            # card footprint before giving up. Never place a card on top of an
            # existing one merely to display every translation.
            original_width, original_height = candidates[0].width(), candidates[0].height()
            for scale in (0.85, 0.70, 0.55, 0.40):
                width = max(96, round(original_width * scale))
                height = max(24, round(original_height * scale))
                if width > image_rect.width() or height > image_rect.height():
                    continue
                for y in range(image_rect.top(), image_rect.bottom() - height + 2, 8):
                    for x in range(image_rect.left(), image_rect.right() - width + 2, 8):
                        candidate = QRect(x, y, width, height)
                        if not _rect_overlaps(candidate, placed):
                            card = candidate
                            break
                    if card is not None:
                        break
                if card is not None:
                    break
        if card is None:
            if is_table_cell:
                # A table cell can only use its own grid slot. Preserve it
                # rather than dropping the translation if OCR produced
                # touching cells.
                card = candidates[0]
            else:
                log.warning("Skipping overlapping screenshot translation card: %s", translated[:80])
                continue

        padding = (6, 3, 6, 3) if is_table_cell else (8, 4, 8, 4)
        text_rect = card.adjusted(*padding)
        max_size = max(5, min(source_font_size, text_rect.height() - 2))
        font = _fit_font(
            translated,
            text_rect,
            max_size=max_size,
            min_size=5,
        )
        placed.append(card)
        result.append((region, translated, card, text_rect, font, is_table_cell))
    return result


def compose_translated_image(original: QImage, regions: list[dict]) -> QImage:
    image = original.convertToFormat(QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    for region, translated, rect, text_rect, font, is_table_cell in _layout_translation_regions(
        image.rect(), regions
    ):
        # An opaque card removes the original text beneath it. A translucent
        # background created distracting double text on long translations.
        painter.setBrush(QColor(8, 10, 16, 248))
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setFont(font)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 240), max(2, font.pointSize() // 7)))
        text_flags = int(Qt.AlignmentFlag.AlignCenter) | _WRAP_FLAGS
        painter.drawText(text_rect, text_flags, translated)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(text_rect, text_flags, translated)
    painter.end()
    return image


class ScreenshotTranslationOverlay(QWidget):
    """Movable translation sticker, created only after the user confirms Paste."""

    result_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, screen_rect, parent=None):
        super().__init__(parent)
        self._screen_rect = screen_rect
        self._frame_margin = 8
        self._drag_offset = None
        self._source_image = None
        self._scale = 1.0
        self._rendered_size = (0, 0)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setContentsMargins(0, 0, 0, 0)
        self._label.setStyleSheet("background: transparent; border: none;")
        # The label fills the overlay, so it must not swallow close gestures
        # intended for the parent layer.
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setToolTip(
            "左键拖动贴图 · 滚轮缩放 · 左键双击关闭 · 右键关闭 · Esc 关闭"
        )
        self.setGeometry(screen_rect.adjusted(
            -self._frame_margin, -self._frame_margin,
            self._frame_margin, self._frame_margin,
        ))
        self.result_ready.connect(self._on_result)
        self.status_changed.connect(self._on_status)

    def resizeEvent(self, event):
        self._label.setGeometry(self.rect().adjusted(
            self._frame_margin, self._frame_margin,
            -self._frame_margin, -self._frame_margin,
        ))
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        content = self.rect().adjusted(
            self._frame_margin, self._frame_margin,
            -self._frame_margin, -self._frame_margin,
        )
        # Soft shadow outside the translated image makes the pasted result
        # legible against arbitrary desktop backgrounds.
        for spread, alpha in ((7, 20), (5, 32), (3, 48)):
            shadow = content.adjusted(-spread, -spread, spread, spread)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, alpha))
            painter.drawRoundedRect(shadow, 6 + spread, 6 + spread)
        painter.setPen(QPen(QColor(150, 215, 255, 230), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(content, 5, 5)
        painter.end()

    def set_status(self, text):
        self.status_changed.emit(str(text))

    def set_result(self, image):
        self.result_ready.emit(image)

    def _on_status(self, text):
        # There is intentionally no separate result dialog. Keep progress and
        # failures in the normal application log instead.
        log.debug("Screenshot overlay: %s", text)

    @pyqtSlot(object)
    def _on_result(self, image):
        self._source_image = image
        self._scale = 1.0
        self._apply_scale(self._scale)
        self.show()
        self.raise_()
        self.activateWindow()

    def _apply_scale(self, scale, global_anchor=None):
        """Resize around the pointer so wheel zoom feels anchored in place."""
        if self._source_image is None or self._source_image.isNull():
            return
        scale = max(0.25, min(3.0, float(scale)))
        old_width, old_height = self._rendered_size
        old_width = max(1, old_width)
        old_height = max(1, old_height)

        if global_anchor is not None:
            local = global_anchor - self.frameGeometry().topLeft()
            ratio_x = max(0.0, min(1.0, (local.x() - self._frame_margin) / old_width))
            ratio_y = max(0.0, min(1.0, (local.y() - self._frame_margin) / old_height))
        else:
            ratio_x = ratio_y = 0.0

        width = max(1, round(self._source_image.width() * scale))
        height = max(1, round(self._source_image.height() * scale))
        pixmap = QPixmap.fromImage(self._source_image).scaled(
            width,
            height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pixmap)
        self._label.setFixedSize(width, height)
        self._rendered_size = (width, height)
        self.resize(width + self._frame_margin * 2, height + self._frame_margin * 2)
        if global_anchor is not None:
            self.move(
                global_anchor
                - QPoint(
                    round(self._frame_margin + ratio_x * width),
                    round(self._frame_margin + ratio_y * height),
                )
            )
        self._scale = scale

    def wheelEvent(self, event):
        if self._source_image is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta:
            # One mouse-wheel notch is a measured 15% change.  Handle large
            # high-resolution wheel deltas smoothly while retaining a sensible
            # lower and upper limit for floating translated screenshots.
            steps = delta / 120.0
            self._apply_scale(
                self._scale * (1.15 ** steps),
                event.globalPosition().toPoint(),
            )
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.close()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.close()
            return
        super().mouseDoubleClickEvent(event)


class ScreenshotTranslationWindow(QDialog):
    """Non-modal result window; OCR/translation runs outside the GUI thread."""

    result_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, original: QImage, parent=None):
        super().__init__(parent)
        self._original = original
        self._translated = None
        self._show_translation = True
        self.setWindowTitle("LiveTranslate - Screenshot Translation")
        self.resize(min(1100, max(640, original.width() + 40)), min(800, max(420, original.height() + 110)))
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self._status = QLabel("OCR / translation pending...")
        toolbar.addWidget(self._status, 1)
        self._toggle_btn = QPushButton("Original")
        self._toggle_btn.clicked.connect(self._toggle_image)
        self._toggle_btn.setEnabled(False)
        toolbar.addWidget(self._toggle_btn)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        toolbar.addWidget(save_btn)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy)
        toolbar.addWidget(copy_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(close_btn)
        layout.addLayout(toolbar)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setBackgroundRole(self._label.backgroundRole())
        self._set_image(original)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._label)
        layout.addWidget(scroll, 1)
        self.result_ready.connect(self._on_result)
        self.status_changed.connect(self._status.setText)

    def set_status(self, text):
        self.status_changed.emit(str(text))

    def set_result(self, image):
        self.result_ready.emit(image)

    def _set_image(self, image):
        self._label.setPixmap(self._pixmap_for(image))

    def _pixmap_for(self, image):
        from PyQt6.QtGui import QPixmap

        return QPixmap.fromImage(image)

    @pyqtSlot(object)
    def _on_result(self, image):
        self._translated = image
        self._show_translation = True
        self._set_image(image)
        self._toggle_btn.setEnabled(True)
        self._toggle_btn.setText("Original")

    def _toggle_image(self):
        self._show_translation = not self._show_translation
        self._set_image(self._translated if self._show_translation else self._original)
        self._toggle_btn.setText("Original" if self._show_translation else "Translated")

    def _current_image(self):
        return self._translated if self._show_translation and self._translated else self._original

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "translated.png", "PNG Image (*.png);;All Files (*)")
        if path:
            self._current_image().save(path, "PNG")

    def _copy(self):
        QApplication.clipboard().setPixmap(self._pixmap_for(self._current_image()))
