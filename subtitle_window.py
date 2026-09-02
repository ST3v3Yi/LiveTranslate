"""
Subtitle window - clean text-only window for OBS capture.
Uses QPainterPath for outlined text rendering.

Usage:
  - Middle-click drag to move the window
  - Configure via tray menu → Subtitle Mode → Settings
  - OBS: Window Capture → select "LiveTranslate Subtitle" → check "Allow Transparency"
"""

import ctypes
import time
from pathlib import Path

import json

from PyQt6.QtCore import (
    Qt, QPoint, QRect, pyqtSignal, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QParallelAnimationGroup, QEasingCurve, QTimer,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QCursor,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QProgressBar,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from i18n import t

_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x20


def _resolve_image_path(path: str) -> str:
    """Resolve image path (relative to project dir or absolute)."""
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return str(p) if p.exists() else ""
    resolved = Path(__file__).parent / p
    return str(resolved) if resolved.exists() else ""

# Default subtitle window settings
DEFAULT_SUBTITLE_WIN_SETTINGS = {
    "enabled": False,
    "sentences": 2,
    "window_width": 1000,
    "window_height": 0,
    "auto_wrap": True,
    "dual_segment_mode": True,
    "line_spacing": 8,
    "bg_color": "#000000",
    "bg_opacity": 76,
    "bg_image": "",
    "border_radius": 8,
    "auto_hide_timeout": 5,
    "auto_hide_animation": "fade",
    "auto_hide_duration": 300,
    "click_through": False,
    "locked": False,
    "always_on_top": True,
    "show_vad_meter": True,
    "lines": [
        {
            "type": "original",
            "enabled": True,
            "font_family": "Microsoft YaHei",
            "font_size": 24,
            "color": "#FFFFFF",
            "opacity": 255,
            "outline_enabled": True,
            "outline_color": "#000000",
            "outline_width": 2,
            "align": "center",
            "bg_image": "",
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        },
        {
            "type": "translation",
            "lang": "zh",
            "enabled": True,
            "font_family": "Microsoft YaHei",
            "font_size": 28,
            "color": "#FFD700",
            "opacity": 255,
            "outline_enabled": True,
            "outline_color": "#000000",
            "outline_width": 2,
            "align": "center",
            "bg_image": "",
            "entry_animation": "none",
            "exit_animation": "none",
            "animation_duration": 300,
        },
    ],
}


def _merge_settings(base, override):
    result = {**base}
    for k, v in (override or {}).items():
        if k == "lines" and isinstance(v, list):
            result["lines"] = v
        else:
            result[k] = v
    return result


def _hex_to_rgba(hex_color: str, opacity: int) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


class _SubtitleTextWidget(QWidget):
    """Renders outlined text using QPainterPath, with automatic word-wrap.
    Supports entry/exit animations via custom properties.
    """

    height_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._wrapped_lines = []
        self._font = QFont("Microsoft YaHei", 24)
        self._color = QColor(255, 255, 255)
        self._outline_enabled = True
        self._outline_color = QColor(0, 0, 0)
        self._outline_width = 2
        self._align = "center"
        self._auto_wrap = True
        self._bg_pixmap = None
        self._text_cache = None
        self._last_width = 0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Animation state
        self._content_opacity_val = 1.0
        self._slide_offset_x_val = 0.0
        self._slide_offset_y_val = 0.0
        self._entry_animation = "none"
        self._exit_animation = "none"
        self._animation_duration = 300
        self._anim_group = None
        self._pending_text = None

    # --- pyqtProperty for content opacity ---
    def _get_content_opacity(self):
        return self._content_opacity_val

    def _set_content_opacity(self, val):
        self._content_opacity_val = val
        self.update()

    content_opacity = pyqtProperty(float, _get_content_opacity, _set_content_opacity)

    # --- pyqtProperty for slide offsets ---
    def _get_slide_offset_x(self):
        return self._slide_offset_x_val

    def _set_slide_offset_x(self, val):
        self._slide_offset_x_val = val
        self.update()

    slide_offset_x = pyqtProperty(float, _get_slide_offset_x, _set_slide_offset_x)

    def _get_slide_offset_y(self):
        return self._slide_offset_y_val

    def _set_slide_offset_y(self, val):
        self._slide_offset_y_val = val
        self.update()

    slide_offset_y = pyqtProperty(float, _get_slide_offset_y, _set_slide_offset_y)

    def set_config(self, cfg: dict):
        self._font = QFont(cfg.get("font_family", "Microsoft YaHei"), cfg.get("font_size", 24))
        c = QColor(cfg.get("color", "#FFFFFF"))
        c.setAlpha(cfg.get("opacity", 255))
        self._color = c
        self._outline_enabled = cfg.get("outline_enabled", True)
        self._outline_color = QColor(cfg.get("outline_color", "#000000"))
        self._outline_width = cfg.get("outline_width", 2)
        self._align = cfg.get("align", "center")
        self._auto_wrap = bool(cfg.get("auto_wrap", True))
        resolved = _resolve_image_path(cfg.get("bg_image", ""))
        self._bg_pixmap = QPixmap(resolved) if resolved else None
        self._entry_animation = cfg.get("entry_animation", "none")
        self._exit_animation = cfg.get("exit_animation", "none")
        self._animation_duration = cfg.get("animation_duration", 300)
        self._text_cache = None
        self._update_height()
        self.update()

    def set_text(self, text: str):
        if self._text and text != self._text and self._exit_animation != "none":
            self._pending_text = text
            self._stop_all_animations()
            self.animate_out(callback=self._apply_pending_text)
            return

        self._apply_text_immediate(text)

    def _apply_pending_text(self):
        text = getattr(self, "_pending_text", "")
        self._pending_text = None
        self._apply_text_immediate(text)

    def _apply_text_immediate(self, text: str):
        # Stop any running animations and reset to final state
        self._stop_all_animations()
        self._content_opacity_val = 1.0
        self._slide_offset_x_val = 0.0
        self._slide_offset_y_val = 0.0
        self._pending_text = None

        self._text = text
        self._text_cache = None
        self._update_height()
        self.update()
        self.height_changed.emit()

        if text:
            self.animate_in()

    def _stop_all_animations(self):
        if self._anim_group and self._anim_group.state() != self._anim_group.State.Stopped:
            self._anim_group.stop()
        self._anim_group = None

    def animate_in(self):
        anim_type = self._entry_animation
        if anim_type == "none":
            self._content_opacity_val = 1.0
            self._slide_offset_x_val = 0.0
            self._slide_offset_y_val = 0.0
            self.update()
            return

        dur = self._animation_duration
        group = QParallelAnimationGroup(self)

        # Opacity animation (all types fade in)
        opacity_anim = QPropertyAnimation(self, b"content_opacity", self)
        opacity_anim.setDuration(dur)
        opacity_anim.setStartValue(0.0)
        opacity_anim.setEndValue(1.0)
        opacity_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(opacity_anim)

        w = self.width() or 200
        h = self.height() or 40

        if anim_type == "slide_left":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(dur)
            slide.setStartValue(float(-w))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_right":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(dur)
            slide.setStartValue(float(w))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_up":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(dur)
            slide.setStartValue(float(h))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_down":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(dur)
            slide.setStartValue(float(-h))
            slide.setEndValue(0.0)
            slide.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(slide)

        self._content_opacity_val = 0.0
        self.update()
        self._anim_group = group
        group.start()

    def animate_out(self, callback=None, anim_type=None, duration=None):
        if anim_type is None:
            anim_type = self._exit_animation
        if duration is None:
            duration = self._animation_duration
        if anim_type == "none":
            self._content_opacity_val = 0.0
            self.update()
            if callback:
                callback()
            return

        self._stop_all_animations()

        group = QParallelAnimationGroup(self)

        opacity_anim = QPropertyAnimation(self, b"content_opacity", self)
        opacity_anim.setDuration(duration)
        opacity_anim.setStartValue(self._content_opacity_val)
        opacity_anim.setEndValue(0.0)
        opacity_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        group.addAnimation(opacity_anim)

        w = self.width() or 200
        h = self.height() or 40

        if anim_type == "slide_left":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(-w))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_right":
            slide = QPropertyAnimation(self, b"slide_offset_x", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(w))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_up":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(-h))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)
        elif anim_type == "slide_down":
            slide = QPropertyAnimation(self, b"slide_offset_y", self)
            slide.setDuration(duration)
            slide.setStartValue(0.0)
            slide.setEndValue(float(h))
            slide.setEasingCurve(QEasingCurve.Type.InCubic)
            group.addAnimation(slide)

        if callback:
            group.finished.connect(callback)

        self._anim_group = group
        group.start()

    def split_text(self, text: str) -> list:
        """Split text into segments that fit within available width."""
        fm = QFontMetrics(self._font)
        ow = self._outline_width if self._outline_enabled else 0
        avail_w = self.width() - ow * 2
        if not self._auto_wrap:
            if avail_w <= 0:
                return [text]
            return [fm.elidedText(text, Qt.TextElideMode.ElideRight, avail_w)]
        if avail_w <= 0 or fm.horizontalAdvance(text) <= avail_w:
            return [text]

        segments = []
        while text:
            if fm.horizontalAdvance(text) <= avail_w:
                segments.append(text)
                break

            best = 0
            for i in range(1, len(text) + 1):
                if fm.horizontalAdvance(text[:i]) > avail_w:
                    break
                best = i
            if best == 0:
                best = 1

            # Prefer breaking at word/punctuation boundary
            break_at = best
            for j in range(best - 1, max(best // 2, 0), -1):
                if text[j] in ' ,，。、!！?？;；:：.':
                    break_at = j + 1
                    break

            segments.append(text[:break_at].rstrip())
            text = text[break_at:].lstrip()

        return segments or [text]

    def _rewrap(self):
        """Recalculate wrapped lines from current text."""
        if not self._text:
            self._wrapped_lines = []
        else:
            self._wrapped_lines = self.split_text(self._text)

    def desired_height(self) -> int:
        fm = QFontMetrics(self._font)
        ow = self._outline_width if self._outline_enabled else 0
        n = max(len(self._wrapped_lines), 1)
        return fm.lineSpacing() * n + ow * 2 + 4

    def _update_height(self):
        self._rewrap()
        self.setFixedHeight(self.desired_height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        if w != self._last_width:
            self._last_width = w
            self._rewrap()
            self._text_cache = None
            self.setFixedHeight(self.desired_height())

    def _render_text_pixmap(self):
        lines = self._wrapped_lines or [self._text]
        w = self.width()
        h = self.desired_height()
        if w <= 0 or h <= 0:
            self._text_cache = None
            return

        dpr = self.devicePixelRatioF()
        pw, ph = int(w * dpr), int(h * dpr)
        if pw <= 0 or ph <= 0:
            self._text_cache = None
            return

        pix = QPixmap(pw, ph)
        pix.setDevicePixelRatio(dpr)
        pix.fill(QColor(0, 0, 0, 0))

        painter = QPainter(pix)
        if not painter.isActive():
            self._text_cache = None
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        fm = QFontMetrics(self._font)
        ow = self._outline_width if self._outline_enabled else 0
        y = ow + fm.ascent()

        path = QPainterPath()
        for line in lines:
            text_w = fm.horizontalAdvance(line)
            if self._align == "center":
                lx = (w - text_w) / 2
            elif self._align == "right":
                lx = w - text_w - ow
            else:
                lx = ow
            path.addText(lx, y, self._font, line)
            y += fm.lineSpacing()

        if self._outline_enabled and self._outline_width > 0:
            pen = QPen(self._outline_color, self._outline_width * 2,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._color))
        painter.drawPath(path)
        painter.end()

        self._text_cache = pix

    def paintEvent(self, event):
        if not self._text:
            return

        if self._text_cache is None:
            self._render_text_pixmap()
        if self._text_cache is None:
            return

        painter = QPainter(self)
        painter.setOpacity(self._content_opacity_val)

        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter.drawPixmap(self.rect(), self._bg_pixmap)

        painter.drawPixmap(
            int(self._slide_offset_x_val),
            int(self._slide_offset_y_val),
            self._text_cache,
        )

        painter.end()


class _SubtitleControlBar(QWidget):
    """Compact draggable title area that leaves the actual controls clickable."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self._owner = owner
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        self._owner._show_controls()
        if (
            not self._owner._click_through
            and not self._owner._locked
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._owner._start_drag(event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._owner._show_controls()
        if self._owner._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self._owner.move(event.globalPosition().toPoint() - self._owner._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._owner._drag_pos:
            self._owner._finish_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self._owner._show_controls()
        super().enterEvent(event)

    def leaveEvent(self, event):
        QTimer.singleShot(0, self._owner._hide_controls_if_pointer_outside)
        super().leaveEvent(event)


class SubtitleWindow(QWidget):
    """Clean text-only subtitle window for OBS capture.

    Drag with the left mouse button to move. Resize from any edge or corner.
    Height still grows automatically when the subtitle text needs more space.
    """

    update_text_signal = pyqtSignal(str, str)  # original, translations_json
    position_changed = pyqtSignal()
    window_closed = pyqtSignal()
    window_options_changed = pyqtSignal(dict)
    update_vad_signal = pyqtSignal(float)
    screenshot_requested = pyqtSignal()

    def __init__(self, settings=None):
        super().__init__()
        self._settings = _merge_settings(DEFAULT_SUBTITLE_WIN_SETTINGS, settings)
        self._text_widgets = []
        self._sentences = []  # [(original, {lang: text, ...}), ...]
        self._drag_pos = None
        self._drag_button = None
        self._resize_edges = set()
        self._resize_start_geometry = None
        self._resize_start_global = None
        self._manual_height = max(0, int(self._settings.get("window_height", 0)))
        self._bg_pixmap = None
        self._click_through = bool(self._settings.get("click_through", False))
        self._locked = bool(self._settings.get("locked", False))
        self._always_on_top = bool(self._settings.get("always_on_top", True))
        # Mouse passthrough prevents ordinary Qt hover events over the subtitle
        # body. Poll the cursor lightly so entering that body still reveals the
        # control bar, while only actual buttons receive mouse input.
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(80)
        self._hover_timer.timeout.connect(self._sync_control_visibility)
        # Follow the main overlay's proven Win32 implementation: toggle the
        # extended transparent style based on the cursor position instead of
        # intercepting native hit-test messages in the Qt message hook.
        self._ct_timer = QTimer(self)
        self._ct_timer.setInterval(50)
        self._ct_timer.timeout.connect(self._check_click_through)
        # Auto-hide state
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self._on_auto_hide_timeout)
        self._is_hidden_by_timeout = False
        # Pending overflow segments for delayed insertion
        self._pending_segment_timers = []
        # Minimum display time: queue rapid updates instead of replacing instantly
        self._last_insert_time = 0.0
        self._min_display_ms = 1500  # minimum ms before a sentence can be replaced
        self._height_anim = None
        self._promoted_current = False

        self._setup_ui()
        self.update_text_signal.connect(self._on_update_text)
        self.update_vad_signal.connect(self._on_update_vad)

    @staticmethod
    def _is_pos_visible(x, y, margin=50):
        for screen in QApplication.screens():
            geo = screen.availableGeometry()
            if geo.left() <= x + margin and x < geo.right() and geo.top() <= y + margin and y < geo.bottom():
                return True
        return False

    def _clamp_to_screen(self):
        x, y = self.x(), self.y()
        if self._is_pos_visible(x, y):
            return
        screen = QApplication.screenAt(QPoint(x, y))
        if screen is None:
            screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        nx = max(geo.left(), min(x, geo.right() - self.width()))
        ny = max(geo.top(), min(y, geo.bottom() - self.height()))
        self.move(nx, ny)

    def _setup_ui(self):
        flags = Qt.WindowType.FramelessWindowHint
        if self._always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setWindowTitle("LiveTranslate Subtitle")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumWidth(240)
        self.setMaximumWidth(3840)

        s = self._settings
        w = s.get("window_width", 1000)
        saved_x = s.get("window_x")
        saved_y = s.get("window_y")
        if saved_x is not None and saved_y is not None:
            if self._is_pos_visible(saved_x, saved_y):
                self.move(saved_x, saved_y)
            else:
                self.move(100, 100)
        else:
            self.move(100, 100)
        self.resize(w, max(20, self._manual_height))

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        self._title_bar = _SubtitleControlBar(self, self)
        self._title_bar.setFixedHeight(28)
        self._title_bar.setStyleSheet(
            "background: rgba(0, 0, 0, 120); border-bottom: 1px solid rgba(255, 255, 255, 45);"
        )
        title_layout = QHBoxLayout(self._title_bar)
        title_layout.setContentsMargins(8, 2, 4, 2)
        title_layout.setSpacing(3)
        title_layout.addStretch()

        self._topmost_btn = QToolButton(self._title_bar)
        self._topmost_btn.setText(t("subwin_pin"))
        self._topmost_btn.setCheckable(True)
        self._topmost_btn.setChecked(self._always_on_top)
        self._topmost_btn.setToolTip(t("subwin_always_on_top_hint"))
        self._topmost_btn.setFixedSize(38, 22)
        self._topmost_btn.toggled.connect(self._on_topmost_toggled)
        title_layout.addWidget(self._topmost_btn)

        self._lock_btn = QToolButton(self._title_bar)
        self._lock_btn.setText(t("subwin_lock"))
        self._lock_btn.setCheckable(True)
        self._lock_btn.setChecked(self._locked)
        self._lock_btn.setToolTip(t("subwin_lock_hint"))
        self._lock_btn.setFixedSize(38, 22)
        self._lock_btn.toggled.connect(self._on_lock_toggled)
        title_layout.addWidget(self._lock_btn)

        self._passthrough_btn = QToolButton(self._title_bar)
        self._passthrough_btn.setText(t("subwin_mouse_passthrough"))
        self._passthrough_btn.setCheckable(True)
        self._passthrough_btn.setChecked(self._click_through)
        self._passthrough_btn.setToolTip(t("subwin_mouse_passthrough_hint"))
        self._passthrough_btn.setFixedSize(38, 22)
        self._passthrough_btn.toggled.connect(self._on_passthrough_toggled)
        title_layout.addWidget(self._passthrough_btn)

        self._screenshot_btn = QToolButton(self._title_bar)
        self._screenshot_btn.setText(t("screenshot"))
        self._screenshot_btn.setToolTip(t("screenshot_hint"))
        self._screenshot_btn.setFixedSize(52, 22)
        self._screenshot_btn.clicked.connect(self.screenshot_requested.emit)
        title_layout.addWidget(self._screenshot_btn)

        self._minimize_btn = QToolButton(self._title_bar)
        self._minimize_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMinButton)
        )
        self._minimize_btn.setToolTip(t("subwin_minimize"))
        self._minimize_btn.setFixedSize(24, 22)
        self._minimize_btn.clicked.connect(self.showMinimized)
        title_layout.addWidget(self._minimize_btn)

        self._close_btn = QToolButton(self._title_bar)
        self._close_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        )
        self._close_btn.setToolTip(t("subwin_close"))
        self._close_btn.setFixedSize(24, 22)
        self._close_btn.clicked.connect(self.close)
        title_layout.addWidget(self._close_btn)
        # Content area
        self._content = QWidget()
        self._content.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 8, 16, 8)
        self._content_layout.setSpacing(s.get("line_spacing", 8))

        self._rebuild_text_widgets()

        self._vad_bar = QProgressBar()
        self._vad_bar.setRange(0, 100)
        self._vad_bar.setTextVisible(False)
        self._vad_bar.setFixedHeight(5)
        self._vad_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._vad_bar.setStyleSheet(
            "QProgressBar { background: rgba(255,255,255,45); border: 0; border-radius: 2px; }"
            "QProgressBar::chunk { background: #dcdcaa; border-radius: 2px; }"
        )
        self._vad_bar.setVisible(bool(s.get("show_vad_meter", True)))
        self._content_layout.addWidget(self._vad_bar)

        self._main_layout.addWidget(self._content)

        self._apply_background()
        self._fit_height_animated()
        self._layout_title_bar()
        self._title_bar.hide()

    def _rebuild_text_widgets(self):
        for w in self._text_widgets:
            self._content_layout.removeWidget(w)
            w.deleteLater()
        self._text_widgets = []

        self._add_text_widgets_for_segments()

    def _add_text_widgets_for_segments(self):
        line_configs = [
            cfg for cfg in self._settings.get("lines", [])
            if cfg.get("enabled", True)
        ]
        segment_count = 2 if self._settings.get("dual_segment_mode", True) else 1
        for _segment_index in range(segment_count):
            for base_cfg in line_configs:
                line_cfg = dict(base_cfg)
                line_cfg["auto_wrap"] = self._settings.get("auto_wrap", True)
                tw = _SubtitleTextWidget()
                tw.set_config(line_cfg)
                tw.height_changed.connect(self._fit_height_animated)
                self._text_widgets.append(tw)
                vad_index = self._content_layout.indexOf(getattr(self, "_vad_bar", None))
                if vad_index >= 0:
                    self._content_layout.insertWidget(vad_index, tw)
                else:
                    self._content_layout.addWidget(tw)

    def _apply_background(self):
        s = self._settings
        resolved = _resolve_image_path(s.get("bg_image", ""))
        if resolved:
            self._bg_pixmap = QPixmap(resolved)
            self._content.setStyleSheet("background: transparent;")
        else:
            self._bg_pixmap = None
            color = s.get("bg_color", "#000000")
            opacity = s.get("bg_opacity", 0)
            if opacity == 0:
                self._content.setStyleSheet("background: transparent;")
            else:
                rgba = _hex_to_rgba(color, opacity)
                radius = s.get("border_radius", 8)
                self._content.setStyleSheet(f"background: {rgba}; border-radius: {radius}px;")
        self.update()

    def _calc_target_height(self):
        margins = self._content_layout.contentsMargins()
        spacing = self._content_layout.spacing()
        items = list(self._text_widgets)
        if getattr(self, "_vad_bar", None) is not None and self._vad_bar.isVisible():
            items.append(self._vad_bar)
        total = margins.top() + margins.bottom()
        for i, widget in enumerate(items):
            total += widget.desired_height() if isinstance(widget, _SubtitleTextWidget) else widget.height()
            if i > 0:
                total += spacing
        return max(total, 20, self._manual_height)

    def _fit_height_snap(self):
        new_h = self._calc_target_height()
        old_h = self.height()
        if new_h == old_h:
            return
        if self._height_anim and self._height_anim.state() != QPropertyAnimation.State.Stopped:
            self._height_anim.stop()
        self.move(self.x(), self.y() - (new_h - old_h) // 2)
        self.setMinimumHeight(self._calc_content_height())
        self.resize(self.width(), new_h)
        self._clamp_to_screen()
        self.position_changed.emit()

    def _fit_height_animated(self):
        new_h = self._calc_target_height()
        old_h = self.height()
        if new_h == old_h:
            return
        if self._height_anim and self._height_anim.state() != QPropertyAnimation.State.Stopped:
            self._height_anim.stop()

        target_y = self.y() - (new_h - old_h) // 2
        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(150)
        anim.setStartValue(QRect(self.x(), self.y(), self.width(), old_h))
        anim.setEndValue(QRect(self.x(), target_y, self.width(), new_h))
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def on_finished():
            self.setMinimumHeight(self._calc_content_height())
            self.setMaximumHeight(16777215)
            self.resize(self.width(), new_h)
            self._clamp_to_screen()
            self.position_changed.emit()
        anim.finished.connect(on_finished)

        self._height_anim = anim
        anim.start()

    def apply_settings(self, settings: dict):
        self._settings = _merge_settings(DEFAULT_SUBTITLE_WIN_SETTINGS, settings)
        self._manual_height = max(0, int(self._settings.get("window_height", 0)))
        self._promoted_current = False

        for w in self._text_widgets:
            self._content_layout.removeWidget(w)
            w.deleteLater()
        self._text_widgets = []

        self._add_text_widgets_for_segments()

        self._content_layout.setSpacing(self._settings.get("line_spacing", 8))

        w = self._settings.get("window_width", 1000)
        self.resize(w, max(self.height(), self._manual_height, 20))

        self._apply_background()
        self._set_always_on_top(self._settings.get("always_on_top", True), notify=False)
        self._set_locked(self._settings.get("locked", False), notify=False)
        self._vad_bar.setVisible(bool(self._settings.get("show_vad_meter", True)))
        self._refresh_display()
        self.set_click_through(self._settings.get("click_through", False))

        # Reset auto-hide timer with new settings
        self._restart_auto_hide_timer()

    # --- Auto-hide ---
    def _restart_auto_hide_timer(self):
        timeout = self._settings.get("auto_hide_timeout", 0)
        self._auto_hide_timer.stop()
        if timeout > 0 and self._sentences:
            self._auto_hide_timer.setInterval(timeout * 1000)
            self._auto_hide_timer.start()

    def _on_auto_hide_timeout(self):
        if self._is_hidden_by_timeout:
            return
        if self._settings.get("dual_segment_mode", True) and self._sentences:
            # Keep the last useful subtitle visible, but move it into the
            # previous segment until a new sentence becomes available.
            self._sentences = [self._sentences[-1]]
            self._promoted_current = True
            self._refresh_display()
            return
        self._is_hidden_by_timeout = True
        anim_type = self._settings.get("auto_hide_animation", "fade")
        duration = self._settings.get("auto_hide_duration", 300)
        for tw in self._text_widgets:
            tw.animate_out(anim_type=anim_type, duration=duration)

    def _restore_from_auto_hide(self):
        if not self._is_hidden_by_timeout:
            return
        self._is_hidden_by_timeout = False
        anim_type = self._settings.get("auto_hide_animation", "fade")
        duration = self._settings.get("auto_hide_duration", 300)
        # Reverse the hide animation type for restore
        restore_type = anim_type
        if anim_type == "slide_down":
            restore_type = "slide_up"
        elif anim_type == "slide_up":
            restore_type = "slide_down"
        elif anim_type == "slide_left":
            restore_type = "slide_right"
        elif anim_type == "slide_right":
            restore_type = "slide_left"

        for tw in self._text_widgets:
            tw._stop_all_animations()
            # Set hidden state
            tw._content_opacity_val = 0.0
            if restore_type == "slide_left":
                tw._slide_offset_x_val = float(-(tw.width() or 200))
            elif restore_type == "slide_right":
                tw._slide_offset_x_val = float(tw.width() or 200)
            elif restore_type == "slide_up":
                tw._slide_offset_y_val = float(tw.height() or 40)
            elif restore_type == "slide_down":
                tw._slide_offset_y_val = float(-(tw.height() or 40))
            tw.update()
            # Use the entry animation mechanism with overridden type
            old_entry = tw._entry_animation
            old_dur = tw._animation_duration
            tw._entry_animation = restore_type if restore_type != "none" else "fade"
            tw._animation_duration = duration
            tw.animate_in()
            tw._entry_animation = old_entry
            tw._animation_duration = old_dur

    # --- Click-through ---
    def set_click_through(self, enabled: bool, notify=False):
        self._click_through = bool(enabled)
        self._settings["click_through"] = self._click_through
        if hasattr(self, "_passthrough_btn"):
            self._passthrough_btn.blockSignals(True)
            self._passthrough_btn.setChecked(self._click_through)
            self._passthrough_btn.blockSignals(False)
        self._sync_control_visibility()
        self._check_click_through()
        if notify:
            self.window_options_changed.emit({"click_through": self._click_through})

    def _layout_title_bar(self):
        if hasattr(self, "_title_bar"):
            self._title_bar.setGeometry(0, 0, self.width(), self._title_bar.height())

    def _show_controls(self):
        if not hasattr(self, "_title_bar"):
            return
        self._layout_title_bar()
        self._title_bar.show()
        self._title_bar.raise_()

    def _hide_controls_if_pointer_outside(self):
        if not hasattr(self, "_title_bar"):
            self._title_bar.hide()
            return
        local_pos = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(local_pos):
            self._title_bar.hide()

    def _sync_control_visibility(self):
        if not self.isVisible() or not hasattr(self, "_title_bar"):
            return
        local_pos = self.mapFromGlobal(QCursor.pos())
        if self.rect().contains(local_pos):
            self._show_controls()
        else:
            self._title_bar.hide()

    def _on_topmost_toggled(self, enabled: bool):
        self._set_always_on_top(enabled, notify=True)

    def _on_lock_toggled(self, enabled: bool):
        self._set_locked(enabled, notify=True)

    def _set_locked(self, enabled: bool, notify: bool):
        enabled = bool(enabled)
        changed = enabled != self._locked
        self._locked = enabled
        self._settings["locked"] = enabled
        if hasattr(self, "_lock_btn"):
            self._lock_btn.blockSignals(True)
            self._lock_btn.setChecked(enabled)
            self._lock_btn.blockSignals(False)
        if enabled:
            self._drag_pos = None
            self._drag_button = None
            self._resize_edges.clear()
        if notify and changed:
            self.window_options_changed.emit({"locked": enabled})

    def _on_passthrough_toggled(self, enabled: bool):
        self.set_click_through(enabled, notify=True)

    def _set_always_on_top(self, enabled: bool, notify: bool):
        enabled = bool(enabled)
        changed = enabled != self._always_on_top
        self._always_on_top = enabled
        self._settings["always_on_top"] = enabled
        if hasattr(self, "_topmost_btn"):
            self._topmost_btn.blockSignals(True)
            self._topmost_btn.setChecked(enabled)
            self._topmost_btn.blockSignals(False)
        if changed:
            # Qt hides a visible frameless window when its window flags change.
            # Capture this before setWindowFlag, otherwise the subsequent
            # isVisible() check is always false and the subtitle looks closed.
            was_visible = self.isVisible()
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
            if was_visible:
                self.show()
                if enabled:
                    self.raise_()
        if notify and changed:
            self.window_options_changed.emit({"always_on_top": enabled})

    def _start_drag(self, global_pos: QPoint):
        self._drag_pos = global_pos - self.frameGeometry().topLeft()
        self._drag_button = Qt.MouseButton.LeftButton
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))

    def _finish_drag(self):
        self._drag_pos = None
        self._drag_button = None
        self.position_changed.emit()
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def showEvent(self, event):
        super().showEvent(event)
        self._hover_timer.start()
        self._ct_timer.start()
        self._sync_control_visibility()
        self._check_click_through()

    def hideEvent(self, event):
        self._hover_timer.stop()
        self._ct_timer.stop()
        self._set_native_transparent(False)
        super().hideEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_title_bar()

    def _calc_content_height(self):
        margins = self._content_layout.contentsMargins()
        spacing = self._content_layout.spacing()
        items = list(self._text_widgets)
        if getattr(self, "_vad_bar", None) is not None and self._vad_bar.isVisible():
            items.append(self._vad_bar)
        total = margins.top() + margins.bottom()
        for index, widget in enumerate(items):
            total += widget.desired_height() if isinstance(widget, _SubtitleTextWidget) else widget.height()
            if index > 0:
                total += spacing
        return max(total, 20)

    def _resize_edges_at(self, pos: QPoint) -> set[str]:
        margin = 8
        edges = set()
        if pos.x() <= margin:
            edges.add("left")
        elif pos.x() >= self.width() - margin:
            edges.add("right")
        if pos.y() <= margin:
            edges.add("top")
        elif pos.y() >= self.height() - margin:
            edges.add("bottom")
        return edges

    @staticmethod
    def _cursor_for_edges(edges: set[str]):
        if {"left", "top"} <= edges or {"right", "bottom"} <= edges:
            return Qt.CursorShape.SizeFDiagCursor
        if {"right", "top"} <= edges or {"left", "bottom"} <= edges:
            return Qt.CursorShape.SizeBDiagCursor
        if "left" in edges or "right" in edges:
            return Qt.CursorShape.SizeHorCursor
        if "top" in edges or "bottom" in edges:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def _update_hover_cursor(self, pos: QPoint):
        if self._click_through or self._locked or self._drag_pos or self._resize_edges:
            return
        self.setCursor(QCursor(self._cursor_for_edges(self._resize_edges_at(pos))))

    # --- Drag and resize ---
    def mousePressEvent(self, event):
        button = event.button()
        if self._click_through:
            super().mousePressEvent(event)
            return
        if button == Qt.MouseButton.LeftButton and not self._locked:
            edges = self._resize_edges_at(event.position().toPoint())
            if edges:
                self._resize_edges = edges
                self._resize_start_geometry = self.geometry()
                self._resize_start_global = event.globalPosition().toPoint()
                event.accept()
                return
        if not self._locked and button in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
            self._drag_button = button
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._show_controls()
        if self._resize_edges and event.buttons() & Qt.MouseButton.LeftButton:
            start = self._resize_start_geometry
            delta = event.globalPosition().toPoint() - self._resize_start_global
            left, top, right, bottom = start.left(), start.top(), start.right(), start.bottom()
            min_width = self.minimumWidth()
            min_height = self._calc_content_height()
            if "left" in self._resize_edges:
                left = min(start.right() - min_width + 1, start.left() + delta.x())
            if "right" in self._resize_edges:
                right = max(start.left() + min_width - 1, start.right() + delta.x())
            if "top" in self._resize_edges:
                top = min(start.bottom() - min_height + 1, start.top() + delta.y())
            if "bottom" in self._resize_edges:
                bottom = max(start.top() + min_height - 1, start.bottom() + delta.y())
            geometry = QRect(QPoint(left, top), QPoint(right, bottom))
            self.setGeometry(geometry)
            self._settings["window_width"] = geometry.width()
            if "top" in self._resize_edges or "bottom" in self._resize_edges:
                self._manual_height = geometry.height()
                self._settings["window_height"] = self._manual_height
            event.accept()
            return
        if self._drag_pos and event.buttons() & self._drag_button:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        self._update_hover_cursor(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._resize_edges:
            self._resize_edges.clear()
            self._resize_start_geometry = None
            self._resize_start_global = None
            self._fit_height_snap()
            self.position_changed.emit()
            self._update_hover_cursor(event.position().toPoint())
            event.accept()
            return
        if event.button() == self._drag_button and self._drag_pos:
            self._drag_pos = None
            self._drag_button = None
            self.position_changed.emit()
            self._update_hover_cursor(event.position().toPoint())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if not self._drag_pos and not self._resize_edges:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        QTimer.singleShot(0, self._hide_controls_if_pointer_outside)
        super().leaveEvent(event)

    def enterEvent(self, event):
        self._show_controls()
        super().enterEvent(event)

    def _set_native_transparent(self, transparent: bool):
        """Toggle Win32 input transparency for this top-level window."""
        if not self.winId():
            return
        try:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            has_style = bool(style & _WS_EX_TRANSPARENT)
            if transparent == has_style:
                return
            if transparent:
                style |= _WS_EX_TRANSPARENT
            else:
                style &= ~_WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style)
        except (AttributeError, OSError, TypeError, ValueError):
            # The window can be destroyed while a timer tick is queued.
            return

    def _check_click_through(self):
        """Keep the control bar interactive while the subtitle body passes through."""
        if not self.isVisible():
            return
        if not self._click_through:
            self._set_native_transparent(False)
            return

        local = self.mapFromGlobal(QCursor.pos())
        in_control_bar = (
            self._title_bar.isVisible()
            and self._title_bar.geometry().contains(local)
        )
        self._set_native_transparent(not in_control_bar)

    def closeEvent(self, event):
        self.window_closed.emit()
        super().closeEvent(event)

    def update_vad(self, confidence: float):
        """Thread-safe VAD confidence update from the audio capture loop."""
        self.update_vad_signal.emit(float(confidence))

    @pyqtSlot(float)
    def _on_update_vad(self, confidence: float):
        if not self._vad_bar.isHidden():
            self._vad_bar.setValue(max(0, min(100, round(confidence * 100))))

    def paintEvent(self, event):
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self._bg_pixmap)
            painter.end()
        super().paintEvent(event)

    # --- Text updates ---
    def update_text(self, original: str, translations: dict | str):
        """Thread-safe text update.

        translations: dict mapping lang code to translated text,
                      or a plain string (backward compat, treated as primary target).
        """
        if isinstance(translations, str):
            # Backward compat: wrap in dict with empty key
            translations = {"": translations}
        self.update_text_signal.emit(original, json.dumps(translations, ensure_ascii=False))

    @pyqtSlot(str, str)
    def _on_update_text(self, original: str, translations_json: str):
        translations = json.loads(translations_json)
        self._cancel_pending_segments()

        # Respect minimum display time: delay if previous sentence was inserted recently
        now_ms = time.monotonic() * 1000
        elapsed = now_ms - self._last_insert_time
        base_delay = max(0, int(self._min_display_ms - elapsed)) if self._last_insert_time > 0 else 0

        if base_delay == 0:
            self._insert_sentence(original, translations)
        else:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(base_delay)
            timer.timeout.connect(lambda o=original, t=translations: self._insert_sentence(o, t))
            timer.start()
            self._pending_segment_timers.append(timer)

    def _insert_sentence(self, original: str, translations: dict):
        """Insert a single sentence and refresh display."""

        max_sentences = 2 if self._settings.get("dual_segment_mode", True) else 1
        self._sentences.append((original, translations))
        if len(self._sentences) > max_sentences:
            self._sentences = self._sentences[-max_sentences:]
        self._promoted_current = False

        if self._is_hidden_by_timeout:
            self._restore_from_auto_hide()

        self._refresh_display()
        self._restart_auto_hide_timer()
        self._last_insert_time = time.monotonic() * 1000

    def _cancel_pending_segments(self):
        """Cancel any pending delayed segment insertions."""
        for timer in self._pending_segment_timers:
            timer.stop()
            timer.deleteLater()
        self._pending_segment_timers.clear()


    def _refresh_display(self):
        if not self._sentences:
            for tw in self._text_widgets:
                tw.set_text("")
            self._fit_height_snap()
            return

        lines_cfg = [
            cfg for cfg in self._settings.get("lines", [])
            if cfg.get("enabled", True)
        ]
        segment_count = 2 if self._settings.get("dual_segment_mode", True) else 1
        display_sentences = list(self._sentences)
        if segment_count == 2 and len(display_sentences) == 1:
            display_sentences = (
                [display_sentences[0], None]
                if self._promoted_current
                else [None, display_sentences[0]]
            )
        elif segment_count == 2:
            display_sentences = display_sentences[-2:]
        else:
            display_sentences = display_sentences[-1:]

        widget_index = 0
        for sentence in display_sentences:
            for cfg in lines_cfg:
                if widget_index >= len(self._text_widgets):
                    break
                text = ""
                if sentence is not None:
                    original, translations = sentence
                    if cfg.get("type", "original") == "original":
                        text = original or ""
                    elif isinstance(translations, str):
                        text = translations
                    else:
                        lang = cfg.get("lang", "")
                        if lang and lang in translations:
                            text = translations[lang]
                        elif translations.get(""):
                            text = translations[""]
                        else:
                            text = next(
                                (value for value in translations.values() if value), ""
                            )
                self._text_widgets[widget_index].set_text(text)
                widget_index += 1

        for text_widget in self._text_widgets[widget_index:]:
            text_widget.set_text("")

    def get_target_languages(self) -> set:
        """Return set of unique target language codes from enabled translation lines."""
        langs = set()
        for cfg in self._settings.get("lines", []):
            if cfg.get("enabled", True) and cfg.get("type") == "translation":
                lang = cfg.get("lang", "")
                if lang:
                    langs.add(lang)
        return langs

    def clear(self):
        self._sentences.clear()
        self._promoted_current = False
        self._cancel_pending_segments()
        self._auto_hide_timer.stop()
        self._is_hidden_by_timeout = False
        for tw in self._text_widgets:
            tw._stop_all_animations()
            tw._content_opacity_val = 1.0
            tw._slide_offset_x_val = 0.0
            tw._slide_offset_y_val = 0.0
            tw.set_text("")
        self._fit_height_snap()
