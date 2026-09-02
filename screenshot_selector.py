"""Screen-region selection overlay for screenshot translation."""

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ScreenRegionSelector(QWidget):
    # Emits the cropped screenshot and its global desktop geometry.  The
    # geometry lets the translation layer put the result back over the exact
    # screen area that was selected, including on a multi-monitor desktop.
    selection_finished = pyqtSignal(object, object)
    selection_cancelled = pyqtSignal()
    translate_requested = pyqtSignal(object)
    paste_requested = pyqtSignal(object, object)
    status_changed = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    error_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._origin = QPoint()
        self._current = QPoint()
        self._selecting = False
        self._selection_rect = QRect()
        self._global_selection_rect = QRect()
        self._processing = False
        self._cancelled = False
        self._selected_image = None
        self._translated_image = None
        self._status = "拖动鼠标框选需要翻译的区域"
        self._error = ""
        self._desktop_image, self._desktop_rect = self._capture_desktop()
        self.setGeometry(self._desktop_rect)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self._build_action_panel()
        self.status_changed.connect(self._on_status)
        self.result_ready.connect(self._on_result)
        self.error_changed.connect(self._on_error)

    def _build_action_panel(self):
        self._action_panel = QFrame(self)
        self._action_panel.setStyleSheet(
            "QFrame { background: rgba(10, 14, 24, 240);"
            " border: 1px solid rgba(110, 190, 255, 210); border-radius: 8px; }"
            "QLabel { color: #f4f7ff; border: none; }"
            "QCheckBox { color: #eaf3ff; border: none; }"
            "QPushButton { color: #f5f8ff; background: rgba(64, 126, 190, 210);"
            " border: 1px solid rgba(180, 220, 255, 180); border-radius: 5px; padding: 4px 10px; }"
            "QPushButton:hover { background: rgba(86, 158, 226, 240); }"
            "QPushButton:disabled { color: rgba(255,255,255,100); background: rgba(80,80,80,120);"
            " border-color: rgba(255,255,255,50); }"
        )
        layout = QVBoxLayout(self._action_panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(5)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: none; background: rgba(255,255,255,35); border-radius: 2px; }"
            "QProgressBar::chunk { background: #5ab9ff; border-radius: 2px; }"
        )
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._auto_translate = QCheckBox("自动翻译")
        self._auto_translate.setChecked(True)
        self._translate_button = QPushButton("翻译")
        self._paste_button = QPushButton("贴图")
        self._save_button = QPushButton("保存")
        self._copy_button = QPushButton("复制")
        self._close_button = QPushButton("关闭")
        self._translate_button.clicked.connect(self._request_translation)
        self._paste_button.clicked.connect(self._request_paste)
        self._save_button.clicked.connect(self._save_result)
        self._copy_button.clicked.connect(self._copy_result)
        self._close_button.clicked.connect(self._cancel_and_close)
        row.addWidget(self._auto_translate)
        row.addWidget(self._translate_button)
        row.addWidget(self._paste_button)
        row.addWidget(self._save_button)
        row.addWidget(self._copy_button)
        row.addWidget(self._close_button)
        layout.addLayout(row)
        self._action_panel.hide()
        self._set_actions_enabled(False)

    def _set_actions_enabled(self, has_result):
        self._translate_button.setEnabled(not self._processing)
        self._paste_button.setEnabled(bool(has_result))
        self._save_button.setEnabled(bool(has_result))
        self._copy_button.setEnabled(bool(has_result))

    def _place_action_panel(self):
        if self._selection_rect.isEmpty():
            return
        self._action_panel.adjustSize()
        width = min(max(430, self._action_panel.sizeHint().width()), self.width() - 24)
        height = self._action_panel.sizeHint().height()
        x = max(12, min(self._selection_rect.left(), self.width() - width - 12))
        y = self._selection_rect.bottom() + 12
        if y + height > self.height() - 12:
            y = self._selection_rect.top() - height - 12
        if y < 12:
            y = max(12, self.height() - height - 12)
        self._action_panel.setGeometry(x, y, width, height)

    def _show_action_panel(self):
        self._status_label.setText(self._status)
        self._place_action_panel()
        self._action_panel.show()
        self._action_panel.raise_()

    @staticmethod
    def _capture_desktop():
        screens = QApplication.screens()
        if not screens:
            return QImage(), QRect()
        rect = screens[0].geometry()
        for screen in screens[1:]:
            rect = rect.united(screen.geometry())
        image = QImage(rect.size(), QImage.Format.Format_RGB32)
        image.fill(QColor("black"))
        painter = QPainter(image)
        for screen in screens:
            geo = screen.geometry()
            pix = screen.grabWindow(0).toImage()
            painter.drawImage(QRect(geo.topLeft() - rect.topLeft(), geo.size()), pix)
        painter.end()
        return image, rect

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._desktop_image)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 105))
        rect = self._selection_rect
        if self._selecting:
            rect = QRect(self._origin, self._current).normalized()
        if not rect.isEmpty():
            # As soon as OCR/translation finishes, show the translated image
            # inside the original selection.  The surrounding desktop remains
            # dimmed until the user either pastes it as a movable sticker or
            # closes this review layer.
            if self._translated_image is not None:
                painter.drawImage(rect, self._translated_image)
            else:
                painter.drawImage(rect, self._desktop_image.copy(rect))
            painter.setPen(QPen(QColor(80, 190, 255), 2))
            painter.drawRect(rect)
            if self._selecting:
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(
                    rect.adjusted(6, 6, -6, -6),
                    Qt.AlignmentFlag.AlignTop,
                    f"{rect.width()} × {rect.height()}",
                )
        painter.end()

    def begin_review(self, image, screen_rect):
        self._selected_image = image
        self._global_selection_rect = QRect(screen_rect)
        self._translated_image = None
        self._error = ""
        self._cancelled = False
        self._status = "框选完成：可点击“翻译”。完成后译文会直接显示在框选区域内。"
        self._set_actions_enabled(False)
        self._show_action_panel()
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def should_auto_translate(self):
        return self._auto_translate.isChecked()

    def begin_processing(self):
        if self._selected_image is None or self._processing:
            return
        self._processing = True
        self._cancelled = False
        self._error = ""
        self._status = "正在准备识别…"
        self._set_actions_enabled(False)
        self._status_label.setText(self._status)
        self._progress_bar.show()
        self._place_action_panel()
        self.setCursor(Qt.CursorShape.BusyCursor)
        self.translate_requested.emit(self._selected_image)

    def set_status(self, text):
        self.status_changed.emit(str(text))

    def set_result(self, image):
        self.result_ready.emit(image)

    def set_error(self, text):
        self.error_changed.emit(str(text))

    @pyqtSlot(str)
    def _on_status(self, text):
        if self._processing and not self._cancelled:
            self._status = text
            self._status_label.setText(text)

    @pyqtSlot(object)
    def _on_result(self, image):
        if self._cancelled:
            return
        self._processing = False
        self._translated_image = image
        self._status = "翻译完成：译文已显示在框选区域内。点击“贴图”可生成可拖动的独立贴图。"
        self._status_label.setText(self._status)
        self._set_actions_enabled(True)
        self._progress_bar.hide()
        self._place_action_panel()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    @pyqtSlot(str)
    def _on_error(self, text):
        if self._cancelled:
            return
        self._processing = False
        self._error = text
        self._status = text
        self._status_label.setText(text)
        self._set_actions_enabled(False)
        self._progress_bar.hide()
        self._place_action_panel()
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _request_translation(self):
        self.begin_processing()

    def _request_paste(self):
        if self._translated_image is not None and not self._processing:
            self.paste_requested.emit(
                self._translated_image, QRect(self._global_selection_rect)
            )

    def _save_result(self):
        if self._translated_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存翻译截图", "translated.png", "PNG 图片 (*.png);;所有文件 (*)"
        )
        if path:
            self._translated_image.save(path, "PNG")

    def _copy_result(self):
        if self._translated_image is not None:
            from PyQt6.QtGui import QPixmap

            QApplication.clipboard().setPixmap(QPixmap.fromImage(self._translated_image))

    def _cancel_and_close(self):
        self._cancelled = True
        self.selection_cancelled.emit()
        self.close()

    def mousePressEvent(self, event):
        if self._processing or not self._selection_rect.isEmpty():
            if event.button() == Qt.MouseButton.RightButton:
                self._cancel_and_close()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self._selecting = True
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.selection_cancelled.emit()
            self.close()

    def mouseMoveEvent(self, event):
        if self._selecting:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._selecting:
            return
        self._current = event.position().toPoint()
        rect = QRect(self._origin, self._current).normalized()
        self._selecting = False
        if rect.width() < 8 or rect.height() < 8:
            self.selection_cancelled.emit()
            self.close()
            return
        global_rect = QRect(
            self._desktop_rect.topLeft() + rect.topLeft(),
            rect.size(),
        )
        self._selection_rect = rect
        self._action_panel.hide()
        self.selection_finished.emit(self._desktop_image.copy(rect), global_rect)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_and_close()
        else:
            super().keyPressEvent(event)
