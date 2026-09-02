import json
import logging
import os
import platform
import threading
from pathlib import Path

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from benchmark import run_benchmark
from dialogs import (
    ModelEditDialog,
    available_screen_height,
    make_scroll_area,
)
from model_manager import (
    DEFAULT_FUNASR_MODEL,
    MODELS_DIR,
    _WHISPER_SIZES,
    dir_size,
    funasr_model_options,
    funasr_supports_padding,
    format_size,
    get_cache_entries,
    list_local_faster_whisper_models,
    migrate_funasr_settings,
    normalize_funasr_model_key,
    qwen3_asr_defaults,
    resolve_custom_whisper_model,
)
from term_glossary import TermGlossary
from i18n import t, LANGUAGES
from subtitle_settings import SubtitleSettingsWidget

log = logging.getLogger("LiveTranslate.Panel")

SETTINGS_FILE = Path(__file__).parent / "user_settings.json"
_GLOSSARY_READY_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def _local_cpu_name() -> str:
    """Return a readable CPU name without requiring an external command."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if value:
                return str(value).strip()
    except (ImportError, OSError):
        pass
    return platform.processor().strip() or "CPU"


def _available_ocr_devices() -> list[tuple[str, str]]:
    """Map friendly OCR device labels to the PaddleX device strings."""
    devices = []
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index).strip()
                devices.append((f"{name} (GPU)", f"gpu:{index}"))
    except (ImportError, RuntimeError):
        pass
    devices.append((f"{_local_cpu_name()} (CPU)", "cpu"))
    return devices


class HoverInfoButton(QToolButton):
    """Information icon backed by an app-owned hover popup."""

    def __init__(self, help_text: str, parent=None):
        super().__init__(parent)
        self._help_text = help_text
        self.setText("ⓘ")
        self.setAutoRaise(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFixedSize(26, 26)
        # Keep the standard property for accessibility and keyboard help events;
        # the visible popup below does not rely on the platform tooltip service.
        self.setToolTip(help_text)

        self._popup = QFrame(self)
        self._popup.setObjectName("hoverHelpPopup")
        self._popup.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._popup.setStyleSheet(
            "QFrame#hoverHelpPopup {"
            " background-color: #fffbe6; color: #202020;"
            " border: 1px solid #8a8a8a; border-radius: 5px;"
            "}"
            "QFrame#hoverHelpPopup QLabel { color: #202020; border: none; }"
        )
        popup_layout = QVBoxLayout(self._popup)
        popup_layout.setContentsMargins(10, 8, 10, 8)
        self._help_label = QLabel(help_text)
        self._help_label.setTextFormat(Qt.TextFormat.RichText)
        self._help_label.setWordWrap(True)
        self._help_label.setFixedWidth(420)
        popup_layout.addWidget(self._help_label)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._popup.hide)
        self.clicked.connect(self._toggle_help)

    def _show_help(self):
        if not self.isVisible() or not self._help_text:
            return
        host = self.window()
        if self._popup.parentWidget() is not host:
            self._popup.setParent(host)
        self._help_label.setFixedWidth(max(260, min(420, host.width() - 40)))
        self._popup.layout().activate()
        self._popup.resize(self._popup.sizeHint())

        anchor = self.mapTo(host, QPoint(0, self.height() + 4))
        margin = 8
        x = min(
            max(anchor.x(), margin),
            max(margin, host.width() - self._popup.width() - margin),
        )
        y = anchor.y()
        if y + self._popup.height() > host.height() - margin:
            y = self.mapTo(host, QPoint(0, -self._popup.height() - 4)).y()
        y = min(
            max(y, margin),
            max(margin, host.height() - self._popup.height() - margin),
        )
        self._popup.move(x, y)
        self._popup.show()
        self._popup.raise_()
        self._hide_timer.start(20000)

    def _toggle_help(self):
        if self._popup.isVisible():
            self._popup.hide()
            self._hide_timer.stop()
        else:
            self._show_help()

    def _hide_if_pointer_left(self):
        if not self.underMouse():
            self._popup.hide()
            self._hide_timer.stop()

    def enterEvent(self, event):
        super().enterEvent(event)
        self._hide_timer.stop()
        # Defer until Qt has completed the enter event so geometry and screen
        # placement are final before the popup is positioned.
        QTimer.singleShot(0, self._show_help)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        QTimer.singleShot(180, self._hide_if_pointer_left)

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            self._show_help()
            event.accept()
            return True
        return super().event(event)


def _load_saved_settings() -> dict | None:
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            migrate_funasr_settings(data)
            log.info(f"Loaded saved settings from {SETTINGS_FILE}")
            return data
    except Exception as e:
        log.warning(f"Failed to load settings: {e}")
    return None


def _save_settings(settings: dict):
    try:
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(SETTINGS_FILE)
        log.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        log.warning(f"Failed to save settings: {e}")


class ControlPanel(QWidget):
    """Settings and monitoring panel."""

    settings_changed = pyqtSignal(dict)
    model_changed = pyqtSignal(dict)
    models_list_changed = pyqtSignal(list, int)
    subtitle_settings_changed = pyqtSignal(dict)
    _bench_result = pyqtSignal(str)
    _cache_result = pyqtSignal(list)
    reset_positions = pyqtSignal()
    visibility_changed = pyqtSignal(bool)

    def __init__(self, config, saved_settings=None):
        super().__init__()
        self._config = config
        self.setWindowTitle(t("window_control_panel"))
        self.setMinimumSize(480, 420)
        self.resize(520, min(650, available_screen_height(self)))

        saved = migrate_funasr_settings(saved_settings) or _load_saved_settings()
        if saved:
            self._current_settings = saved
        else:
            tc = config["translation"]
            self._current_settings = {
                "vad_mode": "silero",
                "vad_threshold": config["asr"]["vad_threshold"],
                "energy_threshold": 0.02,
                "firered_vad_model_path": r"D:\Models\FireRedVAD\Stream-VAD",
                "firered_threshold": 0.5,
                "firered_smooth_window": 5,
                "firered_pad_start_frames": 5,
                "firered_min_speech_frames": 8,
                "firered_min_silence_frames": 20,
                "vad_pre_roll_ms": 160,
                "vad_adaptive_silence_min": 0.3,
                "vad_adaptive_silence_max": 2.0,
                "vad_split_tail_guard": 0.5,
                "vad_progressive_split": True,
                "min_speech_duration": config["asr"]["min_speech_duration"],
                "max_speech_duration": config["asr"]["max_speech_duration"],
                "silence_mode": "auto",
                "silence_duration": 0.8,
                "asr_language": config["asr"].get("language", "auto"),
                "asr_engine": "funasr",
                "funasr_model": config["asr"].get(
                    "funasr_model", DEFAULT_FUNASR_MODEL
                ),
                "asr_device": "cuda",
                "sensevoice_pad_seconds": config["asr"].get(
                    "sensevoice_pad_seconds", 0.5
                ),
                "whisper_pad_seconds": config["asr"].get(
                    "whisper_pad_seconds", 0.5
                ),
                "models": [
                    {
                        "name": f"{tc['model']}",
                        "api_base": tc["api_base"],
                        "api_key": tc["api_key"],
                        "model": tc["model"],
                    }
                ],
                "active_model": 0,
                "hub": "ms",
            }

        if "models" not in self._current_settings:
            tc = config["translation"]
            self._current_settings["models"] = [
                {
                    "name": f"{tc['model']}",
                    "api_base": tc["api_base"],
                    "api_key": tc["api_key"],
                    "model": tc["model"],
                }
            ]
            self._current_settings["active_model"] = 0

        self._current_settings.setdefault(
            "funasr_model",
            config["asr"].get("funasr_model", DEFAULT_FUNASR_MODEL),
        )
        self._current_settings["funasr_model"] = normalize_funasr_model_key(
            self._current_settings.get("funasr_model")
        )
        self._current_settings.setdefault(
            "sensevoice_pad_seconds",
            config["asr"].get("sensevoice_pad_seconds", 0.5),
        )
        self._current_settings.setdefault(
            "whisper_pad_seconds",
            config["asr"].get("whisper_pad_seconds", 0.5),
        )
        qwen_defaults = qwen3_asr_defaults()
        self._current_settings.setdefault("qwen3_python", qwen_defaults["python"])
        self._current_settings.setdefault("qwen3_project", qwen_defaults["project"])
        self._current_settings.setdefault("qwen3_model_path", qwen_defaults["model"])
        self._current_settings.setdefault("qwen3_context_turns", 3)
        self._current_settings.setdefault("qwen3_max_new_tokens", 128)
        self._current_settings.setdefault("qwen3_hotwords", "")
        self._current_settings.setdefault("qwen3_refine_enabled", False)
        self._current_settings.setdefault(
            "firered_vad_model_path", r"D:\Models\FireRedVAD\Stream-VAD"
        )
        self._current_settings.setdefault(
            "firered_threshold", self._current_settings.get("vad_threshold", 0.5)
        )
        self._current_settings.setdefault("firered_smooth_window", 5)
        self._current_settings.setdefault("firered_pad_start_frames", 5)
        self._current_settings.setdefault("firered_min_speech_frames", 8)
        self._current_settings.setdefault("firered_min_silence_frames", 20)
        self._current_settings.setdefault("vad_pre_roll_ms", 160)
        self._current_settings.setdefault("vad_adaptive_silence_min", 0.3)
        self._current_settings.setdefault("vad_adaptive_silence_max", 2.0)
        self._current_settings.setdefault("vad_split_tail_guard", 0.5)
        self._current_settings.setdefault("vad_progressive_split", True)
        self._current_settings.setdefault(
            "glossary_enabled",
            config["translation"].get("glossary_enabled", True),
        )
        ocr_defaults = config.get("ocr") or {}
        self._current_settings.setdefault(
            "ocr_enabled", ocr_defaults.get("enabled", True)
        )
        self._current_settings.setdefault("ocr_python", ocr_defaults.get("python", ""))
        self._current_settings.setdefault(
            "ocr_model_path", ocr_defaults.get("model_path", "")
        )
        self._current_settings.setdefault(
            "ocr_device", ocr_defaults.get("device", "gpu:0")
        )
        self._current_settings.setdefault(
            "ocr_cache_dir", ocr_defaults.get("cache_dir", "paddlex_cache")
        )
        if "glossary_paths" not in self._current_settings:
            default_glossary = config["translation"].get("glossary_path")
            self._current_settings["glossary_paths"] = (
                [str(self._resolve_path(default_glossary))]
                if default_glossary
                else []
            )

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        tabs.addTab(make_scroll_area(self._create_vad_tab()), t("tab_vad_asr"))
        tabs.addTab(
            make_scroll_area(self._create_translation_tab()), t("tab_translation")
        )
        tabs.addTab(make_scroll_area(self._create_style_tab()), t("tab_style"))
        tabs.addTab(make_scroll_area(self._create_subtitle_tab()), t("tab_subtitle"))
        tabs.addTab(self._create_benchmark_tab(), t("tab_benchmark"))
        self._cache_tab_index = tabs.addTab(
            make_scroll_area(self._create_cache_tab()), t("tab_cache")
        )
        tabs.addTab(self._create_changelog_tab(), t("tab_changelog"))
        tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(tabs)

        self._bench_result.connect(self._on_bench_result)
        self._cache_result.connect(self._on_cache_result)

        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._do_auto_save)

        # Fit initial height based on whisper group visibility
        QTimer.singleShot(0, self._fit_height)

    def _fit_height(self):
        """Resize to content height, clamped to the screen (issue #39)."""
        h = min(self.sizeHint().height() + 20, available_screen_height(self))
        self.resize(self.width(), max(h, self.minimumHeight()))

    # ── VAD / ASR Tab ──

    def _create_vad_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings
        qwen_defaults = qwen3_asr_defaults()

        asr_group = QGroupBox(t("group_asr_engine"))
        asr_layout = QGridLayout(asr_group)
        asr_layout.setColumnStretch(0, 1)
        asr_layout.setColumnMinimumWidth(1, 180)

        self._asr_engine = QComboBox()
        self._asr_engine.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._asr_engine.addItems(
            [
                f"[{t('asr_accurate')}] Whisper (faster-whisper)",
                f"[{t('asr_fast')}] FunASR",
                "Anime-Whisper (ja, anime/galgame)",
                "Remote Whisper (remote GPU server)",
                "Qwen3-ASR (local Transformers)",
            ]
        )
        engine_map_idx = {
            "whisper": 0,
            "funasr": 1,
            "anime-whisper": 2,
            "remote-whisper": 3,
            "qwen3-asr": 4,
        }
        engine_idx = engine_map_idx.get(s.get("asr_engine"), 0)
        self._asr_engine.setCurrentIndex(engine_idx)
        asr_layout.addWidget(QLabel(t("label_engine")), 0, 0)
        asr_layout.addWidget(self._asr_engine, 0, 1)
        self._asr_engine.currentIndexChanged.connect(self._auto_save)

        self._asr_lang = QComboBox()
        for code, native in LANGUAGES:
            label = t("asr_lang_auto") if code == "auto" else native
            self._asr_lang.addItem(f"{code} - {label}", code)
        lang = s.get("asr_language", self._config["asr"].get("language", "auto"))
        idx = self._asr_lang.findData(lang)
        if idx >= 0:
            self._asr_lang.setCurrentIndex(idx)
        asr_layout.addWidget(QLabel(t("label_language_hint")), 1, 0)
        asr_layout.addWidget(self._asr_lang, 1, 1)
        self._asr_lang.currentIndexChanged.connect(self._auto_save)

        self._asr_device = QComboBox()
        devices = ["cuda", "cpu"]
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.insert(i, f"cuda:{i} ({name})")
            if torch.cuda.device_count() > 0:
                devices = [d for d in devices if d != "cuda"]
        except Exception:
            pass
        self._asr_device.addItems(devices)
        saved_dev = s.get("asr_device", self._config["asr"].get("device", "cuda"))
        for i in range(self._asr_device.count()):
            if self._asr_device.itemText(i).startswith(saved_dev):
                self._asr_device.setCurrentIndex(i)
                break
        asr_layout.addWidget(QLabel(t("label_device")), 2, 0)
        asr_layout.addWidget(self._asr_device, 2, 1)
        self._asr_device.currentIndexChanged.connect(self._auto_save)

        self._funasr_model_label = QLabel(t("label_funasr_model"))
        self._funasr_model_combo = QComboBox()
        for key, display_name in funasr_model_options():
            self._funasr_model_combo.addItem(display_name, key)
        saved_funasr_model = normalize_funasr_model_key(
            s.get("funasr_model", DEFAULT_FUNASR_MODEL)
        )
        funasr_idx = self._funasr_model_combo.findData(saved_funasr_model)
        if funasr_idx >= 0:
            self._funasr_model_combo.setCurrentIndex(funasr_idx)
        self._funasr_model_combo.currentIndexChanged.connect(
            self._on_funasr_model_changed
        )
        asr_layout.addWidget(self._funasr_model_label, 3, 0)
        asr_layout.addWidget(self._funasr_model_combo, 3, 1)

        self._whisper_pad_label = QLabel(t("label_whisper_padding"))
        self._whisper_pad_seconds = QDoubleSpinBox()
        self._whisper_pad_seconds.setRange(0.0, 5.0)
        self._whisper_pad_seconds.setDecimals(2)
        self._whisper_pad_seconds.setSingleStep(0.1)
        try:
            whisper_pad_seconds = float(s.get("whisper_pad_seconds", 0.5))
        except (TypeError, ValueError):
            whisper_pad_seconds = 0.5
        self._whisper_pad_seconds.setValue(whisper_pad_seconds)
        self._whisper_pad_seconds.setSuffix(" s")
        self._whisper_pad_seconds.setSpecialValueText(t("whisper_padding_off"))
        self._whisper_pad_seconds.setToolTip(t("whisper_padding_tooltip"))
        asr_layout.addWidget(self._whisper_pad_label, 4, 0)
        asr_layout.addWidget(self._whisper_pad_seconds, 4, 1)
        self._whisper_pad_seconds.valueChanged.connect(self._auto_save)

        self._sensevoice_pad_label = QLabel(t("label_sensevoice_padding"))
        self._sensevoice_pad_seconds = QDoubleSpinBox()
        self._sensevoice_pad_seconds.setRange(0.0, 5.0)
        self._sensevoice_pad_seconds.setDecimals(2)
        self._sensevoice_pad_seconds.setSingleStep(0.1)
        try:
            sensevoice_pad_seconds = float(s.get("sensevoice_pad_seconds", 0.5))
        except (TypeError, ValueError):
            sensevoice_pad_seconds = 0.5
        self._sensevoice_pad_seconds.setValue(sensevoice_pad_seconds)
        self._sensevoice_pad_seconds.setSuffix(" s")
        self._sensevoice_pad_seconds.setSpecialValueText(t("sensevoice_padding_off"))
        self._sensevoice_pad_seconds.setToolTip(t("sensevoice_padding_tooltip"))
        asr_layout.addWidget(self._sensevoice_pad_label, 5, 0)
        asr_layout.addWidget(self._sensevoice_pad_seconds, 5, 1)
        self._sensevoice_pad_seconds.valueChanged.connect(self._auto_save)

        self._audio_device = QComboBox()
        self._audio_device.addItem(t("audio_disabled"))
        self._audio_device.addItem(t("system_default"))
        try:
            from audio_capture import list_output_devices

            for name in list_output_devices():
                self._audio_device.addItem(name)
        except Exception:
            pass
        saved_audio = s.get("audio_device")
        if saved_audio == "__disabled__":
            self._audio_device.setCurrentIndex(0)
        elif saved_audio:
            idx = self._audio_device.findText(saved_audio)
            if idx >= 0:
                self._audio_device.setCurrentIndex(idx)
        else:
            self._audio_device.setCurrentIndex(1)  # system default
        asr_layout.addWidget(QLabel(t("label_audio")), 6, 0)
        asr_layout.addWidget(self._audio_device, 6, 1)
        self._audio_device.currentIndexChanged.connect(self._auto_save)

        self._mic_device = QComboBox()
        self._mic_device.addItem(t("mic_disabled"))
        self._mic_device.addItem(t("system_default"))
        try:
            from audio_capture import list_input_devices

            for name in list_input_devices():
                self._mic_device.addItem(name)
        except Exception:
            pass
        saved_mic = s.get("mic_device")
        if saved_mic:
            if saved_mic in ("__default__", "default"):
                self._mic_device.setCurrentIndex(1)
            else:
                idx = self._mic_device.findText(saved_mic)
                if idx >= 0:
                    self._mic_device.setCurrentIndex(idx)
        asr_layout.addWidget(QLabel(t("label_mic")), 7, 0)
        asr_layout.addWidget(self._mic_device, 7, 1)
        self._mic_device.currentIndexChanged.connect(self._auto_save)

        self._hub_combo = QComboBox()
        self._hub_combo.addItems([t("hub_modelscope"), t("hub_huggingface")])
        saved_hub = s.get("hub", "ms")
        self._hub_combo.setCurrentIndex(0 if saved_hub == "ms" else 1)
        asr_layout.addWidget(QLabel(t("label_hub")), 8, 0)
        asr_layout.addWidget(self._hub_combo, 8, 1)
        self._hub_combo.currentIndexChanged.connect(self._auto_save)

        self._ui_lang_combo = QComboBox()
        self._ui_lang_combo.addItems(["English", "中文"])
        from i18n import get_lang

        saved_lang = s.get("ui_lang", get_lang())
        self._ui_lang_combo.setCurrentIndex(0 if saved_lang == "en" else 1)
        asr_layout.addWidget(QLabel(t("label_ui_lang")), 9, 0)
        asr_layout.addWidget(self._ui_lang_combo, 9, 1)
        self._ui_lang_combo.currentIndexChanged.connect(self._on_ui_lang_changed)

        layout.addWidget(asr_group)

        # Whisper model download — only visible when engine is Whisper
        self._whisper_group = QGroupBox(t("group_download_whisper"))
        whisper_layout = QHBoxLayout(self._whisper_group)
        self._whisper_size_combo = QComboBox()
        saved_size = s.get(
            "whisper_model_size", self._config["asr"].get("model_size", "medium")
        )
        self._populate_whisper_models(saved_size)
        self._whisper_size_combo.currentIndexChanged.connect(
            self._on_whisper_size_changed
        )
        whisper_layout.addWidget(self._whisper_size_combo)
        self._whisper_status = QLabel("")
        self._whisper_status.setStyleSheet("color: #888; font-size: 11px;")
        whisper_layout.addWidget(self._whisper_status, 1)
        self._whisper_dl_btn = QPushButton(t("btn_download_whisper"))
        self._whisper_dl_btn.clicked.connect(self._download_whisper)
        whisper_layout.addWidget(self._whisper_dl_btn)
        layout.addWidget(self._whisper_group)
        self._whisper_group.setVisible(engine_idx == 0)
        self._asr_engine.currentIndexChanged.connect(
            self._on_engine_changed_whisper_vis
        )
        self._on_engine_changed_whisper_vis(engine_idx)
        self._update_whisper_size_label()

        # Remote ASR server URL — only visible when engine is Remote Whisper
        self._remote_group = QGroupBox("Remote ASR Server")
        remote_layout = QHBoxLayout(self._remote_group)
        remote_layout.addWidget(QLabel("URL"))
        self._remote_url_edit = QLineEdit(
            s.get("remote_asr_url", "http://127.0.0.1:8765")
        )
        self._remote_url_edit.setPlaceholderText("http://127.0.0.1:8765")
        self._remote_url_edit.editingFinished.connect(self._auto_save)
        remote_layout.addWidget(self._remote_url_edit, 1)
        layout.addWidget(self._remote_group)
        self._remote_group.setVisible(engine_idx == 3)

        self._qwen3_group = QGroupBox(t("group_qwen3_asr"))
        qwen3_layout = QGridLayout(self._qwen3_group)
        self._qwen3_model_path = QLineEdit(s.get("qwen3_model_path", ""))
        self._qwen3_model_path.setPlaceholderText(qwen_defaults["model"])
        self._qwen3_python = QLineEdit(s.get("qwen3_python", ""))
        self._qwen3_python.setPlaceholderText(qwen_defaults["python"])
        self._qwen3_project = QLineEdit(s.get("qwen3_project", ""))
        self._qwen3_project.setPlaceholderText(qwen_defaults["project"])
        qwen3_layout.addWidget(QLabel(t("label_qwen3_model_path")), 0, 0)
        qwen3_layout.addWidget(self._qwen3_model_path, 0, 1)
        qwen3_layout.addWidget(QLabel(t("label_qwen3_python")), 1, 0)
        qwen3_layout.addWidget(self._qwen3_python, 1, 1)
        qwen3_layout.addWidget(QLabel(t("label_qwen3_project")), 2, 0)
        qwen3_layout.addWidget(self._qwen3_project, 2, 1)
        self._qwen3_context_turns = QSpinBox()
        self._qwen3_context_turns.setRange(0, 8)
        self._qwen3_context_turns.setValue(int(s.get("qwen3_context_turns", 3)))
        self._qwen3_context_turns.setToolTip(t("qwen3_context_turns_hint"))
        self._qwen3_context_turns.valueChanged.connect(self._auto_save)
        qwen3_layout.addWidget(QLabel(t("label_qwen3_context_turns")), 3, 0)
        qwen3_layout.addWidget(self._qwen3_context_turns, 3, 1)

        self._qwen3_max_new_tokens = QSpinBox()
        self._qwen3_max_new_tokens.setRange(32, 256)
        self._qwen3_max_new_tokens.setSingleStep(16)
        self._qwen3_max_new_tokens.setValue(int(s.get("qwen3_max_new_tokens", 128)))
        self._qwen3_max_new_tokens.setToolTip(t("qwen3_max_new_tokens_hint"))
        self._qwen3_max_new_tokens.valueChanged.connect(self._auto_save)
        qwen3_layout.addWidget(QLabel(t("label_qwen3_max_new_tokens")), 4, 0)
        qwen3_layout.addWidget(self._qwen3_max_new_tokens, 4, 1)

        self._qwen3_refine_enabled = QCheckBox(t("label_qwen3_refine"))
        self._qwen3_refine_enabled.setChecked(
            s.get("qwen3_refine_enabled", False)
        )
        self._qwen3_refine_enabled.setToolTip(t("qwen3_refine_hint"))
        self._qwen3_refine_enabled.toggled.connect(self._auto_save)

        self._glossary_enabled_cb = QCheckBox(t("label_enable_glossary"))
        self._glossary_enabled_cb.setChecked(s.get("glossary_enabled", True))
        glossary_tooltip = t("glossary_enabled_tooltip")
        self._glossary_enabled_cb.setToolTip(glossary_tooltip)
        self._glossary_enabled_cb.toggled.connect(
            self._on_glossary_enabled_changed
        )

        feature_toggle_row = QHBoxLayout()
        feature_toggle_row.addWidget(self._qwen3_refine_enabled)
        feature_toggle_row.addWidget(self._glossary_enabled_cb)
        feature_toggle_row.addStretch()
        qwen3_layout.addLayout(feature_toggle_row, 5, 0, 1, 2)

        self._qwen3_hotwords = QLineEdit(s.get("qwen3_hotwords", ""))
        self._qwen3_hotwords.setPlaceholderText(t("qwen3_hotwords_placeholder"))
        self._qwen3_hotwords.setToolTip(t("qwen3_hotwords_hint"))
        self._qwen3_hotwords.editingFinished.connect(self._auto_save)
        qwen3_layout.addWidget(QLabel(t("label_qwen3_hotwords")), 6, 0)
        qwen3_layout.addWidget(self._qwen3_hotwords, 6, 1)
        for edit in (self._qwen3_model_path, self._qwen3_python, self._qwen3_project):
            edit.editingFinished.connect(self._auto_save)
        layout.addWidget(self._qwen3_group)
        self._qwen3_group.setVisible(engine_idx == 4)

        mode_group = QGroupBox(t("group_vad_mode"))
        mode_layout = QVBoxLayout(mode_group)
        self._vad_mode = QComboBox()
        self._vad_mode.addItems(
            [t("vad_silero"), t("vad_firered"), t("vad_energy"), t("vad_disabled")]
        )
        mode_map = {"silero": 0, "firered": 1, "energy": 2, "disabled": 3}
        self._vad_mode.setCurrentIndex(mode_map.get(s.get("vad_mode", "energy"), 1))
        self._vad_mode.currentIndexChanged.connect(self._on_vad_mode_changed)
        self._vad_mode.currentIndexChanged.connect(self._auto_save)
        mode_layout.addWidget(self._vad_mode)
        layout.addWidget(mode_group)

        self._silero_group = QGroupBox(t("group_silero_threshold"))
        silero_layout = QGridLayout(self._silero_group)
        self._vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._vad_threshold_slider.setRange(0, 100)
        vad_pct = int(s.get("vad_threshold", 0.5) * 100)
        self._vad_threshold_slider.setValue(vad_pct)
        self._vad_threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._vad_threshold_slider.sliderReleased.connect(self._auto_save)
        self._vad_threshold_label = QLabel(f"{vad_pct}%")
        self._vad_threshold_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        silero_layout.addWidget(QLabel(t("label_threshold")), 0, 0)
        silero_layout.addWidget(self._vad_threshold_slider, 0, 1)
        silero_layout.addWidget(self._vad_threshold_label, 0, 2)
        layout.addWidget(self._silero_group)

        self._firered_group = QGroupBox(t("group_firered_vad"))
        firered_layout = QGridLayout(self._firered_group)
        self._firered_model_path = QLineEdit(
            s.get("firered_vad_model_path", r"D:\Models\FireRedVAD\Stream-VAD")
        )
        self._firered_model_path.setToolTip(t("firered_vad_path_hint"))
        self._firered_model_path.editingFinished.connect(self._auto_save)
        firered_layout.addWidget(QLabel(t("label_firered_vad_path")), 0, 0)
        firered_layout.addWidget(self._firered_model_path, 0, 1)
        self._firered_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._firered_threshold_slider.setRange(1, 99)
        firered_pct = int(s.get("firered_threshold", 0.5) * 100)
        self._firered_threshold_slider.setValue(firered_pct)
        self._firered_threshold_slider.valueChanged.connect(
            self._on_firered_threshold_changed
        )
        self._firered_threshold_slider.sliderReleased.connect(self._auto_save)
        self._firered_threshold_label = QLabel(f"{firered_pct}%")
        self._firered_threshold_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        firered_layout.addWidget(QLabel(t("label_firered_threshold")), 1, 0)
        firered_layout.addWidget(self._firered_threshold_slider, 1, 1)
        firered_layout.addWidget(self._firered_threshold_label, 1, 2)
        self._firered_smooth_window = QSpinBox()
        self._firered_smooth_window.setRange(1, 15)
        self._firered_smooth_window.setValue(s.get("firered_smooth_window", 5))
        self._firered_pad_start_frames = QSpinBox()
        self._firered_pad_start_frames.setRange(0, 30)
        self._firered_pad_start_frames.setValue(
            s.get("firered_pad_start_frames", 5)
        )
        self._firered_min_speech_frames = QSpinBox()
        self._firered_min_speech_frames.setRange(1, 50)
        self._firered_min_speech_frames.setValue(
            s.get("firered_min_speech_frames", 8)
        )
        self._firered_min_silence_frames = QSpinBox()
        self._firered_min_silence_frames.setRange(1, 100)
        self._firered_min_silence_frames.setValue(
            s.get("firered_min_silence_frames", 20)
        )
        for control in (
            self._firered_smooth_window,
            self._firered_pad_start_frames,
            self._firered_min_speech_frames,
            self._firered_min_silence_frames,
        ):
            control.valueChanged.connect(self._on_timing_changed)
            control.valueChanged.connect(self._auto_save)
        firered_layout.addWidget(QLabel(t("label_firered_smooth")), 2, 0)
        firered_layout.addWidget(self._firered_smooth_window, 2, 1)
        firered_layout.addWidget(QLabel(t("label_firered_pad_start")), 3, 0)
        firered_layout.addWidget(self._firered_pad_start_frames, 3, 1)
        firered_layout.addWidget(QLabel(t("label_firered_min_speech")), 4, 0)
        firered_layout.addWidget(self._firered_min_speech_frames, 4, 1)
        firered_layout.addWidget(QLabel(t("label_firered_min_silence")), 5, 0)
        firered_layout.addWidget(self._firered_min_silence_frames, 5, 1)
        layout.addWidget(self._firered_group)
        self._firered_group.setVisible(s.get("vad_mode", "energy") == "firered")

        self._energy_group = QGroupBox(t("group_energy_threshold"))
        energy_layout = QGridLayout(self._energy_group)
        self._energy_slider = QSlider(Qt.Orientation.Horizontal)
        self._energy_slider.setRange(1, 100)
        energy_pm = int(s.get("energy_threshold", 0.03) * 1000)
        self._energy_slider.setValue(energy_pm)
        self._energy_slider.valueChanged.connect(self._on_energy_changed)
        self._energy_slider.sliderReleased.connect(self._auto_save)
        self._energy_label = QLabel(f"{energy_pm}\u2030")
        self._energy_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        energy_layout.addWidget(QLabel(t("label_threshold")), 0, 0)
        energy_layout.addWidget(self._energy_slider, 0, 1)
        energy_layout.addWidget(self._energy_label, 0, 2)
        layout.addWidget(self._energy_group)
        self._update_vad_mode_visibility(s.get("vad_mode", "silero"))

        timing_group = QGroupBox(t("group_timing"))
        timing_layout = QGridLayout(timing_group)
        timing_layout.setColumnStretch(0, 1)
        timing_layout.setColumnMinimumWidth(1, 180)
        self._min_speech = QDoubleSpinBox()
        self._min_speech.setRange(0.1, 5.0)
        self._min_speech.setSingleStep(0.1)
        self._min_speech.setValue(s.get("min_speech_duration", 2.0))
        self._min_speech.setSuffix(" s")
        self._min_speech.valueChanged.connect(self._on_timing_changed)
        self._min_speech.valueChanged.connect(self._auto_save)
        self._max_speech = QDoubleSpinBox()
        self._max_speech.setRange(2.0, 30.0)
        self._max_speech.setSingleStep(1.0)
        self._max_speech.setValue(s.get("max_speech_duration", 6.0))
        self._max_speech.setSuffix(" s")
        self._max_speech.valueChanged.connect(self._on_timing_changed)
        self._max_speech.valueChanged.connect(self._auto_save)
        self._silence_mode = QComboBox()
        self._silence_mode.addItems([t("silence_auto"), t("silence_fixed")])
        saved_smode = s.get("silence_mode", "auto")
        self._silence_mode.setCurrentIndex(0 if saved_smode == "auto" else 1)
        self._silence_mode.currentIndexChanged.connect(self._on_silence_mode_changed)
        self._silence_mode.currentIndexChanged.connect(self._on_timing_changed)
        self._silence_mode.currentIndexChanged.connect(self._auto_save)

        self._silence_duration = QDoubleSpinBox()
        self._silence_duration.setRange(0.1, 3.0)
        self._silence_duration.setSingleStep(0.1)
        self._silence_duration.setValue(s.get("silence_duration", 0.8))
        self._silence_duration.setSuffix(" s")
        self._silence_duration.setEnabled(saved_smode != "auto")
        self._silence_duration.valueChanged.connect(self._on_timing_changed)
        self._silence_duration.valueChanged.connect(self._auto_save)

        timing_layout.addWidget(QLabel(t("label_min_speech")), 0, 0)
        timing_layout.addWidget(self._min_speech, 0, 1)
        timing_layout.addWidget(QLabel(t("label_max_speech")), 1, 0)
        timing_layout.addWidget(self._max_speech, 1, 1)
        timing_layout.addWidget(QLabel(t("label_silence")), 2, 0)
        timing_layout.addWidget(self._silence_mode, 2, 1)
        timing_layout.addWidget(QLabel(t("label_silence_dur")), 3, 0)
        timing_layout.addWidget(self._silence_duration, 3, 1)

        self._incremental_asr_cb = QCheckBox(t("label_incremental_asr"))
        self._incremental_asr_cb.setToolTip(t("incremental_asr_tooltip"))
        self._incremental_asr_cb.setChecked(s.get("incremental_asr", False))
        self._incremental_asr_cb.toggled.connect(self._on_timing_changed)
        self._incremental_asr_cb.toggled.connect(self._auto_save)
        timing_layout.addWidget(self._incremental_asr_cb, 4, 0)

        self._interim_interval_spin = QDoubleSpinBox()
        self._interim_interval_spin.setRange(1.0, 10.0)
        self._interim_interval_spin.setSingleStep(0.5)
        self._interim_interval_spin.setValue(s.get("interim_interval", 2.0))
        self._interim_interval_spin.setSuffix(" s")
        self._interim_interval_spin.setEnabled(s.get("incremental_asr", False))
        self._interim_interval_spin.valueChanged.connect(self._on_timing_changed)
        self._interim_interval_spin.valueChanged.connect(self._auto_save)
        self._incremental_asr_cb.toggled.connect(self._interim_interval_spin.setEnabled)
        timing_layout.addWidget(QLabel(t("label_interim_interval")), 5, 0)
        timing_layout.addWidget(self._interim_interval_spin, 5, 1)

        layout.addWidget(timing_group)

        advanced_vad_group = QGroupBox(t("group_vad_advanced"))
        advanced_vad_layout = QGridLayout(advanced_vad_group)
        self._vad_pre_roll_ms = QSpinBox()
        self._vad_pre_roll_ms.setRange(0, 500)
        self._vad_pre_roll_ms.setSingleStep(20)
        self._vad_pre_roll_ms.setValue(s.get("vad_pre_roll_ms", 160))
        self._vad_adaptive_min = QDoubleSpinBox()
        self._vad_adaptive_min.setRange(0.1, 2.0)
        self._vad_adaptive_min.setSingleStep(0.1)
        self._vad_adaptive_min.setValue(s.get("vad_adaptive_silence_min", 0.3))
        self._vad_adaptive_max = QDoubleSpinBox()
        self._vad_adaptive_max.setRange(0.2, 5.0)
        self._vad_adaptive_max.setSingleStep(0.1)
        self._vad_adaptive_max.setValue(s.get("vad_adaptive_silence_max", 2.0))
        self._vad_split_tail_guard = QDoubleSpinBox()
        self._vad_split_tail_guard.setRange(0.0, 2.0)
        self._vad_split_tail_guard.setSingleStep(0.1)
        self._vad_split_tail_guard.setValue(s.get("vad_split_tail_guard", 0.5))
        self._vad_progressive_split = QCheckBox(t("label_vad_progressive_split"))
        self._vad_progressive_split.setChecked(s.get("vad_progressive_split", True))
        for control in (
            self._vad_pre_roll_ms,
            self._vad_adaptive_min,
            self._vad_adaptive_max,
            self._vad_split_tail_guard,
        ):
            control.valueChanged.connect(self._on_timing_changed)
            control.valueChanged.connect(self._auto_save)
        self._vad_progressive_split.toggled.connect(self._on_timing_changed)
        self._vad_progressive_split.toggled.connect(self._auto_save)
        advanced_vad_layout.addWidget(QLabel(t("label_vad_pre_roll")), 0, 0)
        advanced_vad_layout.addWidget(self._vad_pre_roll_ms, 0, 1)
        advanced_vad_layout.addWidget(QLabel(t("label_vad_adaptive_min")), 1, 0)
        advanced_vad_layout.addWidget(self._vad_adaptive_min, 1, 1)
        advanced_vad_layout.addWidget(QLabel(t("label_vad_adaptive_max")), 2, 0)
        advanced_vad_layout.addWidget(self._vad_adaptive_max, 2, 1)
        advanced_vad_layout.addWidget(QLabel(t("label_vad_tail_guard")), 3, 0)
        advanced_vad_layout.addWidget(self._vad_split_tail_guard, 3, 1)
        advanced_vad_layout.addWidget(self._vad_progressive_split, 4, 0, 1, 2)
        layout.addWidget(advanced_vad_group)

        layout.addStretch()
        return widget

    # ── Translation Tab ──

    def _create_translation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings

        models_group = QGroupBox(t("group_model_configs"))
        models_layout = QVBoxLayout(models_group)

        self._model_list = QListWidget()
        self._model_list.setFont(QFont("Consolas", 9))
        self._model_list.itemDoubleClicked.connect(self._on_model_double_clicked)
        self._refresh_model_list()
        models_layout.addWidget(self._model_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton(t("btn_add"))
        add_btn.clicked.connect(self._add_model)
        btn_row.addWidget(add_btn)
        edit_btn = QPushButton(t("btn_edit"))
        edit_btn.clicked.connect(self._edit_model)
        btn_row.addWidget(edit_btn)
        dup_btn = QPushButton(t("btn_duplicate"))
        dup_btn.clicked.connect(self._dup_model)
        btn_row.addWidget(dup_btn)
        del_btn = QPushButton(t("btn_remove"))
        del_btn.clicked.connect(self._remove_model)
        btn_row.addWidget(del_btn)
        models_layout.addLayout(btn_row)
        layout.addWidget(models_group)

        glossary_group = QGroupBox(t("group_term_glossaries"))
        glossary_layout = QVBoxLayout(glossary_group)

        self._glossary_list = QListWidget()
        self._glossary_list.setMaximumHeight(100)
        self._glossary_list.currentItemChanged.connect(
            self._on_glossary_selection_changed
        )
        glossary_layout.addWidget(self._glossary_list)

        self._glossary_status = QLabel("")
        self._glossary_status.setWordWrap(True)
        self._glossary_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        glossary_layout.addWidget(self._glossary_status)

        self._glossary_path_view = QLineEdit()
        self._glossary_path_view.setReadOnly(True)
        glossary_layout.addWidget(self._glossary_path_view)

        glossary_btn_row = QHBoxLayout()
        self._add_glossary_btn = QPushButton(t("btn_add_glossary"))
        self._add_glossary_btn.clicked.connect(self._add_glossary)
        glossary_btn_row.addWidget(self._add_glossary_btn)
        self._remove_glossary_btn = QPushButton(t("btn_remove_glossary"))
        self._remove_glossary_btn.setToolTip(t("remove_glossary_hint"))
        self._remove_glossary_btn.clicked.connect(self._remove_glossary)
        glossary_btn_row.addWidget(self._remove_glossary_btn)
        self._reload_glossary_btn = QPushButton(t("btn_reload_glossaries"))
        self._reload_glossary_btn.clicked.connect(self._reload_glossary_list)
        glossary_btn_row.addWidget(self._reload_glossary_btn)
        self._open_glossary_file_btn = QPushButton(t("btn_open_glossary_file"))
        self._open_glossary_file_btn.clicked.connect(self._open_glossary_file)
        glossary_btn_row.addWidget(self._open_glossary_file_btn)
        self._open_glossary_folder_btn = QPushButton(
            t("btn_open_glossary_folder")
        )
        self._open_glossary_folder_btn.clicked.connect(self._open_glossary_folder)
        glossary_btn_row.addWidget(self._open_glossary_folder_btn)
        glossary_btn_row.addStretch()
        glossary_layout.addLayout(glossary_btn_row)
        self._refresh_glossary_list()
        layout.addWidget(glossary_group)

        ocr_group = QGroupBox(t("group_ocr_screenshot"))
        ocr_layout = QGridLayout(ocr_group)
        ocr_layout.setColumnStretch(1, 1)

        self._ocr_enabled_cb = QCheckBox(t("label_enable_ocr_screenshot"))
        self._ocr_enabled_cb.setChecked(bool(s.get("ocr_enabled", True)))
        self._ocr_enabled_cb.setToolTip(t("ocr_screenshot_tooltip"))
        self._ocr_enabled_cb.toggled.connect(self._on_ocr_enabled_changed)
        ocr_layout.addWidget(self._ocr_enabled_cb, 0, 0, 1, 2)

        self._ocr_python_edit = QLineEdit(s.get("ocr_python", ""))
        self._ocr_python_edit.setPlaceholderText("C:/.../envs/paddleocr-vl/python.exe")
        ocr_layout.addWidget(QLabel(t("label_ocr_python")), 1, 0)
        ocr_layout.addWidget(self._ocr_python_edit, 1, 1)

        self._ocr_model_path_edit = QLineEdit(s.get("ocr_model_path", ""))
        self._ocr_model_path_edit.setPlaceholderText("D:/Models/PaddleOCR-VL-1.6")
        ocr_layout.addWidget(QLabel(t("label_ocr_model_path")), 2, 0)
        ocr_layout.addWidget(self._ocr_model_path_edit, 2, 1)

        self._ocr_device_combo = QComboBox()
        saved_ocr_device = str(s.get("ocr_device", "gpu:0")).strip() or "gpu:0"
        for label, device in _available_ocr_devices():
            self._ocr_device_combo.addItem(label, device)
        device_index = self._ocr_device_combo.findData(saved_ocr_device)
        if device_index < 0:
            # Keep an unavailable saved selection without exposing its raw ID.
            self._ocr_device_combo.addItem(t("ocr_device_unavailable"), saved_ocr_device)
            device_index = self._ocr_device_combo.count() - 1
        self._ocr_device_combo.setCurrentIndex(device_index)
        self._ocr_device_combo.currentIndexChanged.connect(self._apply_settings)
        self._ocr_device_combo.currentIndexChanged.connect(self._auto_save)
        ocr_layout.addWidget(QLabel(t("label_ocr_device")), 3, 0)
        ocr_layout.addWidget(self._ocr_device_combo, 3, 1)

        self._ocr_cache_dir_edit = QLineEdit(s.get("ocr_cache_dir", "paddlex_cache"))
        self._ocr_cache_dir_edit.setPlaceholderText("paddlex_cache")
        ocr_layout.addWidget(QLabel(t("label_ocr_cache_dir")), 4, 0)
        ocr_layout.addWidget(self._ocr_cache_dir_edit, 4, 1)

        for edit in (
            self._ocr_python_edit,
            self._ocr_model_path_edit,
            self._ocr_cache_dir_edit,
        ):
            edit.editingFinished.connect(self._apply_settings)
            edit.editingFinished.connect(self._auto_save)
        layout.addWidget(ocr_group)

        prompt_group = QGroupBox(t("group_system_prompt"))
        prompt_layout = QVBoxLayout(prompt_group)

        from translator import DEFAULT_PROMPT, PROMPT_PRESETS

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(t("label_prompt_preset")))
        self._prompt_preset = QComboBox()
        self._prompt_preset.addItem(t("prompt_daily"), "daily")
        self._prompt_preset.addItem(t("prompt_esports"), "esports")
        self._prompt_preset.addItem(t("prompt_anime"), "anime")
        self._prompt_preset.addItem(t("prompt_webid"), "webid")
        self._prompt_preset.addItem(t("prompt_custom"), "custom")

        current_prompt = s.get("system_prompt", DEFAULT_PROMPT)
        preset_idx = 4  # default to custom
        for i, key in enumerate(["daily", "esports", "anime", "webid"]):
            if current_prompt.strip() == PROMPT_PRESETS[key].strip():
                preset_idx = i
                break
        if current_prompt.strip() == DEFAULT_PROMPT.strip():
            preset_idx = 0
        self._prompt_preset.setCurrentIndex(preset_idx)
        self._prompt_preset.currentIndexChanged.connect(self._on_prompt_preset_changed)
        preset_row.addWidget(self._prompt_preset, 1)
        prompt_layout.addLayout(preset_row)

        # Prompt text editor
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setFont(QFont("Consolas", 9))
        self._prompt_edit.setMaximumHeight(100)
        self._prompt_edit.setPlainText(current_prompt)
        self._prompt_debounce = QTimer()
        self._prompt_debounce.setSingleShot(True)
        self._prompt_debounce.setInterval(600)
        self._prompt_debounce.timeout.connect(self._apply_prompt)
        self._prompt_edit.textChanged.connect(self._prompt_debounce.start)
        prompt_layout.addWidget(self._prompt_edit)
        layout.addWidget(prompt_group)

        net_group = QGroupBox(t("group_network"))
        net_layout = QGridLayout(net_group)
        net_layout.setColumnStretch(0, 1)
        net_layout.setColumnMinimumWidth(1, 180)
        net_layout.addWidget(QLabel(t("label_timeout")), 0, 0)
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(1, 60)
        self._timeout_spin.setValue(s.get("timeout", 5))
        self._timeout_spin.setSuffix(" s")
        self._timeout_spin.valueChanged.connect(
            lambda v: self._current_settings.update({"timeout": v})
        )
        self._timeout_spin.valueChanged.connect(self._auto_save)
        net_layout.addWidget(self._timeout_spin, 0, 1)
        layout.addWidget(net_group)

        layout.addStretch()
        return widget

    # ── Style Tab ──

    def _create_style_tab(self):
        from subtitle_overlay import DEFAULT_STYLE

        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings.get("style", dict(DEFAULT_STYLE))

        # Preset group
        preset_group = QGroupBox(t("group_preset"))
        preset_layout = QHBoxLayout(preset_group)
        self._style_preset = QComboBox()
        preset_names = [
            ("default", t("preset_default")),
            ("transparent", t("preset_transparent")),
            ("compact", t("preset_compact")),
            ("light", t("preset_light")),
            ("dracula", t("preset_dracula")),
            ("nord", t("preset_nord")),
            ("monokai", t("preset_monokai")),
            ("solarized", t("preset_solarized")),
            ("gruvbox", t("preset_gruvbox")),
            ("tokyo_night", t("preset_tokyo_night")),
            ("catppuccin", t("preset_catppuccin")),
            ("one_dark", t("preset_one_dark")),
            ("everforest", t("preset_everforest")),
            ("kanagawa", t("preset_kanagawa")),
            ("custom", t("preset_custom")),
        ]
        self._preset_keys = [k for k, _ in preset_names]
        for _, label in preset_names:
            self._style_preset.addItem(label)
        current_preset = s.get("preset", "default")
        if current_preset in self._preset_keys:
            self._style_preset.setCurrentIndex(self._preset_keys.index(current_preset))
        else:
            self._style_preset.setCurrentIndex(5)  # custom
        self._style_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._style_preset, 1)
        reset_btn = QPushButton(t("btn_reset_style"))
        reset_btn.clicked.connect(self._reset_style)
        preset_layout.addWidget(reset_btn)
        reset_pos_btn = QPushButton(t("btn_reset_positions"))
        reset_pos_btn.clicked.connect(self.reset_positions.emit)
        preset_layout.addWidget(reset_pos_btn)
        layout.addWidget(preset_group)

        # Background group
        bg_group = QGroupBox(t("group_background"))
        bg_layout = QGridLayout(bg_group)
        bg_layout.setColumnStretch(0, 1)
        bg_layout.setColumnMinimumWidth(1, 180)

        bg_layout.addWidget(QLabel(t("label_bg_color")), 0, 0)
        self._bg_color_btn = self._make_color_btn(
            s.get("bg_color", DEFAULT_STYLE["bg_color"])
        )
        self._bg_color_btn.clicked.connect(lambda: self._pick_color(self._bg_color_btn))
        bg_layout.addWidget(self._bg_color_btn, 0, 1)

        bg_layout.addWidget(QLabel(t("label_bg_opacity")), 1, 0)
        self._bg_opacity = QSpinBox()
        self._bg_opacity.setRange(0, 100)
        self._bg_opacity.setSuffix("%")
        self._bg_opacity.setValue(round(s.get("bg_opacity", DEFAULT_STYLE["bg_opacity"]) / 255 * 100))
        self._bg_opacity.valueChanged.connect(self._on_style_value_changed)
        self._bg_opacity.valueChanged.connect(self._auto_save)
        bg_layout.addWidget(self._bg_opacity, 1, 1)

        bg_layout.addWidget(QLabel(t("label_header_color")), 2, 0)
        self._header_color_btn = self._make_color_btn(
            s.get("header_color", DEFAULT_STYLE["header_color"])
        )
        self._header_color_btn.clicked.connect(
            lambda: self._pick_color(self._header_color_btn)
        )
        bg_layout.addWidget(self._header_color_btn, 2, 1)

        bg_layout.addWidget(QLabel(t("label_header_opacity")), 3, 0)
        self._header_opacity = QSpinBox()
        self._header_opacity.setRange(0, 100)
        self._header_opacity.setSuffix("%")
        self._header_opacity.setValue(round(s.get("header_opacity", DEFAULT_STYLE["header_opacity"]) / 255 * 100))
        self._header_opacity.valueChanged.connect(self._on_style_value_changed)
        self._header_opacity.valueChanged.connect(self._auto_save)
        bg_layout.addWidget(self._header_opacity, 3, 1)

        bg_layout.addWidget(QLabel(t("label_border_radius")), 4, 0)
        self._border_radius = QSpinBox()
        self._border_radius.setRange(0, 30)
        self._border_radius.setValue(
            s.get("border_radius", DEFAULT_STYLE["border_radius"])
        )
        self._border_radius.setSuffix(" px")
        self._border_radius.valueChanged.connect(self._on_style_value_changed)
        self._border_radius.valueChanged.connect(self._auto_save)
        bg_layout.addWidget(self._border_radius, 4, 1)

        layout.addWidget(bg_group)

        # Text group
        text_group = QGroupBox(t("group_text"))
        text_layout = QGridLayout(text_group)
        text_layout.setColumnStretch(0, 1)
        text_layout.setColumnMinimumWidth(1, 180)

        text_layout.addWidget(QLabel(t("label_original_font")), 0, 0)
        self._orig_font_combo = QFontComboBox()
        self._orig_font_combo.setCurrentFont(
            QFont(s.get("original_font_family", DEFAULT_STYLE["original_font_family"]))
        )
        self._orig_font_combo.currentFontChanged.connect(self._on_style_value_changed)
        self._orig_font_combo.currentFontChanged.connect(self._auto_save)
        text_layout.addWidget(self._orig_font_combo, 0, 1)

        text_layout.addWidget(QLabel(t("label_original_font_size")), 1, 0)
        self._orig_font_size = QSpinBox()
        self._orig_font_size.setRange(6, 24)
        self._orig_font_size.setValue(
            s.get("original_font_size", DEFAULT_STYLE["original_font_size"])
        )
        self._orig_font_size.setSuffix(" pt")
        self._orig_font_size.valueChanged.connect(self._on_style_value_changed)
        self._orig_font_size.valueChanged.connect(self._auto_save)
        text_layout.addWidget(self._orig_font_size, 1, 1)

        text_layout.addWidget(QLabel(t("label_original_color")), 2, 0)
        self._orig_color_btn = self._make_color_btn(
            s.get("original_color", DEFAULT_STYLE["original_color"])
        )
        self._orig_color_btn.clicked.connect(
            lambda: self._pick_color(self._orig_color_btn)
        )
        text_layout.addWidget(self._orig_color_btn, 2, 1)

        text_layout.addWidget(QLabel(t("label_translation_font")), 3, 0)
        self._trans_font_combo = QFontComboBox()
        self._trans_font_combo.setCurrentFont(
            QFont(
                s.get(
                    "translation_font_family", DEFAULT_STYLE["translation_font_family"]
                )
            )
        )
        self._trans_font_combo.currentFontChanged.connect(self._on_style_value_changed)
        self._trans_font_combo.currentFontChanged.connect(self._auto_save)
        text_layout.addWidget(self._trans_font_combo, 3, 1)

        text_layout.addWidget(QLabel(t("label_translation_font_size")), 4, 0)
        self._trans_font_size = QSpinBox()
        self._trans_font_size.setRange(6, 24)
        self._trans_font_size.setValue(
            s.get("translation_font_size", DEFAULT_STYLE["translation_font_size"])
        )
        self._trans_font_size.setSuffix(" pt")
        self._trans_font_size.valueChanged.connect(self._on_style_value_changed)
        self._trans_font_size.valueChanged.connect(self._auto_save)
        text_layout.addWidget(self._trans_font_size, 4, 1)

        text_layout.addWidget(QLabel(t("label_translation_color")), 5, 0)
        self._trans_color_btn = self._make_color_btn(
            s.get("translation_color", DEFAULT_STYLE["translation_color"])
        )
        self._trans_color_btn.clicked.connect(
            lambda: self._pick_color(self._trans_color_btn)
        )
        text_layout.addWidget(self._trans_color_btn, 5, 1)

        text_layout.addWidget(QLabel(t("label_timestamp_color")), 6, 0)
        self._ts_color_btn = self._make_color_btn(
            s.get("timestamp_color", DEFAULT_STYLE["timestamp_color"])
        )
        self._ts_color_btn.clicked.connect(lambda: self._pick_color(self._ts_color_btn))
        text_layout.addWidget(self._ts_color_btn, 6, 1)

        layout.addWidget(text_group)

        # Window group
        win_group = QGroupBox(t("group_window"))
        win_layout = QGridLayout(win_group)
        win_layout.setColumnStretch(0, 1)
        win_layout.setColumnMinimumWidth(1, 180)
        win_layout.addWidget(QLabel(t("label_window_opacity")), 0, 0)
        self._window_opacity = QSpinBox()
        self._window_opacity.setRange(30, 100)
        self._window_opacity.setSuffix("%")
        self._window_opacity.setValue(s.get("window_opacity", DEFAULT_STYLE["window_opacity"]))
        self._window_opacity.valueChanged.connect(self._on_style_value_changed)
        self._window_opacity.valueChanged.connect(self._auto_save)
        win_layout.addWidget(self._window_opacity, 0, 1)
        win_layout.addWidget(QLabel(t("label_performance_scope")), 1, 0)
        self._performance_scope = QComboBox()
        self._performance_scope.addItem(t("performance_scope_application"), "application")
        self._performance_scope.addItem(t("performance_scope_system"), "system")
        scope_index = self._performance_scope.findData(
            s.get("performance_scope", DEFAULT_STYLE["performance_scope"])
        )
        self._performance_scope.setCurrentIndex(max(0, scope_index))
        self._performance_scope.currentIndexChanged.connect(self._auto_save)
        win_layout.addWidget(self._performance_scope, 1, 1)
        layout.addWidget(win_group)

        layout.addStretch()
        return widget

    def _make_color_btn(self, color: str) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(60, 24)
        btn.setProperty("hex_color", color)
        btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #888; border-radius: 3px;"
        )
        return btn

    def _pick_color(self, btn: QPushButton):
        from PyQt6.QtGui import QColor as _QColor

        current = _QColor(btn.property("hex_color"))
        color = QColorDialog.getColor(current, self)
        if color.isValid():
            hex_c = color.name()
            btn.setProperty("hex_color", hex_c)
            btn.setStyleSheet(
                f"background-color: {hex_c}; border: 1px solid #888; border-radius: 3px;"
            )
            self._on_style_value_changed()
            self._auto_save()

    def _collect_style(self) -> dict:
        return {
            "preset": self._preset_keys[self._style_preset.currentIndex()],
            "bg_color": self._bg_color_btn.property("hex_color"),
            "bg_opacity": round(self._bg_opacity.value() / 100 * 255),
            "header_color": self._header_color_btn.property("hex_color"),
            "header_opacity": round(self._header_opacity.value() / 100 * 255),
            "border_radius": self._border_radius.value(),
            "original_font_family": self._orig_font_combo.currentFont().family(),
            "translation_font_family": self._trans_font_combo.currentFont().family(),
            "original_font_size": self._orig_font_size.value(),
            "translation_font_size": self._trans_font_size.value(),
            "original_color": self._orig_color_btn.property("hex_color"),
            "translation_color": self._trans_color_btn.property("hex_color"),
            "timestamp_color": self._ts_color_btn.property("hex_color"),
            "window_opacity": self._window_opacity.value(),
            "performance_scope": self._performance_scope.currentData() or "application",
        }

    def _apply_style_to_controls(self, s: dict):
        """Update all style controls to match a style dict, without triggering auto-save."""
        self._bg_color_btn.setProperty("hex_color", s["bg_color"])
        self._bg_color_btn.setStyleSheet(
            f"background-color: {s['bg_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._bg_opacity.setValue(round(s["bg_opacity"] / 255 * 100))
        self._header_color_btn.setProperty("hex_color", s["header_color"])
        self._header_color_btn.setStyleSheet(
            f"background-color: {s['header_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._header_opacity.setValue(round(s["header_opacity"] / 255 * 100))
        self._border_radius.setValue(s["border_radius"])
        self._orig_font_combo.setCurrentFont(QFont(s["original_font_family"]))
        self._trans_font_combo.setCurrentFont(QFont(s["translation_font_family"]))
        self._orig_font_size.setValue(s["original_font_size"])
        self._trans_font_size.setValue(s["translation_font_size"])
        self._orig_color_btn.setProperty("hex_color", s["original_color"])
        self._orig_color_btn.setStyleSheet(
            f"background-color: {s['original_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._trans_color_btn.setProperty("hex_color", s["translation_color"])
        self._trans_color_btn.setStyleSheet(
            f"background-color: {s['translation_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._ts_color_btn.setProperty("hex_color", s["timestamp_color"])
        self._ts_color_btn.setStyleSheet(
            f"background-color: {s['timestamp_color']}; border: 1px solid #888; border-radius: 3px;"
        )
        self._window_opacity.setValue(s["window_opacity"])
        scope_index = self._performance_scope.findData(
            s.get("performance_scope", "application")
        )
        self._performance_scope.setCurrentIndex(max(0, scope_index))

    def _on_preset_changed(self, index):
        from subtitle_overlay import STYLE_PRESETS

        key = self._preset_keys[index]
        if key == "custom":
            return
        preset = STYLE_PRESETS.get(key)
        if not preset:
            return
        self._block_style_signals(True)
        self._apply_style_to_controls(preset)
        self._block_style_signals(False)
        self._auto_save()

    def _on_style_value_changed(self, *_args):
        """When any style control changes manually, switch preset to Custom."""
        custom_idx = len(self._preset_keys) - 1
        if self._style_preset.currentIndex() != custom_idx:
            self._style_preset.blockSignals(True)
            self._style_preset.setCurrentIndex(custom_idx)
            self._style_preset.blockSignals(False)
        self._auto_save()

    def _reset_style(self):
        from subtitle_overlay import DEFAULT_STYLE

        self._style_preset.blockSignals(True)
        self._style_preset.setCurrentIndex(0)  # default
        self._style_preset.blockSignals(False)
        self._block_style_signals(True)
        self._apply_style_to_controls(DEFAULT_STYLE)
        self._block_style_signals(False)
        self._auto_save()

    def _block_style_signals(self, block: bool):
        for w in (
            self._bg_opacity,
            self._header_opacity,
            self._border_radius,
            self._orig_font_combo,
            self._trans_font_combo,
            self._orig_font_size,
            self._trans_font_size,
            self._window_opacity,
            self._performance_scope,
        ):
            w.blockSignals(block)

    # ── Subtitle Tab ──

    def _create_subtitle_tab(self):
        subtitle_settings = self._current_settings.get("subtitle_mode") or {}
        self._subtitle_widget = SubtitleSettingsWidget(subtitle_settings)
        self._subtitle_widget.settings_changed.connect(self._on_subtitle_settings_changed)
        return self._subtitle_widget

    def _on_subtitle_settings_changed(self, s):
        self._current_settings["subtitle_mode"] = s
        self._auto_save()
        self.subtitle_settings_changed.emit(s)

    def update_subtitle_settings(self, s):
        self._current_settings["subtitle_mode"] = s
        self._subtitle_widget.update_settings(s)

    # ── Benchmark Tab ──

    def _create_benchmark_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel(t("label_source")))
        self._bench_lang = QComboBox()
        self._bench_lang.addItems(["ja", "en", "zh", "ko", "fr", "de"])
        self._bench_lang.setCurrentIndex(0)
        ctrl_row.addWidget(self._bench_lang)
        ctrl_row.addWidget(QLabel(t("target_label")))
        self._bench_target = QComboBox()
        self._bench_target.addItems(["zh", "en", "ja", "ko", "fr", "de", "es", "ru"])
        ctrl_row.addWidget(self._bench_target)
        ctrl_row.addStretch()
        self._bench_btn = QPushButton(t("btn_test_all"))
        self._bench_btn.clicked.connect(self._run_benchmark)
        ctrl_row.addWidget(self._bench_btn)
        layout.addLayout(ctrl_row)

        self._bench_output = QTextEdit()
        self._bench_output.setReadOnly(True)
        self._bench_output.setFont(QFont("Consolas", 9))
        self._bench_output.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border: 1px solid #444;"
        )
        layout.addWidget(self._bench_output)

        return widget

    # ── Cache Tab ──

    def _create_changelog_tab(self):
        from dialogs import _load_latest_changelog
        widget = QWidget()
        layout = QVBoxLayout(widget)
        _, html = _load_latest_changelog()
        from PyQt6.QtWidgets import QTextBrowser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        browser.setFont(QFont("Microsoft YaHei UI", 10))
        layout.addWidget(browser)
        return widget

    def _create_cache_tab(self):
        from PyQt6.QtWidgets import QCheckBox

        widget = QWidget()
        layout = QVBoxLayout(widget)
        s = self._current_settings

        # Transcript auto-save group
        ts_group = QGroupBox(t("group_transcript"))
        ts_layout = QHBoxLayout(ts_group)
        self._auto_save_transcript_cb = QCheckBox(t("label_auto_save_transcript"))
        self._auto_save_transcript_cb.setToolTip(t("auto_save_transcript_tooltip"))
        self._auto_save_transcript_cb.setChecked(s.get("auto_save_transcript", True))
        self._auto_save_transcript_cb.toggled.connect(self._auto_save)
        ts_layout.addWidget(self._auto_save_transcript_cb, 1)
        ts_open_btn = QPushButton(t("btn_open_transcripts"))
        ts_open_btn.clicked.connect(self._open_transcripts_folder)
        ts_layout.addWidget(ts_open_btn)
        layout.addWidget(ts_group)

        top_row = QHBoxLayout()
        self._cache_total = QLabel("")
        self._cache_total.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        top_row.addWidget(self._cache_total, 1)
        open_btn = QPushButton(t("btn_open_folder"))
        open_btn.clicked.connect(
            lambda: (
                MODELS_DIR.mkdir(parents=True, exist_ok=True),
                os.startfile(str(MODELS_DIR)),
            )
        )
        top_row.addWidget(open_btn)
        delete_all_btn = QPushButton(t("btn_delete_all_exit"))
        delete_all_btn.clicked.connect(self._delete_all_and_exit)
        top_row.addWidget(delete_all_btn)
        layout.addLayout(top_row)

        self._cache_list = QListWidget()
        self._cache_list.setFont(QFont("Consolas", 9))
        self._cache_list.setAlternatingRowColors(True)
        layout.addWidget(self._cache_list, 1)

        self._cache_entries = []
        self._refresh_cache()

        return widget

    def _open_transcripts_folder(self):
        from pathlib import Path
        ts_dir = Path(__file__).parent / "transcripts"
        ts_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(ts_dir))

    def _on_tab_changed(self, index):
        if index == self._cache_tab_index:
            self._refresh_cache()

    def _refresh_cache(self):
        self._cache_list.clear()
        self._cache_total.setText(t("scanning"))

        def _scan():
            entries = get_cache_entries()
            results = []
            for name, path in entries:
                size = dir_size(path)
                results.append((name, str(path), size))
            self._cache_result.emit(results)

        threading.Thread(target=_scan, daemon=True).start()

    def _on_cache_result(self, results):
        self._cache_list.clear()
        self._cache_entries = results
        total = 0
        for name, path, size in results:
            total += size
            self._cache_list.addItem(f"{name}  —  {format_size(size)}")
        if not results:
            self._cache_list.addItem(t("no_cached_models"))
        self._cache_total.setText(
            t("cache_total").format(size=format_size(total), count=len(results))
        )

    def _delete_all_and_exit(self):
        if not self._cache_entries:
            return
        import shutil

        total_size = sum(s for _, _, s in self._cache_entries)
        ret = QMessageBox.warning(
            self,
            t("dialog_delete_title"),
            t("dialog_delete_msg").format(
                count=len(self._cache_entries), size=format_size(total_size)
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        for name, path, _ in self._cache_entries:
            try:
                shutil.rmtree(path)
                log.info(f"Deleted: {path}")
            except Exception as e:
                log.error(f"Failed to delete {path}: {e}")
        QApplication.instance().quit()

    def _get_asr_lang_code(self) -> str:
        """Get the language code from the ASR language combo (stored as userData)."""
        return self._asr_lang.currentData() or "auto"

    def _on_engine_changed_whisper_vis(self, index):
        self._whisper_group.setVisible(index == 0)
        is_funasr = index == 1
        if hasattr(self, "_funasr_model_combo"):
            self._funasr_model_label.setVisible(is_funasr)
            self._funasr_model_combo.setVisible(is_funasr)
        if hasattr(self, "_whisper_pad_seconds"):
            is_whisper = index == 0
            self._whisper_pad_label.setVisible(is_whisper)
            self._whisper_pad_seconds.setVisible(is_whisper)
        if hasattr(self, "_sensevoice_pad_seconds"):
            show_funasr_pad = is_funasr and funasr_supports_padding(
                self._selected_funasr_model()
            )
            self._sensevoice_pad_label.setVisible(show_funasr_pad)
            self._sensevoice_pad_seconds.setVisible(show_funasr_pad)
        if hasattr(self, "_remote_group"):
            self._remote_group.setVisible(index == 3)
        if hasattr(self, "_qwen3_group"):
            self._qwen3_group.setVisible(index == 4)
        # Resize window to fit content after whisper group visibility change
        QTimer.singleShot(0, self._fit_height)

    def _selected_funasr_model(self) -> str:
        value = self._funasr_model_combo.currentData()
        return normalize_funasr_model_key(str(value) if value else None)

    def _on_funasr_model_changed(self):
        self._current_settings["funasr_model"] = self._selected_funasr_model()
        self._on_engine_changed_whisper_vis(self._asr_engine.currentIndex())
        self._auto_save()

    def _selected_whisper_model(self) -> str:
        value = self._whisper_size_combo.currentData()
        return str(value) if value else self._whisper_size_combo.currentText()

    def _populate_whisper_models(self, saved_value: str):
        self._whisper_size_combo.clear()
        for size in _WHISPER_SIZES:
            self._whisper_size_combo.addItem(size, size)

        local_prefix = t("whisper_local_prefix")
        for item in list_local_faster_whisper_models():
            idx = self._whisper_size_combo.count()
            self._whisper_size_combo.addItem(
                f"{local_prefix}: {item['name']}", item["path"]
            )
            self._whisper_size_combo.setItemData(
                idx, item["path"], Qt.ItemDataRole.ToolTipRole
            )

        selected = resolve_custom_whisper_model(saved_value) or saved_value
        idx = self._whisper_size_combo.findData(selected)
        if idx < 0:
            idx = self._whisper_size_combo.findText(saved_value)
        if idx < 0 and selected:
            label = f"{t('whisper_missing_local')}: {Path(str(selected)).name}"
            idx = self._whisper_size_combo.count()
            self._whisper_size_combo.addItem(label, selected)
            self._whisper_size_combo.setItemData(
                idx, str(selected), Qt.ItemDataRole.ToolTipRole
            )
        if idx >= 0:
            self._whisper_size_combo.setCurrentIndex(idx)

    def _update_whisper_size_label(self):
        from model_manager import is_asr_cached, _MODEL_SIZE_BYTES

        size = self._selected_whisper_model()
        cached = is_asr_cached("whisper", size, self._current_settings.get("hub", "ms"))
        if size not in _WHISPER_SIZES:
            if cached:
                self._whisper_status.setText(t("whisper_local_ready"))
                self._whisper_status.setStyleSheet("color: #4a4; font-size: 11px;")
            else:
                self._whisper_status.setText(t("whisper_invalid_local"))
                self._whisper_status.setStyleSheet("color: #d66; font-size: 11px;")
            self._whisper_dl_btn.setEnabled(False)
            return
        if cached:
            self._whisper_status.setText(t("whisper_already_cached"))
            self._whisper_status.setStyleSheet("color: #4a4; font-size: 11px;")
            self._whisper_dl_btn.setEnabled(False)
        else:
            est = _MODEL_SIZE_BYTES.get(f"whisper-{size}", 0)
            self._whisper_status.setText(f"~{format_size(est)}")
            self._whisper_status.setStyleSheet("color: #888; font-size: 11px;")
            self._whisper_dl_btn.setEnabled(True)

    def _on_whisper_size_changed(self):
        self._current_settings["whisper_model_size"] = (
            self._selected_whisper_model()
        )
        self._update_whisper_size_label()
        # If already cached, switch engine immediately
        from model_manager import is_asr_cached

        size = self._selected_whisper_model()
        if is_asr_cached("whisper", size, self._current_settings.get("hub", "ms")):
            self._auto_save()

    def _download_whisper(self):
        from model_manager import is_asr_cached, get_missing_models

        size = self._selected_whisper_model()
        if size not in _WHISPER_SIZES:
            return
        hub = self._current_settings.get("hub", "ms")
        if is_asr_cached("whisper", size, hub):
            return
        missing = get_missing_models("whisper", size, hub)
        missing = [m for m in missing if m["type"] != "silero-vad"]
        if not missing:
            return
        from dialogs import ModelDownloadDialog

        dlg = ModelDownloadDialog(missing, hub=hub, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self._update_whisper_size_label()
            # Switch to Whisper engine with the downloaded size
            self._auto_save()

    # ── Model Management ──

    def _refresh_model_list(self):
        self._model_list.clear()
        active = self._current_settings.get("active_model", 0)
        for i, m in enumerate(self._current_settings.get("models", [])):
            prefix = ">>> " if i == active else "    "
            proxy = m.get("proxy", "none")
            proxy_tag = f"  [proxy: {proxy}]" if proxy != "none" else ""
            text = (
                f"{prefix}{m['name']}{proxy_tag}\n     {m['api_base']}  |  {m['model']}"
            )
            item = QListWidgetItem(text)
            if i == active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._model_list.addItem(item)

    def _emit_models_list_changed(self):
        models = self._current_settings.get("models", [])
        active_idx = self._current_settings.get("active_model", 0)
        self.models_list_changed.emit(models, active_idx)

    def _add_model(self):
        dlg = ModelEditDialog(self)
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                self._current_settings.setdefault("models", []).append(data)
                self._refresh_model_list()
                _save_settings(self._current_settings)
                self._emit_models_list_changed()

    def _edit_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dlg = ModelEditDialog(self, models[row])
        if dlg.exec():
            data = dlg.get_data()
            if data["name"] and data["model"]:
                models[row] = data
                self._refresh_model_list()
                _save_settings(self._current_settings)
                self._emit_models_list_changed()
                # Re-apply if editing the active model
                active = self._current_settings.get("active_model", 0)
                if row == active:
                    self.model_changed.emit(data)

    def _dup_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models):
            return
        dup = dict(models[row])
        dup["name"] = dup["name"] + " (copy)"
        models.append(dup)
        self._refresh_model_list()
        _save_settings(self._current_settings)
        self._emit_models_list_changed()

    def _remove_model(self):
        row = self._model_list.currentRow()
        models = self._current_settings.get("models", [])
        if row < 0 or row >= len(models) or len(models) <= 1:
            return
        models.pop(row)
        active = self._current_settings.get("active_model", 0)
        if active >= len(models):
            self._current_settings["active_model"] = len(models) - 1
        self._refresh_model_list()
        self._model_list.setCurrentRow(min(row, len(models) - 1))
        _save_settings(self._current_settings)
        self._emit_models_list_changed()

    def _on_model_double_clicked(self, item):
        row = self._model_list.row(item)
        models = self._current_settings.get("models", [])
        if 0 <= row < len(models):
            self._model_list.setCurrentRow(row)
            self._edit_model()

    def _run_benchmark(self):
        models = self._current_settings.get("models", [])
        if not models:
            return

        source_lang = self._bench_lang.currentText()
        target_lang = self._bench_target.currentText()
        timeout_s = self._current_settings.get("timeout", 5)

        self._bench_btn.setEnabled(False)
        self._bench_btn.setText(t("testing"))
        self._bench_output.clear()

        from translator import DEFAULT_PROMPT, LANGUAGE_DISPLAY

        src = LANGUAGE_DISPLAY.get(source_lang, source_lang)
        tgt = LANGUAGE_DISPLAY.get(target_lang, target_lang)
        prompt = self._current_settings.get("system_prompt", DEFAULT_PROMPT)
        try:
            prompt = prompt.format(source_lang=src, target_lang=tgt)
        except (KeyError, IndexError):
            pass

        run_benchmark(
            models, source_lang, target_lang, timeout_s, prompt, self._bench_result.emit
        )

    def _on_bench_result(self, text: str):
        if text == "__DONE__":
            self._bench_btn.setEnabled(True)
            self._bench_btn.setText(t("btn_test_all"))
        else:
            self._bench_output.append(text)

    # ── Shared logic ──

    @staticmethod
    def _resolve_path(value) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = Path(__file__).parent / path
        return path.resolve()

    def _glossary_paths(self):
        output = []
        seen = set()
        for value in self._current_settings.get("glossary_paths", []):
            path = self._resolve_path(value)
            key = str(path).casefold()
            if key not in seen:
                output.append(path)
                seen.add(key)
        return output

    def _on_glossary_enabled_changed(self, enabled: bool):
        self._current_settings["glossary_enabled"] = bool(enabled)
        self._refresh_glossary_list()
        self._apply_settings()
        _save_settings(self._current_settings)

    def _on_ocr_enabled_changed(self, enabled: bool):
        self._current_settings["ocr_enabled"] = bool(enabled)
        self._apply_settings()
        _save_settings(self._current_settings)

    def _selected_glossary_path(self):
        item = self._glossary_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return Path(value) if value else None

    def _refresh_glossary_list(self, select_path=None):
        previous = select_path or self._selected_glossary_path()
        self._glossary_list.blockSignals(True)
        self._glossary_list.clear()
        total_entries = 0
        loaded_files = 0
        failed_files = 0
        selected_row = 0
        for row, path in enumerate(self._glossary_paths()):
            try:
                glossary = TermGlossary.from_file(path)
                count = len(glossary.entries)
                total_entries += count
                loaded_files += 1
                item_text = t("glossary_loaded_item").format(
                    name=path.name, count=count
                )
                ready = True
            except Exception:
                failed_files += 1
                item_text = t("glossary_failed_item").format(name=path.name)
                ready = False
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setData(_GLOSSARY_READY_ROLE, ready)
            item.setToolTip(str(path))
            self._glossary_list.addItem(item)
            if previous is not None and path == previous:
                selected_row = row
        self._glossary_list.blockSignals(False)

        has_glossaries = bool(self._glossary_list.count())
        if has_glossaries:
            self._glossary_list.setCurrentRow(selected_row)
        else:
            self._on_glossary_selection_changed(None, None)

        enabled = (
            self._glossary_enabled_cb.isChecked()
            if hasattr(self, "_glossary_enabled_cb")
            else self._current_settings.get("glossary_enabled", True)
        )
        if not enabled:
            self._glossary_status.setText(
                t("glossary_disabled_status").format(
                    files=loaded_files,
                    count=total_entries,
                    failed=failed_files,
                )
            )
        elif has_glossaries:
            self._glossary_status.setText(
                t("glossary_dynamic_status").format(
                    files=loaded_files,
                    count=total_entries,
                    limit=self._config["translation"].get(
                        "glossary_max_entries", 12
                    ),
                    failed=failed_files,
                )
            )
        else:
            self._glossary_status.setText(t("glossary_none_status"))

    def _on_glossary_selection_changed(self, current, previous=None):
        del previous
        path = self._selected_glossary_path()
        ready = bool(current and current.data(_GLOSSARY_READY_ROLE))
        path_text = str(path) if path else ""
        self._glossary_path_view.setText(path_text)
        self._glossary_path_view.setToolTip(path_text)
        self._remove_glossary_btn.setEnabled(path is not None)
        self._open_glossary_file_btn.setEnabled(
            bool(ready and path is not None and path.is_file())
        )
        self._open_glossary_folder_btn.setEnabled(
            bool(path is not None and path.parent.is_dir())
        )

    def _save_glossary_paths(self, paths, select_path=None):
        self._current_settings["glossary_paths"] = [str(path) for path in paths]
        self._refresh_glossary_list(select_path)
        self._apply_settings()
        _save_settings(self._current_settings)

    def _add_glossary(self):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            t("dialog_add_glossary"),
            str(Path(__file__).parent),
            t("glossary_file_filter"),
        )
        if not selected:
            return
        path = Path(selected).resolve()
        try:
            glossary = TermGlossary.from_file(path)
            if not glossary.entries:
                raise ValueError(t("glossary_no_entries"))
        except Exception as exc:
            QMessageBox.warning(
                self,
                "LiveTranslate",
                t("glossary_failed_status").format(error=str(exc)),
            )
            return
        paths = self._glossary_paths()
        if path not in paths:
            paths.append(path)
        self._save_glossary_paths(paths, path)

    def _remove_glossary(self):
        selected = self._selected_glossary_path()
        if selected is None:
            return
        paths = [path for path in self._glossary_paths() if path != selected]
        self._save_glossary_paths(paths)

    def _reload_glossary_list(self):
        selected = self._selected_glossary_path()
        self._refresh_glossary_list(selected)
        self._apply_settings()

    def _open_glossary_file(self):
        path = self._selected_glossary_path()
        if path is None or not path.is_file():
            QMessageBox.warning(
                self,
                "LiveTranslate",
                t("glossary_path_missing").format(path=path),
            )
            return
        try:
            os.startfile(str(path))
        except OSError as exc:
            QMessageBox.warning(self, "LiveTranslate", str(exc))

    def _open_glossary_folder(self):
        selected = self._selected_glossary_path()
        path = selected.parent if selected is not None else None
        if path is None or not path.is_dir():
            QMessageBox.warning(
                self,
                "LiveTranslate",
                t("glossary_path_missing").format(path=path),
            )
            return
        try:
            os.startfile(str(path))
        except OSError as exc:
            QMessageBox.warning(self, "LiveTranslate", str(exc))

    def _on_silence_mode_changed(self, index):
        self._silence_duration.setEnabled(index == 1)

    def _on_vad_mode_changed(self, index):
        modes = ["silero", "firered", "energy", "disabled"]
        mode = modes[index]
        self._current_settings["vad_mode"] = mode
        self._update_vad_mode_visibility(mode)

    def _update_vad_mode_visibility(self, mode: str):
        """Show only the parameter group used by the selected VAD engine."""
        if hasattr(self, "_silero_group"):
            self._silero_group.setVisible(mode == "silero")
        if hasattr(self, "_firered_group"):
            self._firered_group.setVisible(mode == "firered")
        if hasattr(self, "_energy_group"):
            self._energy_group.setVisible(mode == "energy")
        QTimer.singleShot(0, self._fit_height)

    def _on_threshold_changed(self, value):
        val = value / 100.0
        self._current_settings["vad_threshold"] = val
        self._vad_threshold_label.setText(f"{value}%")
        if not self._vad_threshold_slider.isSliderDown():
            self._auto_save()

    def _on_firered_threshold_changed(self, value):
        self._current_settings["firered_threshold"] = value / 100.0
        self._firered_threshold_label.setText(f"{value}%")
        if not self._firered_threshold_slider.isSliderDown():
            self._auto_save()

    def _on_energy_changed(self, value):
        val = value / 1000.0
        self._current_settings["energy_threshold"] = val
        self._energy_label.setText(f"{value}\u2030")
        if not self._energy_slider.isSliderDown():
            self._auto_save()

    def _on_timing_changed(self):
        self._current_settings["min_speech_duration"] = round(self._min_speech.value(), 2)
        self._current_settings["max_speech_duration"] = round(self._max_speech.value(), 2)
        self._current_settings["silence_mode"] = (
            "auto" if self._silence_mode.currentIndex() == 0 else "fixed"
        )
        self._current_settings["silence_duration"] = round(self._silence_duration.value(), 2)
        self._current_settings["incremental_asr"] = self._incremental_asr_cb.isChecked()
        self._current_settings["interim_interval"] = round(self._interim_interval_spin.value(), 2)
        if hasattr(self, "_firered_model_path"):
            self._current_settings["firered_vad_model_path"] = (
                self._firered_model_path.text().strip()
            )
            self._current_settings["firered_threshold"] = (
                self._firered_threshold_slider.value() / 100.0
            )
            self._current_settings["firered_smooth_window"] = (
                self._firered_smooth_window.value()
            )
            self._current_settings["firered_pad_start_frames"] = (
                self._firered_pad_start_frames.value()
            )
            self._current_settings["firered_min_speech_frames"] = (
                self._firered_min_speech_frames.value()
            )
            self._current_settings["firered_min_silence_frames"] = (
                self._firered_min_silence_frames.value()
            )
        if hasattr(self, "_vad_pre_roll_ms"):
            self._current_settings["vad_pre_roll_ms"] = (
                self._vad_pre_roll_ms.value()
            )
            self._current_settings["vad_adaptive_silence_min"] = round(
                self._vad_adaptive_min.value(), 2
            )
            self._current_settings["vad_adaptive_silence_max"] = round(
                self._vad_adaptive_max.value(), 2
            )
            self._current_settings["vad_split_tail_guard"] = round(
                self._vad_split_tail_guard.value(), 2
            )
            self._current_settings["vad_progressive_split"] = (
                self._vad_progressive_split.isChecked()
            )

    def _on_ui_lang_changed(self, index):
        lang = "en" if index == 0 else "zh"
        self._current_settings["ui_lang"] = lang
        _save_settings(self._current_settings)
        from i18n import set_lang

        set_lang(lang)
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "LiveTranslate",
            "Language changed. Please restart the application.\n"
            "语言已更改，请重启应用程序。",
        )

    def _auto_save(self):
        self._save_timer.start()

    def _do_auto_save(self):
        self._apply_settings()
        _save_settings(self._current_settings)

    def _on_prompt_preset_changed(self, index):
        from translator import DEFAULT_PROMPT, PROMPT_PRESETS
        key = self._prompt_preset.itemData(index)
        if key == "custom":
            return
        prompt = PROMPT_PRESETS.get(key, DEFAULT_PROMPT)
        self._prompt_edit.setPlainText(prompt)
        self._apply_prompt()

    def _apply_prompt(self):
        text = self._prompt_edit.toPlainText().strip()
        if text:
            self._current_settings["system_prompt"] = text
            active = self.get_active_model()
            if active:
                self.model_changed.emit(active)
            _save_settings(self._current_settings)
            log.info("System prompt updated")
            # Update preset combo to reflect current state
            from translator import PROMPT_PRESETS
            self._prompt_preset.blockSignals(True)
            matched = 4  # custom
            for i, key in enumerate(["daily", "esports", "anime", "webid"]):
                if text.strip() == PROMPT_PRESETS[key].strip():
                    matched = i
                    break
            self._prompt_preset.setCurrentIndex(matched)
            self._prompt_preset.blockSignals(False)

    def _apply_settings(self):
        self._current_settings["asr_language"] = self._get_asr_lang_code()
        engine_map = {
            0: "whisper",
            1: "funasr",
            2: "anime-whisper",
            3: "remote-whisper",
            4: "qwen3-asr",
        }
        self._current_settings["asr_engine"] = engine_map.get(
            self._asr_engine.currentIndex(), "whisper"
        )
        self._current_settings["funasr_model"] = self._selected_funasr_model()
        if hasattr(self, "_remote_url_edit"):
            url = self._remote_url_edit.text().strip()
            if url:
                self._current_settings["remote_asr_url"] = url
        if hasattr(self, "_qwen3_model_path"):
            for key, widget in (
                ("qwen3_model_path", self._qwen3_model_path),
                ("qwen3_python", self._qwen3_python),
                ("qwen3_project", self._qwen3_project),
            ):
                value = widget.text().strip()
                if value:
                    self._current_settings[key] = value
            self._current_settings["qwen3_context_turns"] = (
                self._qwen3_context_turns.value()
            )
            self._current_settings["qwen3_max_new_tokens"] = (
                self._qwen3_max_new_tokens.value()
            )
            self._current_settings["qwen3_hotwords"] = (
                self._qwen3_hotwords.text().strip()
            )
            self._current_settings["qwen3_refine_enabled"] = (
                self._qwen3_refine_enabled.isChecked()
            )
        self._current_settings["whisper_model_size"] = (
            self._selected_whisper_model()
        )
        dev_text = self._asr_device.currentText()
        self._current_settings["asr_device"] = dev_text.split(" (")[0]
        audio_idx = self._audio_device.currentIndex()
        if audio_idx == 0:
            self._current_settings["audio_device"] = "__disabled__"
        elif audio_idx == 1:
            self._current_settings["audio_device"] = None
        else:
            self._current_settings["audio_device"] = self._audio_device.currentText()
        mic_idx = self._mic_device.currentIndex()
        if mic_idx == 0:
            self._current_settings["mic_device"] = None
        elif mic_idx == 1:
            self._current_settings["mic_device"] = "__default__"
        else:
            self._current_settings["mic_device"] = self._mic_device.currentText()
        self._current_settings["hub"] = (
            "ms" if self._hub_combo.currentIndex() == 0 else "hf"
        )
        self._current_settings["sensevoice_pad_seconds"] = round(
            self._sensevoice_pad_seconds.value(), 2
        )
        self._current_settings["whisper_pad_seconds"] = round(
            self._whisper_pad_seconds.value(), 2
        )
        prompt_text = self._prompt_edit.toPlainText().strip()
        if prompt_text:
            self._current_settings["system_prompt"] = prompt_text
        if hasattr(self, "_glossary_enabled_cb"):
            self._current_settings["glossary_enabled"] = (
                self._glossary_enabled_cb.isChecked()
            )
        if hasattr(self, "_ocr_enabled_cb"):
            self._current_settings["ocr_enabled"] = self._ocr_enabled_cb.isChecked()
            for key, widget in (
                ("ocr_python", self._ocr_python_edit),
                ("ocr_model_path", self._ocr_model_path_edit),
                ("ocr_cache_dir", self._ocr_cache_dir_edit),
            ):
                self._current_settings[key] = widget.text().strip()
            self._current_settings["ocr_device"] = (
                self._ocr_device_combo.currentData() or "cpu"
            )
        self._current_settings["timeout"] = self._timeout_spin.value()
        if hasattr(self, "_incremental_asr_cb"):
            self._on_timing_changed()
        if hasattr(self, "_auto_save_transcript_cb"):
            self._current_settings["auto_save_transcript"] = (
                self._auto_save_transcript_cb.isChecked()
            )
        if hasattr(self, "_style_preset"):
            self._current_settings["style"] = self._collect_style()
        safe = {
            k: v
            for k, v in self._current_settings.items()
            if k not in ("models", "system_prompt")
        }
        log.info(f"Settings applied: {safe}")
        self.settings_changed.emit(dict(self._current_settings))

    def get_settings(self):
        return dict(self._current_settings)

    def showEvent(self, event):
        super().showEvent(event)
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def get_active_model(self) -> dict | None:
        models = self._current_settings.get("models", [])
        idx = self._current_settings.get("active_model", 0)
        if 0 <= idx < len(models):
            return models[idx]
        return None

    def has_saved_settings(self) -> bool:
        return SETTINGS_FILE.exists()
