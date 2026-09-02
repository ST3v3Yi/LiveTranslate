import ctypes
import os
import subprocess
import time
from ctypes import wintypes

import psutil
from i18n import t, LANGUAGES
from PyQt6.QtCore import QPoint, QPropertyAnimation, QEasingCurve, QSize, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCursor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x20

DEFAULT_STYLE = {
    "preset": "default",
    "bg_color": "#000000",
    "bg_opacity": 240,
    "header_color": "#1a1a2e",
    "header_opacity": 230,
    "border_radius": 8,
    "original_font_family": "Microsoft YaHei",
    "translation_font_family": "Microsoft YaHei",
    "original_font_size": 11,
    "translation_font_size": 14,
    "original_color": "#cccccc",
    "translation_color": "#ffffff",
    "timestamp_color": "#888899",
    "window_opacity": 95,
    "performance_scope": "application",
}

_BASE = DEFAULT_STYLE

STYLE_PRESETS = {
    "default": dict(_BASE),
    "transparent": {
        **_BASE,
        "preset": "transparent",
        "bg_opacity": 120,
        "header_opacity": 120,
        "window_opacity": 70,
    },
    "compact": {
        **_BASE,
        "preset": "compact",
        "original_font_size": 9,
        "translation_font_size": 11,
    },
    "light": {
        **_BASE,
        "preset": "light",
        "bg_color": "#e8e8f0",
        "bg_opacity": 230,
        "header_color": "#c8c8d8",
        "header_opacity": 220,
        "original_color": "#333333",
        "translation_color": "#111111",
        "timestamp_color": "#666688",
    },
    "dracula": {
        **_BASE,
        "preset": "dracula",
        "bg_color": "#282a36",
        "bg_opacity": 235,
        "header_color": "#44475a",
        "header_opacity": 230,
        "original_color": "#f8f8f2",
        "translation_color": "#f8f8f2",
        "timestamp_color": "#6272a4",
    },
    "nord": {
        **_BASE,
        "preset": "nord",
        "bg_color": "#2e3440",
        "bg_opacity": 235,
        "header_color": "#3b4252",
        "header_opacity": 230,
        "original_color": "#d8dee9",
        "translation_color": "#eceff4",
        "timestamp_color": "#4c566a",
    },
    "monokai": {
        **_BASE,
        "preset": "monokai",
        "bg_color": "#272822",
        "bg_opacity": 235,
        "header_color": "#3e3d32",
        "header_opacity": 230,
        "original_color": "#f8f8f2",
        "translation_color": "#f8f8f2",
        "timestamp_color": "#75715e",
    },
    "solarized": {
        **_BASE,
        "preset": "solarized",
        "bg_color": "#002b36",
        "bg_opacity": 235,
        "header_color": "#073642",
        "header_opacity": 230,
        "original_color": "#839496",
        "translation_color": "#eee8d5",
        "timestamp_color": "#586e75",
    },
    "gruvbox": {
        **_BASE,
        "preset": "gruvbox",
        "bg_color": "#282828",
        "bg_opacity": 235,
        "header_color": "#3c3836",
        "header_opacity": 230,
        "original_color": "#ebdbb2",
        "translation_color": "#fbf1c7",
        "timestamp_color": "#928374",
    },
    "tokyo_night": {
        **_BASE,
        "preset": "tokyo_night",
        "bg_color": "#1a1b26",
        "bg_opacity": 235,
        "header_color": "#24283b",
        "header_opacity": 230,
        "original_color": "#a9b1d6",
        "translation_color": "#c0caf5",
        "timestamp_color": "#565f89",
    },
    "catppuccin": {
        **_BASE,
        "preset": "catppuccin",
        "bg_color": "#1e1e2e",
        "bg_opacity": 235,
        "header_color": "#313244",
        "header_opacity": 230,
        "original_color": "#cdd6f4",
        "translation_color": "#cdd6f4",
        "timestamp_color": "#6c7086",
    },
    "one_dark": {
        **_BASE,
        "preset": "one_dark",
        "bg_color": "#282c34",
        "bg_opacity": 235,
        "header_color": "#3e4452",
        "header_opacity": 230,
        "original_color": "#abb2bf",
        "translation_color": "#e5c07b",
        "timestamp_color": "#636d83",
    },
    "everforest": {
        **_BASE,
        "preset": "everforest",
        "bg_color": "#2d353b",
        "bg_opacity": 235,
        "header_color": "#343f44",
        "header_opacity": 230,
        "original_color": "#d3c6aa",
        "translation_color": "#d3c6aa",
        "timestamp_color": "#859289",
    },
    "kanagawa": {
        **_BASE,
        "preset": "kanagawa",
        "bg_color": "#1f1f28",
        "bg_opacity": 235,
        "header_color": "#2a2a37",
        "header_opacity": 230,
        "original_color": "#dcd7ba",
        "translation_color": "#dcd7ba",
        "timestamp_color": "#54546d",
    },
}


def _hex_to_rgba(hex_color: str, opacity: int) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


class ChatMessage(QWidget):
    """Single chat message widget with original + async translation."""

    _current_style = DEFAULT_STYLE
    _compact_mode = False

    def __init__(
        self,
        msg_id: int,
        timestamp: str,
        original: str,
        source_lang: str,
        asr_ms: float,
        parent=None,
    ):
        super().__init__(parent)
        self.msg_id = msg_id
        self._original = original
        self._translated = ""
        self._timestamp = timestamp
        self._source_lang = source_lang
        self._asr_ms = asr_ms
        self._translate_ms = 0.0
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(2)

        s = self._current_style
        self._header_label = QLabel(self._build_header_html(s))
        self._header_label.setFont(
            QFont(s["original_font_family"], s["original_font_size"])
        )
        self._header_label.setTextFormat(Qt.TextFormat.RichText)
        self._header_label.setWordWrap(True)
        self._header_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._header_label)

        self._trans_label = QLabel(
            f'<span style="color:#999; font-style:italic;">{t("translating")}</span>'
        )
        self._trans_label.setFont(
            QFont(s["translation_font_family"], s["translation_font_size"])
        )
        self._trans_label.setTextFormat(Qt.TextFormat.RichText)
        self._trans_label.setWordWrap(True)
        self._trans_label.setStyleSheet("background: transparent;")
        self._layout.addWidget(self._trans_label)

    def _build_header_html(self, s):
        if self._compact_mode:
            return (
                f'<span style="color:#6cf;">[{self._source_lang}]</span> '
                f'<span style="color:{s["original_color"]};">{_escape(self._original)}</span>'
            )
        return (
            f'<span style="color:{s["timestamp_color"]};">[{self._timestamp}]</span> '
            f'<span style="color:#6cf;">[{self._source_lang}]</span> '
            f'<span style="color:{s["original_color"]};">{_escape(self._original)}</span> '
            f'<span style="color:#8b8; font-size:9pt;">ASR {self._asr_ms:.0f}ms</span>'
        )

    def update_streaming(self, partial_text: str):
        """Update translation label with partial streaming text (throttled)."""
        self._pending_streaming = partial_text
        if not hasattr(self, "_streaming_timer"):
            self._streaming_timer = QTimer()
            self._streaming_timer.setSingleShot(True)
            self._streaming_timer.setInterval(50)
            self._streaming_timer.timeout.connect(self._flush_streaming)
        if not self._streaming_timer.isActive():
            self._flush_streaming()
            self._streaming_timer.start()

    def _flush_streaming(self):
        text = getattr(self, "_pending_streaming", None)
        if text is None:
            return
        self._pending_streaming = None
        s = self._current_style
        self._trans_label.setText(
            f'<span style="color:{s["translation_color"]};">&gt; {_escape(text)}</span>'
        )

    def set_translation(self, translated: str, translate_ms: float):
        self._translated = translated or ""
        self._translate_ms = translate_ms
        # Stop streaming throttle if active
        if hasattr(self, "_streaming_timer"):
            self._streaming_timer.stop()
            self._pending_streaming = None
        s = self._current_style
        if translated:
            if self._compact_mode:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(translated)}</span>'
                )
            else:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(translated)}</span> '
                    f'<span style="color:#db8; font-size:9pt;">TL {translate_ms:.0f}ms</span>'
                )
        else:
            self._trans_label.setText(
                f'<span style="color:#aaa; font-style:italic;">&gt; {t("same_language")}</span>'
            )

    def apply_style(self, s: dict):
        self._header_label.setText(self._build_header_html(s))
        self._header_label.setFont(
            QFont(s["original_font_family"], s["original_font_size"])
        )
        self._trans_label.setFont(
            QFont(s["translation_font_family"], s["translation_font_size"])
        )
        if self._translated:
            if self._compact_mode:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(self._translated)}</span>'
                )
            else:
                self._trans_label.setText(
                    f'<span style="color:{s["translation_color"]};">&gt; {_escape(self._translated)}</span> '
                    f'<span style="color:#db8; font-size:9pt;">TL {self._translate_ms:.0f}ms</span>'
                )

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2a2a3a; color: #ddd; border: 1px solid #555; }
            QMenu::item:selected { background: #444; }
            QMenu::separator { height: 1px; background: #555; margin: 4px 0; }
        """)
        copy_orig = menu.addAction(t("copy_original"))
        copy_trans = menu.addAction(t("copy_translation"))
        copy_all = menu.addAction(t("copy_all"))
        menu.addSeparator()
        export_menu = menu.addMenu(t("export_menu"))
        export_orig = export_menu.addAction(t("export_original"))
        export_trans = export_menu.addAction(t("export_translation"))
        export_both = export_menu.addAction(t("export_all"))
        menu.addSeparator()
        clear_list = menu.addAction(t("clear_list"))
        action = menu.exec(event.globalPos())
        if action == copy_orig:
            QApplication.clipboard().setText(self._original)
        elif action == copy_trans:
            QApplication.clipboard().setText(self._translated)
        elif action == copy_all:
            QApplication.clipboard().setText(f"{self._original}\n{self._translated}")
        elif action == clear_list:
            overlay = self.window()
            if hasattr(overlay, '_on_clear'):
                overlay._on_clear()
        elif action in (export_orig, export_trans, export_both):
            mode = {export_orig: "original", export_trans: "translation", export_both: "both"}[action]
            overlay = self.window()
            if hasattr(overlay, "export_messages"):
                overlay.export_messages(mode, parent=self)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_BTN_CSS = """
    QPushButton {
        background: rgba(255,255,255,20);
        border: 1px solid rgba(255,255,255,40);
        border-radius: 3px;
        color: #aaa;
        font-size: 11px;
        padding: 0 6px;
    }
    QPushButton:hover {
        background: rgba(255,255,255,40);
        color: #ddd;
    }
"""

_BAR_CSS_TPL = """
    QProgressBar {{
        background: rgba(255,255,255,15);
        border: 1px solid rgba(255,255,255,30);
        border-radius: 3px;
        text-align: center;
        font-size: 8pt;
        color: #aaa;
    }}
    QProgressBar::chunk {{
        background: {color};
        border-radius: 2px;
    }}
"""


class MonitorBar(QWidget):
    """Compact system monitor displayed in the overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # MIC bar (hidden when mic is disabled)
        self._mic_lbl = QLabel("MIC")
        self._mic_lbl.setFixedWidth(26)
        self._mic_lbl.setFont(QFont("Consolas", 8))
        self._mic_lbl.setStyleSheet("color: #888; background: transparent;")
        self._mic_lbl.setVisible(False)
        row1.addWidget(self._mic_lbl)

        self._mic_bar = QProgressBar()
        self._mic_bar.setRange(0, 100)
        self._mic_bar.setFixedHeight(14)
        self._mic_bar.setTextVisible(True)
        self._mic_bar.setFormat("%v%")
        self._mic_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#c586c0"))
        self._mic_bar.setVisible(False)
        row1.addWidget(self._mic_bar)

        rms_lbl = QLabel("RMS:")
        rms_lbl.setFixedWidth(26)
        rms_lbl.setFont(QFont("Consolas", 8))
        rms_lbl.setStyleSheet("color: #888; background: transparent;")
        row1.addWidget(rms_lbl)

        self._rms_bar = QProgressBar()
        self._rms_bar.setRange(0, 100)
        self._rms_bar.setFixedHeight(14)
        self._rms_bar.setTextVisible(True)
        self._rms_bar.setFormat("%v%")
        self._rms_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#4ec9b0"))
        row1.addWidget(self._rms_bar)

        vad_lbl = QLabel("VAD:")
        vad_lbl.setFixedWidth(26)
        vad_lbl.setFont(QFont("Consolas", 8))
        vad_lbl.setStyleSheet("color: #888; background: transparent;")
        row1.addWidget(vad_lbl)

        self._vad_bar = QProgressBar()
        self._vad_bar.setRange(0, 100)
        self._vad_bar.setFixedHeight(14)
        self._vad_bar.setTextVisible(True)
        self._vad_bar.setFormat("%v%")
        self._vad_bar.setStyleSheet(_BAR_CSS_TPL.format(color="#dcdcaa"))
        row1.addWidget(self._vad_bar)

        layout.addLayout(row1)

        self._stats_label = QLabel()
        self._stats_label.setFont(QFont("Consolas", 8))
        self._stats_label.setStyleSheet("color: #888; background: transparent;")
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        self._proc = psutil.Process(os.getpid())
        self._known_processes = {}
        self._prime_process_cpu([self._proc])
        psutil.cpu_percent(interval=None)
        self._performance_scope = "application"
        self._cpu = 0
        self._ram_commit_mb = 0.0
        self._ram_peak_mb = 0.0
        self._ram_total_mb = 0.0
        self._gpu_text = "N/A"
        self._gpu_tooltip = "GPU memory is unavailable."
        self._last_gpu_sample_at = 0.0
        self._asr_device = ""
        self._asr_count = 0
        self._tl_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0

        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._update_system)
        self._sys_timer.start(1000)
        self._update_system()
        self._refresh_stats()

    def update_audio(self, rms: float, vad: float, mic_rms=None):
        self._rms_bar.setValue(min(100, int(rms * 500)))
        self._vad_bar.setValue(min(100, int(vad * 100)))
        mic_active = mic_rms is not None
        if self._mic_lbl.isVisible() != mic_active:
            self._mic_lbl.setVisible(mic_active)
            self._mic_bar.setVisible(mic_active)
        if mic_active:
            self._mic_bar.setValue(min(100, int(mic_rms * 500)))

    def update_asr_device(self, device: str):
        self._asr_device = device
        self._refresh_stats()

    def set_performance_scope(self, scope: str):
        scope = scope if scope in ("application", "system") else "application"
        if scope == self._performance_scope:
            return
        self._performance_scope = scope
        self._last_gpu_sample_at = 0.0
        psutil.cpu_percent(interval=None)
        self._update_system()

    def update_pipeline_stats(
        self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0
    ):
        self._asr_count = asr_count
        self._tl_count = tl_count
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost = cost
        self._refresh_stats()

    @staticmethod
    def _format_memory(memory_mb: float) -> str:
        if memory_mb >= 1024:
            return f"{memory_mb / 1024:.1f}GB"
        return f"{memory_mb:.0f}MB"

    def _live_process_tree(self):
        """Return this UI process and all live model-worker descendants."""
        try:
            processes = [self._proc, *self._proc.children(recursive=True)]
        except (psutil.Error, OSError):
            processes = [self._proc]

        unique = []
        seen = set()
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            unique.append(process)
        self._prime_process_cpu(unique)
        return unique

    def _prime_process_cpu(self, processes):
        for process in processes:
            if process.pid in self._known_processes:
                continue
            try:
                process.cpu_percent(interval=None)
                self._known_processes[process.pid] = process
            except (psutil.Error, OSError):
                continue

    def _read_gpu_memory(self, pids: set[int], system_scope=False):
        """Read app VRAM when the driver exposes it; otherwise use card total."""
        now = time.monotonic()
        if now - self._last_gpu_sample_at < 2.0:
            return
        self._last_gpu_sample_at = now

        app_mb = self._windows_gpu_process_memory_mb(None if system_scope else pids)
        if app_mb is not None:
            self._gpu_text = self._format_memory(app_mb)
            self._gpu_tooltip = (
                "Dedicated VRAM used by all GPU processes."
                if system_scope
                else "Dedicated VRAM used by LiveTranslate and its model workers."
            )
            return

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            app_mb = 0.0
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 2:
                        continue
                    try:
                        if system_scope or int(parts[0]) in pids:
                            app_mb += float(parts[1])
                    except ValueError:
                        continue
            if app_mb > 0:
                self._gpu_text = self._format_memory(app_mb)
                self._gpu_tooltip = "VRAM used by LiveTranslate and its model workers."
                return
        except (OSError, subprocess.SubprocessError):
            pass

        # WDDM drivers often hide per-process VRAM from nvidia-smi. PyTorch can
        # still report reliable card-wide used/free memory from the same driver.
        try:
            import torch

            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                used_mb = (total_bytes - free_bytes) / 1024 / 1024
                total_mb = total_bytes / 1024 / 1024
                self._gpu_text = (
                    f"{self._format_memory(used_mb)}/{self._format_memory(total_mb)}"
                )
                self._gpu_tooltip = (
                    "Total VRAM used on this GPU. The Windows graphics driver does "
                    "not expose the requested process breakdown."
                )
                return
        except Exception:
            pass
        self._gpu_text = "N/A"
        self._gpu_tooltip = "GPU memory is unavailable."

    @staticmethod
    def _windows_gpu_process_memory_mb(pids: set[int] | None):
        """Use Task Manager's GPU Process Memory counter on Windows."""
        if os.name != "nt":
            return None
        try:
            pdh = ctypes.WinDLL("pdh")
            query = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return None
            try:
                wildcard = r"\GPU Process Memory(*)\Dedicated Usage"
                length = wintypes.DWORD(0)
                pdh.PdhExpandWildCardPathW(
                    None, wildcard, None, ctypes.byref(length), 0
                )
                if not length.value:
                    return None
                paths_buffer = ctypes.create_unicode_buffer(length.value)
                if pdh.PdhExpandWildCardPathW(
                    None, wildcard, paths_buffer, ctypes.byref(length), 0
                ) != 0:
                    return None

                counters = []
                for path in paths_buffer[:].split("\0"):
                    if not path or (
                        pids is not None
                        and not any(f"pid_{pid}_" in path for pid in pids)
                    ):
                        continue
                    counter = ctypes.c_void_p()
                    if pdh.PdhAddEnglishCounterW(
                        query, path, 0, ctypes.byref(counter)
                    ) == 0:
                        counters.append(counter)
                if not counters or pdh.PdhCollectQueryData(query) != 0:
                    return None

                class _PdhValue(ctypes.Union):
                    _fields_ = [("large_value", ctypes.c_longlong)]

                class _PdhFormattedValue(ctypes.Structure):
                    _fields_ = [
                        ("status", wintypes.DWORD),
                        ("value", _PdhValue),
                    ]

                used_bytes = 0
                for counter in counters:
                    value = _PdhFormattedValue()
                    if pdh.PdhGetFormattedCounterValue(
                        counter, 0x00000400, None, ctypes.byref(value)
                    ) == 0 and value.status == 0:
                        used_bytes += max(0, value.value.large_value)
                return used_bytes / 1024 / 1024
            finally:
                pdh.PdhCloseQuery(query)
        except (AttributeError, OSError):
            return None

    def _update_system(self):
        if self._performance_scope == "system":
            memory = psutil.virtual_memory()
            self._cpu = psutil.cpu_percent(interval=None)
            self._ram_commit_mb = memory.used / 1024 / 1024
            self._ram_total_mb = memory.total / 1024 / 1024
            self._ram_peak_mb = 0.0
            self._read_gpu_memory(set(), system_scope=True)
            self._refresh_stats()
            return

        processes = self._live_process_tree()
        cpu_percent = 0.0
        commit_mb = 0.0
        peak_commit_mb = 0.0
        for process in processes:
            try:
                cpu_percent += process.cpu_percent(interval=None)
                memory = process.memory_info()
                # Windows Task Manager reports a model process primarily by its
                # committed private pages (pagefile), not its smaller RSS set.
                pagefile = getattr(memory, "pagefile", memory.vms)
                peak_pagefile = getattr(memory, "peak_pagefile", pagefile)
                commit_mb += pagefile / 1024 / 1024
                peak_commit_mb += peak_pagefile / 1024 / 1024
            except (psutil.Error, OSError):
                continue
        logical_cpus = psutil.cpu_count(logical=True) or 1
        self._cpu = max(0.0, cpu_percent / logical_cpus)
        self._ram_commit_mb = commit_mb
        self._ram_peak_mb = peak_commit_mb
        self._ram_total_mb = 0.0
        self._read_gpu_memory({process.pid for process in processes})
        self._refresh_stats()

    def _refresh_stats(self):
        total = self._prompt_tokens + self._completion_tokens
        tokens_str = f"{total / 1000:.1f}k" if total >= 1000 else str(total)
        dev_str = ""
        if self._asr_device:
            dev_color = "#4ec9b0" if "cuda" in self._asr_device.lower() else "#dcdcaa"
            dev_str = (
                f'<span style="color:{dev_color};">{self._asr_device}</span> '
                f'<span style="color:#555;">|</span> '
            )
        cost_str = ""
        if self._cost > 0:
            from i18n import get_lang
            symbol = "¥" if get_lang() == "zh" else "$"
            cost_str = f' <span style="color:#fa5;">{symbol}{self._cost:.4f}</span>'
        if self._performance_scope == "system":
            ram_text = (
                f'{self._format_memory(self._ram_commit_mb)}/'
                f'{self._format_memory(self._ram_total_mb)}'
            )
            ram_detail = ""
        else:
            ram_text = self._format_memory(self._ram_commit_mb)
            ram_detail = (
                f' <span style="color:#666;">(P{self._format_memory(self._ram_peak_mb)})</span>'
            )
        self._stats_label.setText(
            f"{dev_str}"
            f'<span style="color:#6cf;">CPU</span> {self._cpu:.1f}% '
            f'<span style="color:#6cf;">RAM</span> {ram_text}{ram_detail} '
            f'<span style="color:#6cf;">GPU</span> {self._gpu_text} '
            f'<span style="color:#555;">|</span> '
            f'<span style="color:#8b8;">ASR</span> {self._asr_count} '
            f'<span style="color:#db8;">TL</span> {self._tl_count} '
            f'<span style="color:#c9c;">Tok</span> {tokens_str} '
            f'<span style="color:#666;">({self._prompt_tokens}\u2191{self._completion_tokens}\u2193)</span>'
            f'{cost_str}'
        )
        scope_tip = (
            "CPU, RAM, and GPU report the entire computer.\n"
            if self._performance_scope == "system"
            else "CPU and RAM include the LiveTranslate interface and all local model workers.\n"
        )
        ram_tip = (
            "RAM is currently used / total physical memory.\n"
            if self._performance_scope == "system"
            else "RAM is current committed memory; P is the peak since each worker started.\n"
        )
        self._stats_label.setToolTip(scope_tip + ram_tip + self._gpu_tooltip)


class _DragArea(QWidget):
    """Small draggable area (title + grip)."""

    drag_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self._drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        if self._drag_pos:
            self._drag_pos = None
            self.drag_finished.emit()


_COMBO_CSS = """
    QComboBox {
        background: rgba(255,255,255,20);
        border: 1px solid rgba(255,255,255,40);
        border-radius: 3px;
        color: #aaa;
        font-size: 11px;
        padding: 0 4px;
    }
    QComboBox:hover { background: rgba(255,255,255,40); color: #ddd; }
    QComboBox::drop-down { border: none; width: 14px; }
    QComboBox::down-arrow { image: none; border: none; }
    QComboBox QAbstractItemView {
        background: #2a2a3a; color: #ccc; selection-background-color: #444;
    }
"""

_CHECK_CSS = (
    "QCheckBox { color: #888; background: transparent; spacing: 3px; }"
    "QCheckBox::indicator { width: 12px; height: 12px; }"
)


class DragHandle(QWidget):
    """Top bar: row1=title+buttons, row2=checkboxes+combos."""

    settings_clicked = pyqtSignal()
    subtitle_clicked = pyqtSignal()
    screenshot_clicked = pyqtSignal()
    click_through_toggled = pyqtSignal(bool)
    topmost_toggled = pyqtSignal(bool)
    auto_scroll_toggled = pyqtSignal(bool)
    taskbar_toggled = pyqtSignal(bool)
    target_language_changed = pyqtSignal(str)
    source_language_changed = pyqtSignal(str)
    model_changed = pyqtSignal(int)
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    hide_clicked = pyqtSignal()
    quit_clicked = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    position_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "full"
        self.setFixedHeight(62)
        self.setStyleSheet("background: rgba(60, 60, 80, 200); border-radius: 4px;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(2)

        # Row 1: drag title + action buttons
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(3)

        drag = _DragArea()
        drag.drag_finished.connect(self.position_changed)
        drag.setStyleSheet("background: transparent;")
        drag_layout = QHBoxLayout(drag)
        drag_layout.setContentsMargins(0, 0, 4, 0)
        drag_layout.setSpacing(6)

        title = QLabel("\u2630 LiveTranslate")
        title.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        title.setStyleSheet("color: #aaa; background: transparent;")
        drag_layout.addWidget(title)
        drag_layout.addStretch()
        row1.addWidget(drag, 1)

        def _btn(text, tip=None):
            b = QPushButton(text)
            b.setFixedHeight(20)
            b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            b.setFont(QFont("Consolas", 8))
            b.setStyleSheet(_BTN_CSS)
            if tip:
                b.setToolTip(tip)
            return b

        hide_btn = _btn(t("hide"))
        hide_btn.clicked.connect(self.hide_clicked.emit)
        row1.addWidget(hide_btn)

        self._subtitle_btn = _btn(t("subtitle"))
        self._subtitle_btn.clicked.connect(self.subtitle_clicked.emit)
        row1.addWidget(self._subtitle_btn)

        self._screenshot_btn = _btn(t("screenshot"), t("screenshot_hint"))
        self._screenshot_btn.clicked.connect(self.screenshot_clicked.emit)
        row1.addWidget(self._screenshot_btn)

        self._running = False
        self._start_stop_btn = _btn(t("paused"))
        self._start_stop_btn.clicked.connect(self._on_start_stop)
        row1.addWidget(self._start_stop_btn)

        self._clear_btn = _btn(t("clear"))
        self._clear_btn.clicked.connect(self.clear_clicked.emit)
        row1.addWidget(self._clear_btn)

        # Mode toggle button
        self._mode_btn = _btn(t("mode_full"))
        self._mode_btn.clicked.connect(self._toggle_mode)
        row1.addWidget(self._mode_btn)

        settings_btn = _btn(t("settings"))
        settings_btn.clicked.connect(self.settings_clicked.emit)
        row1.addWidget(settings_btn)

        quit_btn = _btn(t("quit"))
        quit_btn.setStyleSheet(
            _BTN_CSS.replace("rgba(255,255,255,20)", "rgba(200,60,60,40)").replace(
                "rgba(255,255,255,40)", "rgba(200,60,60,80)"
            )
        )
        quit_btn.clicked.connect(self.quit_clicked.emit)
        row1.addWidget(quit_btn)

        outer.addLayout(row1)

        # Row 2 area: checkboxes (row 2a) + model/lang combos (row 2b)
        self._row2_widget = QWidget()
        self._row2_widget.setStyleSheet("background: transparent;")
        row2_outer = QVBoxLayout(self._row2_widget)
        row2_outer.setContentsMargins(0, 0, 0, 0)
        row2_outer.setSpacing(2)

        # Row 2a: checkboxes
        row2a = QHBoxLayout()
        row2a.setContentsMargins(0, 0, 0, 0)
        row2a.setSpacing(6)

        self._ct_check = QCheckBox(t("click_through"))
        self._ct_check.setFont(QFont("Consolas", 8))
        self._ct_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._ct_check.setStyleSheet(_CHECK_CSS)
        self._ct_check.toggled.connect(self.click_through_toggled.emit)
        row2a.addWidget(self._ct_check)

        self._topmost_check = QCheckBox(t("top_most"))
        self._topmost_check.setFont(QFont("Consolas", 8))
        self._topmost_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._topmost_check.setStyleSheet(_CHECK_CSS)
        self._topmost_check.setChecked(True)
        self._topmost_check.toggled.connect(self.topmost_toggled.emit)
        row2a.addWidget(self._topmost_check)

        self._auto_scroll = QCheckBox(t("auto_scroll"))
        self._auto_scroll.setFont(QFont("Consolas", 8))
        self._auto_scroll.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._auto_scroll.setStyleSheet(_CHECK_CSS)
        self._auto_scroll.setChecked(True)
        self._auto_scroll.toggled.connect(self.auto_scroll_toggled.emit)
        row2a.addWidget(self._auto_scroll)

        self._taskbar_check = QCheckBox(t("taskbar"))
        self._taskbar_check.setFont(QFont("Consolas", 8))
        self._taskbar_check.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._taskbar_check.setStyleSheet(_CHECK_CSS)
        self._taskbar_check.setChecked(False)
        self._taskbar_check.toggled.connect(self.taskbar_toggled.emit)
        row2a.addWidget(self._taskbar_check)

        row2a.addStretch()
        row2_outer.addLayout(row2a)

        # Row 2b: model + source language + target language combos (stretch to fill)
        row2b = QHBoxLayout()
        row2b.setContentsMargins(0, 0, 0, 0)
        row2b.setSpacing(4)

        _lbl_css = "color: #888; background: transparent;"
        _lbl_font = QFont("Consolas", 8)
        _combo_font = QFont("Consolas", 8)

        model_lbl = QLabel(t("model_label"))
        model_lbl.setFont(_lbl_font)
        model_lbl.setStyleSheet(_lbl_css)
        row2b.addWidget(model_lbl)

        self._model_combo = QComboBox()
        self._model_combo.setFixedHeight(18)
        self._model_combo.setFont(_combo_font)
        self._model_combo.setStyleSheet(_COMBO_CSS)
        self._model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._model_combo.currentIndexChanged.connect(self.model_changed.emit)
        row2b.addWidget(self._model_combo, 3)

        src_lbl = QLabel(t("source_label"))
        src_lbl.setFont(_lbl_font)
        src_lbl.setStyleSheet(_lbl_css)
        row2b.addWidget(src_lbl)

        self._source_lang = QComboBox()
        self._source_lang.setFixedHeight(18)
        self._source_lang.setFont(_combo_font)
        self._source_lang.setStyleSheet(_COMBO_CSS)
        self._source_lang.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        for code, native in LANGUAGES:
            label = t("asr_lang_auto") if code == "auto" else native
            self._source_lang.addItem(f"{code} - {label}", code)
        self._source_lang.currentIndexChanged.connect(
            lambda idx: self.source_language_changed.emit(
                self._source_lang.currentData() or "auto"
            )
        )
        row2b.addWidget(self._source_lang, 2)

        tgt_lbl = QLabel(t("target_label"))
        tgt_lbl.setFont(_lbl_font)
        tgt_lbl.setStyleSheet(_lbl_css)
        row2b.addWidget(tgt_lbl)

        self._target_lang = QComboBox()
        self._target_lang.setFixedHeight(18)
        self._target_lang.setFont(_combo_font)
        self._target_lang.setStyleSheet(_COMBO_CSS)
        self._target_lang.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        for code, native in LANGUAGES:
            if code == "auto":
                continue
            self._target_lang.addItem(f"{code} - {native}", code)
        self._target_lang.currentIndexChanged.connect(
            lambda idx: self.target_language_changed.emit(
                self._target_lang.currentData() or "zh"
            )
        )
        row2b.addWidget(self._target_lang, 2)

        row2_outer.addLayout(row2b)

        outer.addWidget(self._row2_widget)

    def _on_start_stop(self):
        if self._running:
            self.stop_clicked.emit()
        else:
            self.start_clicked.emit()

    _PAUSED_CSS = _BTN_CSS.replace(
        "rgba(255,255,255,20)", "rgba(220,180,60,50)"
    ).replace("color: #aaa", "color: #ddb")

    def set_target_language(self, lang: str):
        idx = self._target_lang.findData(lang)
        if idx >= 0:
            self._target_lang.blockSignals(True)
            self._target_lang.setCurrentIndex(idx)
            self._target_lang.blockSignals(False)

    def set_source_language(self, lang: str):
        idx = self._source_lang.findData(lang)
        if idx >= 0:
            self._source_lang.blockSignals(True)
            self._source_lang.setCurrentIndex(idx)
            self._source_lang.blockSignals(False)

    def set_models(self, models: list, active_index: int = 0):
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m.get("name", m.get("model", "?")))
        if 0 <= active_index < self._model_combo.count():
            self._model_combo.setCurrentIndex(active_index)
        self._model_combo.blockSignals(False)

    @property
    def auto_scroll(self) -> bool:
        return self._auto_scroll.isChecked()

    def set_running(self, running: bool):
        self._running = running
        if running:
            self._start_stop_btn.setText(t("running"))
            self._start_stop_btn.setStyleSheet(_BTN_CSS)
        else:
            self._start_stop_btn.setText(t("paused"))
            self._start_stop_btn.setStyleSheet(self._PAUSED_CSS)

    def _toggle_mode(self):
        new_mode = "compact" if self._mode == "full" else "full"
        self._apply_mode(new_mode)
        self.mode_changed.emit(new_mode)

    def _apply_mode(self, mode: str):
        self._mode = mode
        compact = mode == "compact"
        self._row2_widget.setVisible(not compact)
        self._clear_btn.setVisible(not compact)
        self._subtitle_btn.setVisible(not compact)
        self._screenshot_btn.setVisible(not compact)
        self._mode_btn.setText(t("mode_compact") if compact else t("mode_full"))
        self.setFixedHeight(24 if compact else 62)

    def set_mode(self, mode: str):
        if mode != self._mode:
            self._apply_mode(mode)

    def set_subtitle_checked(self, checked: bool):
        self._subtitle_btn.setStyleSheet(
            _BTN_CSS.replace("rgba(255,255,255,20)", "rgba(80,180,80,40)").replace(
                "rgba(255,255,255,40)", "rgba(80,180,80,80)"
            ) if checked else _BTN_CSS
        )


class SubtitleOverlay(QWidget):
    """Chat-style overlay window for displaying live transcription."""

    add_message_signal = pyqtSignal(int, str, str, str, float)
    update_translation_signal = pyqtSignal(int, str, float)
    update_streaming_signal = pyqtSignal(int, str)
    clear_signal = pyqtSignal()
    # Monitor signals (thread-safe)
    update_monitor_signal = pyqtSignal(float, float, object)
    update_stats_signal = pyqtSignal(int, int, int, int, float)
    update_asr_device_signal = pyqtSignal(str)

    settings_requested = pyqtSignal()
    target_language_changed = pyqtSignal(str)
    source_language_changed = pyqtSignal(str)
    model_switch_requested = pyqtSignal(int)
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    subtitle_toggled = pyqtSignal()
    screenshot_requested = pyqtSignal()
    mode_changed = pyqtSignal(str)  # "full" or "compact"
    position_changed = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._messages = {}
        self._max_messages = 50
        self._click_through = False
        self._height_before_compact = None
        self._mode_anim = None
        self._pos_save_timer = QTimer(self)
        self._pos_save_timer.setSingleShot(True)
        self._pos_save_timer.setInterval(500)
        self._pos_save_timer.timeout.connect(lambda: self.position_changed.emit())
        self._last_saved_geo = None
        self._setup_ui()

        self.add_message_signal.connect(self._on_add_message)
        self.update_translation_signal.connect(self._on_update_translation)
        self.update_streaming_signal.connect(self._on_update_streaming)
        self.clear_signal.connect(self._on_clear)
        self.update_monitor_signal.connect(self._on_update_monitor)
        self.update_stats_signal.connect(self._on_update_stats)
        self.update_asr_device_signal.connect(self._on_update_asr_device)

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowTitle("LiveTranslate")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        width = 620
        height = 500
        x = geo.right() - width - 20
        y = geo.bottom() - height - 60
        self.setGeometry(x, y, width, height)
        self.setMinimumSize(480, 200)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._container = QWidget()
        self._container.setStyleSheet(
            "background-color: rgba(15, 15, 25, 200); border-radius: 8px;"
        )

        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(0)

        # Drag handle
        self._handle = DragHandle()
        self._handle.settings_clicked.connect(self.settings_requested.emit)
        self._handle.subtitle_clicked.connect(self.subtitle_toggled.emit)
        self._handle.screenshot_clicked.connect(self.screenshot_requested.emit)
        self._handle.click_through_toggled.connect(self._set_click_through)
        self._handle.topmost_toggled.connect(self._set_topmost)
        self._handle.taskbar_toggled.connect(self._set_taskbar)
        self._handle.target_language_changed.connect(self.target_language_changed.emit)
        self._handle.source_language_changed.connect(self.source_language_changed.emit)
        self._handle.model_changed.connect(self.model_switch_requested.emit)
        self._handle.start_clicked.connect(self.start_requested.emit)
        self._handle.stop_clicked.connect(self.stop_requested.emit)
        self._handle.hide_clicked.connect(self.hide_requested.emit)
        self._handle.clear_clicked.connect(self._on_clear)
        self._handle.quit_clicked.connect(self.quit_requested.emit)
        self._handle.mode_changed.connect(self._on_mode_changed)
        self._handle.position_changed.connect(self.position_changed)
        container_layout.addWidget(self._handle)

        # Monitor bar (collapsible)
        self._monitor = MonitorBar()
        container_layout.addWidget(self._monitor)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                width: 6px; background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,60); border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self._msg_container = QWidget()
        self._msg_container.setStyleSheet("background: transparent;")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setContentsMargins(0, 0, 0, 0)
        self._msg_layout.setSpacing(2)
        self._msg_layout.addStretch()

        self._scroll.setWidget(self._msg_container)
        container_layout.addWidget(self._scroll)

        grip_row = QHBoxLayout()
        grip_row.addStretch()
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(22, 22)
        self._grip.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))
        self._grip.setToolTip(t("resize_window_hint"))
        self._grip.setStyleSheet("""
            QSizeGrip {
                background: rgba(255, 255, 255, 42);
                border-top: 1px solid rgba(255, 255, 255, 70);
                border-left: 1px solid rgba(255, 255, 255, 70);
                border-top-left-radius: 8px;
            }
            QSizeGrip:hover {
                background: rgba(100, 180, 255, 110);
                border-color: rgba(150, 210, 255, 180);
            }
        """)
        grip_row.addWidget(self._grip)
        container_layout.addLayout(grip_row)

        main_layout.addWidget(self._container)

        self._ct_timer = QTimer(self)
        self._ct_timer.timeout.connect(self._check_click_through)
        self._ct_timer.start(50)

    def _schedule_pos_save(self):
        if not self.isVisible():
            return
        geo = (self.x(), self.y(), self.width(), self.height())
        if geo != self._last_saved_geo:
            self._last_saved_geo = geo
            self._pos_save_timer.start()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._schedule_pos_save()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_pos_save()

    def set_running(self, running: bool):
        self._handle.set_running(running)

    def _set_topmost(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def _set_taskbar(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags &= ~Qt.WindowType.Tool
        else:
            flags |= Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.show()

    def _set_click_through(self, enabled: bool):
        self._click_through = enabled
        if not enabled:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            if style & _WS_EX_TRANSPARENT:
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style & ~_WS_EX_TRANSPARENT
                )

    def _check_click_through(self):
        if not self._click_through:
            return
        cursor = QCursor.pos()
        local = self.mapFromGlobal(cursor)
        hwnd = int(self.winId())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)

        scroll_top = self._scroll.mapTo(self, QPoint(0, 0)).y()
        in_header = 0 <= local.x() <= self.width() and 0 <= local.y() < scroll_top
        grip_pos = self._grip.mapTo(self, QPoint(0, 0))
        in_grip = (
            grip_pos.x() <= local.x() < grip_pos.x() + self._grip.width()
            and grip_pos.y() <= local.y() < grip_pos.y() + self._grip.height()
        )

        if in_header or in_grip:
            if style & _WS_EX_TRANSPARENT:
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style & ~_WS_EX_TRANSPARENT
                )
        else:
            if not (style & _WS_EX_TRANSPARENT):
                ctypes.windll.user32.SetWindowLongW(
                    hwnd, _GWL_EXSTYLE, style | _WS_EX_TRANSPARENT
                )

    def _on_mode_changed(self, mode: str):
        compact = mode == "compact"
        self._monitor.setVisible(not compact)
        ChatMessage._compact_mode = compact
        s = ChatMessage._current_style
        for msg in self._messages.values():
            msg.apply_style(s)
        self.mode_changed.emit(mode)

        # Animate window height
        if self._mode_anim and self._mode_anim.state() != QPropertyAnimation.State.Stopped:
            self._mode_anim.stop()

        if compact:
            self._height_before_compact = self.height()
            target_h = self.minimumHeight()
        else:
            target_h = self._height_before_compact or 500

        actual_h = self.frameGeometry().height()
        if abs(actual_h - target_h) < 10:
            self.resize(self.width(), target_h)
        else:
            anim = QPropertyAnimation(self, b"size")
            anim.setDuration(200)
            anim.setStartValue(QSize(self.width(), actual_h))
            anim.setEndValue(QSize(self.width(), target_h))
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._mode_anim = anim
            anim.start()

    def set_mode(self, mode: str):
        self._handle.set_mode(mode)

    def set_subtitle_checked(self, checked: bool):
        self._handle.set_subtitle_checked(checked)

    @pyqtSlot(float, float, object)
    def _on_update_monitor(self, rms: float, vad_conf: float, mic_rms):
        self._monitor.update_audio(rms, vad_conf, mic_rms)

    @pyqtSlot(int, int, int, int, float)
    def _on_update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost):
        self._monitor.update_pipeline_stats(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    @pyqtSlot(str)
    def _on_update_asr_device(self, device: str):
        self._monitor.update_asr_device(device)

    @pyqtSlot(int, str, str, str, float)
    def _on_add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        msg = ChatMessage(msg_id, timestamp, original, source_lang, asr_ms)
        self._messages[msg_id] = msg
        self._msg_layout.addWidget(msg)

        if len(self._messages) > self._max_messages:
            oldest_id = min(self._messages.keys())
            old_msg = self._messages.pop(oldest_id)
            self._msg_layout.removeWidget(old_msg)
            old_msg.deleteLater()

        QTimer.singleShot(50, self._scroll_to_bottom)

    @pyqtSlot(int, str, float)
    def _on_update_translation(self, msg_id, translated, translate_ms):
        msg = self._messages.get(msg_id)
        if msg:
            msg.set_translation(translated, translate_ms)
            QTimer.singleShot(50, self._scroll_to_bottom)

    def _on_update_streaming(self, msg_id, partial_text):
        msg = self._messages.get(msg_id)
        if msg:
            msg.update_streaming(partial_text)

    @pyqtSlot()
    def _on_clear(self):
        for msg in self._messages.values():
            self._msg_layout.removeWidget(msg)
            msg.deleteLater()
        self._messages.clear()

    def _scroll_to_bottom(self):
        if not self._handle.auto_scroll:
            return
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def apply_style(self, style: dict):
        s = {**DEFAULT_STYLE, **style}
        # Migrate old single font_family to split fields
        if "font_family" in s and "original_font_family" not in style:
            s["original_font_family"] = s["font_family"]
            s["translation_font_family"] = s["font_family"]
        # Container background
        bg_rgba = _hex_to_rgba(s["bg_color"], s["bg_opacity"])
        self._container.setStyleSheet(
            f"background-color: {bg_rgba}; border-radius: {s['border_radius']}px;"
        )
        # Header background
        hdr_rgba = _hex_to_rgba(s["header_color"], s["header_opacity"])
        self._handle.setStyleSheet(f"background: {hdr_rgba}; border-radius: 4px;")
        # Window opacity
        self.setWindowOpacity(s["window_opacity"] / 100.0)
        self._monitor.set_performance_scope(s.get("performance_scope", "application"))
        # Update all existing messages
        ChatMessage._current_style = s
        for msg in self._messages.values():
            msg.apply_style(s)

    def export_messages(self, mode: str, parent=None):
        """Export captured messages to a .txt file.

        mode: "original" | "translation" | "both"
        """
        if not self._messages:
            QMessageBox.information(parent or self, "LiveTranslate", t("export_empty"))
            return

        from datetime import datetime
        suffix = {"original": "original", "translation": "translation", "both": "all"}.get(mode, "all")
        default_name = f"livetrans_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.txt"
        path, _ = QFileDialog.getSaveFileName(
            parent or self,
            t("export_dialog_title"),
            default_name,
            t("export_filter"),
        )
        if not path:
            return

        lines = []
        for msg_id in sorted(self._messages.keys()):
            msg = self._messages[msg_id]
            ts = msg._timestamp
            orig = msg._original or ""
            trans = msg._translated or ""
            if mode == "original":
                lines.append(f"[{ts}] {orig}")
            elif mode == "translation":
                if trans:
                    lines.append(f"[{ts}] {trans}")
            else:
                lines.append(f"[{ts}] {orig}")
                if trans:
                    lines.append(f"  -> {trans}")
                lines.append("")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines).rstrip() + "\n")
        except OSError as e:
            QMessageBox.critical(
                parent or self,
                "LiveTranslate",
                t("export_failed").format(error=str(e)),
            )

    # Thread-safe public API
    def add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        self.add_message_signal.emit(msg_id, timestamp, original, source_lang, asr_ms)

    def update_translation(self, msg_id, translated, translate_ms):
        self.update_translation_signal.emit(msg_id, translated, translate_ms)

    def update_streaming(self, msg_id, partial_text):
        self.update_streaming_signal.emit(msg_id, partial_text)

    def update_monitor(self, rms, vad_conf, mic_rms=None):
        self.update_monitor_signal.emit(rms, vad_conf, mic_rms)

    def update_stats(self, asr_count, tl_count, prompt_tokens, completion_tokens, cost=0.0):
        self.update_stats_signal.emit(
            asr_count, tl_count, prompt_tokens, completion_tokens, cost
        )

    def update_asr_device(self, device: str):
        self.update_asr_device_signal.emit(device)

    def set_target_language(self, lang: str):
        self._handle.set_target_language(lang)

    def set_source_language(self, lang: str):
        self._handle.set_source_language(lang)

    def set_models(self, models: list, active_index: int = 0):
        self._handle.set_models(models, active_index)

    def clear(self):
        self.clear_signal.emit()
