"""Dynamic terminology lookup for per-request translation prompts."""

import csv
from dataclasses import dataclass
import io
import logging
from pathlib import Path
import re


_PAREN_RE = re.compile(r"[（(]([^）)]*)[）)]")
_EDITORIAL_WORDS = (
    "亦作", "社区", "待官方", "简称", "版本", "限定", "已故", "剧情",
    "称号", "区域", "来源", "阵营", "相关", "also", "pending",
)
log = logging.getLogger("LiveTranslate.Glossary")


def _read_glossary_text(path: Path) -> str:
    """Read UTF-8 glossary files, with a Windows Chinese-codepage fallback."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Excel and some Windows editors save CSV files as ANSI/GBK even when
        # the user intends to use UTF-8. GB18030 is a superset that preserves
        # Chinese, Japanese, and ASCII data from those files.
        text = path.read_text(encoding="gb18030")
        log.warning(
            "Glossary is not UTF-8; read with GB18030 fallback: %s. "
            "Save as UTF-8 for maximum portability.",
            path,
        )
        return text


def _clean_target(value: str) -> str:
    value = str(value or "").strip()
    value = _PAREN_RE.sub("", value)
    # The table occasionally lists multiple Chinese alternatives.  Use the
    # first (preferred) form so the model never emits a slash-separated list.
    value = re.split(r"[/／]", value, maxsplit=1)[0]
    return " ".join(value.split()).strip(" /／")


def _expand_aliases(value: str) -> tuple[str, ...]:
    value = str(value or "").strip()
    if not value or value.startswith("—"):
        return ()
    output = []
    seen = set()
    for part in re.split(r"\s+/\s+|／|、|，|;|；|\|", value):
        part = part.strip()
        if not part or part.startswith("—"):
            continue
        candidates = [_PAREN_RE.sub("", part).strip()]
        for inner in _PAREN_RE.findall(part):
            inner = inner.strip()
            if inner and not any(word in inner.lower() for word in _EDITORIAL_WORDS):
                candidates.append(inner)
        for candidate in candidates:
            candidate = " ".join(candidate.split()).strip(" /／")
            key = candidate.casefold()
            if candidate and key not in seen:
                output.append(candidate)
                seen.add(key)
    return tuple(output)


def _alias_key(value: str) -> str:
    return "".join(ch.casefold() for ch in value if ch.isalnum())


def _is_matchable(alias: str) -> bool:
    key = _alias_key(alias)
    if not key:
        return False
    if any(ch.isdigit() for ch in key):
        return len(key) >= 2
    if alias.isascii():
        return len(key) >= 3
    return len(key) >= 2


def _contains_alias(text: str, alias: str) -> bool:
    if not _is_matchable(alias):
        return False
    if alias.isascii():
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
        )
    return alias.casefold() in text.casefold()


@dataclass(frozen=True)
class GlossaryEntry:
    chinese: str
    english: tuple[str, ...]
    japanese: tuple[str, ...]
    section: str = ""

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.japanese + self.english


class TermGlossary:
    def __init__(self, entries, max_entries=12, max_prompt_chars=1400):
        merged = {}
        order = []
        for entry in entries:
            key = entry.chinese.casefold()
            if not entry.chinese:
                continue
            if key not in merged:
                merged[key] = {
                    "chinese": entry.chinese,
                    "english": [],
                    "japanese": [],
                    "section": entry.section,
                }
                order.append(key)
            item = merged[key]
            for field in ("english", "japanese"):
                for alias in getattr(entry, field):
                    if alias not in item[field]:
                        item[field].append(alias)
        self.entries = tuple(
            GlossaryEntry(
                chinese=merged[key]["chinese"],
                english=tuple(merged[key]["english"]),
                japanese=tuple(merged[key]["japanese"]),
                section=merged[key]["section"],
            )
            for key in order
        )
        self.max_entries = max(1, int(max_entries))
        self.max_prompt_chars = max(200, int(max_prompt_chars))

    @classmethod
    def from_file(cls, path, **kwargs):
        path = Path(path)
        text = _read_glossary_text(path)
        if path.suffix.casefold() == ".csv":
            return cls.from_csv(text, **kwargs)
        return cls.from_markdown(text, **kwargs)

    @classmethod
    def from_files(cls, paths, **kwargs):
        entries = []
        for path in paths:
            entries.extend(cls.from_file(path).entries)
        return cls(entries, **kwargs)

    @classmethod
    def from_markdown(cls, text: str, **kwargs):
        entries = []
        section = ""
        table_ready = False
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if line.startswith("##"):
                section = line.lstrip("# ").strip()
                table_ready = False
                continue
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 3:
                continue
            if cells[0] == "中文" and cells[1].lower() == "english":
                table_ready = True
                continue
            if not table_ready or all(set(cell) <= {"-", ":"} for cell in cells[:3]):
                continue
            chinese = _clean_target(cells[0])
            english = _expand_aliases(cells[1])
            japanese = _expand_aliases(cells[2])
            if chinese and (english or japanese):
                entries.append(
                    GlossaryEntry(
                        chinese=chinese,
                        english=english,
                        japanese=japanese,
                        section=section,
                    )
                )
        return cls(entries, **kwargs)

    @classmethod
    def from_csv(cls, text: str, **kwargs):
        entries = []
        reader = csv.DictReader(io.StringIO(str(text or "").lstrip("\ufeff")))
        for row in reader:
            normalized = {
                str(key or "").strip().casefold(): str(value or "").strip()
                for key, value in row.items()
            }
            chinese = _clean_target(
                normalized.get("中文")
                or normalized.get("chinese")
                or normalized.get("zh")
            )
            english = _expand_aliases(
                normalized.get("english") or normalized.get("英文")
            )
            japanese = _expand_aliases(
                normalized.get("日本語")
                or normalized.get("日文")
                or normalized.get("japanese")
                or normalized.get("ja")
            )
            section = (
                normalized.get("分类")
                or normalized.get("category")
                or normalized.get("section")
                or ""
            )
            if chinese and (english or japanese):
                entries.append(
                    GlossaryEntry(
                        chinese=chinese,
                        english=english,
                        japanese=japanese,
                        section=section,
                    )
                )
        return cls(entries, **kwargs)

    def match(self, text: str) -> list[tuple[GlossaryEntry, str]]:
        text = str(text or "").strip()
        if not text:
            return []
        matches = []
        for entry in self.entries:
            aliases = [alias for alias in entry.aliases if _contains_alias(text, alias)]
            if aliases:
                best = max(aliases, key=lambda value: len(_alias_key(value)))
                matches.append((entry, best))
        matches.sort(
            key=lambda item: (
                "角色" not in item[0].section and "干员" not in item[0].section,
                -len(_alias_key(item[1])),
            )
        )
        return matches[: self.max_entries]

    def normalize_source(
        self, text: str, language: str
    ) -> tuple[str, list[tuple[str, str]]]:
        """Replace explicit source-language aliases with each entry's canonical name.

        The first value in a language column is treated as the canonical spelling;
        later pipe-separated values are aliases, including common ASR variants.
        Only unambiguous aliases are replaced, and overlapping matches prefer the
        longest alias so a short fragment cannot partially rewrite a longer name.
        """
        text = str(text or "")
        lang = str(language or "").strip().casefold().replace("_", "-")
        if lang in {"ja", "jp", "jpn", "japanese"} or lang.startswith("ja-"):
            field = "japanese"
        elif lang in {"en", "eng", "english"} or lang.startswith("en-"):
            field = "english"
        else:
            return text, []

        aliases = {}
        ambiguous = set()
        for entry in self.entries:
            values = getattr(entry, field)
            if len(values) < 2:
                continue
            canonical = values[0]
            for alias in values[1:]:
                if not _is_matchable(alias):
                    continue
                key = alias.casefold()
                if key == canonical.casefold():
                    continue
                previous = aliases.get(key)
                if previous is not None and previous[1].casefold() != canonical.casefold():
                    ambiguous.add(key)
                    continue
                aliases[key] = (alias, canonical)

        for key in ambiguous:
            aliases.pop(key, None)
        if not aliases:
            return text, []

        candidates = []
        for alias, canonical in aliases.values():
            if alias.isascii():
                pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
            else:
                pattern = re.escape(alias)
            for matched in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidates.append(
                    (matched.start(), matched.end(), matched.group(0), canonical)
                )

        if not candidates:
            return text, []

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        selected = []
        occupied_until = -1
        for candidate in candidates:
            if candidate[0] < occupied_until:
                continue
            selected.append(candidate)
            occupied_until = candidate[1]

        output = []
        corrections = []
        cursor = 0
        for start, end, matched_text, canonical in selected:
            output.append(text[cursor:start])
            output.append(canonical)
            corrections.append((matched_text, canonical))
            cursor = end
        output.append(text[cursor:])
        return "".join(output), corrections

    def build_prompt(self, text: str) -> str:
        matches = self.match(text)
        if not matches:
            return ""
        header = (
            "下列为本句命中的《明日方舟：终末地》术语。翻译时必须严格使用箭头右侧的"
            "简体中文名称，不要自行音译，也不要输出术语表："
        )
        lines = [header]
        for entry, matched_alias in matches:
            line = f"- {matched_alias} → {entry.chinese}"
            if len("\n".join(lines + [line])) > self.max_prompt_chars:
                break
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""
