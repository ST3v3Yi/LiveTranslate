import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from term_glossary import TermGlossary


SAMPLE = """
## 5. 干员 / 角色
| 中文 | English | 日本語 |
|---|---|---|
| 佩丽卡 | Perlica | ペルリカ |
| 艾尔黛拉 | Ardelia | アルデリア |
| 思 | Si | シー |

## 8. 元素
| 中文 | English | 日本語 |
|---|---|---|
| 源石技艺 | Arts | アーツ |
"""


class TermGlossaryTests(unittest.TestCase):
    def setUp(self):
        self.glossary = TermGlossary.from_markdown(SAMPLE)

    def test_matches_japanese_and_english_aliases(self):
        prompt = self.glossary.build_prompt("ペルリカとArdeliaが来た。")

        self.assertIn("ペルリカ → 佩丽卡", prompt)
        self.assertIn("Ardelia → 艾尔黛拉", prompt)

    def test_short_latin_alias_does_not_match_inside_word(self):
        prompt = self.glossary.build_prompt("This is a simple test.")

        self.assertNotIn("→ 思", prompt)

    def test_unmatched_text_adds_no_prompt(self):
        self.assertEqual(self.glossary.build_prompt("今日はいい天気です。"), "")

    def test_csv_format_and_alias_columns(self):
        glossary = TermGlossary.from_csv(
            "分类,中文,English,日本語\n"
            "角色,佩丽卡,Perlica | Perurika,ペルリカ\n"
        )

        self.assertIn("佩丽卡", glossary.build_prompt("Perurika"))
        self.assertIn("佩丽卡", glossary.build_prompt("ペルリカ"))

    def test_normalizes_explicit_asr_alias_before_matching(self):
        glossary = TermGlossary.from_csv(
            "分类,中文,English,日本語\n"
            "角色,佩丽卡,Perlica,ペルリカ | ペリカ | ペリ\n"
        )

        normalized, corrections = glossary.normalize_source(
            "この子はペリ。ペリカと同じ種族。", "ja"
        )

        self.assertEqual(
            normalized, "この子はペルリカ。ペルリカと同じ種族。"
        )
        self.assertEqual(
            corrections, [("ペリ", "ペルリカ"), ("ペリカ", "ペルリカ")]
        )
        self.assertIn("ペルリカ → 佩丽卡", glossary.build_prompt(normalized))

    def test_normalization_prefers_longest_overlapping_alias(self):
        glossary = TermGlossary.from_csv(
            "分类,中文,English,日本語\n"
            "角色,佩丽卡,Perlica,ペルリカ | ペリカ | ペリ\n"
        )

        normalized, corrections = glossary.normalize_source("ペリカ", "ja")

        self.assertEqual(normalized, "ペルリカ")
        self.assertEqual(corrections, [("ペリカ", "ペルリカ")])

    def test_multiple_files_are_merged(self):
        with TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.md"
            second = Path(tmp) / "second.csv"
            first.write_text(SAMPLE, encoding="utf-8")
            second.write_text(
                "分类,中文,English,日本語\n"
                "游戏,终末地,Endfield,エンドフィールド\n",
                encoding="utf-8",
            )

            glossary = TermGlossary.from_files([first, second])

        self.assertIn("终末地", glossary.build_prompt("Endfield"))

    def test_csv_gb18030_fallback_keeps_utf8_as_primary_format(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "terms.csv"
            path.write_text(
                "分类,中文,English,日本語\n"
                "角色,佩丽卡,Perlica,ペルリカ\n",
                encoding="gb18030",
            )
            glossary = TermGlossary.from_file(path)

        self.assertIn("佩丽卡", glossary.build_prompt("ペルリカ"))


if __name__ == "__main__":
    unittest.main()
