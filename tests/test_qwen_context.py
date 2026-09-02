import unittest

from asr_qwen3 import Qwen3ASRClient, _normalize_hotwords


class QwenContextTests(unittest.TestCase):
    def test_hotwords_are_normalized_and_deduplicated(self):
        self.assertEqual(
            _normalize_hotwords("ペリカ， Endfield; ペリカ\nArdelia"),
            "ペリカ Endfield Ardelia",
        )

    def test_static_keywords_survive_rolling_context(self):
        client = Qwen3ASRClient(
            {
                "qwen_context_turns": 3,
                "qwen_hotwords": "ペリカ, Endfield",
                "qwen_context_max_chars": 320,
            }
        )
        client.commit_context("で。")
        client.commit_context("へー。")
        client.commit_context("これは前の有用な発言です。")
        client.commit_context("これは前の有用な発言です。")

        context = client._build_context()

        self.assertEqual(
            context,
            "ペリカ Endfield\nこれは前の有用な発言です。",
        )

    def test_keywords_can_be_updated_without_restarting_worker(self):
        client = Qwen3ASRClient({"qwen_hotwords": "old"})
        client.set_static_context("new, name")

        self.assertEqual(client._build_context(), "new name")


if __name__ == "__main__":
    unittest.main()
