import translator as translator_module
from translator import Translator


class _DummyClient:
    pass


def _make_translator(monkeypatch, api_base, model, **kwargs):
    monkeypatch.setattr(
        translator_module,
        "make_openai_client",
        lambda *args, **client_kwargs: _DummyClient(),
    )
    return Translator(
        api_base=api_base,
        api_key="test-key",
        model=model,
        no_think=True,
        **kwargs,
    )


def _extra_body(translator):
    return translator._build_request_kwargs("system", "hello")["extra_body"]


def test_deepseek_v4_uses_nested_thinking_toggle(monkeypatch):
    translator = _make_translator(
        monkeypatch, "https://example.com/v1", "deepseek-v4-pro"
    )

    assert _extra_body(translator) == {"thinking": {"type": "disabled"}}


def test_official_deepseek_endpoint_supports_model_aliases(monkeypatch):
    translator = _make_translator(
        monkeypatch, "https://api.deepseek.com/v1", "production-alias"
    )

    assert _extra_body(translator) == {"thinking": {"type": "disabled"}}


def test_non_deepseek_models_keep_legacy_toggle(monkeypatch):
    translator = _make_translator(
        monkeypatch, "https://example.com/v1", "qwen3"
    )

    assert _extra_body(translator) == {"enable_thinking": False}


def test_explicit_extra_body_overrides_automatic_toggle(monkeypatch):
    translator = _make_translator(
        monkeypatch,
        "https://api.deepseek.com",
        "deepseek-v4-pro",
        extra_body={"thinking": {"type": "enabled"}, "user_id": "test"},
    )

    assert _extra_body(translator) == {
        "thinking": {"type": "enabled"},
        "user_id": "test",
    }


def test_target_language_clone_preserves_thinking_control(monkeypatch):
    translator = _make_translator(
        monkeypatch, "https://api.deepseek.com", "deepseek-v4-pro"
    )

    clone = translator.with_target_language("ja")

    assert _extra_body(clone) == {"thinking": {"type": "disabled"}}
