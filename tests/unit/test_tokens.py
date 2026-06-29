from llm_exec_core.tokens import estimate_tokens


def test_estimate_tokens_empty_text_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_english_uses_default_ratio():
    assert estimate_tokens("a" * 35) == 10


def test_estimate_tokens_chinese_heavy_text_uses_cjk_ratio():
    assert estimate_tokens("科学" * 20) > estimate_tokens("science" * 5)
