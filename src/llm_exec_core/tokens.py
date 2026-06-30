"""Token estimation helpers."""

from .constants import CHAR_TOKEN_RATIO_EN, CHAR_TOKEN_RATIO_ZH


def estimate_tokens(text: str) -> int:
    if not text:
        return 0

    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    total_chars = len(text)
    if total_chars == 0:
        return 0

    chinese_ratio = chinese_chars / total_chars
    if chinese_ratio > 0.2:
        blended_ratio = (
            chinese_ratio * CHAR_TOKEN_RATIO_ZH
            + (1 - chinese_ratio) * CHAR_TOKEN_RATIO_EN
        )
    else:
        blended_ratio = CHAR_TOKEN_RATIO_EN

    return int(total_chars / blended_ratio)
