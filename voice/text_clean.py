"""Text cleaner to prepare LLM output and chat text for natural speech synthesis."""

from __future__ import annotations

import re

# Match <think>...</think> reasoning blocks
_THINK_BLOCK = re.compile(r"(?is)<think>.*?</think>")

# Match markdown code blocks and inline code
_CODE_BLOCK = re.compile(r"(?s)```.*?```")
_INLINE_CODE = re.compile(r"`([^`]+)`")

# Match URLs
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Match markdown images and links
_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]*\)")

# Match markdown table borders and divider lines (e.g. |---|---|)
_TABLE_DIVIDER = re.compile(r"(?m)^\s*\|?[\s\-:|]+\|?\s*$")
_TABLE_ROW_BORDER = re.compile(r"(?m)^\s*\|\s*|\s*\|\s*$")

# Markdown decoration
_HEADING = re.compile(r"(?m)^\s*#{1,6}\s+")
_BLOCKQUOTE = re.compile(r"(?m)^\s*>\s*")
_BULLETS = re.compile(r"(?m)^\s*[-*+]\s+")
_ORDERED_LIST = re.compile(r"(?m)^\s*\d+\.\s+")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)")
_STRIKETHROUGH = re.compile(r"~~")

# Emojis and miscellaneous symbols
_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff"  # Astral planes (surrogate-pair emojis)
    r"\u2600-\u27bf"          # Misc symbols & dingbats
    r"\u2300-\u23ff"          # Misc technical
    r"\u2b50-\u2b55"
    r"\ufe0e\ufe0f"           # Variation selectors
    r"\u200d"                 # Zero-width joiner
    r"]+",
    flags=re.UNICODE,
)

# Currency and common symbols
_DOLLAR_AMOUNT = re.compile(r"\$(\d+(?:\.\d+)?)")
_PERCENT_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def clean_text_for_speech(text: str) -> str:
    """Transform markdown and technical text into clean, pronounceable English sentences."""
    if not text:
        return ""

    # 1. Drop reasoning blocks
    text = _THINK_BLOCK.sub(" ", text)

    # 2. Strip code blocks completely, preserve inline code content
    text = _CODE_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)

    # 3. Strip URLs, images, retain link anchor text
    text = _URL_PATTERN.sub(" ", text)
    text = _IMAGE_PATTERN.sub(" ", text)
    text = _LINK_PATTERN.sub(r"\1", text)

    # 4. Remove tables and table dividers
    text = _TABLE_DIVIDER.sub(" ", text)
    text = _TABLE_ROW_BORDER.sub(" ", text)
    text = text.replace("|", ", ")

    # 5. Remove markdown structure
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _BULLETS.sub("", text)
    text = _ORDERED_LIST.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _STRIKETHROUGH.sub("", text)

    # 6. Currency and spoken math/symbols
    text = _DOLLAR_AMOUNT.sub(r"\1 dollars", text)
    text = _PERCENT_AMOUNT.sub(r"\1 percent", text)
    text = text.replace("%", " percent")
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("°C", " degrees Celsius")
    text = text.replace("°F", " degrees Fahrenheit")
    text = text.replace("°", " degrees")
    text = text.replace("=", " equals ")

    # 7. Strip emojis and odd unicode symbols
    text = _EMOJI_PATTERN.sub("", text)

    # 8. Collapse whitespace and repeated punctuation
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*", "\n", text)
    text = re.sub(r"\n+", ".\n", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:]){2,}", r"\1", text)

    return text.strip()

