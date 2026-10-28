"""Sentence splitter that batches short phrases and splits oversized sentences."""

from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_sentences(text: str, min_chars: int = 25, max_chars: int = 220) -> list[str]:
    """Split text into sentence-sized chunks for low-latency sequential TTS synthesis.
    
    Combines tiny fragments until min_chars is reached, then cleanly splits any
    sentence longer than max_chars on punctuation or whitespace boundaries.
    """
    if not text or not text.strip():
        return []

    parts = [p.strip() for p in _SPLIT.split(text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        buf = f"{buf} {p}".strip()
        if len(buf) >= min_chars:
            out.append(buf)
            buf = ""
    if buf:
        if out:
            out[-1] += " " + buf
        else:
            out.append(buf)

    final: list[str] = []
    for s in out:
        while len(s) > max_chars:
            cut = s.rfind(",", 0, max_chars)
            if cut < 40:
                cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            final.append(s[: cut + 1].strip())
            s = s[cut + 1 :].strip()
        if s:
            final.append(s)

    return [s for s in final if s]
