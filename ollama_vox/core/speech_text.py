"""Keep rich developer responses on screen while preparing natural speech."""

import html
import re
from collections.abc import Iterable

CODE_NOTICE = " The code example is in the response panel. "


def without_fenced_code(tokens: Iterable[str]) -> Iterable[str]:
    """Filter code fences even when markers are split across streaming tokens.

    Only marker state is buffered: prose can still reach TTS immediately and
    long or unfinished code blocks never leak into spoken output.
    """
    fence = None
    pending = ""
    for token in tokens:
        for char in token:
            if char in "`~":
                if pending and pending[0] != char:
                    pending = ""
                pending += char
                if len(pending) == 3:
                    if fence is None:
                        fence = char
                        yield CODE_NOTICE
                    elif fence == char:
                        fence = None
                    pending = ""
                continue
            pending = ""
            if fence is None:
                yield char


def sanitize_for_speech(text: str) -> str:
    """Remove formatting, URLs, and code blocks without altering saved replies."""
    text = re.sub(r"(?s)```.*?(?:```|$)|~~~.*?(?:~~~|$)", CODE_NOTICE, text)
    text = re.sub(r"(?s)<think>.*?(?:</think>|$)", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?:https?://|www\.)[^\s<>]+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)", "", text)
    text = re.sub(r"(?m)^.*\|.*$", "", text)
    text = re.sub(r"(?m)^\s*[-=_]{3,}\s*$", "", text)
    text = text.replace("`", "").replace("*", "").replace("~~", "")
    text = re.sub(r"(?<!\w)_{1,2}|_{1,2}(?!\w)", "", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()
