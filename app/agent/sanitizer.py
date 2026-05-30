import re
import logging

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    r"ignore\s+previous",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"forget\s+(all\s+)?(previous|prior|your)",
    r"new\s+instructions?",
    r"override\s+(all\s+)?(previous|prior)?",
    r"act\s+as\s+if",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def sanitize_tool_result(text: str) -> str:
    if not text:
        return text

    for pattern in _COMPILED:
        if pattern.search(text):
            logger.warning("sanitizer: injection pattern '%s' detected and stripped.", pattern.pattern)
            text = pattern.sub("[REDACTED]", text)
    return text