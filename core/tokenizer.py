"""Conservative token estimation utilities."""

from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


@lru_cache(maxsize=1)
def _encoding():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in text."""
    if not text:
        return 0
    encoding = _encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception:
            pass
    # Conservative fallback: 4 characters per token.
    return max(1, len(text) // 4)


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to a conservative token budget."""
    if max_tokens <= 0 or not text:
        return ""
    encoding = _encoding()
    if encoding is not None:
        try:
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return encoding.decode(tokens[:max_tokens])
        except Exception:
            pass
    max_chars = max_tokens * 4
    return text[:max_chars]