from __future__ import annotations

from typing import Optional


def extract_last_boxed(text: str) -> Optional[str]:
    """Return the inner content of the last ``\\boxed{...}`` span, or None."""
    parts = text.split("\\boxed")
    if len(parts) < 2:
        return None
    last = parts[-1]
    if not last:
        return None
    if last[0] != "{":
        return last.split("$")[0].strip() or None
    depth = 1
    result = ""
    for c in last[1:]:
        if c == "{":
            depth += 1
            result += c
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            result += c
        else:
            result += c
    return result if depth == 0 else None
