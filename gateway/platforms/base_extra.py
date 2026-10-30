"""Numeric coercion for ``config.extra`` values (split from :mod:`gateway.platforms.base`)."""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def coerce_float_extra(extra: dict, key: str, default: float, *,
                       min_value: float = 0.0, max_value: Optional[float] = None) -> float:
    """Float from a ``config.extra`` mapping; NaN/Inf/negative/unparseable → ``default``;
    clamped to ``[min_value, max_value]``."""
    try:  # float(None) → TypeError → default
        parsed = float(extra.get(key))
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed) or parsed < 0:
        parsed = float(default)
    parsed = max(parsed, min_value)
    if max_value is not None and parsed > max_value:
        logger.warning("%s=%s exceeds the %s ceiling; clamped", key, parsed, max_value)
        parsed = max_value
    return parsed
