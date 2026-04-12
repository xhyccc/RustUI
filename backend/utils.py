"""Shared utility functions for the RustUI backend."""

from __future__ import annotations

import math
from typing import Optional


def nan_to_none(value: object) -> Optional[float]:
    """Convert a NaN float to None; return valid floats/ints unchanged."""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 6)


def series_to_list(series, decimals: int = 4) -> list:
    """Convert a pandas Series to a JSON-serialisable list.

    NaN values are replaced with None.
    """
    return [
        None if (v is None or (isinstance(v, float) and math.isnan(v)))
        else round(float(v), decimals)
        for v in series
    ]
