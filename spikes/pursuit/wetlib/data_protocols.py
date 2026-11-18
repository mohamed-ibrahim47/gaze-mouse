"""Stub: the upstream file pulls in mediapipe internals; model_based.py only needs this one name."""
from dataclasses import dataclass
from typing import Any


@dataclass
class PoGResult:
    pog_cm_c: Any = None
    pog_cm_s: Any = None
    pog_mm_s: Any = None
    pog_norm: Any = None
    pog_px: Any = None
