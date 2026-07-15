"""Numeric-layer dataclasses and the tolerant number parser."""

import re
from dataclasses import dataclass, field


@dataclass
class NumericAnomaly:
    page: int
    line_num: int
    text: str               # full line text
    bbox: tuple             # (x0, y0, x1, y1) PDF points
    value: float            # the anomalous number
    group_mean: float       # mean of the group this number belongs to
    group_std: float        # std of the group
    z_score: float          # how many std deviations from mean
    context: str            # what kind of number group (column/label)
    reason: str


@dataclass
class NumericReport:
    anomalies: list[NumericAnomaly]
    groups_analyzed: int
    total_numbers: int
    anomaly_score: int      # 0-100
    signals: list[str]


# ── Number extractor ───────────────────────────────────────────────────────────

# Regex to find numbers including decimals, commas
NUMBER_RE = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b')


def _parse_number(text: str) -> float:
    """Parse number string to float, handling commas."""
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None

