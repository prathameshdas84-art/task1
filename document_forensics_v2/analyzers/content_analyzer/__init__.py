"""
Content Consistency Analyzer — Layer 2
Extracts per-line features from PDF.
Builds document statistical profile.
Finds outlier lines that break consistency.
No training data. No ML. Pure statistics.
"""

from .constants import SCANNER_KEYWORDS
from .models import LineProfile, SuspiciousLine, ContentReport
from .analyzer import ContentAnalyzer

__all__ = [
    "SCANNER_KEYWORDS", "LineProfile", "SuspiciousLine", "ContentReport",
    "ContentAnalyzer",
]
