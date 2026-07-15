"""
Numeric Consistency Analyzer — Layer 4
Extracts all numbers from document, groups them by context,
flags statistical outliers using z-score analysis.
Works on any document type — universal approach.
No training data. No ML. Pure statistics.
"""

from .models import NumericAnomaly, NumericReport, _parse_number
from .analyzer import NumericAnalyzer

__all__ = ["NumericAnomaly", "NumericReport", "NumericAnalyzer"]
