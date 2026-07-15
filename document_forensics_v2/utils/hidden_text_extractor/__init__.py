"""
Hidden Text Extractor — recovers ORIGINAL text that was covered up by edits.

READ-ONLY: never modifies the analyzed PDF or any other file. Three
independent recovery methods are tried and their results merged:

  1. White rectangle cover detection — text sitting under an opaque white
     filled rectangle (a classic "white-out and retype" edit).
  2. Z-order text overlap detection — two different text spans occupying
     the same location (one was drawn over the other).
  3. Incremental update recovery — PDFs with multiple %%EOF markers keep
     every prior revision's bytes; text present in an early revision but
     missing from the latest one was removed/replaced.
"""


from .models import (
    HiddenTextFinding, HiddenTextReport, TextStackingFinding,
)
from .stacking import (
    TEXT_STACKING_MIN_OVERLAP_FRACTION, TEXT_STACKING_MIN_TEXT_LEN,
    TEXT_STACKING_FUSION_SCORE,
)
from .extractor import HiddenTextExtractor

__all__ = [
    "HiddenTextFinding", "HiddenTextReport", "TextStackingFinding",
    "TEXT_STACKING_MIN_OVERLAP_FRACTION", "TEXT_STACKING_MIN_TEXT_LEN",
    "TEXT_STACKING_FUSION_SCORE", "HiddenTextExtractor",
]
