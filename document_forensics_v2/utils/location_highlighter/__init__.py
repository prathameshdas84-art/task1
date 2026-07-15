"""
Location Highlighter — renders PDF pages and draws labeled boxes around
suspicious regions; returns annotated page images for the API/UI.
"""

from .styles import (
    RENDER_DPI,
    COLOR_CONTENT, COLOR_NUMERIC, COLOR_ELA, COLOR_WHITE_RECT,
    COLOR_IMAGE_OVERLAY, COLOR_GHOST, COLOR_TEXT_STACKING,
    COLOR_EMBEDDED_IMAGE,
    _content_label, _numeric_label, _flat_zone_label,
)
from .highlighter import LocationHighlighter

__all__ = [
    "RENDER_DPI", "LocationHighlighter",
]
