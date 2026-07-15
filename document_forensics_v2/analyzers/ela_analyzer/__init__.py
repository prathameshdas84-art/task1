"""
Error Level Analysis (ELA) — Layer 5
Detects localized image-editing artifacts by re-compressing the page as
JPEG and measuring the per-block difference against the original render.
Blocks with abnormally high recompression error (relative to the page's
own block-error distribution) indicate a region that was likely edited
or pasted in after the rest of the page was finalized.
"""

from .constants import RENDER_DPI, VECTOR_PDF_RENDER_DPI
from .models import ELARegion, ELAReport
from .analyzer import ELAAnalyzer

__all__ = ["ELAAnalyzer", "ELAReport", "ELARegion", "RENDER_DPI", "VECTOR_PDF_RENDER_DPI"]
