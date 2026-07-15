"""
Image-Document Forensic Analyzer — dedicated pipeline for DIRECT image
uploads (JPG/PNG).

Targets the specific attack class this engine previously had no coverage
for: a photographed document (ID card, certificate, receipt) where an
editor (a) used AI inpainting to REMOVE original content — which smooths
away the sensor-noise texture in that region — and/or (b) used a phone
app's text/sticker tool to OVERLAY new content — which renders with
mathematically crisp anti-aliased edges that never match the soft
ink-spread + lens-blur + JPEG-blur profile of the photographed original.
Plus stamp/seal/signature paste-in detection (flat ink fill, cutout
boundary).

ROUTING DECISION (Part 1 of the spec): this analyzer runs ONLY for
direct JPG/PNG uploads via POST /analyze-image. PDFs — including
scanned/mixed PDFs with embedded raster pages — stay entirely in the
existing PDF pipeline: ela_analyzer.py already carries the image-based
document checks for those (noise-consistency + digital-erasure), so
routing raster PDF pages here as well would silently run two different
smoothing detectors over the same pixels and double-count the signal.
One document class, one pipeline.

This module NEVER touches, imports from, or alters the behavior of the
six existing PDF layers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 2 — DELIBERATELY NOT IMPLEMENTED (honesty requirement, Part 0/3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every entry below is a technique that CANNOT produce a real confidence
number from a single uploaded image with no reference data. Rather than
implement placeholder math that looks like forensics, each is omitted
and surfaced in the report's `not_implemented` list so a reader knows
what was NOT checked — not just what came back clean.

* PRNU sensor fingerprinting — requires multiple reference images from
  the SAME camera sensor to average out scene content and extract the
  photo-response non-uniformity pattern. A single image has no reference
  to correlate against; any single-image "PRNU-lite" score would be
  fabricated precision.
* ICA/PCA ink source separation — no established reliable method exists
  for separating ink chemical types from a small consumer-resolution RGB
  patch; output would be false confidence, not signal.
* Lighting/shadow-direction consistency — requires 3D scene
  reconstruction a flat document photo gives no basis for; extremely
  high false-positive risk on documents with naturally uniform lighting.
* DCT quantization TABLE extraction / exact resave counting — only the
  categorical single/double/uncertain flag (Check 4) is implemented;
  table-level analysis and precise resave counts are not reliably
  recoverable, especially after social-media recompression.
* Perspective/lens-distortion geometric consistency — requires camera
  calibration data an arbitrary upload doesn't carry. The edge-sharpness
  comparison (Checks 5/9) is the practical substitute for catching flat
  digital overlays.
* Stamp-geometry "pressure deviation" contour fitting — real stamp
  geometry varies enough from paper texture / photo perspective alone
  that this has meaningful false-positive risk; excluded entirely
  rather than shipped as a near-zero-weight decoration.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Numeric discipline used throughout (same rules the earlier flat-edit
spec mandated): all variance math is done in float64 BEFORE squaring;
E[x^2]-E[x]^2 is clipped at 0 before sqrt; every detection threshold is
RELATIVE to the document's own measured baseline, never an absolute
constant applied blind; and if the whole image has a near-zero noise
baseline (born-digital render), the noise-dependent checks gate out to
score 0 instead of manufacturing findings.
"""

from .constants import CHECK_POINTS, DOUBLE_COMPRESSION_POINTS, NOT_IMPLEMENTED
from .report import (
    ImageAnomaly, ImageForensicsReport, normalize_for_fusion, score_anomalies,
)
from .analyzer import ImageDocumentAnalyzer

__all__ = [
    "CHECK_POINTS", "DOUBLE_COMPRESSION_POINTS", "NOT_IMPLEMENTED",
    "ImageAnomaly", "ImageForensicsReport", "normalize_for_fusion",
    "score_anomalies", "ImageDocumentAnalyzer",
]
