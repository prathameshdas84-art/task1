"""
Shared PDF utility helpers usable by any analyzer without circular imports.
"""

import io

import cv2
import fitz
import numpy as np

from PIL import Image

from utils.flat_zone_detection import local_std_map, BORN_DIGITAL_STD_FLOOR

# QR codes are near-square, small relative to the page, and almost pure
# black/white — these heuristics are deliberately generic (not Aadhaar-
# specific) so they catch a QR code on any document type.
QR_ASPECT_MIN     = 0.8
QR_ASPECT_MAX     = 1.2
QR_MAX_AREA_FRAC  = 0.15   # QR code occupies less than 15% of the page area
QR_MIN_BW_RATIO   = 0.75   # fraction of pixels that are near-black or near-white
QR_BLACK_CUTOFF   = 30
QR_WHITE_CUTOFF   = 225


def get_qr_zones(page: "fitz.Page", doc: "fitz.Document") -> list:
    """
    Returns a list of fitz.Rect bounding boxes for likely QR-code image
    regions on a page.

    High-frequency QR pixel data reads as a tamper signal to ELA and
    white-rect detection alike, regardless of document type — so callers
    in each layer use this to exclude QR regions before flagging an
    anomaly there.
    """
    qr_zones = []
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return qr_zones

    try:
        page_images = page.get_images(full=True)
    except Exception:
        return qr_zones

    for img in page_images:
        xref = img[0]
        try:
            bbox = page.get_image_bbox(img)
        except Exception:
            continue
        if not isinstance(bbox, fitz.Rect) or bbox.is_empty or bbox.width <= 0 or bbox.height <= 0:
            continue

        pix = None
        try:
            pix = fitz.Pixmap(doc, xref)
            # Normalize to RGB (no alpha) so the sample buffer reshapes
            # cleanly — source images may be grayscale, CMYK, or have an
            # alpha channel.
            if pix.colorspace is None:
                continue
            if pix.colorspace.n not in (1, 3) or pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            w, h = pix.width, pix.height
            if w == 0 or h == 0:
                continue

            aspect = w / h
            area_frac = (bbox.width * bbox.height) / page_area
            if not (QR_ASPECT_MIN <= aspect <= QR_ASPECT_MAX and area_frac < QR_MAX_AREA_FRAC):
                continue

            img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, pix.n)
            gray = cv2.cvtColor(img_arr[:, :, :3], cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img_arr[:, :, 0]
            bw_ratio = np.sum((gray < QR_BLACK_CUTOFF) | (gray > QR_WHITE_CUTOFF)) / (w * h)

            if bw_ratio > QR_MIN_BW_RATIO:
                qr_zones.append(bbox)
        except Exception:
            continue
        finally:
            pix = None

    return qr_zones


# ── Born-digital gate corroboration (PDF-render paths) ─────────────────────
# A page RENDER is a fresh rasterization: its measured noise floor can
# collapse for reasons that say nothing about the CONTENT being born-digital
# (get_pixmap resampling averages real scan noise away; scanner-app
# background cleanup zeroes it at the source). So the render floor alone
# must not classify a page born-digital — the same one-signal
# misclassification already fixed on the standalone image path, where the
# corroborating signal is grid_phase_z measured on the uploaded pixels.
# On the PDF path the render pixels can't carry that corroboration (a
# fresh rasterization has no compression history), but the page's embedded
# SOURCE image objects — the original bytes — can:
#   1. a DCTDecode/JPX stream: content stored as a lossy-compressed raster
#      necessarily passed through a rasterize+save pipeline (scan / photo /
#      screenshot / paste). Unlike a standalone JPEG *upload* — where JPEG
#      is everyone's default export format and proves nothing (the image
#      pipeline's S5 case) — born-digital PDF page content is vector
#      text/drawings; arriving as a JPEG image object is itself the trace.
#   2. real texture in the source pixels at NATIVE resolution (the render
#      may have resampled it away).
#   3. 8px blocking-grid residuals in a non-JPEG-container source (a
#      recompressed capture re-wrapped as PNG/Flate) — the same
#      grid-phase statistic the standalone gate's second signal uses.
RASTER_EVIDENCE_MAX_PX = 2048   # analyze a center CROP above this (never
                                # resample — resampling erases the noise
                                # this is testing for)
_LOSSY_EXTS = ("jpeg", "jpg", "jpx", "jp2")


def page_raster_source_evidence(page: "fitz.Page", doc: "fitz.Document"):
    """
    Returns (evidence: bool, basis: str) — whether this page's embedded
    source images show the content passed through a raster/photo/scan/
    paste pipeline at any point. Used to corroborate (or veto) the
    born-digital gate when a page render's noise floor is low.
    """
    try:
        images = page.get_images(full=True)
    except Exception:
        images = []
    if not images:
        return False, "page has no embedded raster images"

    for im in images:
        xref = im[0]
        try:
            info = doc.extract_image(xref)
            ext = (info.get("ext") or "").lower()
            if ext in _LOSSY_EXTS:
                return True, (
                    f"embedded image xref {xref} is a lossy-compressed "
                    f"({ext}) stream — content passed through a "
                    f"rasterize+save pipeline before entering the PDF"
                )

            pil = Image.open(io.BytesIO(info["image"])).convert("L")
            w, h = pil.size
            if w * h < 64 * 64:
                continue
            if w > RASTER_EVIDENCE_MAX_PX or h > RASTER_EVIDENCE_MAX_PX:
                cw, ch = min(w, RASTER_EVIDENCE_MAX_PX), min(h, RASTER_EVIDENCE_MAX_PX)
                left, top = (w - cw) // 2, (h - ch) // 2
                pil = pil.crop((left, top, left + cw, top + ch))
            gray = np.asarray(pil, dtype=np.float64)

            _, src_baseline = local_std_map(gray)
            if src_baseline >= BORN_DIGITAL_STD_FLOOR:
                return True, (
                    f"embedded image xref {xref} has real scan/photo "
                    f"texture at native resolution (noise baseline "
                    f"{src_baseline:.2f} >= {BORN_DIGITAL_STD_FLOOR})"
                )

            # Lazy import: pdf_utils must stay importable by any analyzer,
            # and this reuses the standalone gate's exact statistic rather
            # than re-deriving a second implementation of it.
            from analyzers.image_document_analyzer.checks_compression import (
                CompressionChecksMixin,
            )
            from analyzers.image_document_analyzer.constants import (
                BLOCKINESS_Z_THRESHOLD,
            )
            _, m = CompressionChecksMixin._detect_jpeg_history(gray, ext.upper())
            grid_z = m.get("grid_phase_z") or [0.0, 0.0]
            if min(grid_z) > BLOCKINESS_Z_THRESHOLD:
                return True, (
                    f"embedded image xref {xref} carries 8px blocking-grid "
                    f"residuals (grid_phase_z={grid_z}) — a re-compressed "
                    f"capture re-wrapped in a lossless container"
                )
        except Exception:
            continue

    return False, ("embedded raster content shows no scan/photo texture "
                   "and no lossy-compression traces at native resolution")


def bbox_overlaps_qr_zone(bbox, qr_zones: list) -> bool:
    """
    True if bbox (a 4-tuple/list of x0,y0,x1,y1, or a fitz.Rect) intersects
    any QR zone.
    """
    if not qr_zones:
        return False
    rect = bbox if isinstance(bbox, fitz.Rect) else fitz.Rect(bbox)
    return any(rect.intersects(qr) for qr in qr_zones)
