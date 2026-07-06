"""
Shared PDF utility helpers usable by any analyzer without circular imports.
"""

import cv2
import fitz
import numpy as np

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

    High-frequency QR pixel data reads as a tamper signal to OCR color/size
    checks, ELA, and white-rect detection alike, regardless of document
    type — so callers in each layer use this to exclude QR regions before
    flagging an anomaly there.
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


def bbox_overlaps_qr_zone(bbox, qr_zones: list) -> bool:
    """
    True if bbox (a 4-tuple/list of x0,y0,x1,y1, or a fitz.Rect) intersects
    any QR zone.
    """
    if not qr_zones:
        return False
    rect = bbox if isinstance(bbox, fitz.Rect) else fitz.Rect(bbox)
    return any(rect.intersects(qr) for qr in qr_zones)
