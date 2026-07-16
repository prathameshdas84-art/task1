"""
Scanned-page pixel-forensics routing for the PDF pipeline.

Since the OCR layer's removal, pixel forensics is the PRIMARY way scanned
documents catch pasted/flat-region tampering (stamps, signatures, photos,
any inserted element). This module routes every scanned/mixed page RENDER
through the image pipeline's page-render-safe checks
(analyzers/image_document_analyzer.analyze_page_render — Check 5 edge
sharpness, Check 6 copy-move, Checks 7-9 stamp ink texture/boundary),
mapping every finding from render-pixel space into PDF point space.

This fills the one gap the two existing raster paths leave open:
  * ela_analyzer consumes page renders, but only for recompression,
    noise, erasure, and the shared flat-zone (Check 1/2) algorithm —
    Checks 5/6/8/9 never ran on a page render.
  * embedded_image_forensics runs the FULL image pipeline, but only on
    embedded image OBJECTS covering < 85% of a page — a scanned page's
    own full-page raster is deliberately excluded there, so its pixels
    never met Checks 5/6/8/9 either.

Check 1 (flat zones, the GENERAL any-flat-region case) is intentionally
NOT re-run here: ela_analyzer._detect_flat_zone_patches already applies
the same shared algorithm (utils/flat_zone_detection) to the same page
renders, fires on ANY texture-inconsistent flat region, and labels it
generically ("flat/uniform region inconsistent with page texture") unless
a stamp/seal ink contour is also present (Check 7's isolation, the
SPECIFIC case) — running it twice would double-count one physical finding.

Gates:
  * pdf_type routing — "scanned"/"scanned_native" route every page;
    "mixed" routes only its raster-dominated pages (same per-page test
    ela_analyzer._is_image_based_document uses: <20 chars of extractable
    text AND at least one image). native_text/image_document never route
    (byte-identical scores).
  * Born-digital gate — two-signal: a render with no scan-noise baseline
    gates the texture checks out per page (inside analyze_page_render)
    only when the page's embedded source images ALSO show no raster-
    pipeline evidence (utils/pdf_utils.page_raster_source_evidence).
  * QR zones — high-frequency QR pixels read as crisp overlay edges to
    Check 5; findings overlapping a QR zone are dropped BEFORE scoring
    (utils/pdf_utils.get_qr_zones, same exclusion every raster layer uses).

Scoring: the FOLD_* constants produce a capped magnitude the caller adds
to ela_report.anomaly_score BEFORE combine() — the same additive fold
pattern as text-stacking → pymupdf, flat-zone → ELA, and embedded-image →
ELA. No new weighted layer. Findings enter fusion as their own
"scanned_pixel" layer via extra_findings, so they cross-validate with
(and can be contradicted by) other layers rather than overriding them.
"""

import fitz
import numpy as np
from PIL import Image

from analyzers.image_document_analyzer import (
    ImageDocumentAnalyzer, score_anomalies,
)
from utils.pdf_utils import (
    get_qr_zones, bbox_overlaps_qr_zone, page_raster_source_evidence,
)

# pdf_types whose pages carry scan-noise raster content worth routing.
ROUTED_PDF_TYPES = ("scanned", "scanned_native", "mixed")

# Same per-page raster-dominance test as ela_analyzer._is_image_based_document.
MIXED_PAGE_MAX_TEXT_CHARS = 20

# 150 DPI matches ela_analyzer's low-DPI sweep — a letter page renders to
# ~1275x1650 px, comfortably inside the pixel range the image pipeline's
# block/cell checks were tuned for on direct uploads.
RENDER_DPI = 150

# Fold into the ELA layer's score (caller applies) — mirrors
# utils/embedded_image_forensics's constants: each routed page contributes
# its own CHECK_POINTS-weighted score times the multiplier, with the
# established cap for a folded sub-check riding another layer's weight.
FOLD_MULTIPLIER = 0.6
FOLD_CAP        = 80

# Short, specific box/finding labels per evidence check — prefixed so
# they're never confused with embedded-image or page-level ELA findings.
CHECK_LABELS = {
    "check5_edge_sharpness": "Scanned Page: Sharp Overlay Edge",
    "check6_copy_move":      "Scanned Page: Cloned Region",
    "check8_stamp_texture":  "Scanned Page: Flat Ink Fill",
    "check9_stamp_boundary": "Scanned Page: Cutout Boundary",
}


def analyze_scanned_pages(pdf_path: str, pdf_type: str) -> dict:
    """
    Returns {
      "findings":  list of pre-normalized fusion dicts (layer
                   "scanned_pixel", 0-indexed page, bbox in PDF points,
                   plus "label" for the annotation box),
      "signals":   human-readable signal strings,
      "fold_score": capped magnitude to add to the ELA layer's score,
      "pages_analyzed": int, "pages_skipped": int,
    }
    Never raises past a page — an unrenderable page is skipped.
    """
    findings = []
    signals = []
    raw_fold = 0.0
    pages_analyzed = 0
    pages_skipped = 0

    if pdf_type not in ROUTED_PDF_TYPES:
        return {"findings": [], "signals": [], "fold_score": 0,
                "pages_analyzed": 0, "pages_skipped": 0}

    analyzer = ImageDocumentAnalyzer()
    pts_scale = 72.0 / RENDER_DPI
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Mixed docs: text pages belong to the text layers — route only
            # raster-dominated pages.
            if pdf_type == "mixed":
                text = (page.get_text("text") or "").strip()
                if len(text) >= MIXED_PAGE_MAX_TEXT_CHARS or not page.get_images():
                    pages_skipped += 1
                    continue

            try:
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
                rgb = np.asarray(img)
                gray = np.asarray(img.convert("L"), dtype=np.float64)
                # Corroborate the born-digital gate against the page's
                # embedded SOURCE images (original bytes), not just the
                # freshly-resampled render — see analyze_page_render.
                evidence, basis = page_raster_source_evidence(page, doc)
                result = analyzer.analyze_page_render(
                    rgb, gray,
                    raster_source_evidence=evidence,
                    evidence_basis=basis,
                )
            except Exception:
                pages_skipped += 1
                continue

            pages_analyzed += 1
            if result["is_born_digital"]:
                # One explanatory line per gated page, not per check.
                signals.append(
                    f"Page {page_num + 1}: {result['signals'][0]}"
                    if result["signals"] else
                    f"Page {page_num + 1}: born-digital render — texture "
                    f"checks gated out."
                )
                # Check 6 may still have produced anomalies; fall through.

            # Drop QR-zone hits BEFORE scoring so the fold magnitude matches
            # the surviving findings.
            qr_zones = get_qr_zones(page, doc)
            kept = []
            for a in result["anomalies"]:
                x, y, w, h = a.bbox
                bbox_pts = (x * pts_scale, y * pts_scale,
                            (x + w) * pts_scale, (y + h) * pts_scale)
                if bbox_overlaps_qr_zone(bbox_pts, qr_zones):
                    continue
                kept.append((a, bbox_pts))

            if not kept:
                continue

            page_score, _per_check = score_anomalies([a for a, _ in kept])
            raw_fold += page_score * FOLD_MULTIPLIER

            for a, bbox_pts in kept:
                label = CHECK_LABELS.get(a.evidence_check,
                                         "Scanned Page: Anomaly")
                findings.append({
                    "layer": "scanned_pixel",
                    "page": page_num,
                    "bbox": bbox_pts,
                    "line_num": None,
                    "text": f"scanned-page render: {a.detail or a.type}",
                    "score": float(a.confidence),
                    "label": label,
                    "evidence_check": a.evidence_check,
                    "raw": a,
                })
                signals.append(
                    f"Page {page_num + 1} at "
                    f"({bbox_pts[0]:.0f},{bbox_pts[1]:.0f})-"
                    f"({bbox_pts[2]:.0f},{bbox_pts[3]:.0f}): "
                    f"{label.split(': ', 1)[1]} — {a.detail or a.type} "
                    f"(confidence {a.confidence:.2f}) "
                    f"[image-pipeline check on the page render — distinct "
                    f"from page-level ELA and embedded-image findings]"
                )
    finally:
        doc.close()

    return {
        "findings": findings,
        "signals": signals,
        "fold_score": min(FOLD_CAP, raw_fold),
        "pages_analyzed": pages_analyzed,
        "pages_skipped": pages_skipped,
    }
