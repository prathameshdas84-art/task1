"""
Embedded-image forensics for the PDF pipeline.

Extracts every embedded raster image XObject from a PDF (the actual
image bytes via fitz doc.extract_image — NOT a page render) and runs
analyzers/image_document_analyzer.ImageDocumentAnalyzer over each one,
mapping every finding's bbox from the image's own pixel space into the
parent PDF page's point space via the image's placement rect.

This is deliberately DISTINCT from ela_analyzer's raster handling:
  * ela_analyzer consumes page.get_pixmap() RENDERS (whole pages at
    150/300/600 DPI) — recompression, noise, erasure, and flat-zone
    checks over the composed page.
  * This module consumes the embedded image OBJECTS themselves — for a
    DCT-encoded image that's the original JPEG stream, quantization
    tables intact, so the image pipeline's compression-history checks
    see the true bytes rather than a re-render.

Gates (which images qualify):
  * MIN_DIMENSION_PX — intrinsic short side must be >= 100 px. Below
    that sit icons, bullets, and rule graphics; a photo/stamp/signature
    scan with meaningful forensic content is never that small, and the
    analyzer's block/cell-based checks (16px flat blocks, 32px sharp
    cells) barely have statistics to work with under it anyway.
  * MAX_PAGE_COVERAGE — a placement covering >= 85% of its page is the
    page itself (a scanned-page background raster). That is exactly the
    input class ela_analyzer already owns; analyzing it here too would
    run two smoothing detectors over the same pixels and double-count
    (the image pipeline's own routing note warns against precisely this).
  * The analyzer's internal born-digital gate then zeroes flat vector
    logos and similar decoration (e.g. a bank-statement header logo)
    without any special-casing here — verified against a real statement.

Images are deduplicated by content hash: template graphics repeat as
separate XObjects on every page (13 identical logos = 1 analysis), and
each finding is still mapped to every placement of that content.

Scoring: the FOLD_* constants produce a capped magnitude the caller
adds to ela_report.anomaly_score BEFORE combine() — the same additive
fold pattern as text-stacking → pymupdf and flat-zone → ELA. No new
weighted layer. check4's categorical double-compression points are
EXCLUDED from the fold for embedded images: a PDF producer routinely
recompresses images at embed time, so multiple-save evidence is
expected there, not tampering evidence (it still appears in signals).
"""

import hashlib
import os
import tempfile

import fitz

from analyzers.image_document_analyzer import ImageDocumentAnalyzer

MIN_DIMENSION_PX  = 100    # intrinsic short side below this = icon/decoration
MAX_PAGE_COVERAGE = 0.85   # placement covering >= this fraction of the page
                           # = scanned-page raster, ela_analyzer's domain

# Fold into the ELA layer's score (caller applies): each analyzed image
# contributes its own anomaly_score (already per-check weighted and capped
# by the image pipeline, minus the categorical check4 points) times the
# multiplier, with a total cap so one busy image can't dominate. Cap 80
# mirrors TEXT_STACKING_SCORE_CAP — the established ceiling for a folded
# sub-check riding another layer's weight.
FOLD_MULTIPLIER = 0.6
FOLD_CAP        = 80

# Short, specific box/finding labels per evidence check (Part 1 labeling
# convention) — prefixed so they're never confused with page-level ELA.
CHECK_LABELS = {
    "check1_local_variance": "Embedded Image: Flat Region",
    "check5_edge_sharpness": "Embedded Image: Sharp Overlay",
    "check6_copy_move":      "Embedded Image: Cloned Region",
    "check8_stamp_texture":  "Embedded Image: Flat Ink Fill",
    "check9_stamp_boundary": "Embedded Image: Cutout Edge",
}


def analyze_embedded_images(pdf_path: str) -> dict:
    """
    Returns {
      "findings":  list of pre-normalized fusion dicts (layer
                   "embedded_image", 0-indexed page, bbox in PDF points,
                   plus "label" for the annotation box),
      "signals":   human-readable signal strings,
      "fold_score": capped magnitude to add to the ELA layer's score,
      "images_analyzed": int, "images_skipped": int,
    }
    Never raises — an unreadable/undecodable image is skipped.
    """
    findings = []
    signals = []
    raw_fold = 0.0
    images_analyzed = 0
    images_skipped = 0

    doc = fitz.open(pdf_path)
    try:
        # content-hash -> analyzed report (template images repeat per page)
        reports_by_hash = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_area = page.rect.width * page.rect.height

            for item in page.get_images(full=True):
                xref = item[0]
                try:
                    info = doc.extract_image(xref)
                except Exception:
                    images_skipped += 1
                    continue
                width, height = info.get("width", 0), info.get("height", 0)
                if min(width, height) < MIN_DIMENSION_PX:
                    images_skipped += 1
                    continue

                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                rects = [
                    r for r in rects
                    if page_area > 0
                    and (r.width * r.height) / page_area < MAX_PAGE_COVERAGE
                    and r.width > 0 and r.height > 0
                ]
                if not rects:
                    images_skipped += 1
                    continue

                digest = hashlib.md5(info["image"]).hexdigest()
                if digest not in reports_by_hash:
                    report = _analyze_image_bytes(info["image"], info.get("ext", "png"))
                    reports_by_hash[digest] = report
                    if report is not None:
                        images_analyzed += 1
                    else:
                        images_skipped += 1
                report = reports_by_hash[digest]
                if report is None:
                    continue

                # Fold contribution counts once per unique image content,
                # on its FIRST placement — repeated template placements of
                # the same pixels are one piece of evidence, not N.
                if not getattr(report, "_fold_counted", False):
                    raw_fold += _foldable_score(report) * FOLD_MULTIPLIER
                    report._fold_counted = True

                for anomaly in report.anomalies:
                    for rect in rects:
                        bbox_pts = _map_px_bbox_to_page(
                            anomaly.bbox, width, height, rect
                        )
                        label = CHECK_LABELS.get(
                            anomaly.evidence_check, "Embedded Image: Anomaly"
                        )
                        findings.append({
                            "layer": "embedded_image",
                            "page": page_num,
                            "bbox": bbox_pts,
                            "line_num": None,
                            "text": f"embedded image ({width}x{height}px): {anomaly.detail}",
                            "score": float(anomaly.confidence),
                            "label": label,
                            "evidence_check": anomaly.evidence_check,
                            "raw": anomaly,
                        })
                        signals.append(
                            f"Embedded image on page {page_num + 1} at "
                            f"({bbox_pts[0]:.0f},{bbox_pts[1]:.0f})-"
                            f"({bbox_pts[2]:.0f},{bbox_pts[3]:.0f}): "
                            f"{label.split(': ', 1)[1]} — {anomaly.detail} "
                            f"(confidence {anomaly.confidence:.2f}) "
                            f"[distinct from page-level ELA — this is the "
                            f"embedded image object's own pixels]"
                        )
    finally:
        doc.close()

    return {
        "findings": findings,
        "signals": signals,
        "fold_score": min(FOLD_CAP, raw_fold),
        "images_analyzed": images_analyzed,
        "images_skipped": images_skipped,
    }


def _analyze_image_bytes(image_bytes: bytes, ext: str):
    """Run ImageDocumentAnalyzer over raw embedded-image bytes via a temp
    file (its entry point takes a path). Returns the report, or None if
    the bytes can't be analyzed."""
    fd, tmp = tempfile.mkstemp(suffix=f".{ext or 'png'}")
    try:
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(image_bytes)
        return ImageDocumentAnalyzer().analyze(tmp)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _foldable_score(report) -> float:
    """The image's anomaly_score minus the categorical double-compression
    points — recompression at PDF-embed time is expected, so check4 stays
    a signal, never folded score, for embedded images."""
    per_check = (report.metrics or {}).get("per_check_scores", {})
    if per_check:
        return float(sum(
            v for k, v in per_check.items() if k != "check4_double_compression"
        ))
    return float(report.anomaly_score)


def _map_px_bbox_to_page(bbox_px: tuple, img_w: int, img_h: int,
                         rect: "fitz.Rect") -> tuple:
    """Map an (x, y, w, h) bbox in the IMAGE's pixel space onto the PDF
    page's point space through the image's placement rect (axis-aligned;
    fitz's get_image_rects already resolves the placement transform).
    Both spaces are top-left-origin with y increasing downward — the same
    convention location_highlighter draws in."""
    x, y, w, h = bbox_px
    sx = rect.width / max(img_w, 1)
    sy = rect.height / max(img_h, 1)
    return (
        float(rect.x0 + x * sx),
        float(rect.y0 + y * sy),
        float(rect.x0 + (x + w) * sx),
        float(rect.y0 + (y + h) * sy),
    )
