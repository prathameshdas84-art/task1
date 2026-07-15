import fitz
import statistics
from utils.pdf_utils import get_qr_zones
from .models import OverlayRegion, PyMuPDFReport
from .constants import (
    WHITE_RECT_SCORE_PER_REGION,
    WHITE_RECT_SCORE_CAP,
    IMAGE_OVERLAY_SCORE_PER_ITEM,
    IMAGE_OVERLAY_SCORE_CAP,
    CHAR_ANOMALY_SCORE_PER_ITEM,
    CHAR_ANOMALY_SCORE_CAP,
    CHAR_SPACING_Z_THRESHOLD,
    MIN_CHARS_FOR_SPACING_CHECK,
    MIN_WIDTHS_FOR_SPACING_CHECK,
    GHOST_TEXT_SCORE_PER_REGION,
    GHOST_TEXT_SCORE_CAP,
    COORD_OVERWRITE_SCORE_PER_FINDING,
    COORD_OVERWRITE_SCORE_CAP,
)
from .checks import (
    is_covering_rectangle,
    get_text_seqnos,
    find_row_pattern_rects,
    was_drawn_over_existing_text,
    detect_overlapping_text,
    detect_coordinate_overwrites,
    bbox_overlaps,
    is_targeted_overlay_size,
)


class PyMuPDFAnalyzer:

    def analyze(self, pdf_path: str) -> PyMuPDFReport:
        doc = fitz.open(pdf_path)
        all_regions = []
        pages_analyzed = len(doc)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text_blocks = page.get_text("blocks")
            page_area = page.rect.width * page.rect.height

            # CHECK 1 — Covering rectangles overlapping text (white, or
            # filled to match the local page background color), filtered by
            # content-stream z-order so table/panel backgrounds painted
            # BEFORE their text don't get mistaken for a paste-over box
            # drawn AFTER the text it's hiding.
            text_spans = get_text_seqnos(page)
            qr_zones = get_qr_zones(page, doc)
            geometric_candidates = []  # (drawing, drawing_rect, cover_reason)
            for drawing in page.get_drawings():
                is_covering, cover_reason = is_covering_rectangle(drawing, page, page_area)
                if not is_covering:
                    continue
                rect = drawing.get("rect")
                drawing_rect = fitz.Rect(rect)
                if any(drawing_rect.intersects(qr) for qr in qr_zones):
                    continue  # QR code area, not a cover-up
                for block in text_blocks:
                    if drawing_rect.intersects(fitz.Rect(block[:4])):
                        geometric_candidates.append((drawing, drawing_rect, cover_reason))
                        break

            row_pattern_excluded = find_row_pattern_rects(
                [c[1] for c in geometric_candidates]
            )
            for drawing, drawing_rect, cover_reason in geometric_candidates:
                if id(drawing_rect) in row_pattern_excluded:
                    continue  # repeating alternating-row table background
                if not was_drawn_over_existing_text(drawing_rect, drawing.get("seqno"), text_spans):
                    continue  # background painted before text = normal table rendering
                all_regions.append(OverlayRegion(
                    page=page_num,
                    bbox=tuple(drawing_rect),
                    overlay_type="covering_rect",
                    reason=cover_reason,
                ))

            # CHECK 1b — Different text blocks overlapping each other (ghost text)
            ghost_regions = detect_overlapping_text(page, page_num, page_area)
            all_regions.extend(ghost_regions)

            # CHECK 1c — Coordinate-hash duplicate detection at span level.
            # Detects different text strings sharing the exact same bbox —
            # a paste-over that block-level ghost_text can miss (same block,
            # or spans shorter than MIN_GHOST_TEXT_LEN).  Skip any candidate
            # whose bbox already overlaps a ghost_text finding to avoid
            # double-counting the same forgery at two granularities.
            coord_candidates = detect_coordinate_overwrites(page, page_num)
            ghost_bboxes = [r.bbox for r in ghost_regions]
            for r in coord_candidates:
                if not any(bbox_overlaps(r.bbox, gb) for gb in ghost_bboxes):
                    all_regions.append(r)

            # CHECK 2 — Images overlapping text regions
            for img in page.get_images(full=True):
                for img_rect in page.get_image_rects(img[0]):
                    rect_obj = fitz.Rect(img_rect)
                    if not is_targeted_overlay_size(rect_obj, page_area):
                        continue  # background/letterhead image, not a targeted paste-over
                    for block in text_blocks:
                        if rect_obj.intersects(fitz.Rect(block[:4])):
                            all_regions.append(OverlayRegion(
                                page=page_num,
                                bbox=tuple(img_rect),
                                overlay_type="image_overlay",
                                reason=(
                                    f"Image overlapping text at "
                                    f"{tuple(round(v,1) for v in img_rect)} — "
                                    f"possible image pasted over original text"
                                ),
                            ))
                            break

            # CHECK 3 — Character-level spacing anomalies
            rawdict = page.get_text("rawdict")
            for block in rawdict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        chars = span.get("chars", [])
                        if len(chars) < MIN_CHARS_FOR_SPACING_CHECK:
                            continue
                        widths = []
                        for i in range(len(chars) - 1):
                            w = chars[i+1]["origin"][0] - chars[i]["origin"][0]
                            if w > 0:
                                widths.append(w)
                        if len(widths) < MIN_WIDTHS_FOR_SPACING_CHECK:
                            continue
                        mean_w = statistics.mean(widths)
                        std_w  = statistics.stdev(widths) if len(widths) > 1 else 0
                        if mean_w <= 0:
                            continue
                        for i, w in enumerate(widths):
                            z = abs(w - mean_w) / max(std_w, 0.01)
                            if z >= CHAR_SPACING_Z_THRESHOLD:
                                char_bbox = chars[i].get("bbox", (0, 0, 0, 0))
                                all_regions.append(OverlayRegion(
                                    page=page_num,
                                    bbox=tuple(char_bbox),
                                    overlay_type="char_spacing",
                                    reason=(
                                        f"Character spacing anomaly: width {w:.2f} "
                                        f"vs mean {mean_w:.2f} (z={z:.1f}) — "
                                        f"possible character replacement"
                                    ),
                                ))

        doc.close()

        # Score and signals
        covering_rects   = [r for r in all_regions if r.overlay_type == "covering_rect"]
        img_overlays     = [r for r in all_regions if r.overlay_type == "image_overlay"]
        char_anomalies   = [r for r in all_regions if r.overlay_type == "char_spacing"]
        ghost_text       = [r for r in all_regions if r.overlay_type == "ghost_text"]
        coord_overwrites = [r for r in all_regions if r.overlay_type == "coordinate_overwrite"]

        signals = []
        score   = 0

        if covering_rects:
            signals.append(
                f"{len(covering_rects)} covering rectangle(s) overlapping text "
                f"(white or background-color-matched fill) — "
                f"classic cover-and-retype edit technique detected"
            )
            score += min(WHITE_RECT_SCORE_CAP,
                         len(covering_rects) * WHITE_RECT_SCORE_PER_REGION)

        if img_overlays:
            signals.append(
                f"{len(img_overlays)} image(s) overlapping text regions — "
                f"possible image pasted over original text"
            )
            score += min(IMAGE_OVERLAY_SCORE_CAP,
                         len(img_overlays) * IMAGE_OVERLAY_SCORE_PER_ITEM)

        if char_anomalies:
            signals.append(
                f"{len(char_anomalies)} character spacing anomaly(s) — "
                f"possible individual character replacement detected"
            )
            score += min(CHAR_ANOMALY_SCORE_CAP,
                         len(char_anomalies) * CHAR_ANOMALY_SCORE_PER_ITEM)

        if ghost_text:
            signals.append(
                f"{len(ghost_text)} overlapping text block(s) — "
                f"different texts at the same coordinates, "
                f"possible text pasted over original"
            )
            score += min(GHOST_TEXT_SCORE_CAP,
                         len(ghost_text) * GHOST_TEXT_SCORE_PER_REGION)

        if coord_overwrites:
            signals.append(
                f"{len(coord_overwrites)} coordinate-overwrite(s) detected — "
                f"different text at identical span coordinates, "
                f"possible paste-over injection"
            )
            score += min(COORD_OVERWRITE_SCORE_CAP,
                         len(coord_overwrites) * COORD_OVERWRITE_SCORE_PER_FINDING)

        if not any([covering_rects, img_overlays, char_anomalies, ghost_text, coord_overwrites]):
            signals.append(
                "PyMuPDF deep analysis passed — "
                "no hidden overlays or character anomalies detected"
            )

        return PyMuPDFReport(
            pages_analyzed=pages_analyzed,
            overlay_regions=all_regions[:20],
            anomaly_score=min(100, score),
            signals=signals,
        )
