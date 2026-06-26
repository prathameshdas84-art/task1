"""
Layer 6 — PyMuPDF Deep Analysis
Detects hidden overlays, image insertions, and character spacing anomalies.
Uses PyMuPDF at full capability for pixel-level forensic analysis.
"""

import fitz
import hashlib
import statistics
from dataclasses import dataclass, field

from pdf_utils import get_qr_zones

# Scoring constants
WHITE_RECT_SCORE_PER_REGION  = 25
WHITE_RECT_SCORE_CAP         = 70
IMAGE_OVERLAY_SCORE_PER_ITEM = 20
IMAGE_OVERLAY_SCORE_CAP      = 60
CHAR_ANOMALY_SCORE_PER_ITEM  = 5
CHAR_ANOMALY_SCORE_CAP       = 40
CHAR_SPACING_Z_THRESHOLD     = 4.0
WHITE_FILL_THRESHOLD         = 0.85  # RGB values above this = white/near-white
MIN_CHARS_FOR_SPACING_CHECK  = 4
MIN_WIDTHS_FOR_SPACING_CHECK = 3

# A vector rectangle or raster image covering most of the page is a
# background fill / letterhead template, not a cover-and-retype edit —
# without this guard, any document with a plain white page background
# (extremely common — Word, LibreOffice, Canva all draw one) would have
# its background rect intersect every text block and always max out
# WHITE_RECT_SCORE_CAP. Only small, targeted overlays count as suspicious.
MAX_OVERLAY_PAGE_AREA_FRACTION = 0.5

# Decorative panel/card backgrounds (common on bank statements, payslips)
# are well under 50% of the page but still far larger than a targeted
# cover-and-retype box — real-world testing found a 283x148pt summary-panel
# background alone produced 37 false-positive "white rect" hits on a clean
# bank statement. A genuine cover-up box is sized to the field/line it
# hides (roughly one to a few lines of text), not a multi-line panel.
MAX_OVERLAY_ABS_AREA_PT2  = 6000  # ~a 200x30pt box, generous for one field+value
MIN_OVERLAY_DIMENSION_PT  = 6     # excludes hairline table border/gutter rects

# Ghost-text detection: two different, non-empty text blocks occupying the
# same physical space is impossible in a legitimately laid-out document.
# A low-effort forgery pastes replacement text directly over the original
# without removing the original run, leaving both in the content stream.
# GHOST_TEXT_MAX_BLOCK_AREA_FRACTION excludes diagonal "CONFIDENTIAL"/
# "DRAFT" watermark stamps and background labels, which legitimately span
# a large fraction of the page and aren't a targeted paste-over.
GHOST_TEXT_OVERLAP_FRACTION         = 0.3
GHOST_TEXT_MAX_BLOCK_AREA_FRACTION  = 0.3
GHOST_TEXT_SCORE_PER_REGION         = 35
GHOST_TEXT_SCORE_CAP                = 70
MIN_GHOST_TEXT_LEN                  = 2

# Coordinate-overwrite detection: two different span texts at the exact same
# bbox (1pt precision) — a very precise paste-over that block-level ghost_text
# detection misses when the injected span is shorter than MIN_GHOST_TEXT_LEN
# or falls within the same block.
COORD_OVERWRITE_SCORE_PER_FINDING = 25
COORD_OVERWRITE_SCORE_CAP          = 50

# A cover-and-retype edit doesn't have to use a white box — on a colored
# letterhead/panel background, an editor will fill with whatever color
# matches the surrounding page so the patch is invisible. We detect that by
# sampling pixels just outside the rectangle's own edges and comparing.
LOCAL_BG_SAMPLE_DPI               = 150
LOCAL_BG_COLOR_DISTANCE_THRESHOLD = 0.15  # euclidean distance in 0-1 RGB space
LOCAL_BG_MARGIN_PT                = 10    # how far outside the rect to probe

# Z-order check: a table row background is painted BEFORE the text that
# sits on it (lower content-stream position = earlier seqno), so the text
# remains visible. A cover-and-retype edit pastes its box AFTER the text
# it's hiding (higher seqno), burying it. Geometric overlap alone can't
# tell these apart — only stream order can.
ROW_PATTERN_SIZE_TOLERANCE_PT = 2.0   # rects within this size delta count as "same size"
ROW_PATTERN_MIN_COUNT         = 3     # need this many same-sized rects to call it a pattern
ROW_PATTERN_INTERVAL_CV_MAX   = 0.3   # coefficient of variation of vertical spacing
COVERAGE_OVERLAP_FRACTION_MIN = 0.4   # fraction of a text span's own area that must be
                                       # inside the rect to count as "covered" (excludes
                                       # adjacent-row font ascender/descender edge-bleed)


@dataclass
class OverlayRegion:
    page: int
    bbox: tuple          # (x0, y0, x1, y1) in PDF points
    overlay_type: str    # "covering_rect" | "image_overlay" | "char_spacing" | "ghost_text"
    reason: str


@dataclass
class PyMuPDFReport:
    pages_analyzed: int
    overlay_regions: list[OverlayRegion]
    anomaly_score: int
    signals: list[str]


class PyMuPDFAnalyzer:

    def _is_targeted_overlay_size(self, rect: "fitz.Rect", page_area: float) -> bool:
        """
        True only for rects sized like a deliberate cover-and-retype box —
        small in both absolute and page-relative terms, and not a hairline
        border/gutter stroke. See MAX_OVERLAY_* constants above for why.
        """
        if rect.width < MIN_OVERLAY_DIMENSION_PT or rect.height < MIN_OVERLAY_DIMENSION_PT:
            return False
        area = rect.width * rect.height
        if area > MAX_OVERLAY_ABS_AREA_PT2:
            return False
        if page_area > 0 and area / page_area > MAX_OVERLAY_PAGE_AREA_FRACTION:
            return False
        return True

    def _sample_region_color(self, page: "fitz.Page", rect: "fitz.Rect"):
        """Average RGB (0-1 range) of the rendered pixels inside `rect`."""
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            return None
        zoom = LOCAL_BG_SAMPLE_DPI / 72
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
        except Exception:
            return None
        if pix.width == 0 or pix.height == 0:
            return None
        samples = pix.samples
        n = pix.n
        total_r = total_g = total_b = count = 0
        step = max(1, (pix.width * pix.height) // 2000)  # subsample for speed
        for i in range(0, len(samples) - n + 1, n * step):
            total_r += samples[i]
            total_g += samples[i + 1]
            total_b += samples[i + 2]
            count += 1
        if count == 0:
            return None
        return (total_r / count / 255.0, total_g / count / 255.0, total_b / count / 255.0)

    def _is_covering_rectangle(self, drawing: dict, page: "fitz.Page", page_area: float):
        """
        True if a filled rectangle is plausibly a cover-and-retype box:
        solid white/near-white (the classic case), OR solid and closely
        matching the LOCAL background color sampled just outside its own
        edges (a colored letterhead/panel background reused to hide text).
        Returns (is_covering: bool, reason: str).
        """
        rect = drawing.get("rect")
        color = drawing.get("fill")
        if rect is None or color is None:
            return False, ""
        color_vals = color[:3] if len(color) >= 3 else []
        if not color_vals:
            return False, ""

        drawing_rect = fitz.Rect(rect)
        if not self._is_targeted_overlay_size(drawing_rect, page_area):
            return False, ""  # background fill / decorative panel, not a targeted cover-up

        if all(c > WHITE_FILL_THRESHOLD for c in color_vals):
            return True, (
                f"White rectangle overlapping text at "
                f"{tuple(round(v, 1) for v in rect)} — "
                f"classic cover-and-retype edit pattern"
            )

        # Not white — check whether it matches the page's LOCAL background
        # color, probed just outside the rectangle's own edges so the probe
        # never samples the rectangle's own fill.
        margin = LOCAL_BG_MARGIN_PT
        page_rect = page.rect
        probes = [
            fitz.Rect(drawing_rect.x0 - margin, drawing_rect.y0, drawing_rect.x0, drawing_rect.y1),
            fitz.Rect(drawing_rect.x1, drawing_rect.y0, drawing_rect.x1 + margin, drawing_rect.y1),
            fitz.Rect(drawing_rect.x0, drawing_rect.y0 - margin, drawing_rect.x1, drawing_rect.y0),
        ]
        for probe in probes:
            probe = probe & page_rect
            if probe.is_empty or probe.width <= 0 or probe.height <= 0:
                continue
            bg_color = self._sample_region_color(page, probe)
            if bg_color is None:
                continue
            distance = sum((a - b) ** 2 for a, b in zip(color_vals, bg_color)) ** 0.5
            if distance < LOCAL_BG_COLOR_DISTANCE_THRESHOLD:
                return True, (
                    f"Rectangle at {tuple(round(v, 1) for v in rect)} filled with "
                    f"color {tuple(round(c, 2) for c in color_vals)} matches the "
                    f"local page background ({tuple(round(c, 2) for c in bg_color)}) "
                    f"— possible colored cover-and-retype edit"
                )
        return False, ""

    def _get_text_seqnos(self, page: "fitz.Page") -> list:
        """List of (bbox: fitz.Rect, seqno: int) for every text span on the page."""
        spans = []
        try:
            for trace in page.get_texttrace():
                bbox = trace.get("bbox")
                seqno = trace.get("seqno")
                if bbox is not None and seqno is not None:
                    spans.append((fitz.Rect(bbox), seqno))
        except Exception:
            pass
        return spans

    def _was_drawn_over_existing_text(self, rect: "fitz.Rect", drawing_seqno, text_spans: list) -> bool:
        """
        True if at least one text span under this rect was already on the
        page (lower seqno) before the rect was drawn — i.e. the rect was
        painted on top of existing text (cover-up). False if the rect came
        first and text was written on top of it afterwards (normal table/
        panel background rendering), or if seqno info is unavailable.
        """
        if drawing_seqno is None:
            return True  # no stream-order info — fall back to flagging (old behavior)
        for span_rect, seqno in text_spans:
            if seqno >= drawing_seqno or not rect.intersects(span_rect):
                continue
            span_area = span_rect.width * span_rect.height
            if span_area <= 0:
                continue
            overlap = rect & span_rect
            overlap_area = overlap.width * overlap.height
            if overlap_area / span_area >= COVERAGE_OVERLAP_FRACTION_MIN:
                return True
        return False

    def _find_row_pattern_rects(self, candidates: list) -> set:
        """
        Identify rects that are part of a repeating same-size, regularly-
        spaced vertical pattern (alternating table row backgrounds) and
        should be excluded as a group regardless of z-order ambiguity.
        Returns a set of id(rect) for excluded rects.
        """
        groups = {}
        for rect in candidates:
            key = (round(rect.width / ROW_PATTERN_SIZE_TOLERANCE_PT),
                   round(rect.height / ROW_PATTERN_SIZE_TOLERANCE_PT))
            groups.setdefault(key, []).append(rect)

        excluded = set()
        for group in groups.values():
            if len(group) < ROW_PATTERN_MIN_COUNT:
                continue
            ys = sorted(r.y0 for r in group)
            intervals = [b - a for a, b in zip(ys, ys[1:])]
            if not intervals:
                continue
            mean_interval = statistics.mean(intervals)
            if mean_interval <= 0:
                continue
            stdev_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
            cv = stdev_interval / mean_interval
            if cv <= ROW_PATTERN_INTERVAL_CV_MAX:
                excluded.update(id(r) for r in group)
        return excluded

    def _detect_overlapping_text(self, page: "fitz.Page", page_num: int, page_area: float) -> list:
        """
        Flag pairs of different, non-empty text blocks whose bounding boxes
        substantially overlap — "ghost text" left behind when a forger
        pastes replacement text directly over the original instead of
        removing it first.
        """
        blocks = page.get_text("blocks")
        regions = []

        for i, block_a in enumerate(blocks):
            text_a = block_a[4].strip() if len(block_a) > 4 else ""
            if len(text_a) < MIN_GHOST_TEXT_LEN:
                continue
            ax0, ay0, ax1, ay1 = block_a[:4]
            area_a = (ax1 - ax0) * (ay1 - ay0)
            if page_area > 0 and area_a / page_area > GHOST_TEXT_MAX_BLOCK_AREA_FRACTION:
                continue  # watermark/background label, not a targeted paste-over

            for block_b in blocks[i + 1:]:
                text_b = block_b[4].strip() if len(block_b) > 4 else ""
                if len(text_b) < MIN_GHOST_TEXT_LEN or text_b == text_a:
                    continue
                bx0, by0, bx1, by1 = block_b[:4]
                area_b = (bx1 - bx0) * (by1 - by0)
                if page_area > 0 and area_b / page_area > GHOST_TEXT_MAX_BLOCK_AREA_FRACTION:
                    continue

                x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
                y_overlap = max(0, min(ay1, by1) - max(ay0, by0))
                if x_overlap <= 5 or y_overlap <= 5:
                    continue

                overlap_area = x_overlap * y_overlap
                smaller_area = min(area_a, area_b)
                if smaller_area <= 0:
                    continue
                overlap_pct = overlap_area / smaller_area
                if overlap_pct > GHOST_TEXT_OVERLAP_FRACTION:
                    regions.append(OverlayRegion(
                        page=page_num,
                        bbox=(min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)),
                        overlay_type="ghost_text",
                        reason=(
                            f"Two different text blocks occupy the same location "
                            f"({overlap_pct*100:.0f}% overlap): "
                            f"'{text_a[:30]}' vs '{text_b[:30]}' — "
                            f"possible text pasted over original (ghost text)"
                        ),
                    ))
        return regions

    def _bbox_overlaps(self, a: tuple, b: tuple) -> bool:
        """True if two (x0, y0, x1, y1) bboxes share any area."""
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    def _detect_coordinate_overwrites(self, page: "fitz.Page", page_num: int) -> list:
        """
        Flag spans where two DIFFERENT texts share the exact same bounding
        box (rounded to 1pt to absorb anti-aliasing drift) — a strong signal
        that text was pasted directly over existing content at the span level
        without removing the original.  Block-level ghost_text detection
        catches the coarser case; this catches same-block or very-short
        overwrites that slip through MIN_GHOST_TEXT_LEN.
        """
        spatial_registry = {}  # bbox_key → sha256 of first text seen there
        findings = []

        # Use "dict" mode (not rawdict) — "dict" spans carry a "text" key
        # directly; rawdict spans only expose "chars" without a top-level text.
        textdict = page.get_text("dict")
        for block in textdict.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    # Round to 1pt so floating-point anti-aliasing noise
                    # in identical glyphs doesn't create false mismatches.
                    bbox_key = tuple(round(c, 1) for c in span["bbox"])

                    text_hash = hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest()

                    if bbox_key in spatial_registry:
                        if spatial_registry[bbox_key] != text_hash:
                            findings.append(OverlayRegion(
                                page=page_num,
                                bbox=span["bbox"],
                                overlay_type="coordinate_overwrite",
                                reason=(
                                    "Different text content at identical coordinates — "
                                    "possible paste-over injection"
                                ),
                            ))
                    else:
                        spatial_registry[bbox_key] = text_hash

        return findings

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
            text_spans = self._get_text_seqnos(page)
            qr_zones = get_qr_zones(page, doc)
            geometric_candidates = []  # (drawing, drawing_rect, cover_reason)
            for drawing in page.get_drawings():
                is_covering, cover_reason = self._is_covering_rectangle(drawing, page, page_area)
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

            row_pattern_excluded = self._find_row_pattern_rects(
                [c[1] for c in geometric_candidates]
            )
            for drawing, drawing_rect, cover_reason in geometric_candidates:
                if id(drawing_rect) in row_pattern_excluded:
                    continue  # repeating alternating-row table background
                if not self._was_drawn_over_existing_text(drawing_rect, drawing.get("seqno"), text_spans):
                    continue  # background painted before text = normal table rendering
                all_regions.append(OverlayRegion(
                    page=page_num,
                    bbox=tuple(drawing_rect),
                    overlay_type="covering_rect",
                    reason=cover_reason,
                ))

            # CHECK 1b — Different text blocks overlapping each other (ghost text)
            ghost_regions = self._detect_overlapping_text(page, page_num, page_area)
            all_regions.extend(ghost_regions)

            # CHECK 1c — Coordinate-hash duplicate detection at span level.
            # Detects different text strings sharing the exact same bbox —
            # a paste-over that block-level ghost_text can miss (same block,
            # or spans shorter than MIN_GHOST_TEXT_LEN).  Skip any candidate
            # whose bbox already overlaps a ghost_text finding to avoid
            # double-counting the same forgery at two granularities.
            coord_candidates = self._detect_coordinate_overwrites(page, page_num)
            ghost_bboxes = [r.bbox for r in ghost_regions]
            for r in coord_candidates:
                if not any(self._bbox_overlaps(r.bbox, gb) for gb in ghost_bboxes):
                    all_regions.append(r)

            # CHECK 2 — Images overlapping text regions
            for img in page.get_images(full=True):
                for img_rect in page.get_image_rects(img[0]):
                    rect_obj = fitz.Rect(img_rect)
                    if not self._is_targeted_overlay_size(rect_obj, page_area):
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
