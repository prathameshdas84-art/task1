import fitz
import hashlib
import statistics
from .models import OverlayRegion
from .constants import (
    MIN_OVERLAY_DIMENSION_PT,
    MAX_OVERLAY_ABS_AREA_PT2,
    MAX_OVERLAY_PAGE_AREA_FRACTION,
    LOCAL_BG_SAMPLE_DPI,
    WHITE_FILL_THRESHOLD,
    LOCAL_BG_MARGIN_PT,
    LOCAL_BG_COLOR_DISTANCE_THRESHOLD,
    COVERAGE_OVERLAP_FRACTION_MIN,
    ROW_PATTERN_SIZE_TOLERANCE_PT,
    ROW_PATTERN_MIN_COUNT,
    ROW_PATTERN_INTERVAL_CV_MAX,
    MIN_GHOST_TEXT_LEN,
    GHOST_TEXT_MAX_BLOCK_AREA_FRACTION,
    GHOST_TEXT_OVERLAP_FRACTION,
)


def is_targeted_overlay_size(rect: fitz.Rect, page_area: float) -> bool:
    """
    True only for rects sized like a deliberate cover-and-retype box —
    small in both absolute and page-relative terms, and not a hairline
    border/gutter stroke. See MAX_OVERLAY_* constants for why.
    """
    if rect.width < MIN_OVERLAY_DIMENSION_PT or rect.height < MIN_OVERLAY_DIMENSION_PT:
        return False
    area = rect.width * rect.height
    if area > MAX_OVERLAY_ABS_AREA_PT2:
        return False
    if page_area > 0 and area / page_area > MAX_OVERLAY_PAGE_AREA_FRACTION:
        return False
    return True


def sample_region_color(page: fitz.Page, rect: fitz.Rect) -> tuple | None:
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


def is_covering_rectangle(drawing: dict, page: fitz.Page, page_area: float) -> tuple[bool, str]:
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
    if not is_targeted_overlay_size(drawing_rect, page_area):
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
        bg_color = sample_region_color(page, probe)
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


def get_text_seqnos(page: fitz.Page) -> list:
    """List of (bbox: fitz.Rect, seqno: int) for every text span on the page."""
    spans = []
    try:
        for trace in page.get_texttrace():
            if not isinstance(trace, dict):
                continue
            bbox = trace.get("bbox")
            seqno = trace.get("seqno")
            if bbox is not None and seqno is not None:
                spans.append((fitz.Rect(bbox), seqno))
    except Exception:
        pass
    return spans


def was_drawn_over_existing_text(rect: fitz.Rect, drawing_seqno, text_spans: list) -> bool:
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


def find_row_pattern_rects(candidates: list) -> set:
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


def detect_overlapping_text(page: fitz.Page, page_num: int, page_area: float) -> list:
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


def bbox_overlaps(a: tuple, b: tuple) -> bool:
    """True if two (x0, y0, x1, y1) bboxes share any area."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def detect_coordinate_overwrites(page: fitz.Page, page_num: int) -> list:
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
