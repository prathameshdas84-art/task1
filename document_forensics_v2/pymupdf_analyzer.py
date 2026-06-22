"""
Layer 6 — PyMuPDF Deep Analysis
Detects hidden overlays, image insertions, and character spacing anomalies.
Uses PyMuPDF at full capability for pixel-level forensic analysis.
"""

import fitz
import statistics
from dataclasses import dataclass, field

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

# A cover-and-retype edit doesn't have to use a white box — on a colored
# letterhead/panel background, an editor will fill with whatever color
# matches the surrounding page so the patch is invisible. We detect that by
# sampling pixels just outside the rectangle's own edges and comparing.
LOCAL_BG_SAMPLE_DPI               = 150
LOCAL_BG_COLOR_DISTANCE_THRESHOLD = 0.15  # euclidean distance in 0-1 RGB space
LOCAL_BG_MARGIN_PT                = 10    # how far outside the rect to probe


@dataclass
class OverlayRegion:
    page: int
    bbox: tuple          # (x0, y0, x1, y1) in PDF points
    overlay_type: str    # "covering_rect" | "image_overlay" | "char_spacing"
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

    def analyze(self, pdf_path: str) -> PyMuPDFReport:
        doc = fitz.open(pdf_path)
        all_regions = []
        pages_analyzed = len(doc)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text_blocks = page.get_text("blocks")
            page_area = page.rect.width * page.rect.height

            # CHECK 1 — Covering rectangles overlapping text (white, or
            # filled to match the local page background color)
            for drawing in page.get_drawings():
                is_covering, cover_reason = self._is_covering_rectangle(drawing, page, page_area)
                if not is_covering:
                    continue
                rect = drawing.get("rect")
                drawing_rect = fitz.Rect(rect)
                for block in text_blocks:
                    if drawing_rect.intersects(fitz.Rect(block[:4])):
                        all_regions.append(OverlayRegion(
                            page=page_num,
                            bbox=tuple(rect),
                            overlay_type="covering_rect",
                            reason=cover_reason,
                        ))
                        break

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
        covering_rects = [r for r in all_regions if r.overlay_type == "covering_rect"]
        img_overlays  = [r for r in all_regions if r.overlay_type == "image_overlay"]
        char_anomalies = [r for r in all_regions if r.overlay_type == "char_spacing"]

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

        if not any([covering_rects, img_overlays, char_anomalies]):
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
