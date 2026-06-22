"""
Location Highlighter — Phase 2
Renders PDF pages and draws red boxes around suspicious lines.
Returns annotated page images for display in Streamlit.
"""

import fitz
from PIL import Image, ImageDraw, ImageFont


def _age_color_intensity(age_days: int) -> tuple:
    """
    Returns (red, green, blue) intensity multiplier based on edit age.
    Recent edits = brighter/redder. Old edits = faded.
    """
    if age_days is None:
        return (1.0, 1.0, 1.0)  # default full intensity
    if age_days == 0:
        return (1.0, 0.3, 0.3)  # bright red — edited today
    elif age_days < 7:
        return (1.0, 0.5, 0.3)  # orange-red — this week
    elif age_days < 30:
        return (1.0, 0.7, 0.4)  # orange — this month
    elif age_days < 180:
        return (0.9, 0.8, 0.5)  # yellow — within 6 months
    elif age_days < 365:
        return (0.8, 0.8, 0.6)  # faded yellow — this year
    else:
        return (0.6, 0.6, 0.6)  # gray — old edit


RENDER_DPI = 150  # page render resolution, matches the DPI used for box-coordinate scaling

# Box colors per signal source, RGB. Each layer gets a distinct color so a
# page with multiple anomaly types is still visually distinguishable.
COLOR_CONTENT = (255, 50, 50)    # red    — font/spacing anomaly (content_analyzer)
COLOR_NUMERIC = (255, 220, 0)    # yellow — numeric outlier
COLOR_ELA     = (180, 0, 255)    # purple — ELA outlier (image edit)
COLOR_WHITE_RECT    = (0, 200, 255)   # cyan    — white rect overlay (pymupdf_analyzer)
COLOR_IMAGE_OVERLAY = (255, 0, 200)   # magenta — image overlay (pymupdf_analyzer)
COLOR_OCR_SIZE  = (255, 100, 0)   # orange-red — OCR word font-size anomaly
COLOR_OCR_COLOR = (255, 0, 200)   # magenta    — OCR word color anomaly
COLOR_OCR_CONF  = (255, 140, 0)   # orange     — OCR word low-confidence anomaly

# ELA blocks are noisy on logos/dense text even in clean documents, so this
# draw filter exists to avoid flooding the page with low-confidence boxes.
# It must match the detection Z_THRESHOLD (3.0) the ELA analyzer flags at:
# the analyzer already keeps only the MAX_REGIONS_PER_PAGE (10) strongest
# outliers per page, and per-page-normalized z-scores cluster in the 3.0-3.5
# band — rarely reaching the old 4.0 cut. At 4.0 this filter silently dropped
# every ELA box on pages whose edits were real but not extreme (the "no
# purple boxes on page 2+" bug), since later pages seldom produce a z>=4
# block. Aligning it with the detection threshold means "what ELA flagged is
# what gets drawn." The ELA score used in the verdict is unaffected — this is
# purely a display filter.
ELA_BOX_MIN_ZSCORE = 3.0

BOX_PADDING        = 4   # px padding added around each drawn box
LABEL_HEIGHT        = 16  # px height of the label background strip
LABEL_CHAR_WIDTH     = 6   # px width estimate per label character (monospace-ish)
LABEL_VERTICAL_OFFSET = 18  # px the label sits above the box's top edge


class LocationHighlighter:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc      = fitz.open(pdf_path)
        self.scale    = RENDER_DPI / 72  # points to pixels

    def highlight_pages(
        self,
        suspicious_lines: list = None,
        ocr_word_anomalies: list = None,
        numeric_anomalies: list = None,
        ela_regions: list = None,
        overlay_regions: list = None,
        age_days: int = None,
    ) -> dict:
        """
        Returns dict of {page_num: PIL.Image} with red boxes drawn.
        Only returns pages that have at least one suspicious region.

        suspicious_lines:    list of SuspiciousLine from content_analyzer
        ocr_word_anomalies:  list of OCRWordAnomaly from ocr_analyzer
        numeric_anomalies:   list of NumericAnomaly from numeric_analyzer
        ela_regions:         list of ELARegion from ela_analyzer
        overlay_regions:     list of OverlayRegion from pymupdf_analyzer
        age_days:          document's last-modification age in days. Box colors
                           are blended toward red for recent edits and faded
                           for old ones, and a "Modified: …" badge is drawn in
                           the top-right corner. NOTE: a PDF stores only ONE
                           modification date for the whole file, so this age
                           applies to the entire document's last edit — it
                           cannot date individual edits.
        """
        # Blend factor applied to box base colors (recent = redder/brighter)
        age_mult = _age_color_intensity(age_days)

        # ALL individual per-layer findings are always drawn. Cross-validated
        # fusion is surfaced separately (as an additional highlighted section in
        # the UI Overview tab) and never suppresses these per-layer markings.

        # Group by page
        lines_by_page = {}
        for sl in (suspicious_lines or []):
            lines_by_page.setdefault(sl.page, []).append(sl)

        ocr_by_page = {}
        if ocr_word_anomalies:
            for r in ocr_word_anomalies:
                ocr_by_page.setdefault(r.page, []).append(r)

        numeric_by_page = {}
        if numeric_anomalies:
            for r in numeric_anomalies:
                numeric_by_page.setdefault(r.page, []).append(r)

        ela_by_page = {}
        if ela_regions:
            for r in ela_regions:
                ela_by_page.setdefault(r.page, []).append(r)

        overlay_by_page = {}
        if overlay_regions:
            for r in overlay_regions:
                overlay_by_page.setdefault(r.page, []).append(r)

        all_pages = (set(lines_by_page.keys()) |
                     set(ocr_by_page.keys()) |
                     set(numeric_by_page.keys()) |
                     set(ela_by_page.keys()) |
                     set(overlay_by_page.keys()))
        result = {}

        for page_num in sorted(all_pages):
            page     = self.doc[page_num]
            page_h   = page.rect.height  # PDF points

            # Render page to image
            mat = fitz.Matrix(self.scale, self.scale)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
            draw = ImageDraw.Draw(img)

            # Draw content layer suspicious lines (RED boxes)
            for sl in lines_by_page.get(page_num, []):
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=sl.bbox,
                    page_h_pts=page_h,
                    color=self._blend_age_color(COLOR_CONTENT, age_mult),
                    label=f"Line {sl.line_num+1}: {sl.anomalies[0][:40] if sl.anomalies else '?'}",
                    label_color=COLOR_CONTENT,
                    thickness=2,
                )

            # Draw OCR word anomalies — orange-red for size, magenta for
            # color, orange for low confidence (size/color take priority
            # in the box color when a word has more than one anomaly type).
            for r in ocr_by_page.get(page_num, []):
                if "size" in r.anomaly_types:
                    color = COLOR_OCR_SIZE
                elif "color" in r.anomaly_types:
                    color = COLOR_OCR_COLOR
                else:
                    color = COLOR_OCR_CONF
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=r.bbox,
                    page_h_pts=page_h,
                    color=color,
                    label=f"OCR:{','.join(r.anomaly_types)}",
                    label_color=color,
                    thickness=2,
                )

            # Draw numeric anomalies (YELLOW boxes)
            for r in numeric_by_page.get(page_num, []):
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=r.bbox,
                    page_h_pts=page_h,
                    color=self._blend_age_color(COLOR_NUMERIC, age_mult),
                    label=f"Numeric: z={r.z_score:.1f}",
                    label_color=COLOR_NUMERIC,
                    thickness=2,
                )

            # Draw ELA anomalies (PURPLE) — only high confidence (z >= 5.0)
            for r in ela_by_page.get(page_num, []):
                if r.z_score >= ELA_BOX_MIN_ZSCORE:  # only show strong signals
                    self._draw_box(
                        draw=draw,
                        img_size=img.size,
                        bbox=r.bbox,
                        page_h_pts=page_h,
                        color=self._blend_age_color(COLOR_ELA, age_mult),
                        label=f"ELA z={r.z_score:.1f}",
                        label_color=COLOR_ELA,
                        thickness=2,
                    )

            # Draw PyMuPDF overlay regions — CYAN for white-rect cover-ups,
            # MAGENTA for image overlays. char_spacing regions are too small
            # (single character bboxes) to usefully draw, so they're skipped.
            for r in overlay_by_page.get(page_num, []):
                if r.overlay_type == "white_rect":
                    self._draw_box(
                        draw=draw,
                        img_size=img.size,
                        bbox=r.bbox,
                        page_h_pts=page_h,
                        color=COLOR_WHITE_RECT,
                        label="White rect overlay",
                        label_color=COLOR_WHITE_RECT,
                        thickness=2,
                    )
                elif r.overlay_type == "image_overlay":
                    self._draw_box(
                        draw=draw,
                        img_size=img.size,
                        bbox=r.bbox,
                        page_h_pts=page_h,
                        color=COLOR_IMAGE_OVERLAY,
                        label="Image overlay",
                        label_color=COLOR_IMAGE_OVERLAY,
                        thickness=2,
                    )

            # Age indicator badge — top-right corner of the page. Shows the
            # document's last-modification age (whole-file, not per-edit).
            self._draw_age_badge(draw, img.size[0], age_days)

            result[page_num] = img

        self.doc.close()
        return result

    def _draw_age_badge(self, draw, img_w, age_days):
        """Draw the top-right 'Modified: …' age badge. No-op if age unknown."""
        if age_days is None:
            return
        age_label = self._format_age_label(age_days)
        badge_color = self._age_badge_color(age_days)
        badge_w = 200
        badge_h = 32
        badge_x = img_w - badge_w - 10
        badge_y = 10
        draw.rectangle(
            [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
            fill=badge_color,
            outline=(0, 0, 0),
            width=2
        )
        draw.text(
            (badge_x + 10, badge_y + 8),
            f"Modified: {age_label}",
            fill=(255, 255, 255),
            font=self._get_font(12)
        )

    # ── Age-based color helpers ─────────────────────────────────────────────────

    def _blend_age_color(self, base_color, age_mult):
        r = int(base_color[0] * age_mult[0])
        g = int(base_color[1] * age_mult[1])
        b = int(base_color[2] * age_mult[2])
        return (r, g, b)

    def _format_age_label(self, age_days):
        if age_days == 0:
            return "Today"
        elif age_days == 1:
            return "Yesterday"
        elif age_days < 7:
            return f"{age_days} days ago"
        elif age_days < 30:
            return f"{age_days // 7} weeks ago"
        elif age_days < 365:
            return f"{age_days // 30} months ago"
        else:
            return f"{age_days // 365} years ago"

    def _age_badge_color(self, age_days):
        if age_days == 0:
            return (220, 30, 30)   # bright red
        elif age_days < 7:
            return (220, 100, 30)  # orange-red
        elif age_days < 30:
            return (220, 150, 30)  # orange
        elif age_days < 365:
            return (180, 150, 30)  # yellow-brown
        else:
            return (100, 100, 100)  # gray

    def _get_font(self, size: int):
        """Load a TrueType font at the requested size, falling back to PIL's
        built-in bitmap font if no system TTF is available."""
        for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _draw_box(
        self,
        draw: ImageDraw.Draw,
        img_size: tuple,
        bbox: tuple,
        page_h_pts: float,
        color: tuple,
        label: str,
        label_color: tuple,
        thickness: int = 2,
    ):
        """
        Draw a colored rectangle on the image.
        Converts PDF points (top-left origin) to pixels (top-left origin).
        """
        x0, y0, x1, y1 = bbox

        # PDF uses top-left origin (pdfplumber) — convert to pixels directly
        px0 = int(x0 * self.scale)
        py0 = int(y0 * self.scale)
        px1 = int(x1 * self.scale)
        py1 = int(y1 * self.scale)

        # Add padding around the box
        pad = BOX_PADDING
        px0 = max(0, px0 - pad)
        py0 = max(0, py0 - pad)
        px1 = min(img_size[0], px1 + pad)
        py1 = py1 + pad

        # Draw rectangle outline
        for i in range(thickness):
            draw.rectangle(
                [px0 - i, py0 - i, px1 + i, py1 + i],
                outline=color
            )

        # Draw label above the box
        label_y = max(0, py0 - LABEL_VERTICAL_OFFSET)
        label_x = px0

        # Background for label
        draw.rectangle(
            [label_x, label_y, label_x + len(label) * LABEL_CHAR_WIDTH + pad, label_y + LABEL_HEIGHT],
            fill=color
        )
        # Label text
        draw.text(
            (label_x + 2, label_y + 2),
            label,
            fill=(255, 255, 255)
        )
