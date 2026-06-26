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
COLOR_GHOST         = (255, 200, 0)   # gold    — ghost text / overlapping text layers (pymupdf_analyzer)
COLOR_OCR_SIZE  = (255, 100, 0)   # orange-red — OCR word font-size anomaly
COLOR_OCR_COLOR = (255, 0, 200)   # magenta    — OCR word color anomaly
COLOR_OCR_CONF  = (255, 140, 0)   # orange     — OCR word low-confidence anomaly
COLOR_OCR_BASELINE = (0, 255, 120)  # green      — OCR word baseline-misalignment anomaly

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
        fused_findings: list = None,
    ) -> dict:
        """
        Returns dict of {page_num: PIL.Image} with colored boxes drawn.
        Only returns pages that have at least one suspicious region.

        suspicious_lines:    list of SuspiciousLine from content_analyzer
        ocr_word_anomalies:  list of OCRWordAnomaly from ocr_analyzer
        numeric_anomalies:   list of NumericAnomaly from numeric_analyzer
        ela_regions:         list of ELARegion from ela_analyzer (not drawn,
                             but counted for strong-page detection)
        overlay_regions:     list of OverlayRegion from pymupdf_analyzer
        age_days:            document's last-modification age in days
        fused_findings:      list of FusedFinding from signal_fusion
        """
        # Blend factor applied to box base colors (recent = redder/brighter)
        age_mult = _age_color_intensity(age_days)

        # ── Signal-strength filtering ───────────────────────────────────────
        # Pages with cross-validated (2+ layer) findings are always drawn in
        # full. Single-layer weak signals are suppressed to reduce noise.

        # Pages that appear in fused (cross-validated) findings
        cross_validated_pages = set()
        for ff in (fused_findings or []):
            cross_validated_pages.add(ff.page)

        # Count distinct layers that fired on each page (using raw signals
        # before any filtering, so the page-strength criterion is unbiased).
        layer_hits_by_page = {}
        for sl in (suspicious_lines or []):
            layer_hits_by_page.setdefault(sl.page, set()).add("content")
        for r in (ocr_word_anomalies or []):
            layer_hits_by_page.setdefault(r.page, set()).add("ocr")
        for r in (numeric_anomalies or []):
            layer_hits_by_page.setdefault(r.page, set()).add("numeric")
        for r in (ela_regions or []):
            layer_hits_by_page.setdefault(r.page, set()).add("ela")
        for r in (overlay_regions or []):
            layer_hits_by_page.setdefault(r.page, set()).add("pymupdf")

        # Pages where 3+ distinct layers fired — likely genuinely suspicious
        strong_pages = set(
            p for p, layers in layer_hits_by_page.items() if len(layers) >= 3
        )
        strong_pages.update(cross_validated_pages)

        # Group by page
        lines_by_page = {}
        for sl in (suspicious_lines or []):
            lines_by_page.setdefault(sl.page, []).append(sl)

        ocr_by_page = {}
        for r in (ocr_word_anomalies or []):
            ocr_by_page.setdefault(r.page, []).append(r)

        numeric_by_page = {}
        for r in (numeric_anomalies or []):
            numeric_by_page.setdefault(r.page, []).append(r)

        overlay_by_page = {}
        for r in (overlay_regions or []):
            overlay_by_page.setdefault(r.page, []).append(r)

        all_pages = (set(lines_by_page.keys()) |
                     set(ocr_by_page.keys()) |
                     set(numeric_by_page.keys()) |
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

            # Draw content layer suspicious lines (RED boxes).
            # Only drawn when the signal clears the strength threshold or the
            # page is already known-suspicious from multiple layers.
            for sl in lines_by_page.get(page_num, []):
                if not self._should_draw_signal(
                    page_num, strong_pages, cross_validated_pages,
                    signal_type="content", score=sl.score,
                ):
                    continue
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

            # Draw OCR word anomalies — MAGENTA only for color/digital_paste
            # anomalies. Size (font-height) and confidence boxes are skipped:
            # they're too noisy on ID cards and scanned documents to annotate
            # reliably (bbox measurement artifacts). The scores from those
            # signals still flow through to the verdict — only the visual box
            # is suppressed here.
            for r in ocr_by_page.get(page_num, []):
                if not any(t in r.anomaly_types for t in ("color", "digital_paste")):
                    continue  # skip size / confidence / baseline — too noisy
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=r.bbox,
                    page_h_pts=page_h,
                    color=COLOR_OCR_COLOR,
                    label=f"OCR:{','.join(r.anomaly_types)}",
                    label_color=COLOR_OCR_COLOR,
                    thickness=2,
                )

            # Draw numeric anomalies (YELLOW boxes).
            # running_balance and arithmetic findings are always drawn (very
            # reliable signals). Other numeric outliers require a high z-score
            # or a strongly suspicious page.
            for r in numeric_by_page.get(page_num, []):
                if not self._should_draw_signal(
                    page_num, strong_pages, cross_validated_pages,
                    signal_type="numeric", z_score=r.z_score,
                    numeric_context=getattr(r, "context", ""),
                ):
                    continue
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

            # ELA findings still score and still appear in the text report's
            # signals — they're just not drawn here. ELA's pixel-noise
            # regions are too imprecise spatially (logos, dense text, scan
            # artifacts) to annotate reliably, so this loop is intentionally
            # removed rather than filtered by z-score.

            # Draw PyMuPDF overlay regions — CYAN for white-rect cover-ups,
            # MAGENTA for image overlays, GOLD for ghost/overlapping text.
            # These are rare and reliable signals — always drawn.
            # char_spacing regions are too small (single character bboxes)
            # to usefully draw, so they're skipped.
            for r in overlay_by_page.get(page_num, []):
                if r.overlay_type == "covering_rect":
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
                elif r.overlay_type == "ghost_text":
                    self._draw_box(
                        draw=draw,
                        img_size=img.size,
                        bbox=r.bbox,
                        page_h_pts=page_h,
                        color=COLOR_GHOST,
                        label="Ghost text overlap",
                        label_color=COLOR_GHOST,
                        thickness=2,
                    )

            # Age indicator badge — top-right corner of the page. Shows the
            # document's last-modification age (whole-file, not per-edit).
            self._draw_age_badge(draw, img.size[0], age_days)

            result[page_num] = img

        self.doc.close()
        return result

    # ── Signal-strength gate ────────────────────────────────────────────────

    def _should_draw_signal(
        self,
        page_num: int,
        strong_pages: set,
        cross_validated_pages: set,
        signal_type: str,
        score: float = None,
        z_score: float = None,
        numeric_context: str = "",
    ) -> bool:
        """
        Returns True if a signal is strong enough to deserve a visual box.

        Criterion A: page has a cross-validated (2+ layer) finding → always draw
        Criterion B: signal individually clears a type-specific threshold
        Criterion C: 3+ distinct layers fired on this page → draw everything
        """
        # A — cross-validated page: always annotate
        if page_num in cross_validated_pages:
            return True

        # B — high-reliability numeric sub-types
        if signal_type == "numeric":
            if (numeric_context == "running_balance"
                    or numeric_context.startswith("arithmetic_")):
                return True  # running-balance and cross-field arithmetic always
            if z_score is not None and z_score >= 5.0:
                return True

        # B — high-score content anomaly
        if signal_type == "content" and score is not None and score >= 0.40:
            return True

        # C — page is strongly suspicious (3+ layers agree)
        if page_num in strong_pages:
            return True

        return False

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
