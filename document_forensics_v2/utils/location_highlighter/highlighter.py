"""LocationHighlighter core — renders PDF pages and draws labeled boxes
around every finding the drawing gate lets through."""

import fitz
from PIL import Image, ImageDraw

from .styles import (
    RENDER_DPI,
    COLOR_CONTENT, COLOR_NUMERIC, COLOR_ELA, COLOR_WHITE_RECT,
    COLOR_IMAGE_OVERLAY, COLOR_GHOST, COLOR_TEXT_STACKING,
    COLOR_EMBEDDED_IMAGE,
    _content_label, _numeric_label, _flat_zone_label, _bbox_overlaps,
    _age_color_intensity, _OVERLAY_LABELS,
)
from .drawing import OverlayDrawingMixin


class LocationHighlighter(OverlayDrawingMixin):
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc      = fitz.open(pdf_path)
        self.scale    = RENDER_DPI / 72  # points to pixels

    def highlight_pages(
        self,
        suspicious_lines: list = None,
        numeric_anomalies: list = None,
        ela_regions: list = None,
        overlay_regions: list = None,
        age_days: int = None,
        fused_findings: list = None,
        text_stacking_findings: list = None,
        hidden_text_findings: list = None,
        embedded_image_findings: list = None,
    ) -> dict:
        """
        Returns dict of {page_num: PIL.Image} with colored boxes drawn.
        Only returns pages that have at least one suspicious region.

        suspicious_lines:    list of SuspiciousLine from content_analyzer
        numeric_anomalies:   list of NumericAnomaly from numeric_analyzer
        ela_regions:         list of ELARegion from ela_analyzer. Block-grid
                             regions are counted for strong-page detection but
                             not drawn (too imprecise spatially); flat-zone
                             regions (flat_zone_anomaly=True) ARE drawn in
                             purple — their bbox is the pixel-refined flat
                             patch itself
        overlay_regions:     list of OverlayRegion from pymupdf_analyzer
        age_days:            document's last-modification age in days
        fused_findings:      list of FusedFinding from signal_fusion
        text_stacking_findings: list of TextStackingFinding from
                             hidden_text_extractor.detect_stacked_text (page is
                             0-indexed, bbox in PDF points — the identical
                             coordinate space as pymupdf's own overlay findings,
                             so no conversion is needed)
        hidden_text_findings: list of HiddenTextFinding from
                             hidden_text_extractor.analyze (page is 1-INDEXED —
                             unlike every other list here — so it is converted
                             to 0-indexed below; bbox is in PDF points)
        embedded_image_findings: list of normalized dicts from
                             utils/embedded_image_forensics (0-indexed page,
                             bbox already mapped into PDF points, "label"
                             carries the specific finding text)

        Hidden-text and text-stacking findings are the most specific,
        authoritative explanation for a location. They are merged into a single
        set of "authoritative boxes" per page; any content/numeric/pymupdf
        finding that TRULY OVERLAPS one is folded into that box's "also flagged
        by" label instead of drawing a second, redundant box (Part 2). This is a
        pure DRAWING decision — no findings list, score, or fusion result is
        altered.
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

        # ELA flat/pasted-patch regions ARE drawn (rare, high-precision,
        # pixel-exact bbox) — unlike ELA's block-grid regions, whose spatial
        # imprecision keeps them out of the annotation entirely (see the
        # comment where their loop used to be).
        flat_zone_by_page = {}
        for r in (ela_regions or []):
            if getattr(r, "flat_zone_anomaly", False):
                flat_zone_by_page.setdefault(r.page, []).append(r)

        embedded_by_page = {}
        for f in (embedded_image_findings or []):
            embedded_by_page.setdefault(f["page"], []).append(f)

        numeric_by_page = {}
        for r in (numeric_anomalies or []):
            numeric_by_page.setdefault(r.page, []).append(r)

        overlay_by_page = {}
        for r in (overlay_regions or []):
            overlay_by_page.setdefault(r.page, []).append(r)

        stacking_by_page = {}
        for r in (text_stacking_findings or []):
            stacking_by_page.setdefault(r.page, []).append(r)

        # HiddenTextFinding.page is 1-indexed — convert to the 0-indexed page
        # space every other list here uses.
        hidden_by_page = {}
        for r in (hidden_text_findings or []):
            hidden_by_page.setdefault(r.page - 1, []).append(r)

        all_pages = (set(lines_by_page.keys()) |
                     set(numeric_by_page.keys()) |
                     set(overlay_by_page.keys()) |
                     set(flat_zone_by_page.keys()) |
                     set(embedded_by_page.keys()) |
                     set(stacking_by_page.keys()) |
                     set(hidden_by_page.keys()))
        result = {}

        for page_num in sorted(all_pages):
            page     = self.doc[page_num]
            page_h   = page.rect.height  # PDF points

            # Render page to image
            mat = fitz.Matrix(self.scale, self.scale)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
            draw = ImageDraw.Draw(img)

            # Authoritative boxes for this page — hidden-text + text-stacking
            # findings, merged where they overlap. Built BEFORE the other-layer
            # loops so those loops can fold overlapping findings into a box's
            # "also flagged by" list instead of drawing a redundant box, and
            # drawn LAST (below) so their labels reflect every fold.
            auth_boxes = self._build_auth_boxes(
                stacking_by_page.get(page_num, []),
                hidden_by_page.get(page_num, []),
            )

            # Draw content layer suspicious lines (RED boxes).
            # Only drawn when the signal clears the strength threshold or the
            # page is already known-suspicious from multiple layers.
            for sl in lines_by_page.get(page_num, []):
                if not self._should_draw_signal(
                    page_num, strong_pages, cross_validated_pages,
                    signal_type="content", score=sl.score,
                ):
                    continue
                # Part 2 — if this line overlaps an authoritative hidden/
                # stacking box, fold it into that box's label instead of
                # drawing a redundant red box on top.
                if self._absorb_into_auth(auth_boxes, sl.bbox, "content"):
                    continue
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=sl.bbox,
                    page_h_pts=page_h,
                    color=self._blend_age_color(COLOR_CONTENT, age_mult),
                    label=_content_label(sl),
                    label_color=COLOR_CONTENT,
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
                if self._absorb_into_auth(auth_boxes, r.bbox, "numeric"):
                    continue
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=r.bbox,
                    page_h_pts=page_h,
                    color=self._blend_age_color(COLOR_NUMERIC, age_mult),
                    label=_numeric_label(r),
                    label_color=COLOR_NUMERIC,
                    thickness=2,
                )

            # ELA block-grid findings still score and still appear in the
            # text report's signals — they're just not drawn here. ELA's
            # pixel-noise regions are too imprecise spatially (logos, dense
            # text, scan artifacts) to annotate reliably, so this loop is
            # intentionally removed rather than filtered by z-score. The
            # exception is the flat/pasted-patch regions, drawn below —
            # those bboxes are pixel-refined and precise.

            # Draw PyMuPDF overlay regions — CYAN for white-rect cover-ups,
            # MAGENTA for image overlays, GOLD for ghost/overlapping text.
            # These are rare and reliable signals — always drawn.
            # char_spacing regions are too small (single character bboxes)
            # to usefully draw, so they're skipped.
            for r in overlay_by_page.get(page_num, []):
                # A pymupdf overlay that sits on an authoritative hidden/
                # stacking box is the same edit seen by another layer — fold it
                # in rather than stacking a second box on top.
                if r.overlay_type in ("covering_rect", "image_overlay", "ghost_text") \
                        and self._absorb_into_auth(auth_boxes, r.bbox, "pymupdf"):
                    continue
                if r.overlay_type == "covering_rect":
                    self._draw_box(
                        draw=draw,
                        img_size=img.size,
                        bbox=r.bbox,
                        page_h_pts=page_h,
                        color=COLOR_WHITE_RECT,
                        label=_OVERLAY_LABELS["covering_rect"],
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
                        label=_OVERLAY_LABELS["image_overlay"],
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
                        label=_OVERLAY_LABELS["ghost_text"],
                        label_color=COLOR_GHOST,
                        thickness=2,
                    )

            # Draw embedded-image forensic findings (GREEN boxes) — the image
            # pipeline's checks run on the embedded image OBJECT, bbox already
            # mapped into page space by utils/embedded_image_forensics. Rare
            # and internally gated (born-digital, glare) — always drawn.
            for f in embedded_by_page.get(page_num, []):
                if self._absorb_into_auth(auth_boxes, f["bbox"], "embedded_image"):
                    continue
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=f["bbox"],
                    page_h_pts=page_h,
                    color=COLOR_EMBEDDED_IMAGE,
                    label=f.get("label", "Embedded Image: Anomaly"),
                    label_color=COLOR_EMBEDDED_IMAGE,
                    thickness=2,
                )

            # Draw ELA flat/pasted-patch regions (PURPLE boxes) — the box is
            # the ACTUAL flat patch (pixel-refined by the shared detector),
            # not the stamp graphic sitting on it. Rare and high-precision
            # (born-digital + glare gates already applied) — always drawn.
            for r in flat_zone_by_page.get(page_num, []):
                if self._absorb_into_auth(auth_boxes, r.bbox, "ela"):
                    continue
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=r.bbox,
                    page_h_pts=page_h,
                    color=COLOR_ELA,
                    label=_flat_zone_label(r),
                    label_color=COLOR_ELA,
                    thickness=2,
                )

            # Draw the authoritative hidden-text / text-stacking boxes LAST, so
            # each label already reflects every other-layer finding folded into
            # it above. MAGENTA DASHED, label below — always drawn (a strong,
            # reliable signal), dashed so it stays legible where it lands on top
            # of a pymupdf overlay box for the same edit. Label reads "Missing
            # Data" / "Replaced Data" (not a generic "Hidden Text Found"), plus
            # an "(also flagged by: …)" suffix when other layers agreed.
            for box in auth_boxes:
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=box["bbox"],
                    page_h_pts=page_h,
                    color=COLOR_TEXT_STACKING,
                    label=self._auth_label(box),
                    label_color=COLOR_TEXT_STACKING,
                    thickness=2,
                    dashed=True,
                    label_below=True,
                )

            # Age indicator badge — top-right corner of the page. Shows the
            # document's last-modification age (whole-file, not per-edit).
            self._draw_age_badge(draw, img.size[0], age_days)

            result[page_num] = img

        self.doc.close()
        return result

