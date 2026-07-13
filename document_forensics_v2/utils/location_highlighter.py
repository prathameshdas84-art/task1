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
# COLOR = which layer flagged it; the box LABEL states the specific finding
# (derived from the finding's own reason/type, not a generic layer name) —
# see the _*_label helpers below.
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
# Coordinate-collision text stacking (utils/hidden_text_extractor.detect_stacked_text)
# — 2+ DIFFERENT text values at the same coordinates. Drawn in bright magenta
# with a DASHED border so it reads as its own category even when it lands on
# top of pymupdf's gold "ghost text" box (the two frequently co-locate on a
# genuine paste-over) — the dash pattern lets the underlying box show through
# rather than one solidly hiding the other.
COLOR_TEXT_STACKING = (255, 0, 255)   # magenta (dashed) — hidden text stacking
# Embedded-image forensics (utils/embedded_image_forensics) — the image
# pipeline's checks run on an image OBJECT extracted from the PDF, with
# findings mapped into page space. Green: no other drawn layer uses it.
COLOR_EMBEDDED_IMAGE = (0, 190, 90)


# ── Box-label helpers ───────────────────────────────────────────────────────
# Each label states WHAT was found, short enough for the one-line label
# strip; the full detail lives in the findings list / signals, not here.

# Content-layer anomaly strings (content_analyzer) → short specific labels.
# Matched by prefix against the finding's own first anomaly reason.
_CONTENT_LABEL_PREFIXES = [
    ("font size",            "Font Size Mismatch"),
    ("font:",                "Font Mismatch"),
    ("char spacing",         "Char Spacing Anomaly"),
    ("word spacing",         "Word Spacing Anomaly"),
    ("line height",          "Line Height Anomaly"),
    ("visual noise",         "Visual Noise Outlier"),
    ("sharpness",            "Sharpness Outlier"),
    ("unnaturally uniform",  "Uniform Spacing (Retyped?)"),
    ("replacement character", "Font Encoding Mismatch"),
    ("[line_gap]",           "Abnormal Line Gap"),
]


def _content_label(sl) -> str:
    reason = (sl.anomalies[0] if getattr(sl, "anomalies", None) else "").strip()
    low = reason.lower()
    for prefix, label in _CONTENT_LABEL_PREFIXES:
        if low.startswith(prefix):
            return f"Line {sl.line_num + 1}: {label}"
    # Unknown anomaly type — fall back to the finding's own reason text.
    return f"Line {sl.line_num + 1}: {reason[:34] if reason else 'Content Anomaly'}"


def _numeric_label(r) -> str:
    ctx = (getattr(r, "context", "") or "")
    if ctx.startswith("running_balance"):
        return "Balance Mismatch"
    if ctx.startswith("arithmetic_"):
        return "Arithmetic Inconsistency"
    return f"Statistical Outlier (z={r.z_score:.1f})"


def _ocr_label(r) -> str:
    # Only color/digital_paste anomalies are drawn (see the loop below);
    # digital_paste is the stronger, more specific claim when both fired.
    if "digital_paste" in r.anomaly_types:
        return "Digital Paste Artifact"
    return "Font Color Mismatch"


# PyMuPDF overlay_type → specific label.
_OVERLAY_LABELS = {
    "covering_rect": "White-Out Cover-Up",
    "image_overlay": "Hidden Image Overlay",
    "ghost_text":    "Ghost Text (Layered)",
}


def _flat_zone_label(r) -> str:
    # ELA-layer flat/pasted-patch regions (ela_analyzer flat-zone check).
    if getattr(r, "stamp_associated", False):
        return "Pasted Stamp: Flat Background"
    return "Flat Region: Texture Mismatch"


def _bbox_overlaps(b1, b2) -> bool:
    """TRUE rectangle intersection (any shared area) — the same semantics as
    signal_fusion.SignalFusion._bbox_overlaps. Used to decide "is this the SAME
    location" for authoritative-box absorption; deliberately NOT a proximity or
    same-row test (this is identity, not corroboration-by-nearness)."""
    if not b1 or not b2:
        return False
    x0_1, y0_1, x1_1, y1_1 = b1
    x0_2, y0_2, x1_2, y1_2 = b2
    return (x0_1 < x1_2 and x1_1 > x0_2 and
            y0_1 < y1_2 and y1_1 > y0_2)


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
        text_stacking_findings: list = None,
        hidden_text_findings: list = None,
        embedded_image_findings: list = None,
    ) -> dict:
        """
        Returns dict of {page_num: PIL.Image} with colored boxes drawn.
        Only returns pages that have at least one suspicious region.

        suspicious_lines:    list of SuspiciousLine from content_analyzer
        ocr_word_anomalies:  list of OCRWordAnomaly from ocr_analyzer
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
        set of "authoritative boxes" per page; any content/numeric/OCR/pymupdf
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
                     set(ocr_by_page.keys()) |
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

            # Draw OCR word anomalies — MAGENTA only for color/digital_paste
            # anomalies. Size (font-height) and confidence boxes are skipped:
            # they're too noisy on ID cards and scanned documents to annotate
            # reliably (bbox measurement artifacts). The scores from those
            # signals still flow through to the verdict — only the visual box
            # is suppressed here.
            for r in ocr_by_page.get(page_num, []):
                if not any(t in r.anomaly_types for t in ("color", "digital_paste")):
                    continue  # skip size / confidence / baseline — too noisy
                if self._absorb_into_auth(auth_boxes, r.bbox, "ocr"):
                    continue
                self._draw_box(
                    draw=draw,
                    img_size=img.size,
                    bbox=r.bbox,
                    page_h_pts=page_h,
                    color=COLOR_OCR_COLOR,
                    label=_ocr_label(r),
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

    # ── Authoritative (hidden-text / text-stacking) box handling ────────────

    def _build_auth_boxes(self, stacking_findings: list, hidden_findings: list) -> list:
        """Merge this page's text-stacking and hidden-text findings into one set
        of authoritative boxes (overlapping ones combined into a single box, so
        an edit caught by BOTH detectors draws once, not twice).

        Each box is a dict: bbox, kind ("missing" | "replaced"), values (the
        text values to show), also_flagged_by (set, filled in later)."""
        raw = []
        for r in (stacking_findings or []):
            raw.append({
                "bbox": tuple(r.bbox),
                "kind": "replaced",   # a collision always has 2+ visible values
                "values": list(getattr(r, "texts", []) or []),
                "also_flagged_by": set(),
            })
        for r in (hidden_findings or []):
            vals = [r.original_text]
            if getattr(r, "replacement_type", "replaced") == "replaced" and r.covering_text:
                vals.append(r.covering_text)
            raw.append({
                "bbox": tuple(r.bbox),
                "kind": getattr(r, "replacement_type", "replaced"),
                "values": vals,
                "also_flagged_by": set(),
            })

        merged = []
        for box in raw:
            for m in merged:
                if _bbox_overlaps(box["bbox"], m["bbox"]):
                    m["bbox"] = (
                        min(m["bbox"][0], box["bbox"][0]),
                        min(m["bbox"][1], box["bbox"][1]),
                        max(m["bbox"][2], box["bbox"][2]),
                        max(m["bbox"][3], box["bbox"][3]),
                    )
                    # Any covering content present anywhere in the cluster means
                    # this location is a replacement, not a pure removal.
                    if box["kind"] == "replaced":
                        m["kind"] = "replaced"
                    for v in box["values"]:
                        if v not in m["values"]:
                            m["values"].append(v)
                    break
            else:
                merged.append(box)
        return merged

    def _absorb_into_auth(self, auth_boxes: list, bbox, layer_name: str) -> bool:
        """If `bbox` truly overlaps an authoritative box, record `layer_name` on
        it (for the "also flagged by" label) and return True so the caller skips
        drawing a redundant box. Returns False when there's no overlap."""
        if not bbox or not auth_boxes:
            return False
        bbox = tuple(bbox)
        for box in auth_boxes:
            if _bbox_overlaps(bbox, box["bbox"]):
                box["also_flagged_by"].add(layer_name)
                return True
        return False

    def _auth_label(self, box: dict) -> str:
        base = "Missing Data" if box["kind"] == "missing" else "Replaced Data"
        vals = box.get("values") or []
        if box["kind"] == "replaced" and len(vals) >= 2:
            detail = f": {vals[0][:14]} -> {vals[1][:14]}"
        elif vals:
            detail = f": {vals[0][:20]}"
        else:
            detail = ""
        label = base + detail
        if box["also_flagged_by"]:
            label += f" (also flagged by: {', '.join(sorted(box['also_flagged_by']))})"
        return label

    def _draw_dashed_rect(self, draw, x0, y0, x1, y1, color, dash=6, gap=4):
        """Draw a dashed rectangle outline (PIL has no native dashed stroke)."""
        def dashed_line(a, b, horizontal):
            start = a
            while start < b:
                end = min(start + dash, b)
                if horizontal is not None:  # horizontal edge at y=horizontal
                    draw.line([start, horizontal, end, horizontal], fill=color)
                start = end + gap
        def dashed_vline(a, b, x):
            start = a
            while start < b:
                end = min(start + dash, b)
                draw.line([x, start, x, end], fill=color)
                start = end + gap
        if x1 < x0 or y1 < y0:
            return
        dashed_line(x0, x1, y0)   # top
        dashed_line(x0, x1, y1)   # bottom
        dashed_vline(y0, y1, x0)  # left
        dashed_vline(y0, y1, x1)  # right

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
        dashed: bool = False,
        label_below: bool = False,
    ):
        """
        Draw a colored rectangle on the image.
        Converts PDF points (top-left origin) to pixels (top-left origin).

        dashed:      draw a dashed outline instead of a solid one — used so a
                     box that co-locates with another layer's solid box stays
                     distinguishable (the underlying box shows through the gaps).
        label_below: place the label strip just below the box instead of above
                     it — used to keep two co-located findings' labels from
                     overprinting each other.
        """
        x0, y0, x1, y1 = bbox
        # Normalize ordering — PIL's rectangle() raises on x1<x0 / y1<y0,
        # and a single malformed bbox from any layer would 500 the whole
        # annotated-image request.
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

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
        # A bbox lying fully off the page's right edge inverts after the
        # clamp above — collapse it to a zero-width box instead of crashing.
        px0 = min(px0, px1)

        # Draw rectangle outline (solid, or dashed when requested)
        if dashed:
            for i in range(thickness):
                self._draw_dashed_rect(draw, px0 - i, py0 - i, px1 + i, py1 + i, color)
        else:
            for i in range(thickness):
                draw.rectangle(
                    [px0 - i, py0 - i, px1 + i, py1 + i],
                    outline=color
                )

        # Draw label — above the box by default, or just below it when
        # label_below is set (so co-located findings' labels don't overprint).
        if label_below:
            label_y = min(img_size[1] - LABEL_HEIGHT, py1 + 2)
        else:
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
