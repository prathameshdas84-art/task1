"""Low-level drawing for the annotated overlay: signal-strength gate,
age badge, fonts, authoritative-box merging, dashed/solid box + label."""

from PIL import ImageDraw, ImageFont

from .styles import (
    COLOR_TEXT_STACKING, COLOR_WHITE_RECT,
    BOX_PADDING, LABEL_HEIGHT, LABEL_CHAR_WIDTH, LABEL_VERTICAL_OFFSET,
    _age_color_intensity, _bbox_overlaps,
)


class OverlayDrawingMixin:
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
