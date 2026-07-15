"""The three hidden-text recovery methods: white-rectangle covers,
z-order text overlap, and incremental-update revision recovery."""

import re

import fitz

from .models import HiddenTextFinding


class HiddenTextExtractionMixin:
    # ── Shared helpers ───────────────────────────────────────────────────

    def _is_white_or_near_white(self, fill) -> bool:
        if not fill:
            return False
        if isinstance(fill, (tuple, list)) and len(fill) >= 3:
            r, g, b = fill[0], fill[1], fill[2]
            # Pure white OR very light (>= 0.85 on all channels) — some
            # PDF writers emit off-white cover boxes instead of pure (1,1,1).
            return r >= 0.85 and g >= 0.85 and b >= 0.85
        return False

    # ── Method 1 — White Rectangle Cover Detection ──────────────────────

    def _extract_covered_text(self, pdf_path) -> list:
        findings = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):

            # Get all drawings (rectangles, lines etc)
            drawings = page.get_drawings()

            # Find WHITE (or near-white) filled rectangles
            white_rects = []
            for drawing in drawings:
                fill = drawing.get("fill")
                stroke = drawing.get("color")
                is_cover = (
                    self._is_white_or_near_white(fill) or
                    self._is_white_or_near_white(stroke)
                )
                if is_cover:
                    rect = drawing.get("rect")
                    if rect:
                        white_rects.append(fitz.Rect(rect))

            if not white_rects:
                continue

            # Get all text on this page. NOTE: "dict" (not "rawdict") is
            # required here — rawdict spans only expose a "chars" list of
            # per-character dicts, they have no "text" key at all, so any
            # span["text"] access against rawdict output raises KeyError.
            textdict = page.get_text(
                "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
            )

            for white_rect in white_rects:

                # Skip very small rects (not cover-ups)
                if white_rect.width < 10 or white_rect.height < 5:
                    continue

                # Find every text span whose bbox overlaps the white rect —
                # this catches BOTH the original text sitting under the
                # cover box AND any replacement text typed on top of it
                # (the cover box is usually just large enough to bound the
                # new text, so the overlay also overlaps the same rect).
                overlapping_spans = []
                for block in textdict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            span_rect = fitz.Rect(span["bbox"])
                            overlap = white_rect & span_rect
                            if not overlap.is_empty:
                                text = span["text"].strip()
                                if text:
                                    overlapping_spans.append({
                                        "text": text,
                                        "bbox": span["bbox"],
                                        "color": span.get("color", 0),
                                    })

                if not overlapping_spans:
                    continue

                # Split by color: the darkest (usually black, color 0) text
                # is the original drawn before the cover box; any span with
                # a different, lighter/tinted color is the replacement text
                # typed on top of it — same z-order heuristic used by
                # _extract_overlapping_text() below.
                base_color = min(s["color"] for s in overlapping_spans)
                hidden_texts = [
                    s for s in overlapping_spans if s["color"] == base_color
                ]
                covering_texts_full = [
                    s for s in overlapping_spans if s["color"] != base_color
                ]

                if hidden_texts:
                    hidden_combined = " | ".join(
                        h["text"] for h in hidden_texts
                    )
                    # Empty (not the "unknown" sentinel) when nothing was typed
                    # over the cover box — a genuine "missing" white-out, which
                    # analyze() classifies via covering_text below.
                    covering_combined = " | ".join(
                        c["text"] for c in covering_texts_full[:3]
                    )

                    findings.append(HiddenTextFinding(
                        page=page_num + 1,
                        method="white_rectangle_cover",
                        original_text=hidden_combined,
                        covering_text=covering_combined,
                        bbox=tuple(white_rect),
                        confidence="HIGH",
                        # Placeholder — analyze() rewrites this from
                        # replacement_type so missing vs replaced reads clearly.
                        description="",
                    ))

        doc.close()
        return findings

    # ── Method 2 — Y-Coordinate Text Overlap Detection ──────────────────
    #
    # Uses "dict" (not "rawdict") — rawdict spans have no "text" key (only
    # a per-character "chars" list), so any span["text"] lookup against it
    # raises KeyError. Two texts covering the same edit are frequently
    # returned by PyMuPDF as separate blocks that merely SHARE a Y position
    # rather than truly overlapping bboxes, so spans are grouped by Y
    # coordinate (rounded to a small bucket) instead of requiring bbox
    # intersection.

    def _extract_overlapping_text(self, pdf_path) -> list:
        findings = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):

            textdict = page.get_text(
                "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
            )

            # Group spans by Y position (same line = same Y). Round to
            # 2pt buckets to absorb sub-pixel baseline differences.
            Y_BUCKET = 2.0

            lines_by_y = {}
            for block in textdict.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if not text or len(text) < 2:
                            continue

                        y_center = (
                            span["bbox"][1] + span["bbox"][3]
                        ) / 2

                        y_key = round(y_center / Y_BUCKET) * Y_BUCKET

                        lines_by_y.setdefault(y_key, []).append({
                            "text": text,
                            "bbox": span["bbox"],
                            "color": span.get("color", 0),
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                            "x0": span["bbox"][0],
                        })

            # Find lines with MULTIPLE DIFFERENT texts at the same Y position
            for y_key, spans in lines_by_y.items():

                if len(spans) < 2:
                    continue

                unique_texts = list(dict.fromkeys(
                    s["text"] for s in spans
                ))

                if len(unique_texts) < 2:
                    continue

                # Check for X overlap between different texts (they must
                # actually occupy the same horizontal space).
                for i in range(len(spans)):
                    for j in range(i + 1, len(spans)):
                        sa = spans[i]
                        sb = spans[j]

                        if sa["text"] == sb["text"]:
                            continue

                        x_overlap = (
                            min(sa["bbox"][2], sb["bbox"][2]) -
                            max(sa["bbox"][0], sb["bbox"][0])
                        )
                        x_distance = abs(sa["x0"] - sb["x0"])

                        # A genuine cover-up sits almost exactly on top of
                        # what it replaces — either the boxes actually
                        # intersect (x_overlap > 5) or they start at nearly
                        # the same X origin (x_distance < 5, i.e. same
                        # column). A larger x_distance threshold (e.g. 20pt)
                        # also matches ordinary adjacent words in flowing
                        # prose sharing the same text line/Y-bucket ("in"
                        # next to "the"), which produced dozens of false
                        # positives on unmodified body text.
                        if x_overlap > 5 or x_distance < 5:
                            # Determine which is original: lower color
                            # value = darker = more likely original (black
                            # text). Lighter/colored text was placed on top.
                            if sa["color"] <= sb["color"]:
                                original = sa
                                covering = sb
                            else:
                                original = sb
                                covering = sa

                            union_bbox = (
                                min(sa["bbox"][0], sb["bbox"][0]),
                                min(sa["bbox"][1], sb["bbox"][1]),
                                max(sa["bbox"][2], sb["bbox"][2]),
                                max(sa["bbox"][3], sb["bbox"][3]),
                            )

                            findings.append(HiddenTextFinding(
                                page=page_num + 1,
                                method="text_overlap",
                                original_text=original["text"],
                                covering_text=covering["text"],
                                bbox=union_bbox,
                                confidence="HIGH",
                                description=(
                                    f"Two different texts at same "
                                    f"location on line y={y_key:.0f}: "
                                    f"'{original['text']}' hidden under "
                                    f"'{covering['text']}'"
                                ),
                            ))

        doc.close()

        # Deduplicate — keep unique (original, covering) pairs
        seen = set()
        unique = []
        for f in findings:
            key = (f.page, f.original_text[:30], f.covering_text[:30])
            if key not in seen:
                seen.add(key)
                unique.append(f)

        return unique

    # ── Method 3 — Incremental Update Recovery ──────────────────────────

    def _extract_revision_text(self, pdf_path) -> list:
        findings = []

        # Check for multiple %%EOF markers
        with open(pdf_path, 'rb') as f:
            content = f.read()

        eof_positions = [
            m.start() for m in re.finditer(b'%%EOF', content)
        ]

        if len(eof_positions) <= 1:
            return []  # No incremental updates

        # Extract text from each revision
        revision_texts = {}

        for i, eof_pos in enumerate(eof_positions):
            revision_bytes = content[:eof_pos + 6]
            try:
                doc = fitz.open(stream=revision_bytes, filetype="pdf")
                texts = {}
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    blocks = page.get_text("dict")["blocks"]
                    page_texts = []
                    for block in blocks:
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    page_texts.append({
                                        "text": text,
                                        "bbox": span["bbox"],
                                    })
                    texts[page_num] = page_texts

                revision_texts[i] = texts
                doc.close()
            except Exception:
                continue

        if len(revision_texts) < 2:
            return []

        # Compare revision 0 (original) vs the latest revision (edited)
        rev_0 = revision_texts.get(0, {})
        rev_1 = revision_texts.get(len(revision_texts) - 1, {})

        for page_num in rev_0:
            if page_num not in rev_1:
                continue

            texts_0 = set(s["text"] for s in rev_0[page_num])
            texts_1 = set(s["text"] for s in rev_1[page_num])

            # Text in original but NOT in edited version = removed/replaced
            removed = texts_0 - texts_1
            added = texts_1 - texts_0

            for removed_text in removed:
                if len(removed_text) < 2:
                    continue

                original_span = next(
                    (s for s in rev_0[page_num]
                     if s["text"] == removed_text), None
                )

                # Empty (not the "unknown" sentinel) when the revision removed
                # text without adding anything back — a "missing" edit.
                replacing_text = ", ".join(
                    t for t in added if len(t) >= 2
                )[:100]

                findings.append(HiddenTextFinding(
                    page=page_num + 1,
                    method="incremental_update",
                    original_text=removed_text,
                    covering_text=replacing_text,
                    bbox=tuple(
                        original_span["bbox"]
                    ) if original_span else (0, 0, 0, 0),
                    confidence="HIGH",
                    # Placeholder — analyze() rewrites this from replacement_type.
                    description="",
                ))

        return findings

