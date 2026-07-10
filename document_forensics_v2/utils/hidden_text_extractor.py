"""
Hidden Text Extractor — recovers ORIGINAL text that was covered up by edits.

READ-ONLY: never modifies the analyzed PDF or any other file. Three
independent recovery methods are tried and their results merged:

  1. White rectangle cover detection — text sitting under an opaque white
     filled rectangle (a classic "white-out and retype" edit).
  2. Z-order text overlap detection — two different text spans occupying
     the same location (one was drawn over the other).
  3. Incremental update recovery — PDFs with multiple %%EOF markers keep
     every prior revision's bytes; text present in an early revision but
     missing from the latest one was removed/replaced.
"""

from dataclasses import dataclass
import re

import fitz


@dataclass
class HiddenTextFinding:
    page: int
    method: str         # how it was found
    original_text: str  # the hidden/original text
    covering_text: str  # what was placed on top
    bbox: tuple          # location on page
    confidence: str      # HIGH / MEDIUM / LOW
    description: str     # human readable explanation
    field_type: str = "unknown"   # auto-detected: name/amount/date/id_number/address/score/unknown
    plain_explanation: str = ""   # human readable explanation of HOW it was done
    # "replaced" — original was hidden AND different visible text was put in its
    # place (the classic, already-working case). "missing" — original was
    # hidden/removed with NOTHING visibly put in its place (covering_text is
    # empty/whitespace after normalization). Classified centrally in analyze();
    # defaults to "replaced" so any finding built without going through analyze()
    # keeps the historical behavior.
    replacement_type: str = "replaced"   # "replaced" | "missing"


@dataclass
class HiddenTextReport:
    findings: list        # list of HiddenTextFinding
    total_found: int
    recovery_summary: str
    signals: list          # for main report
    conclusion: str = ""    # plain-English summary of the tampering


@dataclass
class TextStackingFinding:
    """A location where 2+ DISTINCT text runs occupy the SAME coordinates.

    NOTE: `page` is 0-indexed (matching the internal fusion/analysis-route
    convention where response building adds +1), NOT 1-indexed like
    HiddenTextFinding — this structure's only consumer is signal_fusion via
    the extra_findings path, which works in 0-indexed page space."""
    page: int              # 0-indexed
    bbox: tuple            # union of the colliding runs, (x0, y0, x1, y1) PDF points
    texts: list            # the distinct colliding text values (raw, order preserved)
    overlap_fraction: float  # strongest pairwise overlap in the cluster (0.0-1.0)
    confidence: str        # always "HIGH" — a coordinate collision is a strong signal
    score: float           # 0.0-1.0 fusion score
    description: str        # human readable explanation


# ── Coordinate-collision text-stacking thresholds ──────────────────────────
# Two runs "collide" when their bboxes overlap by at least this fraction of
# the SMALLER box's area. 0.5 (half the smaller box) is deliberately tight:
# a genuine paste-over sits almost exactly on top of what it replaces (the
# smaller run is typically 90-100% inside the larger — see the "8,50,000" vs
# "18,50,000" case, where the shorter box is fully contained), while
# legitimately adjacent-but-distinct text (a label beside its value, two
# table cells) does not overlap by anywhere near half its own area. Uses a
# FRACTIONAL-area test rather than signal_fusion._bbox_overlaps (a bare
# boolean intersection that would also fire on font-ascender/descender
# edge-bleed between neighbouring rows) — same overlap-of-smaller-area
# semantics pymupdf_analyzer's ghost_text detection already uses.
TEXT_STACKING_MIN_OVERLAP_FRACTION = 0.5
# Ignore runs shorter than this (after whitespace normalization) — single
# stray glyphs are too noisy to treat as a stacked edit on their own.
TEXT_STACKING_MIN_TEXT_LEN = 2
# Fusion score for a text-stacking finding — deliberately high: two DIFFERENT
# texts at one coordinate cannot happen in a legitimately laid-out document,
# so this is a much stronger signal than a bare hidden-text run.
TEXT_STACKING_FUSION_SCORE = 0.95


class HiddenTextExtractor:

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

    # ── Missing-vs-replaced classification ──────────────────────────────

    def _classify_replacement_type(self, covering_text: str) -> str:
        """Classify a hidden-text finding by whether anything visible was put
        in place of the hidden original.

        "missing"  — covering_text is empty/whitespace after normalization (the
                     legacy "unknown" sentinel is treated as missing too, for
                     any finding produced before this field existed). The
                     original was removed/covered with nothing visibly typed
                     over it.
        "replaced" — covering_text carries actual content (the existing,
                     already-working case)."""
        norm = re.sub(r"\s+", "", covering_text or "")
        if not norm or norm.lower() == "unknown":
            return "missing"
        return "replaced"

    def _compose_hidden_text_description(self, f: "HiddenTextFinding") -> str:
        """Human-readable description that reads clearly for each case."""
        orig = f.original_text[:60]
        if f.replacement_type == "missing":
            return (
                f"Original data hidden — no replacement text visible: "
                f"'{orig}' (content was removed, nothing put in its place)"
            )
        return (
            f"Original data hidden and replaced with different visible text: "
            f"'{orig}' → '{f.covering_text[:60]}'"
        )

    # ── Field-type classification & plain-English explanations ──────────

    def _classify_field_type(self, text) -> str:
        text_lower = text.lower().strip()

        # Amount/number fields
        if re.match(r'^[\d,\.\s₹$€£]+$', text):
            return "amount"

        # Date fields
        if re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', text):
            return "date"

        # ID/Reference numbers
        if re.match(r'^[A-Z]{1,4}[\d\/\-]+$', text.upper()):
            return "id_number"

        # Name fields (words only, mixed case)
        if re.match(r'^[A-Za-z\s\.]+$', text) and len(text) > 3:
            return "name"

        # Address
        if any(w in text_lower for w in
               ['street', 'road', 'avenue', 'city',
                'state', 'country', 'pin', 'postal']):
            return "address"

        return "unknown"

    def _get_plain_explanation(self, method, field_type) -> str:
        explanations = {
            "white_rectangle_cover": (
                "A white box was placed over the original "
                "text and new content was typed on top. "
                "Visual PDF readers only show the top layer, "
                "but the original text remains hidden in "
                "the file's data."
            ),
            "text_overlap": (
                "New text was placed directly over the "
                "original text without using a white box. "
                "Both versions exist in the file — the newer "
                "text appears on top when the document is "
                "opened normally."
            ),
            "incremental_update": (
                "The document was edited and re-saved. "
                "The original version is preserved in the "
                "file's edit history, revealing what the "
                "content looked like before it was changed."
            ),
        }

        return explanations.get(
            method,
            "Original content was found beneath "
            "the visible text in this document.",
        )

    def _generate_conclusion(self, findings) -> str:
        if not findings:
            return "No hidden content detected. The visible text appears to be the original."

        n = len(findings)
        pages = sorted(set(f.page for f in findings))
        methods = set(f.method for f in findings)

        method_descriptions = {
            "white_rectangle_cover": "white boxes placed over original text",
            "text_overlap": "new text layered over original text",
            "incremental_update": "content changed between saved versions",
        }

        method_text = " and ".join(
            method_descriptions.get(m, m) for m in methods
        )

        page_text = (
            f"page {pages[0]}" if len(pages) == 1
            else f"pages {', '.join(str(p) for p in pages)}"
        )

        return (
            f"{n} hidden text region{'s' if n > 1 else ''} "
            f"found on {page_text}. "
            f"The document appears to have been altered "
            f"using {method_text}. "
            f"The original content shown above was present "
            f"in the document before it was modified."
        )

    # ── NEW CHECK — Coordinate-collision text stacking ──────────────────
    #
    # Layered ON TOP of the three recovery methods above WITHOUT touching
    # their output or scoring. Where the recovery methods answer "what did
    # the original text say?", this answers a narrower, stronger question:
    # "are two DIFFERENT texts stacked at the exact same spot?". It reports
    # ONLY genuine collisions (2+ distinct runs at one location) and NEVER a
    # lone hidden run with no counterpart — that case stays entirely the
    # recovery methods' job. It compares EVERY text run against every other
    # (via fitz "dict" spans, which expose hidden-under-white-rect runs too),
    # so it catches both hidden+visible and visible+visible stacked pairs,
    # and — working at the span level with a tolerance band — it catches
    # paste-overs whose two runs have different-width bboxes (e.g.
    # "8,50,000" vs "18,50,000"), which an exact-bbox match would miss.

    def _stacking_normalize(self, text: str) -> str:
        """Collapse ALL whitespace and strip, for the "same string / only a
        whitespace difference" test — so "8,50,000" and "8,50,000 " compare
        equal and are never treated as a collision."""
        return re.sub(r"\s+", "", text)

    def _stacking_numeric_value(self, text: str):
        """Parsed numeric value after stripping currency symbols, commas and
        whitespace, or None if the run isn't a plain number. Used so the SAME
        amount formatted differently ("₹8,50,000" vs "8,50,000") is treated as
        identical, not a collision — while "8,50,000" vs "18,50,000" (a real
        value change) stays different."""
        cleaned = re.sub(r"[,\s]", "", text)
        # Strip common currency-word prefixes and currency-symbol / trailing-
        # dot decoration from the ends, so "Rs.8,50,000", "₹8,50,000" and
        # "8,50,000.00" all reduce to the same value — but a run containing a
        # LETTER in its core (e.g. "A1" vs "B1") is left non-numeric so those
        # stay meaningfully different.
        cleaned = re.sub(r"(?i)^(?:rs\.?|inr|usd|eur|gbp)", "", cleaned)
        cleaned = cleaned.strip("₹$€£¥.")
        if re.fullmatch(r"[-+]?\d*\.?\d+", cleaned or ""):
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _stacking_texts_differ(self, a: str, b: str) -> bool:
        """True only when two runs are MEANINGFULLY different content — not
        identical, not a whitespace-only difference, not the same numeric
        value formatted differently."""
        na, nb = self._stacking_normalize(a), self._stacking_normalize(b)
        if not na or not nb:
            return False           # whitespace-only run
        if na == nb:
            return False           # identical / whitespace-only difference
        va, vb = self._stacking_numeric_value(a), self._stacking_numeric_value(b)
        if va is not None and vb is not None and va == vb:
            return False           # same value, different formatting
        return True

    def _stacking_overlap_fraction(self, a: tuple, b: tuple) -> float:
        """Intersection area as a fraction of the SMALLER box's area."""
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        if ix <= 0 or iy <= 0:
            return 0.0
        inter = ix * iy
        smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        if smaller <= 0:
            return 0.0
        return inter / smaller

    def detect_stacked_text(self, pdf_path: str) -> list:
        """Return a list of TextStackingFinding — one per location where 2+
        distinct text runs collide. Empty when nothing collides (the common
        case, including every legitimately laid-out document)."""
        findings = []
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return findings

        try:
            for page_num, page in enumerate(doc):
                textdict = page.get_text(
                    "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
                )
                spans = []
                for block in textdict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "")
                            if len(self._stacking_normalize(text)) < TEXT_STACKING_MIN_TEXT_LEN:
                                continue
                            spans.append({"text": text.strip(), "bbox": tuple(span["bbox"])})

                if len(spans) < 2:
                    continue

                # Collect qualifying collision edges (different content AND
                # sufficient overlap) once.
                edges = []  # (i, j, overlap_fraction)
                for i in range(len(spans)):
                    for j in range(i + 1, len(spans)):
                        if not self._stacking_texts_differ(spans[i]["text"], spans[j]["text"]):
                            continue
                        frac = self._stacking_overlap_fraction(spans[i]["bbox"], spans[j]["bbox"])
                        if frac >= TEXT_STACKING_MIN_OVERLAP_FRACTION:
                            edges.append((i, j, frac))

                if not edges:
                    continue

                # Union-find over the edges, so three runs stacked at one spot
                # become ONE finding rather than three overlapping pairs.
                parent = list(range(len(spans)))

                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i

                strongest = {}   # root → strongest overlap fraction in cluster
                for i, j, frac in edges:
                    parent[find(i)] = find(j)
                for i, j, frac in edges:
                    r = find(i)
                    strongest[r] = max(strongest.get(r, 0.0), frac)

                # Gather the members of each collision cluster.
                clusters = {}
                for idx in range(len(spans)):
                    root = find(idx)
                    if root in strongest:   # only roots that carry a collision
                        clusters.setdefault(root, []).append(idx)

                for root, members in clusters.items():
                    # Distinct texts (dedup by normalized form, keep first raw).
                    seen_norm = set()
                    distinct_texts = []
                    for idx in members:
                        t = spans[idx]["text"]
                        n = self._stacking_normalize(t)
                        if n in seen_norm:
                            continue
                        seen_norm.add(n)
                        distinct_texts.append(t)
                    if len(distinct_texts) < 2:
                        continue
                    boxes = [spans[idx]["bbox"] for idx in members]
                    union_bbox = (
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes),
                    )
                    frac = strongest[root]
                    findings.append(TextStackingFinding(
                        page=page_num,   # 0-indexed (see dataclass note)
                        bbox=union_bbox,
                        texts=distinct_texts,
                        overlap_fraction=frac,
                        confidence="HIGH",
                        score=TEXT_STACKING_FUSION_SCORE,
                        description=(
                            f"{len(distinct_texts)} different text runs occupy the "
                            f"same location ({frac*100:.0f}% overlap): "
                            + " vs ".join(f"'{t[:40]}'" for t in distinct_texts)
                            + " — new text placed over original without removing it"
                        ),
                    ))
        finally:
            doc.close()

        return findings

    # ── Main entry point ────────────────────────────────────────────────

    def analyze(self, pdf_path: str) -> HiddenTextReport:
        all_findings = []

        # Method 1 — white rectangle cover-ups
        try:
            covered = self._extract_covered_text(pdf_path)
            all_findings.extend(covered)
        except Exception:
            pass

        # Method 2 — z-order text overlaps
        try:
            overlapping = self._extract_overlapping_text(pdf_path)
            all_findings.extend(overlapping)
        except Exception:
            pass

        # Method 3 — incremental update recovery
        try:
            revisions = self._extract_revision_text(pdf_path)
            all_findings.extend(revisions)
        except Exception:
            pass

        # Deduplicate findings at the same location (methods 1 and 2 may
        # both catch the same cover-up).
        seen_locations = set()
        unique_findings = []
        for f in all_findings:
            key = (f.page, f.original_text[:20])
            if key not in seen_locations:
                seen_locations.add(key)
                unique_findings.append(f)

        # Classify field type + missing/replaced, and attach a clear
        # description and plain-English explanation for each case.
        for f in unique_findings:
            f.field_type = self._classify_field_type(f.original_text)
            f.replacement_type = self._classify_replacement_type(f.covering_text)
            f.description = self._compose_hidden_text_description(f)
            if f.replacement_type == "missing":
                f.plain_explanation = (
                    "The original content was hidden or removed with nothing "
                    "visible put in its place. It still exists in the file's "
                    "underlying data even though the page shows a blank or "
                    "covered area where it used to be."
                )
            else:
                f.plain_explanation = self._get_plain_explanation(
                    f.method, f.field_type
                )

        # Build signals for the main report
        signals = []
        for f in unique_findings:
            if f.replacement_type == "missing":
                signals.append(
                    f"[HIDDEN TEXT] Page {f.page} "
                    f"({f.method}): "
                    f"Original='{f.original_text[:50]}' "
                    f"— data removed, no replacement visible"
                )
            else:
                signals.append(
                    f"[HIDDEN TEXT] Page {f.page} "
                    f"({f.method}): "
                    f"Original='{f.original_text[:50]}' "
                    f"Replaced by='{f.covering_text[:50]}'"
                )

        if unique_findings:
            methods_used = set(f.method for f in unique_findings)
            summary = (
                f"Found {len(unique_findings)} hidden text "
                f"region(s) via: "
                f"{', '.join(methods_used)}"
            )
        else:
            summary = "No hidden original text detected"

        report = HiddenTextReport(
            findings=unique_findings,
            total_found=len(unique_findings),
            recovery_summary=summary,
            signals=signals,
        )
        report.conclusion = self._generate_conclusion(unique_findings)
        return report
