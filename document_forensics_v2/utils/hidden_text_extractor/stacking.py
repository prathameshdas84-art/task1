"""Coordinate-collision text stacking: 2+ DIFFERENT text runs at the
same coordinates — cannot happen in a legitimately laid-out document."""

import re

import fitz

from .models import TextStackingFinding

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


class TextStackingMixin:
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

