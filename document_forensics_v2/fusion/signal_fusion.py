"""
Cross-Layer Signal Fusion — Document Forensics Engine

Each analysis layer (content / numeric / ELA / PyMuPDF) reports anomalies
independently, which is noisy: a region flagged by only one layer is often a
false positive. This module cross-validates findings ACROSS layers and keeps
only regions that 2+ independent layers agree on, so the UI can surface
high-confidence tamper signals and suppress single-layer noise.

Implementation notes (deviations from the original drop-in spec, made so the
engine actually runs and meets its stated goal):

  * The layer inputs are dataclasses (SuspiciousLine, NumericAnomaly,
    ELARegion, OverlayRegion), NOT dicts. The original normalization
    called ``.get()`` on the findings, which raises AttributeError on a
    dataclass and crashes fusion. ``_field()`` reads from either an object
    attribute or a dict key, so both work.

  * Content and numeric findings DO carry a real ``bbox`` (the spec assumed
    they didn't and used ``bbox=None``). Populating it is essential: the core
    cross-layer case — a numeric change AND an ELA artifact at the same region —
    can only fuse if the text-layer findings have spatial coordinates to match
    against the visual layers. ``line_num`` is kept as well, so content/numeric
    still fuse with each other line-wise.

Contradiction modeling (additive): agreement (above) isn't the only useful
cross-layer signal — one layer's independent evidence can also UNDERMINE
another layer's finding. ``detect_contradictions()`` runs AFTER ``fuse()``,
never deletes a finding (weight reduction only, evidence stays visible), and
currently implements:

  * Rule 2/3 (generalized cross-page-repetition contradiction): a finding
    from ANY layer whose bbox overlaps a location content_analyzer.py already
    classified as structural/repeated page furniture (exposed via
    ``ContentReport.structural_line_locations``) is self-contradicting as
    "an edit" — edits are typically localized, not uniformly repeated. This
    generalizes content_analyzer's own per-line suppression (which only
    catches its OWN font-mismatch findings) to any layer's finding on the
    same kind of region. Numeric findings get their own rule tag
    ("numeric_vs_structural_context") since a numeric anomaly on page
    furniture is specifically a disagreement about what the region even IS,
    not just a repeated-edit pattern.

  * Rule 1 (metadata vs. structural fingerprint) is NOT implemented — it
    depends on a structural PDF fingerprinting system (font-subsetting
    pattern, xref structure, trailer ordering, etc., with HIGH/MEDIUM
    confidence levels) that does not exist anywhere in this codebase yet.
    That work was proposed separately and never built. TODO once it exists:
    reduce metadata's contribution when producer/creator is unrecognized AND
    the fingerprint matches a known-legitimate pattern AND content has no
    strong anomalies.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple


def _field(obj, name, default=None):
    """Read a field from either a dataclass/object (attribute) or a dict (key).

    The analyzers hand us dataclass instances, but normalized findings flowing
    back in (e.g. from the API layer) may be plain dicts — support both."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class FusedFinding:
    """A region flagged by multiple layers — high confidence tamper signal."""
    page: int
    bbox: tuple              # (x0, y0, x1, y1) in PDF points
    confirming_layers: list  # layers that flagged this region
    confidence: str          # "HIGH", "MEDIUM", "LOW"
    score: int               # 0-100
    description: str         # human readable explanation

    # Original findings from each layer
    content_finding: dict = None
    numeric_finding: dict = None
    ela_finding: dict = None
    pymupdf_finding: dict = None


@dataclass
class ContradictedFinding:
    """A finding from one layer that independent structural evidence from
    another layer undermines. Weight is REDUCED, never zeroed — the original
    finding stays fully described here so a human reviewer can see it existed
    and judge the contradiction themselves."""
    page: int
    bbox: tuple
    layer: str                     # the layer whose finding is being contradicted
    original_description: str      # the original finding's own text/description, preserved
    contradiction_rule: str        # "cross_page_repetition" | "numeric_vs_structural_context"
    contradicting_evidence: str    # human-readable explanation of what undermines it
    weight_reduction_points: int   # points subtracted from that layer's anomaly_score (0-100 scale)


class SignalFusion:
    """
    Cross-layer signal fusion engine.

    Validates findings from multiple layers against each other.
    Real tampering tends to leave traces in MULTIPLE layers:
    - Font change → caught by content layer
    - Number change → caught by numeric layer
    - Visual artifact → caught by ELA layer
    - Hidden overlay → caught by pymupdf layer

    If only ONE layer flags a region → likely false positive
    If TWO layers flag the same region → moderate confidence
    If THREE+ layers flag → high confidence real tamper
    """

    # Distance threshold for spatial matching (in PDF points)
    OVERLAP_THRESHOLD = 50  # points = ~17mm

    # Score multipliers
    SCORE_PER_LAYER = 30
    PROXIMITY_BONUS = 10  # if findings are very close

    # ── detect_contradictions() thresholds ──────────────────────────────
    # Same-row test: the two bboxes' y-ranges must overlap by at least
    # this fraction of the SMALLER box's height. Cells in the same table
    # row share nearly their whole y-range even across font-size
    # differences, while adjacent rows in dense tables graze each other
    # by at most a few points of descender/padding — half the smaller
    # height cleanly separates the two. (A 50pt center-radius was used
    # before and vetoed genuine findings one row away from any label.)
    SAME_ROW_MIN_Y_OVERLAP_FRACTION = 0.5
    # Cross-page self-repetition: a finding whose text sits at the same
    # position on this many distinct pages is page furniture (headers/
    # footers/table headers), not an edit — edits are localized. Two
    # pages can legitimately repeat (a duplicated summary page); three
    # is header behavior.
    CROSS_PAGE_REPEAT_MIN_PAGES = 3

    def fuse(self,
             suspicious_lines: list = None,
             numeric_anomalies: list = None,
             ela_regions: list = None,
             overlay_regions: list = None,
             metadata_findings: list = None,
             extra_findings: list = None) -> Tuple[List[FusedFinding], dict]:
        """
        Returns:
        - List of high-confidence FusedFinding objects
        - Dict with stats: confirmed, suppressed, single_layer

        metadata_findings: optional list of pre-normalized dicts representing
        document-level metadata anomalies. Metadata has no spatial bbox, so it
        is treated as a GLOBAL page signal that can cross-validate ANY
        location-based finding (see _findings_match). This lets a metadata
        anomaly + one visual/content anomaly form a high-confidence pair.

        extra_findings: optional list of ALREADY-normalized finding dicts
        (each carrying its own "layer"/"page"/"bbox"/"score"/"text") appended
        verbatim to the grouping pass. This is how the image-document pipeline
        (analyzers/image_document_analyzer.py) participates: each of its
        checks enters as its own layer, so two checks co-locating on the same
        region cross-validate through the exact same 2+-layer agreement logic
        as the PDF layers — no special-casing here.
        """

        # Step 1: Normalize all findings to a common format
        all_findings = []

        for sl in (suspicious_lines or []):
            bbox = _field(sl, "bbox")
            all_findings.append({
                "layer": "content",
                "page": _field(sl, "page", 0),
                "bbox": tuple(bbox) if bbox else None,
                "line_num": _field(sl, "line_num"),
                "text": _field(sl, "text"),
                # SuspiciousLine.score is already 0.0-1.0
                "score": _field(sl, "score", 0) or 0,
                "raw": sl,
            })

        for na in (numeric_anomalies or []):
            bbox = _field(na, "bbox")
            all_findings.append({
                "layer": "numeric",
                "page": _field(na, "page", 0),
                "bbox": tuple(bbox) if bbox else None,
                "line_num": _field(na, "line_num"),
                "text": _field(na, "text"),
                "score": 0.9,  # numeric outliers are usually significant
                "raw": na,
            })

        for ela in (ela_regions or []):
            bbox = _field(ela, "bbox")
            all_findings.append({
                "layer": "ela",
                "page": _field(ela, "page", 0),
                "bbox": tuple(bbox) if bbox else None,
                "line_num": None,
                "text": None,
                "score": min(1.0, (_field(ela, "z_score", 0) or 0) / 10),
                "raw": ela,
            })

        for ov in (overlay_regions or []):
            bbox = _field(ov, "bbox")
            all_findings.append({
                "layer": "pymupdf",
                "page": _field(ov, "page", 0),
                "bbox": tuple(bbox) if bbox else None,
                "line_num": None,
                "text": None,
                "score": 1.0,
                "raw": ov,
            })

        # Pre-normalized findings from the image-document pipeline (or any
        # future caller) — appended as-is; each dict already carries its
        # own layer name.
        for ef in (extra_findings or []):
            all_findings.append({
                "layer": ef.get("layer", "image"),
                "page": ef.get("page", 0),
                "bbox": tuple(ef["bbox"]) if ef.get("bbox") else None,
                "line_num": ef.get("line_num"),
                "text": ef.get("text"),
                "score": ef.get("score", 0) or 0,
                "raw": ef.get("raw", ef),
            })

        # Metadata findings: document-level, no bbox/line_num. Normalized to the
        # common shape and appended LAST so a location finding always anchors
        # its group — metadata then joins as a confirming layer rather than
        # absorbing everything as the group anchor.
        for md in (metadata_findings or []):
            all_findings.append({
                "layer": "metadata",
                "page": _field(md, "page", 0),
                "bbox": None,
                "line_num": None,
                "text": _field(md, "text"),
                "score": _field(md, "score", 0) or 0,
                "raw": md,
            })

        # Step 2: Group findings by spatial/line proximity
        # Two findings are "matched" if:
        # - Same page AND
        # - (Same line_num OR overlapping bbox OR bbox within OVERLAP_THRESHOLD)

        fused_groups = []
        used_indices = set()

        for i, f1 in enumerate(all_findings):
            if i in used_indices:
                continue

            group = [f1]
            used_indices.add(i)

            for j, f2 in enumerate(all_findings[i+1:], start=i+1):
                if j in used_indices:
                    continue
                # Page gating lives in _findings_match now, so metadata (a
                # global page-less signal) can match findings on any page.
                if self._findings_match(f1, f2):
                    group.append(f2)
                    used_indices.add(j)

            fused_groups.append(group)

        # Step 3: Convert groups to FusedFindings
        # Only keep groups with 2+ layers (cross-validated)
        # Single-layer groups are tracked separately for stats

        high_confidence = []
        single_layer_count = 0

        for group in fused_groups:
            layers = set(f["layer"] for f in group)

            if len(layers) >= 2:
                # Multiple layers agree → HIGH confidence
                fused = self._build_fused_finding(group, layers)
                high_confidence.append(fused)
            else:
                single_layer_count += 1

        stats = {
            "total_findings_input": len(all_findings),
            "high_confidence_findings": len(high_confidence),
            "single_layer_suppressed": single_layer_count,
            "fusion_groups": len(fused_groups),
        }

        return high_confidence, stats

    def _findings_match(self, f1: dict, f2: dict) -> bool:
        """Check if two findings refer to the same region.

        Matching is lenient on purpose: real tampering leaves traces across
        layers that rarely land on pixel-identical coordinates. Being on the
        same page is enough for a weak match; matching line_num or close bboxes
        is a stronger one. Metadata is a document-level signal with no location,
        so it is allowed to fuse with any finding on any page.
        """
        # Metadata is a GLOBAL page signal — fuse with any location-based
        # finding regardless of page.
        if f1.get("layer") == "metadata" or f2.get("layer") == "metadata":
            return True

        # Different pages → no match
        if f1["page"] != f2["page"]:
            return False

        # Same line number → match
        if (f1.get("line_num") is not None
                and f1["line_num"] == f2.get("line_num")):
            return True

        # Both have bbox → check overlap/proximity
        if f1.get("bbox") and f2.get("bbox"):
            return self._bbox_close(f1["bbox"], f2["bbox"])

        # If one side has no bbox, same-page is acceptable as a weak match.
        if not f1.get("bbox") or not f2.get("bbox"):
            return True

        return False

    def _bbox_overlaps(self, b1, b2) -> bool:
        """TRUE rectangle intersection only — no proximity fallback."""
        if not b1 or not b2:
            return False
        x0_1, y0_1, x1_1, y1_1 = b1
        x0_2, y0_2, x1_2, y1_2 = b2
        return (x0_1 < x1_2 and x1_1 > x0_2 and
                y0_1 < y1_2 and y1_1 > y0_2)

    def _same_row(self, b1, b2) -> bool:
        """Do these two bboxes share a table/text row? True when their
        y-ranges overlap by >= SAME_ROW_MIN_Y_OVERLAP_FRACTION of the
        smaller box's height, regardless of horizontal distance — a header
        and a data cell in the same row are the same structural element; a
        label one row above or below is not, however close in raw distance.
        Used by detect_contradictions(), where the question is identity
        ("this finding sits ON page furniture"), not corroboration."""
        if not b1 or not b2:
            return False
        y_overlap = min(b1[3], b2[3]) - max(b1[1], b2[1])
        if y_overlap <= 0:
            return False
        smaller_h = min(b1[3] - b1[1], b2[3] - b2[1])
        if smaller_h <= 0:
            return True  # degenerate zero-height box that still intersects
        return y_overlap >= self.SAME_ROW_MIN_Y_OVERLAP_FRACTION * smaller_h

    def _bbox_close(self, b1, b2) -> bool:
        """Check if two bounding boxes overlap or are within threshold."""
        if self._bbox_overlaps(b1, b2):
            return True
        if not b1 or not b2:
            return False
        x0_1, y0_1, x1_1, y1_1 = b1
        x0_2, y0_2, x1_2, y1_2 = b2

        # Proximity check (centers)
        cx1, cy1 = (x0_1+x1_1)/2, (y0_1+y1_1)/2
        cx2, cy2 = (x0_2+x1_2)/2, (y0_2+y1_2)/2
        dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        return dist < self.OVERLAP_THRESHOLD

    def _build_fused_finding(self, group: list, layers: set) -> FusedFinding:
        """Build a FusedFinding from a group of layer findings."""
        # Use the first finding as anchor
        anchor = group[0]

        # Determine confidence based on layer count
        if len(layers) >= 3:
            confidence = "HIGH"
            base_score = 90
        elif len(layers) == 2:
            confidence = "MEDIUM"
            base_score = 70
        else:
            confidence = "LOW"
            base_score = 40

        # A metadata anomaly that co-occurs with ANY other layer is a strong
        # tamper signal (document-level edit trace + a concrete visual/content
        # anomaly), so promote those pairs to HIGH confidence.
        if "metadata" in layers and len(layers) >= 2:
            confidence = "HIGH"
            base_score = 85

        # Build description
        layer_descs = []
        for f in group:
            layer_name = f["layer"].upper()
            if f.get("text"):
                layer_descs.append(
                    f"{layer_name}: {f['text'][:60]}"
                )
            else:
                layer_descs.append(layer_name)

        description = (
            f"Region flagged by {len(layers)} independent layers "
            f"({', '.join(sorted(layers))}). "
            f"Cross-validated tamper signal."
        )

        # Union the bboxes of every layer in the group that has one — a
        # region two layers flagged independently rarely lands on the exact
        # same pixel rect (e.g. content's line bbox vs ELA's block bbox for
        # the same edited figure), so picking just one layer's box can crop
        # the drawn highlight to less than what was actually flagged.
        group_bboxes = [f["bbox"] for f in group if f.get("bbox")]
        if group_bboxes:
            bbox = (
                min(b[0] for b in group_bboxes),
                min(b[1] for b in group_bboxes),
                max(b[2] for b in group_bboxes),
                max(b[3] for b in group_bboxes),
            )
        else:
            bbox = None

        # Map layer findings
        findings_by_layer = {f["layer"]: f["raw"] for f in group}

        return FusedFinding(
            page=anchor["page"],
            bbox=bbox or (0, 0, 100, 20),
            confirming_layers=sorted(list(layers)),
            confidence=confidence,
            score=base_score,
            description=description,
            content_finding=findings_by_layer.get("content"),
            numeric_finding=findings_by_layer.get("numeric"),
            ela_finding=findings_by_layer.get("ela"),
            pymupdf_finding=findings_by_layer.get("pymupdf"),
        )

    # ── Contradiction detection (additive) — runs AFTER fuse(), before the
    # caller recomputes combined_score ──────────────────────────────────────

    # Points subtracted from a layer's anomaly_score per contradicted
    # finding (0-100 scale) — same fixed-points-per-finding shape as the
    # Gemini Layer 7 downweight in main.py, kept as its own separate,
    # tunable constant since these are independent mechanisms.
    CONTRADICTION_DOWNWEIGHT_POINTS = 10

    def detect_contradictions(self,
                               structural_line_locations: list = None,
                               suspicious_lines: list = None,
                               numeric_anomalies: list = None,
                               ela_regions: list = None,
                               overlay_regions: list = None) -> Tuple[List[ContradictedFinding], dict]:
        """
        Detects when a finding from one layer is undermined by independent
        structural evidence, via two tests — NEVER deleting the original
        finding:

        1. Same-row furniture: the finding shares a text/table row
           (_same_row — substantial y-range overlap, any horizontal
           distance) with a location content_analyzer.py classified as
           structural/repeated page furniture. Unlike fuse(), no
           center-proximity fallback: proximity is right for "these
           anomalies corroborate" but wrong for "this finding IS page
           furniture" — the old 50pt radius let label lines one row
           above/below veto a genuine finding between them.
        2. Cross-page self-repetition: the finding's own text appears at
           the same position on CROSS_PAGE_REPEAT_MIN_PAGES+ distinct
           pages. Repeating identically across pages is the definition of
           page furniture (running table headers, footers) regardless of
           whether content_analyzer's structural classifier caught that
           specific line.

        structural_line_locations: list of {"page", "bbox", "text"} dicts —
        ContentReport.structural_line_locations, i.e. every line
        content_analyzer already classifies (via _is_structural_line) as a
        header/footer/label/repeated line, whether or not it became a
        SuspiciousLine finding.

        Returns (contradicted_findings, stats).
        """
        locations = structural_line_locations or []
        contradicted: List[ContradictedFinding] = []

        def _repeat_key(it, bbox):
            # Position rounded to a 10pt cell + text prefix: identical
            # running headers land on the same key across pages even with
            # sub-point layout jitter. Returns None for text-less findings
            # (ELA blocks, overlay rects): position-only repetition is
            # trivially satisfied by grid-aligned visual regions and would
            # let per-page noise suppress the visual layers wholesale —
            # verbatim TEXT repeating at the same position is the actual
            # page-furniture signature.
            text = str(_field(it, "text", None) or _field(it, "word", None) or "")
            if not text.strip():
                return None
            return (round(bbox[0] / 10), round(bbox[1] / 10), text[:25])

        def check(items, layer_name, rule_name):
            # Pass 1 — group this layer's findings by (position, text) and
            # count distinct pages, for the self-repetition test.
            pages_by_key = {}
            for it in items or []:
                bbox = _field(it, "bbox")
                if not bbox:
                    continue
                key = _repeat_key(it, tuple(bbox))
                if key is not None:
                    pages_by_key.setdefault(key, set()).add(_field(it, "page", 0))

            for it in items or []:
                bbox = _field(it, "bbox")
                page = _field(it, "page", 0)
                if not bbox:
                    continue
                bbox = tuple(bbox)

                # Both contradiction tests are claims about TEXT-row identity
                # ("this finding IS that header/label/repeated line"), so they
                # only apply to text-bearing findings. Visual-layer regions
                # (ELA blocks, overlay rects) inevitably y-align with SOME
                # text row on a dense page — row alignment says nothing about
                # them being page furniture.
                key = _repeat_key(it, bbox)
                if key is None:
                    continue
                original_desc = (
                    _field(it, "text", None) or _field(it, "reason", None)
                    or _field(it, "word", None) or f"{layer_name} finding"
                )

                # Test 2 — cross-page self-repetition.
                n_pages = len(pages_by_key.get(key, ()))
                if n_pages >= self.CROSS_PAGE_REPEAT_MIN_PAGES:
                    contradicted.append(ContradictedFinding(
                        page=page,
                        bbox=bbox,
                        layer=layer_name,
                        original_description=str(original_desc)[:200],
                        contradiction_rule=rule_name,
                        contradicting_evidence=(
                            f"The same finding repeats at the same position on "
                            f"{n_pages} pages — edits are typically localized; "
                            f"uniform repetition is page furniture."
                        ),
                        weight_reduction_points=self.CONTRADICTION_DOWNWEIGHT_POINTS,
                    ))
                    continue  # one contradiction per finding

                # Test 1 — same-row overlap with classified page furniture.
                for loc in locations:
                    if loc.get("page") != page or not loc.get("bbox"):
                        continue
                    if not self._same_row(bbox, tuple(loc["bbox"])):
                        continue
                    contradicted.append(ContradictedFinding(
                        page=page,
                        bbox=bbox,
                        layer=layer_name,
                        original_description=str(original_desc)[:200],
                        contradiction_rule=rule_name,
                        contradicting_evidence=(
                            f"Shares a row with a line content_analyzer classified as "
                            f"structural/repeated page furniture (\"{str(loc.get('text', ''))[:60]}\") "
                            f"— edits are typically localized, not uniformly repeated."
                        ),
                        weight_reduction_points=self.CONTRADICTION_DOWNWEIGHT_POINTS,
                    ))
                    break  # one match is enough to flag this finding once

        check(suspicious_lines, "content", "cross_page_repetition")
        check(ela_regions, "ela", "cross_page_repetition")
        check(overlay_regions, "pymupdf", "cross_page_repetition")
        # Numeric gets its own rule tag: a numeric anomaly on page furniture
        # is a disagreement about what the region IS, not just a repeated-
        # edit pattern (see rule 3 in the module docstring).
        check(numeric_anomalies, "numeric", "numeric_vs_structural_context")

        stats = {
            "contradictions_found": len(contradicted),
            "layers_affected": sorted(set(c.layer for c in contradicted)),
        }
        return contradicted, stats
