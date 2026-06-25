"""
Cross-Layer Signal Fusion — Document Forensics Engine

Each analysis layer (content / numeric / OCR / ELA / PyMuPDF) reports anomalies
independently, which is noisy: a region flagged by only one layer is often a
false positive. This module cross-validates findings ACROSS layers and keeps
only regions that 2+ independent layers agree on, so the UI can surface
high-confidence tamper signals and suppress single-layer noise.

Implementation notes (deviations from the original drop-in spec, made so the
engine actually runs and meets its stated goal):

  * All five inputs are dataclasses (SuspiciousLine, NumericAnomaly, ELARegion,
    SuspiciousRegion, OverlayRegion), NOT dicts. The original normalization
    called ``.get()`` on the OCR/content findings, which raises AttributeError
    on a dataclass and crashes fusion on any OCR/scanned document. ``_field()``
    reads from either an object attribute or a dict key, so both work.

  * Content and numeric findings DO carry a real ``bbox`` (the spec assumed
    they didn't and used ``bbox=None``). Populating it is essential: the core
    cross-layer case — a numeric change AND an ELA artifact at the same region —
    can only fuse if the text-layer findings have spatial coordinates to match
    against the visual layers. ``line_num`` is kept as well, so content/numeric
    still fuse with each other line-wise.
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
    ocr_finding: dict = None
    pymupdf_finding: dict = None


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

    def fuse(self,
             suspicious_lines: list = None,
             numeric_anomalies: list = None,
             ela_regions: list = None,
             ocr_regions: list = None,
             overlay_regions: list = None,
             metadata_findings: list = None) -> Tuple[List[FusedFinding], dict]:
        """
        Returns:
        - List of high-confidence FusedFinding objects
        - Dict with stats: confirmed, suppressed, single_layer

        metadata_findings: optional list of pre-normalized dicts representing
        document-level metadata anomalies. Metadata has no spatial bbox, so it
        is treated as a GLOBAL page signal that can cross-validate ANY
        location-based finding (see _findings_match). This lets a metadata
        anomaly + one visual/content anomaly form a high-confidence pair.
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

        for ocr in (ocr_regions or []):
            bbox = _field(ocr, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0)
            all_findings.append({
                "layer": "ocr",
                "page": _field(ocr, "page", 0),
                "bbox": tuple(bbox),
                "line_num": None,
                "text": _field(ocr, "word") or _field(ocr, "text"),
                "score": 1.0 - (_field(ocr, "confidence", 100) or 100) / 100,
                "raw": ocr,
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

    def _bbox_close(self, b1, b2) -> bool:
        """Check if two bounding boxes overlap or are within threshold."""
        if not b1 or not b2:
            return False
        x0_1, y0_1, x1_1, y1_1 = b1
        x0_2, y0_2, x1_2, y1_2 = b2

        # Overlap check
        if (x0_1 < x1_2 and x1_1 > x0_2 and
                y0_1 < y1_2 and y1_1 > y0_2):
            return True

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
            ocr_finding=findings_by_layer.get("ocr"),
            pymupdf_finding=findings_by_layer.get("pymupdf"),
        )
