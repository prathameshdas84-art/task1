"""
Image-Document Forensic Analyzer — dedicated pipeline for DIRECT image
uploads (JPG/PNG).

Targets the specific attack class this engine previously had no coverage
for: a photographed document (ID card, certificate, receipt) where an
editor (a) used AI inpainting to REMOVE original content — which smooths
away the sensor-noise texture in that region — and/or (b) used a phone
app's text/sticker tool to OVERLAY new content — which renders with
mathematically crisp anti-aliased edges that never match the soft
ink-spread + lens-blur + JPEG-blur profile of the photographed original.
Plus stamp/seal/signature paste-in detection (flat ink fill, cutout
boundary).

ROUTING DECISION (Part 1 of the spec): this analyzer runs ONLY for
direct JPG/PNG uploads via POST /analyze-image. PDFs — including
scanned/mixed PDFs with embedded raster pages — stay entirely in the
existing PDF pipeline: ela_analyzer.py already carries the image-based
document checks for those (noise-consistency + digital-erasure), so
routing raster PDF pages here as well would silently run two different
smoothing detectors over the same pixels and double-count the signal.
One document class, one pipeline.

This module NEVER touches, imports from, or alters the behavior of the
six existing PDF layers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 2 — DELIBERATELY NOT IMPLEMENTED (honesty requirement, Part 0/3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every entry below is a technique that CANNOT produce a real confidence
number from a single uploaded image with no reference data. Rather than
implement placeholder math that looks like forensics, each is omitted
and surfaced in the report's `not_implemented` list so a reader knows
what was NOT checked — not just what came back clean.

* PRNU sensor fingerprinting — requires multiple reference images from
  the SAME camera sensor to average out scene content and extract the
  photo-response non-uniformity pattern. A single image has no reference
  to correlate against; any single-image "PRNU-lite" score would be
  fabricated precision.
* ICA/PCA ink source separation — no established reliable method exists
  for separating ink chemical types from a small consumer-resolution RGB
  patch; output would be false confidence, not signal.
* Lighting/shadow-direction consistency — requires 3D scene
  reconstruction a flat document photo gives no basis for; extremely
  high false-positive risk on documents with naturally uniform lighting.
* DCT quantization TABLE extraction / exact resave counting — only the
  categorical single/double/uncertain flag (Check 4) is implemented;
  table-level analysis and precise resave counts are not reliably
  recoverable, especially after social-media recompression.
* Perspective/lens-distortion geometric consistency — requires camera
  calibration data an arbitrary upload doesn't carry. The edge-sharpness
  comparison (Checks 5/9) is the practical substitute for catching flat
  digital overlays.
* Stamp-geometry "pressure deviation" contour fitting — real stamp
  geometry varies enough from paper texture / photo perspective alone
  that this has meaningful false-positive risk; excluded entirely
  rather than shipped as a near-zero-weight decoration.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Numeric discipline used throughout (same rules the earlier flat-edit
spec mandated): all variance math is done in float64 BEFORE squaring;
E[x^2]-E[x]^2 is clipped at 0 before sqrt; every detection threshold is
RELATIVE to the document's own measured baseline, never an absolute
constant applied blind; and if the whole image has a near-zero noise
baseline (born-digital render), the noise-dependent checks gate out to
score 0 instead of manufacturing findings.
"""

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# Check 1 (local variance / flat-zone), Check 2 (glare discrimination), and
# Check 7's ink isolation now live in utils/flat_zone_detection.py so
# ela_analyzer.py can run the SAME algorithm on scanned/mixed PDF raster
# pages — one implementation, two pipelines. Their tuning constants moved
# with them; BORN_DIGITAL_STD_FLOOR is re-imported because the born-digital
# gate is applied here, by the caller.
from utils.flat_zone_detection import (
    local_std_map, detect_flat_zones, isolate_ink_regions,
    BORN_DIGITAL_STD_FLOOR,
)

# ── Check 3: JPEG compression history (works inside PNG containers too) ────
BLOCKINESS_Z_THRESHOLD  = 3.5    # grid-phase diff z-score vs other 7 phases

# ── Check 4: double-compression flag (categorical ONLY — see Tier 2 note) ──
DC_COEFS                = [(0, 1), (1, 0), (1, 1)]  # low-freq AC coefficients analyzed
DC_HIST_RANGE           = 60     # histogram over [-range, +range]
DC_MIN_COEF_AGREEMENT   = 2      # coefs that must agree for a single/double verdict

# ── Check 5: glyph/edge rendering sharpness (PRIMARY overlay signal) ───────
EDGE_AMP_WINDOW         = 7      # local amplitude (max-min) window
EDGE_GRAD_FLOOR         = 60.0   # min Sobel magnitude for a pixel to count as an edge
EDGE_AMP_FLOOR          = 70.0   # min local amplitude (strong edges only)
SHARP_CELL              = 32     # aggregation cell (px)
SHARP_CELL_MIN_EDGES    = 12     # min edge pixels for a cell to be scored
SHARP_RATIO             = 1.45   # cell flagged if p90 sharpness > ratio × image baseline
SHARP_ABS_MIN           = 0.50   # and above this absolute floor (1.0 = perfect step edge)
SHARP_MIN_CELLS         = 2      # min connected flagged cells (single cells = noise)
SHARP_BASELINE_GATE     = 0.70   # baseline above this = whole image crisp → check meaningless

# ── Check 6: copy-move with offset-vector consensus ────────────────────────
CM_ORB_FEATURES         = 3000
CM_MIN_SPATIAL_DIST     = 30     # px — matches closer than this are the same structure
CM_MAX_HAMMING          = 40
CM_OFFSET_BIN           = 8      # px — displacement-vector quantization
CM_MERGE_BIN_DIST       = 2     # adjacent bins (chebyshev) merged into one cluster
CM_MIN_PAIRS            = 30     # pairs sharing one offset required (consensus)
CM_NCC_VERIFY           = 0.85   # patch correlation confirmation
CM_NCC_PATCH            = 16
CM_MIN_REGION_DIM       = 20     # px — thinner regions are glyph/substring repeats
# Repeated template text produces a FAMILY of harmonically-related offsets
# (one per line-pitch multiple); a genuine clone produces exactly one.
CM_HARMONIC_ANGLE_COS   = 0.985  # ~10° — offsets more parallel than this…
CM_HARMONIC_RATIO_TOL   = 0.15   # …whose length ratio is near-integer = lattice

# ── Checks 7-9: stamp/signature ink texture + boundary ─────────────────────
# (ink isolation constants INK_*/SIG_MAX_STROKE_HALFW moved to
# utils/flat_zone_detection.py with the shared isolate_ink_regions)
FLAT_INK_ABS_FLOOR      = 4.0    # ink-density std below max(this, rel) = flat fill
FLAT_INK_REL            = 1.0    # × image noise baseline
ORGANIC_INK_MIN_STD     = 8.0    # metrics/reporting aid only, not a flag threshold
STAMP_BOUNDARY_RATIO    = 1.45   # boundary sharpness vs image edge baseline (same as Check 5)
STAMP_BOUNDARY_ABS      = 0.55

# ── Check 10: near-white micro-contrast heatmap (display only) ──────────────
HEATMAP_BAND_LOW        = 240
HEATMAP_BAND_HIGH       = 255

# ── Scoring weights (Part 4) ─────────────────────────────────────────────────
# Per Part 0's honesty requirement, the two signals most durable against
# social-media recompression — glyph/edge sharpness (Checks 5/9) and
# variance smoothing (Check 1) — carry the highest weight. Copy-move gets
# a solid mid weight ONLY because the offset-consensus + NCC verification
# makes a fired detection high-precision. The double-compression flag is
# categorical and recompression-fragile, so it contributes almost nothing.
# The stamp-geometry check is NOT implemented at all (see Tier 2 block),
# so it carries no weight rather than a decorative near-zero one.
# per_hit is multiplied by the finding's 0-1 confidence. Sized so a single
# high-confidence hit from one of the two primary checks clears the
# MODIFIED threshold (20) plus the uncertain band (±5) on its own — each
# was validated for specificity against the clean/glare/born-digital
# false-positive suite before being trusted with that weight.
CHECK_POINTS = {
    "check5_edge_sharpness":   {"per_hit": 40, "cap": 55},   # PRIMARY
    "check1_local_variance":   {"per_hit": 35, "cap": 45},   # PRIMARY
    "check9_stamp_boundary":   {"per_hit": 20, "cap": 35},   # same mechanism as 5
    "check6_copy_move":        {"per_hit": 15, "cap": 30},
    "check8_stamp_texture":    {"per_hit": 12, "cap": 20},
}
DOUBLE_COMPRESSION_POINTS = 8    # categorical, recompression-fragile → tiny


def score_anomalies(anomalies: list) -> tuple:
    """CHECK_POINTS-weighted scoring — per_hit × confidence per anomaly,
    capped per check. Shared by analyze() and the scanned-page routing
    (utils/scanned_page_forensics), which must re-score after filtering
    QR-zone hits out so its fold magnitude matches its surviving findings.
    Returns (raw_score_float, per_check_scores_dict)."""
    score = 0.0
    per_check_scores = {}
    for check, cfg in CHECK_POINTS.items():
        hits = [a for a in anomalies if a.evidence_check == check]
        s = min(cfg["cap"], sum(cfg["per_hit"] * a.confidence for a in hits))
        per_check_scores[check] = round(s, 1)
        score += s
    return score, per_check_scores

NOT_IMPLEMENTED = [
    {
        "technique": "prnu_sensor_fingerprint",
        "reason": "Requires multiple reference images from the same camera "
                  "sensor to extract a fingerprint; a single uploaded image "
                  "has nothing to correlate against — any single-image PRNU "
                  "number would be fabricated precision. NOT IMPLEMENTED.",
    },
    {
        "technique": "ink_source_separation_ica_pca",
        "reason": "No established reliable method for separating ink chemical "
                  "types from a single small RGB patch at consumer camera "
                  "resolution — would produce false confidence. NOT IMPLEMENTED.",
    },
    {
        "technique": "lighting_shadow_direction_consistency",
        "reason": "Requires 3D scene reconstruction a flat document photo "
                  "gives no basis for; extremely high false-positive risk on "
                  "uniformly lit documents. NOT IMPLEMENTED.",
    },
    {
        "technique": "dct_quant_table_extraction_resave_count",
        "reason": "Only the categorical single/double/uncertain flag (Check 4) "
                  "is implemented; quantization-table extraction and precise "
                  "resave counts are not reliably recoverable, especially "
                  "after social-media recompression. NOT IMPLEMENTED beyond "
                  "the categorical flag.",
    },
    {
        "technique": "perspective_lens_distortion_consistency",
        "reason": "Requires camera calibration data an arbitrary upload does "
                  "not carry. Edge-sharpness comparison (Checks 5/9) is the "
                  "practical substitute for catching flat digital overlays. "
                  "NOT IMPLEMENTED.",
    },
    {
        "technique": "stamp_geometry_pressure_deviation",
        "reason": "Real stamp geometry varies enough from paper texture and "
                  "photo perspective alone that contour-deviation fitting has "
                  "meaningful false-positive risk on genuine stamps — excluded "
                  "entirely rather than shipped as a near-zero-weight signal. "
                  "NOT IMPLEMENTED.",
    },
]


@dataclass
class ImageAnomaly:
    type: str                # e.g. "inpaint_smoothing", "sharp_overlay_edge"
    bbox: tuple              # (x, y, w, h) in image pixels
    confidence: float        # 0.0-1.0
    evidence_check: str      # which check produced it (see CHECK_POINTS keys)
    page: int = 1            # always 1 for a single image
    detail: str = ""


@dataclass
class ImageForensicsReport:
    is_born_digital: bool
    jpeg_history_detected: bool
    compression_history: str   # single_compression | double_compression_suspected | uncertain | not_applicable
    stamp_detected: bool
    signature_detected: bool
    anomalies: list = field(default_factory=list)          # list[ImageAnomaly]
    not_implemented: list = field(default_factory=lambda: list(NOT_IMPLEMENTED))
    metrics: dict = field(default_factory=dict)
    anomaly_score: int = 0
    signals: list = field(default_factory=list)
    heatmap_png: bytes = None   # Check 10 — display-only evidence, never scored


def normalize_for_fusion(report: ImageForensicsReport) -> list:
    """Convert this report's anomalies into signal_fusion's normalized
    finding shape (dicts with layer/page/bbox/score/text). Each CHECK is
    its own fusion 'layer', so two different checks co-locating on the
    same region cross-validate through the existing 2+-layer agreement
    logic with zero special-casing. bboxes convert (x,y,w,h) → (x0,y0,x1,y1);
    pages stay 0-indexed inside fusion like every PDF layer's findings."""
    findings = []
    for a in report.anomalies:
        x, y, w, h = a.bbox
        findings.append({
            "layer": f"image_{a.evidence_check}",
            "page": a.page - 1,
            "bbox": (float(x), float(y), float(x + w), float(y + h)),
            "line_num": None,
            "text": a.detail or a.type,
            "score": float(a.confidence),
            "raw": a,
        })
    return findings


class ImageDocumentAnalyzer:

    def analyze(self, image_path: str) -> ImageForensicsReport:
        pil = Image.open(image_path)
        container_format = (pil.format or "").upper()
        # The file's OWN quantization tables (JPEG only) — Check 4 compares
        # the histogram's comb period against what THIS save applied.
        try:
            qtables = dict(pil.quantization) if container_format == "JPEG" else None
        except Exception:
            qtables = None
        rgb = np.asarray(pil.convert("RGB"))
        gray = np.asarray(pil.convert("L"), dtype=np.float64)

        signals = []
        metrics = {"image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
                   "container_format": container_format}
        anomalies = []

        # ── Shared primitives ──────────────────────────────────────────
        std_map, baseline = self._local_std_map(gray)
        metrics["noise_baseline_std"] = round(float(baseline), 3)

        # Born-digital gate: a render with no sensor noise makes every
        # noise-relative check meaningless — flat zones and uniform fills
        # are NORMAL there, not evidence. Gate to zero, say so.
        is_born_digital = baseline < BORN_DIGITAL_STD_FLOOR
        if is_born_digital:
            signals.append(
                f"Born-digital gate: image noise baseline "
                f"{baseline:.2f} < {BORN_DIGITAL_STD_FLOOR} — no sensor-noise "
                f"texture exists, so variance/texture checks (1, 2, 5, 8, 9) "
                f"are gated out rather than scored against a baseline of zero."
            )

        # ── Check 3: any lossy compression history at all? ─────────────
        jpeg_history, blockiness_metrics = self._detect_jpeg_history(
            gray, container_format
        )
        metrics["blockiness"] = blockiness_metrics
        metrics["jpeg_dependent_checks"] = (
            "APPLICABLE" if jpeg_history else "NOT_APPLICABLE"
        )
        if jpeg_history:
            signals.append(
                "JPEG compression history detected "
                f"({blockiness_metrics.get('basis', 'container')}) — "
                "compression-dependent checks applicable."
            )
        else:
            signals.append(
                "No lossy compression history found — JPEG-dependent checks "
                "report NOT_APPLICABLE (not a clean 0; a very high-quality "
                "lossy save can be below pixel-level detectability)."
            )

        # ── Check 4: double-compression flag (categorical ONLY) ────────
        if not jpeg_history:
            compression_history = "not_applicable"
        else:
            compression_history, dc_metrics = self._double_compression_flag(
                gray, container_format, qtables
            )
            metrics["double_compression"] = dc_metrics
            if compression_history == "double_compression_suspected":
                signals.append(
                    "DCT histogram periodicity consistent with more than one "
                    "JPEG save (categorical flag only — an exact resave count "
                    "is not recoverable and is deliberately not claimed)."
                )

        # ── Checks 1+2: flat zones + glare discrimination ──────────────
        glare_regions = []
        if not is_born_digital:
            flat_anoms, glare_regions = self._flat_zone_check(gray, std_map, baseline)
            anomalies.extend(flat_anoms)
            if glare_regions:
                signals.append(
                    f"{len(glare_regions)} bright low-variance region(s) "
                    f"classified as physical glare (soft variance falloff at "
                    f"boundary) and excluded from scoring."
                )
            for a in flat_anoms:
                signals.append(
                    f"Check 1: smoothed region at {a.bbox} — local noise "
                    f"variance collapsed vs document baseline ({a.detail})"
                )
        metrics["glare_regions_excluded"] = [list(g) for g in glare_regions]

        # ── Check 5: edge/glyph rendering sharpness (PRIMARY) ──────────
        sharp_map = None
        edge_baseline = None
        if not is_born_digital:
            sharp_anoms, sharp_map, edge_baseline, sharp_metrics = \
                self._edge_sharpness_check(gray)
            metrics["edge_sharpness"] = sharp_metrics
            anomalies.extend(sharp_anoms)
            for a in sharp_anoms:
                signals.append(
                    f"Check 5: anomalously crisp edges at {a.bbox} vs the "
                    f"image's own blur baseline ({a.detail})"
                )

        # ── Check 6: copy-move with offset-vector consensus ────────────
        cm_anoms, cm_metrics = self._copy_move_check(gray)
        metrics["copy_move"] = cm_metrics
        anomalies.extend(cm_anoms)
        for a in cm_anoms:
            signals.append(f"Check 6: clone consensus at {a.bbox} ({a.detail})")

        # ── Checks 7-9: stamp/signature ink isolation, texture, boundary ─
        stamp_detected = False
        signature_detected = False
        ink_components = self._isolate_ink_regions(rgb)
        metrics["ink_components"] = len(ink_components)
        for comp in ink_components:
            if comp["kind"] == "stamp":
                stamp_detected = True
            else:
                signature_detected = True

            if is_born_digital:
                continue  # texture/boundary comparisons are gated out

            # Check 8 — ink texture variance inside the mask
            tex_anom = self._stamp_texture_check(gray, comp, baseline)
            if tex_anom:
                anomalies.append(tex_anom)
                signals.append(
                    f"Check 8: flat uniform ink fill inside {comp['kind']} at "
                    f"{tex_anom.bbox} ({tex_anom.detail})"
                )

            # Check 9 — mask-boundary sharpness (same mechanism as Check 5)
            if sharp_map is not None and edge_baseline is not None:
                bnd_anom = self._stamp_boundary_check(
                    comp, sharp_map, edge_baseline
                )
                if bnd_anom:
                    anomalies.append(bnd_anom)
                    signals.append(
                        f"Check 9: cutout-sharp {comp['kind']} boundary at "
                        f"{bnd_anom.bbox} ({bnd_anom.detail})"
                    )

        if stamp_detected:
            signals.append("Check 7: colored stamp/seal ink region isolated.")
        if signature_detected:
            signals.append("Check 7: thin-stroke colored ink (signature-like) isolated. "
                           "Note: black/graphite signatures are not separable by "
                           "ink color and are not detected by this check.")

        # ── Check 10: near-white micro-contrast heatmap (display only) ─
        heatmap_png = self._near_white_heatmap(gray)

        # ── Scoring — weighted per CHECK_POINTS, capped per check ──────
        score, per_check_scores = score_anomalies(anomalies)
        if compression_history == "double_compression_suspected":
            score += DOUBLE_COMPRESSION_POINTS
            per_check_scores["check4_double_compression"] = DOUBLE_COMPRESSION_POINTS
        score = int(round(min(100, score)))
        metrics["per_check_scores"] = per_check_scores
        metrics["check_weights"] = {
            k: dict(v) for k, v in CHECK_POINTS.items()
        }

        return ImageForensicsReport(
            is_born_digital=is_born_digital,
            jpeg_history_detected=jpeg_history,
            compression_history=compression_history,
            stamp_detected=stamp_detected,
            signature_detected=signature_detected,
            anomalies=anomalies,
            not_implemented=list(NOT_IMPLEMENTED),
            metrics=metrics,
            anomaly_score=score,
            signals=signals,
            heatmap_png=heatmap_png,
        )

    def analyze_page_render(self, rgb: np.ndarray, gray: np.ndarray) -> dict:
        """Run the page-render-safe SUBSET of this pipeline's checks over a
        rendered PDF page (the scanned/mixed routing in
        utils/scanned_page_forensics): Check 5 (edge sharpness), Check 6
        (copy-move), and Checks 7-9 (stamp/signature ink isolation +
        texture + boundary).

        Deliberately EXCLUDED from this entry point:
          * Checks 1+2 (flat zones + glare): ela_analyzer's
            _detect_flat_zone_patches already runs the SAME shared
            algorithm (utils/flat_zone_detection) on the same page
            renders — running both would double-count one physical
            finding into two scores.
          * Checks 3+4 (compression history): a get_pixmap render is a
            fresh rasterization, so those checks would measure the
            render, not the document; compression evidence on PDFs is
            the ELA layer's whole job.
          * Check 10 (near-white heatmap): display-only, never scored.

        Scoring is left to the caller (score_anomalies) so it can filter
        QR-zone false positives out first. Returns
        {"anomalies", "signals", "is_born_digital", "noise_baseline"}.
        """
        std_map, baseline = self._local_std_map(gray)
        is_born_digital = baseline < BORN_DIGITAL_STD_FLOOR

        anomalies = []
        signals = []
        sharp_map = None
        edge_baseline = None

        if is_born_digital:
            signals.append(
                f"Born-digital gate: page-render noise baseline "
                f"{baseline:.2f} < {BORN_DIGITAL_STD_FLOOR} — no scan-noise "
                f"texture exists, so the texture checks (5, 8, 9) are gated "
                f"out for this page."
            )
        else:
            sharp_anoms, sharp_map, edge_baseline, _ = \
                self._edge_sharpness_check(gray)
            anomalies.extend(sharp_anoms)
            for a in sharp_anoms:
                signals.append(
                    f"Check 5: anomalously crisp edges at {a.bbox} vs the "
                    f"page's own blur baseline ({a.detail})"
                )

        # Check 6 works from structure, not noise — runs either way (same
        # as analyze()).
        cm_anoms, _ = self._copy_move_check(gray)
        anomalies.extend(cm_anoms)
        for a in cm_anoms:
            signals.append(f"Check 6: clone consensus at {a.bbox} ({a.detail})")

        if not is_born_digital:
            for comp in self._isolate_ink_regions(rgb):
                tex_anom = self._stamp_texture_check(gray, comp, baseline)
                if tex_anom:
                    anomalies.append(tex_anom)
                    signals.append(
                        f"Check 8: flat uniform ink fill inside {comp['kind']} "
                        f"at {tex_anom.bbox} ({tex_anom.detail})"
                    )
                if sharp_map is not None and edge_baseline is not None:
                    bnd_anom = self._stamp_boundary_check(
                        comp, sharp_map, edge_baseline
                    )
                    if bnd_anom:
                        anomalies.append(bnd_anom)
                        signals.append(
                            f"Check 9: cutout-sharp {comp['kind']} boundary at "
                            f"{bnd_anom.bbox} ({bnd_anom.detail})"
                        )

        return {
            "anomalies": anomalies,
            "signals": signals,
            "is_born_digital": is_born_digital,
            "noise_baseline": float(baseline),
        }

    # ── Shared primitives ────────────────────────────────────────────────

    @staticmethod
    def _local_std_map(gray: np.ndarray):
        """Shared implementation — see utils/flat_zone_detection.local_std_map."""
        return local_std_map(gray)

    # ── Checks 1 + 2 ─────────────────────────────────────────────────────

    def _flat_zone_check(self, gray, std_map, baseline):
        """Check 1 (flat zones) + Check 2 (glare discrimination) — the
        algorithm lives in utils/flat_zone_detection.detect_flat_zones,
        shared with ela_analyzer's raster-page check; this wrapper only
        converts the shared dict shape into this pipeline's ImageAnomaly."""
        zones, glare_regions = detect_flat_zones(gray, std_map, baseline)
        anomalies = [
            ImageAnomaly(
                type="inpaint_smoothing",
                bbox=z["bbox"],
                confidence=z["confidence"],
                evidence_check="check1_local_variance",
                detail=z["detail"],
            )
            for z in zones
        ]
        return anomalies, glare_regions

    # ── Check 3 ──────────────────────────────────────────────────────────

    @staticmethod
    def _detect_jpeg_history(gray, container_format):
        """Detect whether ANY lossy compression history exists — including
        inside a PNG container — via 8px-grid blocking residuals. The grid
        phase's mean adjacent-pixel difference is compared against the other
        7 phases (self-calibrating; no absolute threshold on image content)."""
        metrics = {}
        if container_format == "JPEG":
            metrics["basis"] = "container is JPEG"
            history = True
        else:
            history = False

        g = gray.astype(np.float64)
        zs = []
        for axis in (0, 1):
            d = np.abs(np.diff(g, axis=axis)).mean(axis=1 - axis)
            phases = np.array([d[p::8].mean() for p in range(8)])
            others_mask = np.ones(8, bool)
            best_p = int(np.argmax(phases))
            others_mask[best_p] = False
            spread = phases[others_mask].std()
            z = (phases[best_p] - phases[others_mask].mean()) / max(spread, 1e-9)
            zs.append(float(z))
        metrics["grid_phase_z"] = [round(z, 2) for z in zs]
        if not history and min(zs) > BLOCKINESS_Z_THRESHOLD:
            history = True
            metrics["basis"] = "8px blocking-grid residual inside non-JPEG container"
        elif container_format == "JPEG":
            pass
        elif not history:
            metrics["basis"] = "no grid residual found"
        return history, metrics

    # ── Check 4 ──────────────────────────────────────────────────────────

    # JPEG zigzag index of the analyzed low-frequency AC coefficients —
    # used to read the matching entry out of the file's quantization table.
    _ZIGZAG_IDX = {(0, 1): 1, (1, 0): 2, (1, 1): 4}

    @staticmethod
    def _double_compression_flag(gray, container_format, qtables):
        """Categorical double-compression suspicion. A single JPEG save
        already combs the coefficient histogram (values at multiples of its
        quantization step) — the honest evidence for a PRIOR save is a comb
        COARSER than what this file's own quantization table applied: the
        final save's step is known (qtables), so a detected period well
        above it can only come from an earlier, stronger quantization.

        Deliberately outputs ONLY single/double/uncertain — an exact resave
        count is not recoverable (see Tier 2 block). Non-JPEG containers
        return 'uncertain': the DCT grid alignment and final step are
        unknown after a container conversion, so comb evidence can't be
        attributed to one save vs another."""
        g = gray.astype(np.float64) - 128.0
        h, w = g.shape
        nby, nbx = h // 8, w // 8
        if nby < 8 or nbx < 8:
            return "uncertain", {"basis": "image too small for DCT statistics"}
        blocks = g[: nby * 8, : nbx * 8].reshape(nby, 8, nbx, 8).transpose(0, 2, 1, 3)
        # 8x8 DCT-II basis
        k = np.arange(8)
        D = np.cos((2 * k[None, :] + 1) * k[:, None] * np.pi / 16.0)
        D[0, :] *= 1 / np.sqrt(2)
        D *= 0.5
        coefs = np.einsum("ij,abjk,lk->abil", D, blocks, D)

        if container_format != "JPEG" or not qtables:
            return "uncertain", {
                "basis": "non-JPEG container — final quantization step and "
                         "grid alignment unknown, comb evidence unattributable"
            }
        lum_table = qtables.get(0) or list(qtables.values())[0]

        detail = {}
        votes_double, votes_single = 0, 0
        for (u, v) in DC_COEFS:
            c = np.rint(coefs[:, :, u, v]).astype(np.int64).ravel()
            c = c[np.abs(c) <= DC_HIST_RANGE]
            hist, _ = np.histogram(c, bins=2 * DC_HIST_RANGE + 1,
                                   range=(-DC_HIST_RANGE - 0.5, DC_HIST_RANGE + 0.5))
            hist = hist.astype(np.float64)
            center = DC_HIST_RANGE
            half = hist[center + 1:] + hist[:center][::-1]  # symmetrize, drop 0-bin
            if half.sum() < 500:
                continue

            # Detect the histogram's comb period: the q>=2 whose multiples
            # concentrate the mass vs non-multiples.
            q_detected = 1
            best_sep = 0.0
            for q in range(2, 17):
                on = half[q - 1::q]           # bins at multiples of q (1-indexed values)
                off_mask = np.ones(len(half), bool)
                off_mask[q - 1::q] = False
                off = half[off_mask]
                if on.size < 4 or off.size < 4:
                    continue
                sep = (on.mean() - off.mean()) / max(half.mean(), 1e-9)
                if sep > max(0.5, best_sep):
                    best_sep = sep
                    q_detected = q

            q_file = int(lum_table[ImageDocumentAnalyzer._ZIGZAG_IDX[(u, v)]])
            detail[f"coef_{u}{v}"] = {"comb_period_detected": q_detected,
                                      "file_qtable_step": q_file}
            if q_detected >= max(2.0, 1.8 * q_file):
                votes_double += 1
            elif q_detected <= 1.3 * q_file:
                votes_single += 1

        if votes_double + votes_single == 0:
            return "uncertain", {"basis": "insufficient DCT statistics", **detail}
        if votes_double >= DC_MIN_COEF_AGREEMENT:
            return "double_compression_suspected", detail
        if votes_single >= DC_MIN_COEF_AGREEMENT and votes_double == 0:
            return "single_compression", detail
        return "uncertain", detail

    # ── Check 5 ──────────────────────────────────────────────────────────

    def _edge_sharpness_check(self, gray):
        """Per-pixel edge sharpness = Sobel magnitude / (4 × local amplitude):
        1.0 ≈ a mathematically perfect 1-2px step, lower = softer transition.
        Cells are flagged only RELATIVE to the image's own blur baseline —
        a crisp overlay is anomalous because the rest of the same photo is
        soft, never because of an absolute sharpness bar."""
        g = gray.astype(np.float64)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        gmag = np.hypot(gx, gy)
        k = np.ones((EDGE_AMP_WINDOW, EDGE_AMP_WINDOW), np.uint8)
        amp = cv2.dilate(g, k) - cv2.erode(g, k)
        edge_px = (gmag > EDGE_GRAD_FLOOR) & (amp > EDGE_AMP_FLOOR)
        sharp = np.zeros_like(g)
        sharp[edge_px] = np.clip(gmag[edge_px] / (4.0 * amp[edge_px]), 0, 1.2)

        h, w = g.shape
        ncy, ncx = h // SHARP_CELL, w // SHARP_CELL
        cell_sharp = np.full((ncy, ncx), np.nan)
        for cy in range(ncy):
            for cx in range(ncx):
                ys, xs = cy * SHARP_CELL, cx * SHARP_CELL
                cell_edges = sharp[ys:ys + SHARP_CELL, xs:xs + SHARP_CELL]
                cell_mask = edge_px[ys:ys + SHARP_CELL, xs:xs + SHARP_CELL]
                if cell_mask.sum() >= SHARP_CELL_MIN_EDGES:
                    cell_sharp[cy, cx] = np.percentile(cell_edges[cell_mask], 90)

        valid = ~np.isnan(cell_sharp)
        m = {"cells_with_edges": int(valid.sum())}
        if valid.sum() < 4:
            m["basis"] = "too few edge-bearing cells to establish a baseline"
            return [], sharp, None, m
        baseline = float(np.median(cell_sharp[valid]))
        m["edge_sharpness_baseline"] = round(baseline, 3)
        if baseline > SHARP_BASELINE_GATE:
            m["basis"] = (f"whole-image edge baseline {baseline:.2f} > "
                          f"{SHARP_BASELINE_GATE} — uniformly crisp image, no "
                          f"soft photographic baseline to compare overlays against")
            return [], sharp, baseline, m

        flag_grid = np.zeros((ncy, ncx), np.uint8)
        flag_grid[valid & (cell_sharp > SHARP_RATIO * baseline)
                  & (cell_sharp > SHARP_ABS_MIN)] = 1
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(flag_grid, 8)
        anomalies = []
        for i in range(1, n_lbl):
            if stats[i, cv2.CC_STAT_AREA] < SHARP_MIN_CELLS:
                continue
            region_vals = cell_sharp[lbl == i]
            mean_sharp = float(np.nanmean(region_vals))
            conf = float(np.clip((mean_sharp / baseline - 1.0) / 1.2, 0.0, 1.0))
            anomalies.append(ImageAnomaly(
                type="sharp_overlay_edge",
                bbox=(int(stats[i, cv2.CC_STAT_LEFT] * SHARP_CELL),
                      int(stats[i, cv2.CC_STAT_TOP] * SHARP_CELL),
                      int(stats[i, cv2.CC_STAT_WIDTH] * SHARP_CELL),
                      int(stats[i, cv2.CC_STAT_HEIGHT] * SHARP_CELL)),
                confidence=round(conf, 2),
                evidence_check="check5_edge_sharpness",
                detail=(f"edge sharpness {mean_sharp:.2f} vs image baseline "
                        f"{baseline:.2f} ({mean_sharp / baseline:.1f}x)"),
            ))
        return anomalies, sharp, baseline, m

    # ── Check 6 ──────────────────────────────────────────────────────────

    @staticmethod
    def _copy_move_check(gray):
        """Copy-move with OFFSET-VECTOR CONSENSUS: a genuine clone is many
        keypoint pairs sharing ONE displacement vector; scattered matches
        (repeated glyphs in any normal document) never converge on a single
        offset with this much support. Fired clusters are then verified by
        raw patch correlation before being reported."""
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        orb = cv2.ORB_create(nfeatures=CM_ORB_FEATURES)
        kps, des = orb.detectAndCompute(g8, None)
        metrics = {"keypoints": 0 if kps is None else len(kps)}
        if des is None or len(kps) < 20:
            return [], metrics

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf.knnMatch(des, des, k=4)
        pairs = []
        for ms in knn:
            for mtc in ms:
                if mtc.queryIdx == mtc.trainIdx or mtc.distance > CM_MAX_HAMMING:
                    continue
                p1 = np.array(kps[mtc.queryIdx].pt)
                p2 = np.array(kps[mtc.trainIdx].pt)
                if np.hypot(*(p2 - p1)) < CM_MIN_SPATIAL_DIST:
                    continue
                dx, dy = p2 - p1
                if dx < 0 or (dx == 0 and dy < 0):   # canonical direction
                    dx, dy, p1, p2 = -dx, -dy, p2, p1
                pairs.append((dx, dy, tuple(p1), tuple(p2)))
                break  # one non-self match per keypoint is enough

        metrics["candidate_pairs"] = len(pairs)
        if not pairs:
            return [], metrics

        raw_clusters = {}
        for dx, dy, p1, p2 in pairs:
            key = (round(dx / CM_OFFSET_BIN), round(dy / CM_OFFSET_BIN))
            raw_clusters.setdefault(key, []).append((p1, p2))

        # Merge clusters in adjacent bins (the same physical offset lands in
        # neighboring bins through keypoint jitter) so a real clone is ONE
        # cluster — otherwise the harmonic filter below would see its own
        # bin-split halves as a "lattice" and reject it.
        merged = []   # list of [sum_key(px), members]
        for key, members in sorted(raw_clusters.items(), key=lambda kv: -len(kv[1])):
            placed = False
            for mc in merged:
                mk = mc["key"]
                if (abs(mk[0] - key[0]) <= CM_MERGE_BIN_DIST
                        and abs(mk[1] - key[1]) <= CM_MERGE_BIN_DIST):
                    mc["members"].extend(members)
                    placed = True
                    break
            if not placed:
                merged.append({"key": key, "members": list(members)})
        metrics["largest_offset_cluster"] = (
            max(len(mc["members"]) for mc in merged) if merged else 0
        )

        candidates = [mc for mc in merged if len(mc["members"]) >= CM_MIN_PAIRS]

        # Lattice/harmonic rejection: repeated template text (line pitch,
        # column grid) produces a FAMILY of near-parallel offsets at integer
        # multiples of one base vector. A genuine clone is a single offset.
        def _is_lattice(a, b):
            va = np.array(a["key"], float) * CM_OFFSET_BIN
            vb = np.array(b["key"], float) * CM_OFFSET_BIN
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1 or nb < 1:
                return False
            cos = abs(float(va @ vb) / (na * nb))
            if cos < CM_HARMONIC_ANGLE_COS:
                return False
            ratio = max(na, nb) / min(na, nb)
            return abs(ratio - round(ratio)) < CM_HARMONIC_RATIO_TOL
        lattice = set()
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if _is_lattice(candidates[i], candidates[j]):
                    lattice.add(i)
                    lattice.add(j)
        candidates = [c for i, c in enumerate(candidates) if i not in lattice]
        metrics["lattice_clusters_rejected"] = len(lattice)

        anomalies = []
        for mc in candidates[:3]:
            members = mc["members"]
            key = mc["key"]
            src = np.array([m[0] for m in members])
            dst = np.array([m[1] for m in members])
            sx0, sy0 = src.min(axis=0); sx1, sy1 = src.max(axis=0)
            dx0, dy0 = dst.min(axis=0); dx1, dy1 = dst.max(axis=0)

            # Thin regions = a repeated glyph run / substring, not a clone.
            if (min(sx1 - sx0, sy1 - sy0) < CM_MIN_REGION_DIM
                    or min(dx1 - dx0, dy1 - dy0) < CM_MIN_REGION_DIM):
                continue

            # Source/dest overlap = periodic structure matching itself one
            # period over (text lines, table rows) — a clone's source and
            # destination are disjoint regions.
            ox = min(sx1, dx1) - max(sx0, dx0)
            oy = min(sy1, dy1) - max(sy0, dy0)
            if ox > 0 and oy > 0:
                inter = ox * oy
                smaller = min((sx1 - sx0) * (sy1 - sy0), (dx1 - dx0) * (dy1 - dy0))
                if smaller > 0 and inter / smaller > 0.2:
                    continue

            # NCC verification on sample keypoint patches
            ok = 0
            checked = 0
            half = CM_NCC_PATCH // 2
            for p1, p2 in members[:6]:
                x1, y1 = int(p1[0]), int(p1[1])
                x2, y2 = int(p2[0]), int(p2[1])
                if (min(x1, x2) < half or min(y1, y2) < half or
                        max(x1, x2) >= g8.shape[1] - half or
                        max(y1, y2) >= g8.shape[0] - half):
                    continue
                a = g8[y1 - half:y1 + half, x1 - half:x1 + half].astype(np.float64)
                b = g8[y2 - half:y2 + half, x2 - half:x2 + half].astype(np.float64)
                a -= a.mean(); b -= b.mean()
                denom = np.sqrt((a * a).sum() * (b * b).sum())
                checked += 1
                if denom > 1e-9 and (a * b).sum() / denom > CM_NCC_VERIFY:
                    ok += 1
            if checked == 0 or ok / checked < 0.5:
                continue

            conf = float(np.clip(len(members) / (3.0 * CM_MIN_PAIRS), 0.3, 1.0))
            for (x0, y0, x1, y1), tag in (((sx0, sy0, sx1, sy1), "source"),
                                          ((dx0, dy0, dx1, dy1), "clone")):
                anomalies.append(ImageAnomaly(
                    type="copy_move_region",
                    bbox=(int(x0), int(y0), int(max(8, x1 - x0)), int(max(8, y1 - y0))),
                    confidence=round(conf, 2),
                    evidence_check="check6_copy_move",
                    detail=(f"{tag} of {len(members)} keypoint pairs sharing "
                            f"offset ~({key[0] * CM_OFFSET_BIN},{key[1] * CM_OFFSET_BIN})px, "
                            f"NCC-verified"),
                ))
        return anomalies, metrics

    # ── Check 7 ──────────────────────────────────────────────────────────

    @staticmethod
    def _isolate_ink_regions(rgb):
        """Shared implementation — see utils/flat_zone_detection.isolate_ink_regions."""
        return isolate_ink_regions(rgb)

    # ── Check 8 ──────────────────────────────────────────────────────────

    @staticmethod
    def _stamp_texture_check(gray, comp, noise_baseline):
        """Ink-density variance inside the (eroded) ink mask — float64,
        relative to the image's own noise baseline. Genuine wet ink has
        organic density variation (pressure, bleed); a digitally filled
        stamp is uniform down to compression noise."""
        inner = cv2.erode(comp["mask"], np.ones((3, 3), np.uint8))
        vals = gray[inner > 0].astype(np.float64)
        if vals.size < 200:
            return None
        ink_std = float(np.sqrt(np.clip((vals ** 2).mean() - vals.mean() ** 2, 0, None)))
        threshold = max(FLAT_INK_ABS_FLOOR, FLAT_INK_REL * noise_baseline)
        if ink_std >= threshold:
            return None
        conf = float(np.clip(1.0 - ink_std / max(threshold, 1e-6), 0.2, 1.0))
        return ImageAnomaly(
            type="flat_ink_fill",
            bbox=comp["bbox"],
            confidence=round(conf, 2),
            evidence_check="check8_stamp_texture",
            detail=(f"ink-density std {ink_std:.2f} < flat-fill threshold "
                    f"{threshold:.2f} (image noise baseline {noise_baseline:.2f})"),
        )

    # ── Check 9 ──────────────────────────────────────────────────────────

    @staticmethod
    def _stamp_boundary_check(comp, sharp_map, edge_baseline):
        """Boundary sharpness of the ink mask — the SAME transition-profile
        mechanism as Check 5, restricted to the stamp/signature contour.
        Organic ink bleeds into paper fiber over several pixels; a pasted
        cutout transitions in one."""
        contour = comp["mask"] - cv2.erode(comp["mask"], np.ones((3, 3), np.uint8))
        band = cv2.dilate(contour, np.ones((3, 3), np.uint8))
        vals = sharp_map[(band > 0) & (sharp_map > 0)]
        if vals.size < 30:
            return None
        boundary_sharp = float(np.percentile(vals, 75))
        if (boundary_sharp <= STAMP_BOUNDARY_RATIO * edge_baseline
                or boundary_sharp <= STAMP_BOUNDARY_ABS):
            return None
        conf = float(np.clip((boundary_sharp / edge_baseline - 1.0) / 1.2, 0.0, 1.0))
        return ImageAnomaly(
            type="pasted_stamp_boundary",
            bbox=comp["bbox"],
            confidence=round(conf, 2),
            evidence_check="check9_stamp_boundary",
            detail=(f"{comp['kind']} boundary sharpness {boundary_sharp:.2f} vs "
                    f"image edge baseline {edge_baseline:.2f}"),
        )

    # ── Check 10 ─────────────────────────────────────────────────────────

    @staticmethod
    def _near_white_heatmap(gray):
        """Near-white micro-contrast stretch (240-255 band → full range),
        float math before the uint8 cast, COLORMAP_JET. Display-only
        evidence for a human reviewer — deliberately NOT a scoring input:
        it visualizes residue, it doesn't measure it."""
        band = (gray.astype(np.float64) - HEATMAP_BAND_LOW) * (
            255.0 / (HEATMAP_BAND_HIGH - HEATMAP_BAND_LOW)
        )
        band = np.clip(band, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(band, cv2.COLORMAP_JET)
        ok, buf = cv2.imencode(".png", heat)
        return buf.tobytes() if ok else None
