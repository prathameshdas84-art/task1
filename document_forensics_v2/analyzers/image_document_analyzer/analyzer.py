"""ImageDocumentAnalyzer core: the analyze() orchestration for direct
JPG/PNG uploads, the analyze_page_render() subset for scanned/mixed PDF
page renders, and the shared noise/flat-zone primitives. The individual
evidence checks live in the checks_* modules as mixins."""

import cv2
import numpy as np
from PIL import Image

from utils.flat_zone_detection import (
    local_std_map, detect_flat_zones,
    BORN_DIGITAL_STD_FLOOR,
)

from .constants import (
    BLOCKINESS_Z_THRESHOLD, CHECK_POINTS, DOUBLE_COMPRESSION_POINTS,
    NOT_IMPLEMENTED,
)
from .report import ImageAnomaly, ImageForensicsReport, score_anomalies
from .checks_compression import CompressionChecksMixin
from .checks_edges import EdgeChecksMixin
from .checks_copy_move import CopyMoveCheckMixin
from .checks_stamp import StampChecksMixin
from .checks_background import BackgroundColorCheckMixin


class ImageDocumentAnalyzer(CompressionChecksMixin, EdgeChecksMixin,
                            CopyMoveCheckMixin, StampChecksMixin,
                            BackgroundColorCheckMixin):

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

        # ── Check 3: any lossy compression history at all? ─────────────
        # Computed BEFORE the born-digital gate because the gate uses its
        # grid-phase statistic as a second signal (below).
        jpeg_history, blockiness_metrics = self._detect_jpeg_history(
            gray, container_format
        )
        metrics["blockiness"] = blockiness_metrics
        metrics["jpeg_dependent_checks"] = (
            "APPLICABLE" if jpeg_history else "NOT_APPLICABLE"
        )

        # Born-digital gate: a render with no sensor noise makes every
        # noise-relative check meaningless — flat zones and uniform fills
        # are NORMAL there, not evidence. Gate to zero, say so.
        #
        # A LOW noise floor alone does not prove "born digital". ANY
        # detected lossy-compression history vetoes the classification —
        # JPEG history (container or the 8px blocking-grid residual inside
        # other containers) proves the content passed through a raster
        # save pipeline, which a true vector render never does. A true
        # born-digital render has NEITHER noise NOR JPEG history.
        # Consequence (accepted): a pristine render exported AS JPEG is no
        # longer classified born-digital — its texture checks run instead
        # of gating. That is safe because every check is self-limiting on
        # a zero-noise image (Check 1's threshold is relative to the
        # floor; Check 5 gates itself on a uniformly-crisp baseline), and
        # it is what makes weak-grain genuine scans (noise floor < 1.2
        # after a high-quality JPEG save) detectable at all.
        low_noise_floor = baseline < BORN_DIGITAL_STD_FLOOR
        grid_z = blockiness_metrics.get("grid_phase_z") or [0.0, 0.0]
        compression_erased_noise = min(grid_z) > BLOCKINESS_Z_THRESHOLD
        is_born_digital = low_noise_floor and not jpeg_history
        if is_born_digital:
            signals.append(
                f"Born-digital gate: image noise baseline "
                f"{baseline:.2f} < {BORN_DIGITAL_STD_FLOOR} and no lossy-"
                f"compression history — no sensor-noise texture exists, so "
                f"variance/texture checks (1, 2, 5, 8, 9) are gated out "
                f"rather than scored against a baseline of zero."
            )
        elif low_noise_floor and compression_erased_noise:
            signals.append(
                f"NOT born-digital despite low noise baseline "
                f"{baseline:.2f} < {BORN_DIGITAL_STD_FLOOR}: 8px blocking-grid "
                f"residuals (grid_phase_z={[round(z, 2) for z in grid_z]}, both > "
                f"{BLOCKINESS_Z_THRESHOLD}) show the noise floor was erased by "
                f"JPEG quantization — this is a re-compressed capture, not a "
                f"vector render. Flat-zone checks (1, 2) still gate out (the "
                f"quantization that erased the noise also manufactures flat "
                f"zones, so low variance is not evidence here), but the "
                f"self-calibrating checks (5, 8, 9) remain active."
            )
        elif low_noise_floor:
            signals.append(
                f"NOT born-digital despite low noise baseline "
                f"{baseline:.2f} < {BORN_DIGITAL_STD_FLOOR}: JPEG compression "
                f"history is present "
                f"({blockiness_metrics.get('basis', 'container')}) — the "
                f"content passed through a raster save pipeline, so this is "
                f"a weak-grain scan/photo, not a vector render. All texture "
                f"checks (1, 2, 5, 8, 9) stay active; Check 1's flat "
                f"threshold stays relative to the measured floor "
                f"({baseline:.2f})."
            )

        # Checks 1+2 flat-zone gating: still gate when the image is truly
        # born-digital (flat is normal on a render) OR when quantization
        # ERASED the noise floor (the same quantization manufactures flat
        # zones, so low variance is not evidence — the recompressed-capture
        # case above). But on a weak-grain scan whose floor is low WITHOUT
        # grid residuals (high-quality JPEG of a real scan), the floor is a
        # genuine — just quiet — texture baseline, and a pasted flat patch
        # sitting below FLAT_RATIO of it is real evidence.
        # Checks 5/8/9 gate only on true born-digital: they calibrate
        # against the image's own blur baseline / an absolute ink-texture
        # floor, so they stay valid — and useful — on re-compressed
        # captures (an overlay added AFTER recompression is still crisper
        # than the degraded original).
        flat_zone_checks_gated = low_noise_floor and (
            is_born_digital or compression_erased_noise
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
        if not flat_zone_checks_gated:
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

        # ── Check 11: background color consistency ─────────────────────
        # Gated with the corrected born-digital logic, BOTH branches: a
        # true vector render has a perfectly uniform background by
        # construction (nothing to compare against), and a quantization-
        # erased capture (low floor + grid residuals) has block-scale
        # color plateaus MANUFACTURED by the low-quality JPEG — the same
        # reason Checks 1/2 gate out on that class. The check applies to
        # scanned/photographed content whose background variation is
        # genuine, however quiet.
        if not (is_born_digital or flat_zone_checks_gated):
            bg_anoms, bg_metrics = self._background_color_check(rgb, gray)
            metrics["background_color"] = bg_metrics
            anomalies.extend(bg_anoms)
            for a in bg_anoms:
                signals.append(
                    f"Check 11: background color mismatch at {a.bbox} — "
                    f"region reads as the same background to the eye but "
                    f"its color value differs from its surroundings "
                    f"({a.detail})"
                )

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

    def analyze_page_render(self, rgb: np.ndarray, gray: np.ndarray,
                            raster_source_evidence: bool = False,
                            evidence_basis: str = "") -> dict:
        """Run the page-render-safe SUBSET of this pipeline's checks over a
        rendered PDF page (the scanned/mixed routing in
        utils/scanned_page_forensics): Check 5 (edge sharpness), Check 6
        (copy-move), and Checks 7-9 (stamp/signature ink isolation +
        texture + boundary).

        raster_source_evidence corroborates the born-digital gate the same
        way grid_phase_z does in analyze(): a RENDER's noise floor can
        collapse without the content being born-digital (resampling, or
        scanner-app cleanup at the source), so the caller checks the page's
        embedded SOURCE images (utils/pdf_utils.page_raster_source_evidence)
        and a low render floor only gates when that evidence is absent too.

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
        low_noise_floor = baseline < BORN_DIGITAL_STD_FLOOR
        # Two-signal gate, mirroring analyze(): the render floor alone is
        # not proof of born-digital — only gate when the page's embedded
        # source images ALSO show no raster-pipeline history.
        is_born_digital = low_noise_floor and not raster_source_evidence

        anomalies = []
        signals = []
        sharp_map = None
        edge_baseline = None

        if is_born_digital:
            signals.append(
                f"Born-digital gate: page-render noise baseline "
                f"{baseline:.2f} < {BORN_DIGITAL_STD_FLOOR} and no raster "
                f"source evidence — no scan-noise texture exists, so the "
                f"texture checks (5, 8, 9) are gated out for this page."
            )
        else:
            if low_noise_floor:
                signals.append(
                    f"NOT born-digital despite low render noise baseline "
                    f"{baseline:.2f} < {BORN_DIGITAL_STD_FLOOR}: "
                    f"{evidence_basis or 'page source images carry raster-pipeline evidence'} "
                    f"— the collapsed floor reflects render resampling or "
                    f"source-side cleanup, not a vector render. The "
                    f"self-calibrating checks (5, 8, 9) stay active; each "
                    f"still applies its own internal gates."
                )
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

