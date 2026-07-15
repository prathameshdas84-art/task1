"""ELAAnalyzer core — the three-phase DPI sweep orchestration, DCT
analysis, and cross-page consistency. Individual evidence checks live
in the mixin modules."""

import re
import statistics

import fitz
import numpy as np
from PIL import Image

from utils.pdf_utils import get_qr_zones, bbox_overlaps_qr_zone

from .constants import *
from .models import ELARegion, ELAReport
from .structure_forensics import ObjectForensicsMixin
from .signatures import SignatureChecksMixin
from .pixel_grid import ElaGridMixin
from .page_filters import PageFilterMixin
from .raster_checks import RasterChecksMixin


class ELAAnalyzer(ObjectForensicsMixin, SignatureChecksMixin, ElaGridMixin,
                  PageFilterMixin, RasterChecksMixin):
    def analyze(self, pdf_path: str, pdf_type: str = "native_text") -> ELAReport:
        doc = fitz.open(pdf_path)

        # Use higher DPI for vector PDFs (no compression artifacts at low
        # DPI, need more pixels for meaningful ELA).
        from analyzers.content_analyzer import ContentAnalyzer
        try:
            is_vector = ContentAnalyzer()._is_vector_pdf(pdf_path)
        except Exception:
            is_vector = False
        render_scales = RENDER_SCALES_VECTOR if is_vector else RENDER_SCALES
        (low_name, low_dpi), (med_name, med_dpi), (high_name, high_dpi) = render_scales

        is_image_doc = self._is_image_based_document(doc, pdf_type)
        is_scanned_type = pdf_type in ("scanned", "scanned_native")
        is_compiled = self._is_compiled_document(pdf_path, pdf_type)
        page_heights = {p: doc[p].rect.height for p in range(len(doc))}

        signals = []
        total_blocks       = 0   # phase-1 blocks scanned, for diagnostics only
        total_phase1_hits   = 0   # phase-1 raw candidates, for diagnostics only
        total_dct_regions   = 0

        # ── PHASE 1: low-DPI sweep across every page — fast candidate scan ──
        page_low_imgs  = {}             # page_num -> rendered low-DPI image (reused below)
        page_candidates = {}            # page_num -> list[ELARegion] (phase-1 hits)
        mat_low = fitz.Matrix(low_dpi / 72, low_dpi / 72)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat_low, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
            page_low_imgs[page_num] = img

            regions, n_blocks, n_flagged = self._analyze_page(img, page_num, low_dpi)
            total_blocks      += n_blocks
            total_phase1_hits += n_flagged
            regions = self._restrict_to_raster_content(page, regions)
            if is_scanned_type:
                page_h = page_heights[page_num]
                signature_y0 = page_h * (1 - SCANNED_SIGNATURE_ZONE_FRACTION)
                regions = [r for r in regions if r.bbox[1] < signature_y0]
            if regions:
                for r in regions:
                    r.confirmed_scales = [low_name]
                page_candidates[page_num] = regions

            dct_regions = self._dct_analysis(img, page_num, low_dpi)
            total_dct_regions += len(dct_regions)

        signals.append(
            f"Phase 1 ({low_name}/{low_dpi}dpi): {total_phase1_hits} candidate block(s) "
            f"out of {total_blocks} scanned across {len(doc)} page(s)"
        )

        # Compiled/merged scanned documents: apply a stricter z-threshold to
        # phase-1 candidates.  Each source page was compressed independently,
        # so page-boundary compression differences look like "edits" at the
        # normal Z_THRESHOLD=3.0.  Raising the bar to COMPILED_PHASE1_Z_THRESHOLD
        # keeps only blocks with a genuinely anomalous error level.
        if is_compiled and is_scanned_type:
            for page_num in list(page_candidates.keys()):
                strong = [r for r in page_candidates[page_num]
                          if r.z_score >= COMPILED_PHASE1_Z_THRESHOLD]
                if strong:
                    page_candidates[page_num] = strong
                else:
                    del page_candidates[page_num]

        # ── PHASE 2: medium-DPI confirmation — ONLY pages with candidates ──
        confirmed_after_medium = {}     # page_num -> list[ELARegion]
        mat_med = fitz.Matrix(med_dpi / 72, med_dpi / 72)

        for page_num, candidates in page_candidates.items():
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat_med, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)

            med_regions, _, _ = self._analyze_page(img, page_num, med_dpi)
            qr_zones = get_qr_zones(page, doc)

            confirmed = []
            for cand in candidates:
                if not self._region_confirmed_by(cand.bbox, med_regions):
                    continue
                if bbox_overlaps_qr_zone(cand.bbox, qr_zones):
                    continue  # QR code's high-frequency pixels, not an edit
                cand.confirmed_scales.append(med_name)
                confirmed.append(cand)
            if confirmed:
                confirmed_after_medium[page_num] = confirmed

        n_confirmed_medium = sum(len(v) for v in confirmed_after_medium.values())
        signals.append(
            f"Phase 2 ({med_name}/{med_dpi}dpi): {n_confirmed_medium} block(s) confirmed "
            f"at {SCALE_CONFIRM_MIN_AGREEMENT}+ scales across "
            f"{len(confirmed_after_medium)} page(s)"
        )

        # Scanned docs: drop scattered isolated/paired hits (paper-texture
        # noise), keep only blocks that are part of a contiguous cluster
        # large enough to plausibly be one edited word/line. See
        # SCANNED_MIN_CLUSTER_SIZE above for why.
        # Compiled docs use a larger minimum cluster (COMPILED_MIN_CLUSTER_SIZE)
        # because their higher per-page noise floor means small clusters are
        # almost certainly compression boundary artifacts, not real edits.
        if is_scanned_type:
            cluster_min = COMPILED_MIN_CLUSTER_SIZE if is_compiled else SCANNED_MIN_CLUSTER_SIZE
            for page_num in list(confirmed_after_medium.keys()):
                clustered = self._filter_to_significant_clusters(
                    confirmed_after_medium[page_num], min_size=cluster_min
                )
                if clustered:
                    confirmed_after_medium[page_num] = clustered
                else:
                    del confirmed_after_medium[page_num]

        # ── PHASE 3: high-DPI exact-location refinement + text sharpness ──
        # ONLY for pages that survived phase 2 — confirmed-block bboxes are
        # cropped, not the whole page (cheap); sharpness needs a page-wide
        # baseline so that one render is per-page, not per-region.
        all_regions = []
        for page_num, regions in confirmed_after_medium.items():
            page = doc[page_num]

            sharp_anomaly_bboxes = []
            try:
                text_blocks = [b[:4] for b in page.get_text("blocks")]
                sharpness_scores = self._analyze_text_sharpness(page, text_blocks)
                sharp_anomalies = self._detect_sharpness_anomalies(sharpness_scores)
                sharp_anomaly_bboxes = [
                    text_blocks[a["block_id"]] for a in sharp_anomalies
                    if a["block_id"] < len(text_blocks)
                ]
            except Exception:
                pass

            for region in regions:
                refined_bbox, high_confirmed = self._refine_region_at_high_dpi(
                    page, region.bbox, high_dpi
                )
                if high_confirmed:
                    region.confirmed_scales.append(high_name)
                if refined_bbox is not None:
                    region.bbox = refined_bbox

                if any(self._bbox_overlaps(region.bbox, b) for b in sharp_anomaly_bboxes):
                    region.sharpness_anomaly = True

                if is_scanned_type:
                    page_h = page_heights[page_num]
                    region.score_weight = (
                        SCANNED_HEADER_WEIGHT_MULTIPLIER
                        if region.bbox[1] < page_h * SCANNED_HEADER_ZONE_FRACTION
                        else 1.0
                    )

                all_regions.append(region)

        total_confirmed = len(all_regions)

        # ── Image-based document: noise-consistency check ──
        # Camera/scanner sensor noise is spatially correlated and consistent;
        # a digital edit either smooths it out (too clean) or introduces a
        # different noise pattern (pasted-in content) — ELA's compression-
        # error model doesn't apply to a document that's a photograph of
        # pages, so this runs as a parallel check rather than a replacement.
        noise_regions = []
        if is_image_doc:
            for page_num, img in page_low_imgs.items():
                arr = np.asarray(img)
                hits = self._analyze_noise_consistency(arr)
                pts_scale = 72 / low_dpi
                for hit in hits:
                    x0 = hit["bx"] * pts_scale
                    y0 = hit["by"] * pts_scale
                    noise_regions.append(ELARegion(
                        page=page_num,
                        bbox=(x0, y0, x0 + BLOCK_SIZE * pts_scale, y0 + BLOCK_SIZE * pts_scale),
                        mean_error=hit["variance"],
                        z_score=hit["z_score"],
                        render_dpi=low_dpi,
                        noise_anomaly=True,
                    ))
            if noise_regions:
                signals.append(
                    "Image-based document detected — noise pattern analysis used "
                    f"({len(noise_regions)} inconsistent region(s) found)"
                )
            all_regions.extend(noise_regions)

        # ── Image-based document: digital-erasure / "too clean" check ──
        # Narrower and complementary to the noise-consistency check above —
        # see ERASURE_* constants for why this targets a separate case.
        erasure_regions = []
        if is_image_doc:
            for page_num, img in page_low_imgs.items():
                arr = np.asarray(img)
                hits = self._detect_erased_regions(arr, page_num, low_dpi)
                for hit in hits:
                    erasure_regions.append(ELARegion(
                        page=page_num,
                        bbox=hit["bbox"],
                        mean_error=hit["std_val"],
                        z_score=hit["median_std"] / max(hit["std_val"], 0.01),
                        render_dpi=low_dpi,
                        erasure_anomaly=True,
                    ))
            if erasure_regions:
                signals.append(
                    f"{len(erasure_regions)} suspiciously uniform background "
                    f"region(s) found — possible digital erasure or clone stamp"
                )
            all_regions.extend(erasure_regions)

        # ── Flat/pasted-patch check (scanned/mixed raster pages only) ──
        # GENUINELY skipped for pdf_type="native_text" — a born-digital text
        # page has no scan-noise texture for a flat patch to be inconsistent
        # WITH, so running the detector there could only manufacture noise
        # (and the born-digital gate inside would zero it anyway). See the
        # FLAT_ZONE_* constants for what this catches and why the two
        # existing image-doc checks structurally could not.
        flat_zone_regions = []
        if pdf_type != "native_text" and (is_image_doc or is_scanned_type):
            for page_num, img in page_low_imgs.items():
                flat_zone_regions.extend(self._detect_flat_zone_patches(
                    doc[page_num], img, page_num, low_dpi
                ))
            if flat_zone_regions:
                n_stamp = sum(1 for r in flat_zone_regions if r.stamp_associated)
                signals.append(
                    f"{len(flat_zone_regions)} flat/texture-less region(s) "
                    f"inconsistent with the page's scan noise found "
                    f"({n_stamp} containing a stamp/seal-shaped graphic) — "
                    f"possible pasted-in patch"
                )
            all_regions.extend(flat_zone_regions)

        # Cap how many boxes get drawn per page — keep only the strongest
        # outliers so the UI doesn't flood with low-confidence boxes.
        regions_by_page = {}
        for r in all_regions:
            regions_by_page.setdefault(r.page, []).append(r)
        all_regions = []
        for page_regions in regions_by_page.values():
            page_regions.sort(key=lambda r: r.z_score, reverse=True)
            all_regions.extend(page_regions[:MAX_REGIONS_PER_PAGE])

        doc.close()

        for r in all_regions:
            if r.flat_zone_anomaly:
                kind = ("pasted stamp — flat background"
                        if r.stamp_associated
                        else "flat/uniform region inconsistent with page texture")
                signals.append(
                    f"Page {r.page + 1}: {kind} at "
                    f"({r.bbox[0]:.0f},{r.bbox[1]:.0f})-({r.bbox[2]:.0f},{r.bbox[3]:.0f}) "
                    f"confidence={r.flat_confidence:.2f} — {r.detail}"
                )
            elif r.erasure_anomaly:
                signals.append(
                    f"Page {r.page + 1}: erased region detected at "
                    f"({r.bbox[0]:.0f},{r.bbox[1]:.0f})-({r.bbox[2]:.0f},{r.bbox[3]:.0f}) "
                    f"std={r.mean_error:.2f} — possible digital erasure"
                )
            elif r.noise_anomaly:
                signals.append(
                    f"Page {r.page + 1}: noise-consistency anomaly at "
                    f"({r.bbox[0]:.0f},{r.bbox[1]:.0f})-({r.bbox[2]:.0f},{r.bbox[3]:.0f}) "
                    f"z={r.z_score:.1f}"
                )
            else:
                scales_desc = "+".join(s.upper() for s in r.confirmed_scales)
                extra = []
                if r.sharpness_anomaly:
                    extra.append("sharpness anomaly")
                extra_desc = f" [{', '.join(extra)}]" if extra else ""
                confidence = "HIGH" if len(r.confirmed_scales) >= 3 else "MEDIUM"
                signals.append(
                    f"Page {r.page + 1}: edit confirmed at {scales_desc} DPI scales "
                    f"({confidence} confidence) at "
                    f"({r.bbox[0]:.0f},{r.bbox[1]:.0f})-({r.bbox[2]:.0f},{r.bbox[3]:.0f}) "
                    f"z={r.z_score:.1f}{extra_desc}"
                )

        # Score off the CONFIRMED block count (rare, high-precision) rather
        # than the raw phase-1 fraction — a block surviving 2+ independent
        # DPI scales is a much stronger signal than "this fraction of blocks
        # looked busy at one resolution."
        ela_confirmed_regions = [
            r for r in all_regions
            if not r.noise_anomaly and not r.erasure_anomaly and not r.flat_zone_anomaly
        ]
        n_ela_confirmed = sum(r.score_weight for r in ela_confirmed_regions)
        ela_confirmed_score = min(CONFIRMED_BLOCK_SCORE_CAP, n_ela_confirmed * CONFIRMED_BLOCK_SCORE_PER_BLOCK)
        if is_scanned_type:
            # Compiled/merged docs use a lower multiplier — surviving blocks are
            # still more likely to be compression-boundary artifacts than real edits
            # even after the stricter phase-1/phase-2 gates above.
            score_mult = COMPILED_SCORE_MULTIPLIER if is_compiled else SCANNED_SCORE_MULTIPLIER
            ela_confirmed_score *= score_mult
            if len(ela_confirmed_regions) < SCANNED_LOW_HIT_COUNT:
                ela_confirmed_score *= SCANNED_LOW_HIT_MULTIPLIER

        anomaly_score = min(
            100,
            ela_confirmed_score
            + min(NOISE_SCORE_CAP, len(noise_regions) * NOISE_SCORE_PER_REGION)
            + min(ERASURE_SCORE_CAP, len(erasure_regions) * ERASURE_SCORE_PER_REGION)
            + min(FLAT_ZONE_SCORE_CAP,
                  sum(FLAT_ZONE_SCORE_PER_REGION * r.flat_confidence
                      for r in flat_zone_regions))
        )

        # Cross-page consistency check (for multi-page documents)
        cp_anomalies, cp_signals, cp_score = self._cross_page_consistency(pdf_path)
        for s in cp_signals:
            signals.append(s)
        anomaly_score = min(100, anomaly_score + cp_score // CROSS_PAGE_MERGE_DIVISOR)

        # PDF object fingerprinting
        obj_signals, obj_score = self._pdf_object_fingerprint(pdf_path)
        for s in obj_signals:
            signals.append(f"[OBJECT] {s}")
        anomaly_score = min(100, anomaly_score + obj_score // OBJECT_MERGE_DIVISOR)

        # Incremental-update / old-object recovery
        incremental = self._detect_incremental_updates(pdf_path)
        for s in incremental.get("signals", []):
            signals.append(f"[INCREMENTAL] {s}")
        anomaly_score = min(100, anomaly_score + incremental.get("score", 0) // INCREMENTAL_MERGE_DIVISOR)

        if total_dct_regions:
            dct_score = min(DCT_SCORE_CAP, total_dct_regions * DCT_SCORE_PER_REGION)
            anomaly_score = min(100, anomaly_score + dct_score // DCT_MERGE_DIVISOR)

        # Shadow attack detection
        shadow_signals, shadow_score = self._detect_shadow_attack(pdf_path)
        for s in shadow_signals:
            signals.append(f"[SHADOW] {s}")
        anomaly_score = min(100, anomaly_score + shadow_score // SHADOW_ATTACK_SCORE_DIVISOR)

        # Digital signature validation
        sig_signals, sig_score = self._validate_digital_signature(pdf_path)
        for s in sig_signals:
            signals.append(f"[SIGNATURE] {s}")
        anomaly_score = min(100, anomaly_score + sig_score // SIGNATURE_SCORE_DIVISOR)

        return ELAReport(
            pdf_type=pdf_type,
            anomaly_score=int(round(anomaly_score)),
            regions=all_regions,
            signals=signals,
            incremental_updates=incremental,
        )

    def _dct_analysis(self, img: Image.Image, page_num: int, render_dpi: float = RENDER_DPI) -> list:
        """
        Analyze DCT coefficient distribution across 8x8 blocks.
        JPEG compression works in 8x8 DCT blocks.
        Edited/pasted regions have different coefficient distributions
        than organically compressed regions.

        Returns list of suspicious block coordinates.
        """
        try:
            import cv2 as _cv2

            # Convert to YCbCr (JPEG native color space)
            arr = np.asarray(img.convert("YCbCr"), dtype=np.float32)
            y_channel = arr[:, :, 0]  # Luma channel

            h, w = y_channel.shape
            dct_block_size = DCT_BLOCK_SIZE
            n_rows = h // dct_block_size
            n_cols = w // dct_block_size

            if n_rows < 4 or n_cols < 4:
                return []

            # Compute DCT energy per block
            block_energies = []
            block_coords   = []

            for r in range(n_rows):
                for c in range(n_cols):
                    block = y_channel[
                        r*dct_block_size:(r+1)*dct_block_size,
                        c*dct_block_size:(c+1)*dct_block_size,
                    ]
                    dct = _cv2.dct(block)
                    # High-frequency energy (bottom-right of DCT matrix)
                    hf_energy = float(np.sum(np.abs(dct[4:, 4:])))
                    block_energies.append(hf_energy)
                    block_coords.append((c * dct_block_size, r * dct_block_size))

            if len(block_energies) < DCT_MIN_BLOCKS:
                return []

            energies = np.array(block_energies)
            mean_e   = energies.mean()
            std_e    = max(energies.std(), 0.01)

            suspicious = []
            for i, (energy, (bx, by)) in enumerate(zip(block_energies, block_coords)):
                z = abs(energy - mean_e) / std_e
                if z >= DCT_Z_THRESHOLD:  # higher threshold than ELA — DCT energy is noisier
                    pts_scale = 72 / render_dpi
                    x0 = bx * pts_scale
                    y0 = by * pts_scale
                    x1 = (bx + dct_block_size) * pts_scale
                    y1 = (by + dct_block_size) * pts_scale
                    suspicious.append({
                        "page": page_num,
                        "bbox": (x0, y0, x1, y1),
                        "energy": energy,
                        "z_score": round(z, 2),
                    })

            return suspicious

        except Exception:
            return []


    def _cross_page_consistency(
        self,
        pdf_path: str,
    ) -> tuple[list, list[str], int]:
        """
        Compare noise texture fingerprint across all pages.
        Genuine scanned documents have consistent noise patterns
        (same scanner, same settings, same paper).
        A replaced/substituted page shows different noise texture.

        Returns: (anomaly_list, signals, score)
        """
        try:
            doc   = fitz.open(pdf_path)
            scale = RENDER_DPI / 72
            mat   = fitz.Matrix(scale, scale)

            page_fingerprints = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img  = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
                arr  = np.asarray(img.convert("L"), dtype=np.float32)

                # Noise fingerprint = std of high-frequency component
                # Use Laplacian to extract high-frequency noise
                import cv2 as _cv2
                lap = _cv2.Laplacian(arr, _cv2.CV_64F)

                # Divide page into 4 quadrants, compute noise per quadrant
                h, w = lap.shape
                quadrants = [
                    lap[:h//2, :w//2],
                    lap[:h//2, w//2:],
                    lap[h//2:, :w//2],
                    lap[h//2:, w//2:],
                ]
                fingerprint = [float(np.std(q)) for q in quadrants]
                page_fingerprints.append({
                    "page": page_num,
                    "fingerprint": fingerprint,
                    "mean_noise": float(np.std(lap)),
                })

            doc.close()

            if len(page_fingerprints) < CROSS_PAGE_MIN_PAGES:
                # Need at least 3 pages to compare
                return [], ["Cross-page check skipped — document has fewer than 3 pages"], 0

            # Compare each page's noise against document average
            all_noise = [p["mean_noise"] for p in page_fingerprints]
            doc_mean  = statistics.mean(all_noise)
            doc_std   = max(statistics.stdev(all_noise), 0.01)

            anomalies = []
            for p in page_fingerprints:
                z = abs(p["mean_noise"] - doc_mean) / doc_std
                if z >= CROSS_PAGE_Z_THRESHOLD:
                    anomalies.append({
                        "page": p["page"],
                        "noise": p["mean_noise"],
                        "doc_mean": doc_mean,
                        "z_score": round(z, 2),
                        "reason": (
                            f"Page {p['page']+1} noise texture ({p['mean_noise']:.1f}) "
                            f"differs from document average ({doc_mean:.1f}) "
                            f"by z={z:.1f} — possible page substitution"
                        )
                    })

            signals = []
            score   = 0

            if anomalies:
                signals.append(
                    f"{len(anomalies)} page(s) have inconsistent noise texture — "
                    f"possible page substitution or different scan source"
                )
                score = min(CROSS_PAGE_SCORE_CAP, len(anomalies) * CROSS_PAGE_SCORE_PER_PAGE)
            else:
                signals.append(
                    "Cross-page scan consistency check passed — "
                    "all pages show uniform noise texture"
                )

            return anomalies, signals, score

        except Exception:
            return [], [], 0

