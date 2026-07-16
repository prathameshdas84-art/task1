"""Raster-page checks for scanned/image documents: text sharpness,
noise consistency, digital erasure, and flat/pasted-patch detection
(shared algorithm with the image pipeline)."""

import statistics

import fitz

import numpy as np
from PIL import Image

from utils.flat_zone_detection import (
    local_std_map, detect_flat_zones, isolate_ink_regions,
    BORN_DIGITAL_STD_FLOOR,
)
from utils.pdf_utils import page_raster_source_evidence

from .constants import *
from .models import ELARegion


class RasterChecksMixin:
    def _analyze_text_sharpness(self, page, text_blocks: list) -> dict:
        """
        At 600 DPI, measure edge sharpness (Laplacian gradient variance)
        around each text block on the page.

        Original text: consistent sharpness across blocks using the same
        font/renderer. Edited text: different sharpness, because it was
        rendered by a different tool or with different antialiasing.

        Returns {block_index: sharpness_score} — block_index is the index
        into `text_blocks`, so callers can map an anomaly back to a bbox.
        """
        import cv2

        mat = fitz.Matrix(SHARPNESS_RENDER_DPI / 72, SHARPNESS_RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        scale = SHARPNESS_RENDER_DPI / 72  # px per PDF point
        sharpness_scores = {}

        for i, block in enumerate(text_blocks):
            if len(block) < 4:
                continue
            x0, y0, x1, y1 = block[:4]
            px0 = max(0, int(x0 * scale))
            py0 = max(0, int(y0 * scale))
            px1 = min(gray.shape[1], int(x1 * scale))
            py1 = min(gray.shape[0], int(y1 * scale))
            if px1 <= px0 or py1 <= py0:
                continue
            sharpness_scores[i] = float(np.var(laplacian[py0:py1, px0:px1]))

        return sharpness_scores

    def _detect_sharpness_anomalies(self, sharpness_scores: dict) -> list:
        """
        Flag text blocks whose sharpness is a statistical outlier against
        the rest of the page's text blocks — a different renderer/AA
        setting on one block among an otherwise-consistent page is a sign
        that block was edited in after the fact.
        """
        if len(sharpness_scores) < 3:
            return []

        values = list(sharpness_scores.values())
        mean_sharpness = np.mean(values)
        std_sharpness = np.std(values)
        if std_sharpness < 1:
            return []

        anomalies = []
        for block_id, sharpness in sharpness_scores.items():
            z = abs(sharpness - mean_sharpness) / std_sharpness
            if z > SHARPNESS_Z_THRESHOLD:
                anomalies.append({
                    "block_id": block_id,
                    "sharpness": sharpness,
                    "z_score": z,
                    "reason": (
                        f"Text sharpness anomaly (z={z:.1f}) — rendering style "
                        f"differs from document baseline, possible edit with "
                        f"a different tool"
                    ),
                })
        return anomalies

    def _analyze_noise_consistency(self, img_array: np.ndarray) -> list:
        """
        For scanned/photographed documents: analyze noise-pattern
        consistency across regions. Camera sensor noise is a consistent
        Gaussian floor across the whole frame; a digital edit either wipes
        it out locally (too clean — digital insertion) or introduces a
        different noise pattern (pasted-in content from another source).
        """
        import cv2

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(float) - blurred.astype(float)

        h, w = noise.shape
        block_size = BLOCK_SIZE
        noise_variances = {}
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = noise[by:by + block_size, bx:bx + block_size]
                noise_variances[(bx, by)] = float(np.var(block))

        if len(noise_variances) < MIN_BLOCKS:
            return []

        values = list(noise_variances.values())
        mean_var = np.mean(values)
        std_var = np.std(values)
        if std_var < 0.1:
            return []

        anomaly_regions = []
        for (bx, by), var in noise_variances.items():
            z = abs(var - mean_var) / std_var
            if z > NOISE_Z_THRESHOLD:
                if var < mean_var - 2 * std_var:
                    reason = "Suspiciously clean region — possible digital insertion"
                else:
                    reason = "Noise pattern inconsistency — possible pasted content"
                anomaly_regions.append({
                    "bx": bx, "by": by,
                    "variance": var, "z_score": z,
                    "reason": reason,
                })

        return anomaly_regions

    def _detect_erased_regions(self, img_array: np.ndarray, page_num: int, render_dpi: int) -> list:
        """
        Flag background-colored regions with near-zero pixel variance —
        see ERASURE_* constants above for why this targets a narrower case
        than _analyze_noise_consistency.
        """
        import cv2

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        variances = {}
        for y in range(0, h - ERASURE_BLOCK_SIZE, ERASURE_STRIDE):
            for x in range(0, w - ERASURE_BLOCK_SIZE, ERASURE_STRIDE):
                block = gray[y:y + ERASURE_BLOCK_SIZE, x:x + ERASURE_BLOCK_SIZE]
                if float(np.mean(block)) < ERASURE_BG_MIN_BRIGHTNESS:
                    continue  # text stroke or dense image content, not background
                variances[(x, y)] = float(np.std(block))

        if len(variances) < 10:
            return []

        values = list(variances.values())
        median_std = statistics.median(values)
        if median_std < ERASURE_MIN_MEDIAN_STD:
            # Page background itself is already near-flat (digital render,
            # not a real scan) -- nothing to compare an "erased" block against.
            return []

        pts_scale = 72 / render_dpi
        candidates = []
        for (x, y), std_val in variances.items():
            if std_val < median_std * ERASURE_RATIO_THRESHOLD:
                candidates.append({
                    "page": page_num,
                    "bbox": (
                        x * pts_scale, y * pts_scale,
                        (x + ERASURE_BLOCK_SIZE) * pts_scale,
                        (y + ERASURE_BLOCK_SIZE) * pts_scale,
                    ),
                    "std_val": std_val,
                    "median_std": median_std,
                    "reason": (
                        f"Suspiciously uniform background region "
                        f"(std={std_val:.2f} vs page median={median_std:.2f}) — "
                        f"possible digital erasure or clone stamp"
                    ),
                })

        return self._cluster_erasure_regions(candidates)[:ERASURE_MAX_REGIONS]

    def _detect_flat_zone_patches(self, page, img: Image.Image,
                                  page_num: int, render_dpi: float) -> list:
        """
        Run the shared flat-zone detector (utils/flat_zone_detection — the
        image pipeline's Check 1/2 algorithm, including its glare-vs-edit
        boundary discrimination) over this page's low-DPI render, then
        classify each surviving patch by whether it geometrically contains
        a stamp/seal-shaped ink component (Check 7's isolation, also shared).

        Returns list[ELARegion] with flat_zone_anomaly=True, bboxes in PDF
        points. On a MIXED page (extractable text present) the patches are
        additionally restricted to embedded raster content, same as the ELA
        block candidates — a flat vector-drawn rectangle is normal design,
        not a paste-in.
        """
        gray = np.asarray(img.convert("L"), dtype=np.float64)
        std_map, baseline = local_std_map(gray)
        # Born-digital gate — two-signal, like the image pipeline's: a low
        # RENDER noise floor alone is not proof of born-digital (the 150 DPI
        # resampling can average real scan noise away, and scanner-app
        # background cleanup collapses it at the source), so a low floor
        # only gates when the page's embedded SOURCE images also carry no
        # raster-pipeline evidence (utils/pdf_utils.page_raster_source_
        # evidence). When evidence exists the detector runs with the
        # measured baseline as-is: every threshold inside is RELATIVE to
        # it, so a truly-zero floor still cannot manufacture findings —
        # the bypass only revives detection on low-but-nonzero floors.
        if baseline < BORN_DIGITAL_STD_FLOOR:
            evidence, _basis = page_raster_source_evidence(page, page.parent)
            if not evidence:
                return []

        zones, _glare = detect_flat_zones(gray, std_map, baseline)
        if not zones:
            return []

        # A "flat patch" covering most of the page is the page stock itself.
        page_area_px = gray.shape[0] * gray.shape[1]
        zones = [
            z for z in zones
            if (z["bbox"][2] * z["bbox"][3]) / max(page_area_px, 1)
            <= FLAT_ZONE_MAX_PAGE_FRACTION
        ]
        if not zones:
            return []

        stamp_bboxes = [
            c["bbox"] for c in isolate_ink_regions(np.asarray(img.convert("RGB")))
            if c["kind"] == "stamp"
        ]

        pts_scale = 72 / render_dpi
        regions = []
        for z in zones:
            x, y, w, h = z["bbox"]
            stamp_assoc = any(
                self._overlap_fraction((x, y, w, h), sb) >= FLAT_ZONE_STAMP_OVERLAP_MIN
                for sb in stamp_bboxes
            )
            regions.append(ELARegion(
                page=page_num,
                bbox=(x * pts_scale, y * pts_scale,
                      (x + w) * pts_scale, (y + h) * pts_scale),
                mean_error=z["region_std"],
                # Pseudo z for the per-page strongest-N sort — scaled so a
                # high-confidence patch outranks marginal ELA blocks.
                z_score=z["confidence"] * 10,
                render_dpi=render_dpi,
                flat_zone_anomaly=True,
                stamp_associated=stamp_assoc,
                flat_confidence=z["confidence"],
                detail=z["detail"],
            ))

        return self._restrict_to_raster_content(page, regions)

    @staticmethod
    def _overlap_fraction(flat_bbox_xywh: tuple, stamp_bbox_xywh: tuple) -> float:
        """Fraction of the STAMP's bbox area inside the flat patch — the
        'flat patch contains/closely surrounds the seal' test."""
        fx, fy, fw, fh = flat_bbox_xywh
        sx, sy, sw, sh = stamp_bbox_xywh
        ix = max(0, min(fx + fw, sx + sw) - max(fx, sx))
        iy = max(0, min(fy + fh, sy + sh) - max(fy, sy))
        return (ix * iy) / max(sw * sh, 1)

    @staticmethod
    def _cluster_erasure_regions(regions: list) -> list:
        """Merge nearby erasure candidates into single larger regions so one
        flat area doesn't get reported as dozens of overlapping small boxes."""
        if not regions:
            return []

        clustered = []
        used = set()
        for i, r1 in enumerate(regions):
            if i in used:
                continue
            group = [r1]
            used.add(i)
            x0_1, y0_1 = r1["bbox"][0], r1["bbox"][1]
            for j in range(i + 1, len(regions)):
                if j in used:
                    continue
                r2 = regions[j]
                dist = ((x0_1 - r2["bbox"][0]) ** 2 + (y0_1 - r2["bbox"][1]) ** 2) ** 0.5
                if dist < ERASURE_CLUSTER_DIST_PT:
                    group.append(r2)
                    used.add(j)

            x0 = min(r["bbox"][0] for r in group)
            y0 = min(r["bbox"][1] for r in group)
            x1 = max(r["bbox"][2] for r in group)
            y1 = max(r["bbox"][3] for r in group)
            worst = min(group, key=lambda r: r["std_val"])
            clustered.append({
                "page": r1["page"],
                "bbox": (x0, y0, x1, y1),
                "std_val": worst["std_val"],
                "median_std": worst["median_std"],
                "reason": worst["reason"],
            })

        return clustered
