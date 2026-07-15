"""Checks 7-9: stamp/signature ink isolation (shared HSV algorithm),
ink-texture flatness, and cutout-boundary sharpness."""

import cv2
import numpy as np

from utils.flat_zone_detection import isolate_ink_regions

from .constants import (
    FLAT_INK_ABS_FLOOR, FLAT_INK_REL, STAMP_BOUNDARY_RATIO, STAMP_BOUNDARY_ABS,
)
from .report import ImageAnomaly


class StampChecksMixin:
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

