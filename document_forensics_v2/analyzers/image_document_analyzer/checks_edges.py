"""Check 5: glyph/edge rendering sharpness (primary overlay signal) and
Check 10: near-white micro-contrast heatmap (display-only evidence)."""

import cv2
import numpy as np

from .constants import (
    EDGE_AMP_WINDOW, EDGE_GRAD_FLOOR, EDGE_AMP_FLOOR, SHARP_CELL,
    SHARP_CELL_MIN_EDGES, SHARP_RATIO, SHARP_ABS_MIN, SHARP_MIN_CELLS,
    SHARP_BASELINE_GATE, HEATMAP_BAND_LOW, HEATMAP_BAND_HIGH,
)
from .report import ImageAnomaly


class EdgeChecksMixin:
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
