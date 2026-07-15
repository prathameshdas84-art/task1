"""Core ELA math: noise-floor estimation and the per-block JPEG
recompression-error grid."""

import io

import numpy as np
from PIL import Image

from .constants import *
from .models import ELARegion


class ElaGridMixin:
    def _estimate_noise_floor(self, img: Image.Image, quality: int = ELA_QUALITY) -> float:
        """
        Estimate document noise floor from blank margin regions.
        Margins (top 5%, bottom 5%, left 5%, right 5%) are typically blank
        and represent the natural noise level of the document/scan.
        Returns mean ELA error in margin regions.
        """
        try:
            w, h = img.size
            margin_x = max(1, w // NOISE_FLOOR_MARGIN_FRACTION)  # 5% margin
            margin_y = max(1, h // NOISE_FLOOR_MARGIN_FRACTION)

            # Compute ELA on full image
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            recompressed = Image.open(buf)

            orig_arr  = np.asarray(img, dtype=np.int16)
            recom_arr = np.asarray(recompressed, dtype=np.int16)
            diff = np.abs(orig_arr - recom_arr).mean(axis=2).astype(np.float32)

            # Sample margin regions
            top_margin    = diff[:margin_y, :]
            bottom_margin = diff[h-margin_y:, :]
            left_margin   = diff[:, :margin_x]
            right_margin  = diff[:, w-margin_x:]

            margin_values = np.concatenate([
                top_margin.flatten(),
                bottom_margin.flatten(),
                left_margin.flatten(),
                right_margin.flatten(),
            ])

            return float(np.mean(margin_values)) + float(np.std(margin_values))
        except Exception:
            return 0.0

    @staticmethod
    def _block_px_for_dpi(dpi: float) -> int:
        """
        Pixel block size that covers the SAME physical PDF-point area
        (BLOCK_SIZE_PT) regardless of which DPI the page was rendered at.
        Without this, a fixed pixel block size at a higher DPI zooms into a
        smaller and smaller physical region — at 600 DPI a constant-pixel
        block lands on individual glyph edges instead of "the same patch of
        the page" used at 150/300 DPI, breaking the cross-scale comparison.
        """
        return max(8, round(BLOCK_SIZE_PT * dpi / 72))

    def _ela_block_grid(self, img: Image.Image, gray: np.ndarray, quality: int,
                         block_px: int = BLOCK_SIZE):
        """
        Recompress the page at one JPEG quality level and compute the
        per-block normalized error/z-score grids. Factored out of
        _analyze_page so it can be run once per entry in JPEG_QUALITIES.

        block_px is the block size in PIXELS for the DPI `img` was rendered
        at — callers comparing across DPIs must pass a DPI-scaled value
        (see _block_px_for_dpi) so each block covers the same physical area.

        Returns (z_flat, errors_flat, n_rows, n_cols), or (None, None, 0, 0)
        if the page is too small to block out at this resolution.
        """
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig_arr  = np.asarray(img, dtype=np.int16)
        recom_arr = np.asarray(recompressed, dtype=np.int16)
        diff = np.abs(orig_arr - recom_arr).sum(axis=2)  # H x W error map

        h, w = diff.shape
        n_rows = h // block_px
        n_cols = w // block_px
        if n_rows < 2 or n_cols < 2:
            return None, None, 0, 0

        # Normalize each block's recompression error by its own texture
        # (grayscale std-dev) — a block of dense text/logo legitimately
        # recompresses with more error than blank page space, so raw error
        # alone can't tell "naturally busy" apart from "edited."
        ratios = np.zeros((n_rows, n_cols), dtype=np.float64)
        errors = np.zeros((n_rows, n_cols), dtype=np.float64)
        for r in range(n_rows):
            for c in range(n_cols):
                block_diff = diff[
                    r * block_px:(r + 1) * block_px,
                    c * block_px:(c + 1) * block_px,
                ]
                block_gray = gray[
                    r * block_px:(r + 1) * block_px,
                    c * block_px:(c + 1) * block_px,
                ]
                err = block_diff.mean()
                texture = block_gray.std()
                errors[r, c] = err
                ratios[r, c] = err / (texture + 1.0)

        flat_ratios = ratios.flatten()
        flat_errors = errors.flatten()
        if flat_ratios.size < MIN_BLOCKS:
            return None, None, n_rows, n_cols

        noise_floor = self._estimate_noise_floor(img, quality)
        # Subtract noise floor from errors before scoring
        # This removes the baseline noise common to all blocks
        flat_errors_norm = np.maximum(0, flat_errors - noise_floor)
        flat_ratios_norm = flat_errors_norm / (flat_ratios + 1e-6) * flat_ratios

        mean = flat_ratios_norm.mean()
        std  = flat_ratios_norm.std()
        z = (
            np.zeros_like(flat_ratios_norm) if std < 1e-6
            else (flat_ratios_norm - mean) / std
        )

        return z, flat_errors, n_rows, n_cols

    def _analyze_page(self, img: Image.Image, page_num: int, render_dpi: float = RENDER_DPI):
        gray = np.asarray(img.convert("L"), dtype=np.float64)

        # Block size must be scaled to this page's render DPI so a block
        # covers the same physical page area regardless of resolution —
        # see _block_px_for_dpi.
        block_px = self._block_px_for_dpi(render_dpi)

        # Run ELA at multiple JPEG quality levels and only keep the block
        # grids that came back with usable stats (page big enough to block
        # out at block_px).
        z_grids, error_grids = [], []
        n_rows = n_cols = 0
        for quality in JPEG_QUALITIES:
            z, errors, nr, nc = self._ela_block_grid(img, gray, quality, block_px)
            if z is None:
                continue
            z_grids.append(z)
            error_grids.append(errors)
            n_rows, n_cols = nr, nc

        if len(z_grids) < MULTI_QUALITY_MIN_AGREEMENT:
            return [], 0, 0

        # Coordinate conversion must use the DPI this page was actually
        # rendered at (render_dpi), not a fixed instance default — vector
        # PDFs render at VECTOR_PDF_RENDER_DPI, not RENDER_DPI, and using
        # the wrong scale here would put boxes in the wrong place.
        pts_scale = 72 / render_dpi
        n_blocks = z_grids[0].size

        # Only flag a block if it's a statistical outlier at >=2 of the 3
        # quality levels. A logo/header block recompresses badly at low
        # quality (75) but settles down at high quality (95) — it won't
        # clear the threshold consistently. A genuinely pasted-in region's
        # compression-history mismatch shows up at every quality level.
        regions   = []
        n_flagged = 0
        for idx in range(n_blocks):
            agreeing = [z[idx] for z in z_grids if z[idx] > Z_THRESHOLD]
            if len(agreeing) >= MULTI_QUALITY_MIN_AGREEMENT:
                n_flagged += 1
                avg_z     = sum(z[idx] for z in z_grids) / len(z_grids)
                avg_error = sum(e[idx] for e in error_grids) / len(error_grids)
                r, c = divmod(idx, n_cols)
                x0 = c * block_px * pts_scale
                y0 = r * block_px * pts_scale
                x1 = (c + 1) * block_px * pts_scale
                y1 = (r + 1) * block_px * pts_scale
                regions.append(ELARegion(
                    page=page_num,
                    bbox=(x0, y0, x1, y1),
                    mean_error=float(avg_error),
                    z_score=float(avg_z),
                    render_dpi=render_dpi,
                ))

        # Cap how many boxes get drawn per page — keep only the strongest
        # outliers so the UI doesn't flood with low-confidence boxes.
        regions.sort(key=lambda r: r.z_score, reverse=True)
        regions = regions[:MAX_REGIONS_PER_PAGE]

        return regions, n_blocks, n_flagged

