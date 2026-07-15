"""Checks 3-4: JPEG compression history + categorical double-compression
flag (see the package docstring for the Tier 2 honesty limits)."""

import numpy as np

from .constants import (
    BLOCKINESS_Z_THRESHOLD, DC_COEFS, DC_HIST_RANGE, DC_MIN_COEF_AGREEMENT,
)


class CompressionChecksMixin:
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

            q_file = int(lum_table[CompressionChecksMixin._ZIGZAG_IDX[(u, v)]])
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

