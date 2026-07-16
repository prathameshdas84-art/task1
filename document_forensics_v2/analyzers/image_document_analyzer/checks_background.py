"""Check 11: background color consistency.

Targets the painted-over / pasted-patch signature the other checks are
blind to: a region whose fill is the SAME to a human eye but measurably
a different color value than the background around it (253,253,253 vs
253,253,251). This is a COLOR-VALUE comparison between a region and its
local surroundings — deliberately not a texture/variance measurement
(Check 1's property) and it never looks at ink, stamps, or signatures
(Checks 7-9's domain): foreground pixels are masked OUT before any
color is computed.

False-positive discipline, in order of defense:
  1. Per-block color = median of BACKGROUND pixels only (paper-tone
     mask, eroded so anti-aliased text halos and printed-line edges
     never contaminate a block's color).
  2. Each block is compared against a LOCAL surround baseline (masked
     sliding median over BG_BASELINE_KERNEL blocks) — a smooth lighting
     gradient moves the baseline with the block, so gradients
     self-cancel; only a LOCALIZED deviation survives.
  3. The candidate threshold is relative to the document's OWN
     background variation (median + BG_DEV_RATIO × MAD of the deviation
     field), so scan noise and JPEG chroma wobble raise the bar
     document-by-document. An absolute floor of BG_DEV_ABS_FLOOR merely
     keeps uint8 quantization dust from ever qualifying.
  4. Surviving connected regions must look like a digital fill: minimum
     size, rectangular-ish fill ratio, per-block deltas forming a
     coherent PLATEAU (one consistent color offset, not blotch), and a
     SHARP step at the boundary — the ring of background immediately
     outside the region must already sit at the far-surround color
     (BG_STEP_NEAR_MAX). Gradual transitions (lighting, paper aging,
     glare falloff) fail the ring test.
"""

import cv2
import numpy as np

from .constants import (
    BG_BLOCK, BG_MIN_BG_FRAC, BG_PAPER_DELTA, BG_INK_ERODE,
    BG_BASELINE_KERNEL, BG_BASELINE_MIN_VALID, BG_DEV_RATIO,
    BG_DEV_ABS_FLOOR, BG_MIN_BLOCKS, BG_MIN_FILL, BG_PLATEAU_SPREAD,
    BG_STEP_NEAR_MAX, BG_RING_MIN_BLOCKS, BG_MIN_NATURAL_VARIATION,
    BG_MAX_SUBTLE_STEP,
)
from .report import ImageAnomaly


class BackgroundColorCheckMixin:
    # ── Check 11 ─────────────────────────────────────────────────────────

    def _background_color_check(self, rgb: np.ndarray, gray: np.ndarray):
        """Returns (anomalies, metrics). rgb is HxWx3 uint8, gray float64."""
        h, w = gray.shape
        nby, nbx = h // BG_BLOCK, w // BG_BLOCK
        metrics = {}
        if nby < BG_BASELINE_KERNEL // 2 or nbx < BG_BASELINE_KERNEL // 2:
            metrics["basis"] = "image too small for a local-surround baseline"
            return [], metrics

        # 1. Background mask: within reach of the document's paper tone
        #    (median of the light half), eroded so pixels bordering ink /
        #    printed lines never contribute a color sample.
        paper_med = float(np.median(gray[gray >= np.median(gray)]))
        bg = (gray >= paper_med - BG_PAPER_DELTA).astype(np.uint8)
        bg = cv2.erode(bg, np.ones((3, 3), np.uint8), iterations=BG_INK_ERODE)
        metrics["paper_tone"] = round(paper_med, 1)

        # 2. Per-block representative color: MEAN RGB of the block's
        #    background pixels; blocks dominated by foreground get no
        #    color. Mean, not median: the eroded mask already keeps ink
        #    out, and the mean has sub-uint8 resolution — a 2-value color
        #    offset must stay measurable, while medians of uint8 samples
        #    quantize to 0.5 steps and drown it.
        color = np.full((nby, nbx, 3), np.nan)
        for by in range(nby):
            ys = by * BG_BLOCK
            for bx in range(nbx):
                xs = bx * BG_BLOCK
                m = bg[ys:ys + BG_BLOCK, xs:xs + BG_BLOCK] > 0
                if m.mean() < BG_MIN_BG_FRAC:
                    continue
                color[by, bx] = rgb[ys:ys + BG_BLOCK,
                                    xs:xs + BG_BLOCK][m].mean(axis=0)
        valid = ~np.isnan(color[:, :, 0])
        metrics["background_blocks"] = int(valid.sum())
        if valid.sum() < BG_BASELINE_MIN_VALID:
            metrics["basis"] = "too little background to establish a baseline"
            return [], metrics

        # 3. Local surround baseline: masked sliding median per channel.
        #    A smooth gradient moves the baseline along with the block, so
        #    only LOCALIZED deviations survive the subtraction.
        pad = BG_BASELINE_KERNEL // 2
        padded = np.pad(color, ((pad, pad), (pad, pad), (0, 0)),
                        constant_values=np.nan)
        win = np.lib.stride_tricks.sliding_window_view(
            padded, (BG_BASELINE_KERNEL, BG_BASELINE_KERNEL), axis=(0, 1)
        )  # (nby, nbx, 3, K, K)
        win = win.reshape(nby, nbx, 3, -1)
        n_neigh = np.sum(~np.isnan(win[:, :, 0, :]), axis=-1)
        with np.errstate(all="ignore"):
            baseline = np.nanmedian(win, axis=-1)
        base_ok = valid & (n_neigh >= BG_BASELINE_MIN_VALID)

        delta = color - baseline                      # signed, per channel
        dev = np.max(np.abs(delta), axis=-1)          # Chebyshev distance
        dev_vals = dev[base_ok]

        # 4. Adaptive threshold from the document's own background wobble.
        med_dev = float(np.median(dev_vals))
        mad = float(np.median(np.abs(dev_vals - med_dev)))
        sigma = 1.4826 * mad
        threshold = max(BG_DEV_ABS_FLOOR, med_dev + BG_DEV_RATIO * sigma)
        metrics["bg_dev_median"] = round(med_dev, 3)
        metrics["bg_dev_sigma"] = round(sigma, 3)
        metrics["bg_dev_threshold"] = round(threshold, 3)

        # No measurable natural variation = vector-render content (even a
        # weak-grain scan shows ~0.03 here; a render shows exactly 0.0
        # because flat fills survive JPEG untouched). "Deviates more than
        # natural variation explains" is undefined against a variation of
        # zero — and a render's light-tinted fills/bands are legitimate
        # design, not paste — so the check abstains rather than flagging
        # every design element on born-digital-style content.
        if med_dev < BG_MIN_NATURAL_VARIATION and sigma < BG_MIN_NATURAL_VARIATION:
            metrics["basis"] = ("no measurable background variation "
                                "(vector-render content) — abstained")
            return [], metrics

        cand = (base_ok & (dev > threshold)).astype(np.uint8)
        metrics["candidate_blocks"] = int(cand.sum())
        if not cand.any():
            return [], metrics

        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
        anomalies = []
        rejections = {"size": 0, "fill": 0, "plateau": 0, "ring_data": 0,
                      "step_small": 0, "visible_step": 0, "gradual": 0}
        for i in range(1, n_lbl):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < BG_MIN_BLOCKS:
                rejections["size"] += 1
                continue
            bx0 = int(stats[i, cv2.CC_STAT_LEFT])
            by0 = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            fill = area / (bw * bh)
            if fill < BG_MIN_FILL:
                rejections["fill"] += 1
                continue
            region = lbl == i

            # A patch boundary rarely lands on the 16px grid, so the
            # region's rim blocks mix patch and surround colors. Judge the
            # plateau and the region color on the INTERIOR when the region
            # is big enough to have one.
            reg_u8 = region.astype(np.uint8)
            interior = cv2.erode(reg_u8, np.ones((3, 3), np.uint8)) > 0
            core = interior if interior.sum() >= BG_MIN_BLOCKS else region

            # Plateau coherence: one consistent color offset across the
            # region (a digital fill), not incoherent blotch. Judged on
            # the dominant channel's signed per-block deltas.
            core_delta = delta[core]                        # (n, 3) signed
            dom = int(np.argmax(np.abs(np.median(core_delta, axis=0))))
            d_dom = core_delta[:, dom]
            d_med = float(np.median(d_dom))
            if np.std(d_dom) > max(0.75, BG_PLATEAU_SPREAD * abs(d_med)):
                rejections["plateau"] += 1
                continue

            # Sharp-step ring test: background just outside the region
            # must already be AT the far-surround color; a gradual
            # transition (lighting, glare falloff) fails here.
            near = (cv2.dilate(reg_u8, np.ones((3, 3), np.uint8)) > 0) \
                & ~region & valid
            far_out = cv2.dilate(reg_u8, np.ones((11, 11), np.uint8)) > 0
            far_in = cv2.dilate(reg_u8, np.ones((5, 5), np.uint8)) > 0
            far = far_out & ~far_in & valid
            if near.sum() < BG_RING_MIN_BLOCKS or far.sum() < BG_RING_MIN_BLOCKS:
                rejections["ring_data"] += 1
                continue
            c_in = np.median(color[core], axis=0)
            c_near = np.median(color[near], axis=0)
            c_far = np.median(color[far], axis=0)
            step_total = float(np.max(np.abs(c_in - c_far)))
            near_resid = float(np.max(np.abs(c_near - c_far)))
            if step_total < threshold:
                rejections["step_small"] += 1
                continue
            if step_total > BG_MAX_SUBTLE_STEP:
                # An eye-visible color difference is not this check's
                # signature (glare, shadow, sticker, printed panel) —
                # the paste case is the SUBTLE one the eye reads as
                # "the same background".
                rejections["visible_step"] += 1
                continue
            if near_resid > BG_STEP_NEAR_MAX * step_total:
                rejections["gradual"] += 1
                continue

            conf = float(np.clip(step_total / (2.5 * threshold), 0.35, 0.95))
            anomalies.append(ImageAnomaly(
                type="background_color_mismatch",
                bbox=(bx0 * BG_BLOCK, by0 * BG_BLOCK,
                      bw * BG_BLOCK, bh * BG_BLOCK),
                confidence=round(conf, 2),
                evidence_check="check11_background_color",
                detail=(
                    f"background color ({c_in[0]:.0f},{c_in[1]:.0f},"
                    f"{c_in[2]:.0f}) vs surrounding ({c_far[0]:.0f},"
                    f"{c_far[1]:.0f},{c_far[2]:.0f}) — max-channel step "
                    f"{step_total:.1f} over sharp boundary (document "
                    f"background varies {med_dev:.2f}±{sigma:.2f}, "
                    f"threshold {threshold:.2f})"
                ),
            ))
        metrics["rejections"] = rejections
        return anomalies, metrics
