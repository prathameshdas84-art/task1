"""
Shared flat-zone / pasted-patch detection primitives.

Extracted from analyzers/image_document_analyzer.py so the SAME algorithm
serves both pipelines instead of being duplicated:

  * image_document_analyzer.py (direct JPG/PNG uploads) — Check 1's local
    variance / flat-zone detection, Check 2's glare-vs-digital-edit
    boundary discrimination, and Check 7's stamp/signature ink isolation
    now call these functions; behavior is unchanged.
  * ela_analyzer.py (raster pages of scanned/mixed PDFs via /analyze) —
    a stamp/seal sitting on a flat, texture-less rectangular patch that's
    visibly inconsistent with the surrounding scan noise was previously
    never localized or boxed there (its noise-consistency z-score check
    can't statistically fire on LOW variance, and its erasure check only
    examines near-white blocks); it now applies this same detector to
    each raster page.

Numeric discipline (same rules as the image pipeline): variance math in
float64 BEFORE squaring; E[x^2]-E[x]^2 clipped at 0 before sqrt; every
threshold RELATIVE to the document's own measured baseline. The
born-digital gate is the CALLER's responsibility — skip calling
detect_flat_zones() when the baseline from local_std_map() is below
BORN_DIGITAL_STD_FLOOR (a render with no sensor noise makes flat zones
normal, not evidence).
"""

import cv2
import numpy as np

# ── Flat-zone (inpaint smoothing / pasted patch) detection ──────────────────
VAR_WINDOW              = 7      # local-std window (px)
FLAT_BLOCK              = 16     # analysis block size (px)
BASELINE_PERCENTILE     = 30     # block-std percentile used as background noise baseline
FLAT_RATIO              = 0.35   # block flagged flat if its std < ratio * baseline
FLAT_MIN_BLOCKS         = 4      # min connected flagged blocks to form a region
BORN_DIGITAL_STD_FLOOR  = 1.2    # baseline below this = born-digital render → gate out
PIXEL_FLAT_RATIO        = 0.5    # pixel-level mask refinement threshold (× baseline)

# ── Glare vs digital-edit boundary discrimination ───────────────────────────
GLARE_BRIGHT_DELTA      = 20     # interior brighter than the global background by this
GLARE_MIN_BRIGHTNESS    = 230    # and near saturation
STEP_RATIO_THRESHOLD    = 0.60   # std recovery in the inner ring: >= this = sharp step
# A bright saturated region gets a much stricter step requirement before it
# can be called an edit: genuine glare clips the sensor gradually (variance
# falls off over tens of px), while a pasted white patch is a true step
# (ratio ~1.0). Between the two sits "bright but ambiguous" — excluded.
BRIGHT_STEP_THRESHOLD   = 0.85
RING_INNER_PX           = 6      # ring band just outside the flat mask
RING_OUTER_PX           = 24     # outer band = local recovery target for the ratio

# ── Stamp/signature ink isolation ───────────────────────────────────────────
INK_SAT_MIN             = 70     # HSV saturation floor for colored ink
INK_VAL_MIN             = 50
INK_MIN_AREA            = 400    # px — smaller components are specks
INK_HUE_BANDS           = [(0, 10), (100, 165), (170, 180)]  # red, blue→purple, red-wrap
SIG_MAX_STROKE_HALFW    = 3.5    # distance-transform max ≤ this (px) = pen-stroke thin


def local_std_map(gray: np.ndarray):
    """Local std via E[x^2]-E[x]^2 — float64 BEFORE squaring, clipped
    at 0 BEFORE sqrt (negative epsilons from float rounding otherwise
    become NaNs). Baseline = low percentile of blockwise medians, i.e.
    the document's own background noise floor, NOT an absolute constant."""
    g = gray.astype(np.float64)
    m = cv2.blur(g, (VAR_WINDOW, VAR_WINDOW))
    m2 = cv2.blur(g * g, (VAR_WINDOW, VAR_WINDOW))
    var = np.clip(m2 - m * m, 0.0, None)
    std_map = np.sqrt(var)

    h, w = std_map.shape
    nby, nbx = h // FLAT_BLOCK, w // FLAT_BLOCK
    if nby == 0 or nbx == 0:
        return std_map, float(np.median(std_map))
    blocks = std_map[: nby * FLAT_BLOCK, : nbx * FLAT_BLOCK].reshape(
        nby, FLAT_BLOCK, nbx, FLAT_BLOCK
    )
    block_med = np.median(blocks, axis=(1, 3))
    baseline = float(np.percentile(block_med, BASELINE_PERCENTILE))
    return std_map, baseline


def detect_flat_zones(gray: np.ndarray, std_map: np.ndarray, baseline: float):
    """Flat-zone detection with glare discrimination (image pipeline's
    Checks 1+2): blocks whose local std collapsed relative to the
    document's own baseline; each candidate region's boundary is
    classified sharp-step (digital edit) vs gradual (physical glare/
    defocus) by measuring how much of the local std recovery happens in
    the INNER ring (1..RING_INNER_PX outside the mask) vs the outer ring
    band — a digital edit's variance snaps back to baseline within the
    std window's width; glare's falloff spans tens of px. Bright
    saturated regions (glare candidates) must show an emphatic step
    (BRIGHT_STEP_THRESHOLD) before being called an edit — a pasted
    white patch is a true step, glare never is.

    Returns (zones, glare_regions):
      zones: list of dicts {"bbox": (x, y, w, h) px, "confidence": 0-1,
             "region_std": float, "step_ratio": float, "detail": str}
      glare_regions: list of (x, y, w, h) px excluded as physical glare.
    """
    h, w = std_map.shape
    nby, nbx = h // FLAT_BLOCK, w // FLAT_BLOCK
    blocks = std_map[: nby * FLAT_BLOCK, : nbx * FLAT_BLOCK].reshape(
        nby, FLAT_BLOCK, nbx, FLAT_BLOCK
    )
    block_med = np.median(blocks, axis=(1, 3))
    flat_grid = (block_med < FLAT_RATIO * baseline).astype(np.uint8)

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(flat_grid, 8)
    zones, glare_regions = [], []
    global_bg = float(np.median(gray))

    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] < FLAT_MIN_BLOCKS:
            continue
        bx = stats[i, cv2.CC_STAT_LEFT] * FLAT_BLOCK
        by = stats[i, cv2.CC_STAT_TOP] * FLAT_BLOCK
        bw = stats[i, cv2.CC_STAT_WIDTH] * FLAT_BLOCK
        bh = stats[i, cv2.CC_STAT_HEIGHT] * FLAT_BLOCK

        # Pixel-level mask refinement inside an expanded bbox, so the
        # boundary rings measure the TRUE flat edge, not the 16px
        # block-grid quantization of it.
        pad = RING_OUTER_PX + VAR_WINDOW
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        sub_std = std_map[y0:y1, x0:x1]
        sub_gray = gray[y0:y1, x0:x1]
        grid_region = (lbl == i).astype(np.uint8)
        region_px = np.kron(grid_region, np.ones((FLAT_BLOCK, FLAT_BLOCK), np.uint8))
        region_px = np.pad(
            region_px,
            ((0, h - region_px.shape[0]), (0, w - region_px.shape[1])),
            constant_values=0,
        )[y0:y1, x0:x1]
        mask = ((sub_std < PIXEL_FLAT_RATIO * baseline) &
                (cv2.dilate(region_px, np.ones((FLAT_BLOCK, FLAT_BLOCK), np.uint8)) > 0))
        mask = mask.astype(np.uint8)
        if mask.sum() < FLAT_MIN_BLOCKS * FLAT_BLOCK * FLAT_BLOCK * 0.25:
            continue

        k_in = np.ones((2 * RING_INNER_PX + 1,) * 2, np.uint8)
        k_out = np.ones((2 * RING_OUTER_PX + 1,) * 2, np.uint8)
        dil_in = cv2.dilate(mask, k_in)
        dil_out = cv2.dilate(mask, k_out)
        ring1 = (dil_in > 0) & (mask == 0)
        ring2 = (dil_out > 0) & (dil_in == 0)
        if ring1.sum() < 20 or ring2.sum() < 20:
            continue

        s_in = float(np.median(sub_std[mask > 0]))
        # p75, not median: the inner ring band spans 1..RING_INNER_PX
        # outside the mask, and a step edge only reaches baseline in the
        # band's outer half (the local-std window smears the first few
        # px) — the median would under-read a genuine step.
        s_ring1 = float(np.percentile(sub_std[ring1], 75))
        s_ring2 = float(np.median(sub_std[ring2]))
        b_in = float(np.mean(sub_gray[mask > 0]))
        # Recovery target is LOCAL (the outer ring, floored at the global
        # baseline): "how much of the way back to the surrounding noise
        # level does std get within RING_INNER_PX of the boundary?"
        target = max(baseline, s_ring2)
        step_ratio = float(np.clip(
            (s_ring1 - s_in) / max(target - s_in, 1e-6), 0.0, 1.5
        ))

        # Bright saturated candidates (glare-shaped) need an emphatic
        # step to still be called an edit; everything else uses the
        # normal step threshold. Below threshold = physical (glare) or
        # indiscriminable (defocus/shadow falloff) — honestly excluded
        # rather than scored.
        is_bright = (b_in > global_bg + GLARE_BRIGHT_DELTA
                     and b_in > GLARE_MIN_BRIGHTNESS)
        threshold = BRIGHT_STEP_THRESHOLD if is_bright else STEP_RATIO_THRESHOLD
        if step_ratio < threshold:
            glare_regions.append((int(bx), int(by), int(bw), int(bh)))
            continue

        depth = 1.0 - min(1.0, s_in / max(baseline, 1e-6))
        conf = float(np.clip(0.4 * depth + 0.6 * min(step_ratio, 1.0), 0.0, 1.0))
        zones.append({
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "confidence": round(conf, 2),
            "region_std": s_in,
            "step_ratio": step_ratio,
            "detail": (f"region std {s_in:.2f} vs baseline {baseline:.2f}, "
                       f"boundary step-ratio {step_ratio:.2f} (sharp)"),
        })
    return zones, glare_regions


def isolate_ink_regions(rgb: np.ndarray):
    """HSV isolation of high-saturation stamp/signature ink (red/blue/
    purple bands) — the image pipeline's Check 7. Components are
    classified stamp (thick/solid) vs signature-like (thin pen stroke,
    via distance-transform thickness). Black/graphite ink has no color
    separation and is out of scope here."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hch, sch, vch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_mask = np.zeros(hch.shape, bool)
    for lo, hi in INK_HUE_BANDS:
        hue_mask |= (hch >= lo) & (hch <= hi)
    mask = (hue_mask & (sch >= INK_SAT_MIN) & (vch >= INK_VAL_MIN)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] < INK_MIN_AREA:
            continue
        comp_mask = (lbl == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 3)
        max_halfwidth = float(dist.max())
        kind = "signature" if max_halfwidth <= SIG_MAX_STROKE_HALFW else "stamp"
        comps.append({
            "mask": comp_mask,
            "bbox": (int(stats[i, cv2.CC_STAT_LEFT]),
                     int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]),
                     int(stats[i, cv2.CC_STAT_HEIGHT])),
            "kind": kind,
            "stroke_halfwidth": max_halfwidth,
        })
    return comps
