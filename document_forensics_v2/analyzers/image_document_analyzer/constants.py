"""Tuning constants, scoring weights, and the honesty-required
NOT_IMPLEMENTED manifest for the image-document pipeline."""

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
