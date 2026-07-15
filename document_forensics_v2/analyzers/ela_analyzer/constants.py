"""ELA layer tuning constants: DPI ladder, block sizing, scoring
weights, and incremental-update/shadow-attack scoring."""


RENDER_DPI    = 150
# Vector PDFs (text outlined to paths — Canva/Figma/Illustrator exports)
# have no raster compression artifacts to find at low DPI since there's no
# embedded detail to begin with; rendering at higher DPI gives ELA enough
# pixels per block to produce a meaningful signal.
VECTOR_PDF_RENDER_DPI = 300
BLOCK_SIZE    = 32
ELA_QUALITY   = 75
ELA_AMPLIFY   = 15
Z_THRESHOLD   = 3.0
MIN_BLOCKS    = 4

# BLOCK_SIZE (32) is in PIXELS at RENDER_DPI — the canonical physical block
# size it represents is therefore fixed in PDF points, not pixels. Any code
# that blocks out an image rendered at a DIFFERENT dpi must scale the pixel
# block size accordingly (see _block_px_for_dpi) — otherwise a "32px block"
# at 600 DPI covers 1/4 the physical area it does at 150 DPI, zooms into
# individual glyph edges, and flags ordinary text as an outlier at every
# scale (the exact false positive multi-scale analysis is meant to remove).
BLOCK_SIZE_PT = BLOCK_SIZE * 72 / RENDER_DPI

# Multi-quality ELA: recompress at several JPEG quality levels instead of
# just one. A logo or dense-text block recompresses with high error at low
# quality but settles down at high quality (it's just naturally busy, not
# edited) — a genuinely pasted-in region recompresses abnormally at EVERY
# quality level because its underlying compression history doesn't match
# the rest of the page. Requiring agreement across quality levels filters
# out the former while still catching the latter.
JPEG_QUALITIES              = [75, 85, 95]
MULTI_QUALITY_MIN_AGREEMENT = 2   # block must be flagged at >=2 of 3 levels

# Margin sampled to estimate the document's baseline recompression noise
# floor — top/bottom/left/right 5% of the page, assumed blank.
NOISE_FLOOR_MARGIN_FRACTION = 20  # margin = page_dim // this value (5%)

# Per-page boxes are capped to the strongest N outliers so the UI doesn't
# flood with low-confidence boxes when a page has many flagged blocks.
MAX_REGIONS_PER_PAGE = 10

# ── Multi-scale (multi-DPI) analysis ────────────────────────────────────────
#
# A single render resolution can't tell "naturally busy content" (logo,
# dense header) apart from "genuinely pasted-in content" reliably — both
# produce high-error blocks at low DPI. Rendering the SAME page at multiple
# resolutions and requiring agreement does: a logo is a fixed-size raster
# asset, so its apparent recompression-error signature shifts as block
# boundaries land on different sub-pixel detail at each DPI; a genuine edit's
# compression-history mismatch is structural, not resolution-dependent, so it
# keeps reappearing at every scale.
#
# Phased like a search, not a brute-force triple-render of every page:
#   Phase 1 (low DPI):    fast full-document sweep for candidate blocks.
#   Phase 2 (medium DPI): re-render ONLY pages with candidates, confirm.
#   Phase 3 (high DPI):   crop-render ONLY confirmed blocks for exact
#                         location + a page-wide text-sharpness check.
# This keeps clean documents cheap (phase 1 only) while still paying the
# 600 DPI cost on documents that actually have something to confirm.
RENDER_SCALES        = [("low", 150), ("medium", 300), ("high", 600)]
# Vector PDFs have no raster compression artifacts at low DPI to begin with
# (see VECTOR_PDF_RENDER_DPI below) — shift the whole scale ladder up so the
# "low" tier still produces a meaningful signal.
RENDER_SCALES_VECTOR = [("low", 300), ("medium", 450), ("high", 600)]
SCALE_CONFIRM_MIN_AGREEMENT = 2     # block must hold up at >=2 of 3 DPIs

# Tolerance (PDF points) when checking whether a block flagged at one DPI's
# block grid overlaps a block flagged at a different DPI's (coarser/finer)
# block grid — grid boundaries don't land on identical points across scales.
SCALE_MATCH_PADDING_PT = 3.0

# Confirmed-block scoring: a block that survives 2+ independent DPI scales
# is a high-precision signal (rare, specific) rather than the noisy raw
# single-scale fraction the old FRACTION_TO_SCORE_MULTIPLIER scheme scored
# off of — score per confirmed block instead of per fraction-of-page-blocks.
CONFIRMED_BLOCK_SCORE_PER_BLOCK = 15
CONFIRMED_BLOCK_SCORE_CAP       = 70

# Scanned-document calibration: scanner compression artifacts and paper
# texture create a baseline ELA noise floor scattered across the WHOLE
# page as many small, spatially-isolated 1-2-block hits (confirmed by
# testing: 25 such hits spread across all 3 pages of a real scanned
# payslip, no two near each other). A genuine edit covers a whole word or
# line of replaced text, so it shows up as ONE contiguous cluster of
# several touching blocks. Requiring the high-DPI 3rd scale to also agree
# was tried first and rejected — testing showed it almost never confirms
# ANYTHING, even on a document with a known real edit, making it too blunt
# a gate (would suppress real positives as readily as noise). Cluster size
# is the discriminator that actually separates the two cases.
SCANNED_MIN_CLUSTER_SIZE          = 3     # blocks; smaller clusters = scattered noise, not an edit
SCANNED_CLUSTER_GAP_TOLERANCE_PT  = 4.0   # how close two blocks must be to count as "touching"
SCANNED_SCORE_MULTIPLIER          = 0.6
SCANNED_LOW_HIT_COUNT             = 5     # fewer confirmed blocks than this = likely noise, not edit
SCANNED_LOW_HIT_MULTIPLIER        = 0.5
SCANNED_SIGNATURE_ZONE_FRACTION   = 0.15  # bottom of page — signatures always cause ELA noise
SCANNED_HEADER_ZONE_FRACTION      = 0.12  # top of page — printed logos/letterhead on a scan
SCANNED_HEADER_WEIGHT_MULTIPLIER  = 0.6   # reduce (not drop) header-zone hits by 40%

# Compiled/merged-document calibration: when multiple separately-scanned
# source documents are merged into one PDF (via pypdf, PDF24, etc.) each
# source page has its own independent JPEG compression baseline.  ELA sees
# those page-boundary differences as "edits" and fires false positives on
# every page.  Applying stricter thresholds dramatically reduces this noise
# without suppressing genuine single-page edit signals.
COMPILED_PHASE1_Z_THRESHOLD  = 4.5   # vs Z_THRESHOLD=3.0 — only keep strong phase-1 hits
COMPILED_MIN_CLUSTER_SIZE    = 5     # vs SCANNED_MIN_CLUSTER_SIZE=3 — require larger clusters
COMPILED_SCORE_MULTIPLIER    = 0.35  # vs SCANNED_SCORE_MULTIPLIER=0.6 — reduce score weight
COMPILED_MIN_PAGES           = 4     # multi-page scanned doc threshold for compiled detection

# High-DPI region refinement: how much padding (in low-DPI block-equivalents)
# to render around a confirmed block when cropping for exact-location
# refinement, so the crop has enough surrounding context to compute a
# meaningful local mean/std rather than just the suspicious block itself.
HIGH_DPI_CROP_PADDING_BLOCKS = 2

# Text-sharpness anomaly (edited text rendered by a different tool/AA
# settings than the rest of the page) — z-score cutoff against the page's
# OWN text-block sharpness distribution.
SHARPNESS_Z_THRESHOLD = 3.5
SHARPNESS_RENDER_DPI  = 600

# Image-document noise-consistency check (scanned/photographed pages) —
# z-score cutoff for a 32x32 noise-variance block to count as anomalous,
# plus the score weights for however many such blocks get found.
#
# Empirically validated against a real photographed government-ID page
# (dense security micro-print/hologram texture) plus a synthetic
# cover-and-retype tamper built from the same page: natural ID-card
# texture alone produces block z-scores up to 9.2 with zero tampering,
# while the tamper's blocks peaked at 13.0 (several blocks landing
# 9.9-13.0) — a threshold of 4.0 flagged 384 untampered blocks across a
# 12-page real scan, all false positives. 9.5 sits just above the
# observed natural ceiling and below the tamper's confirmed hits.
NOISE_Z_THRESHOLD       = 9.5
NOISE_SCORE_PER_REGION  = 8
NOISE_SCORE_CAP         = 40

# "Too clean" / digital-erasure detection. A digital eraser or clone stamp
# leaves a region with near-zero pixel variance — real paper/background,
# even on a clean scan, has microscopic sensor noise. This is restricted to
# LIGHT pixels only (background, not text strokes or dense image content),
# unlike _analyze_noise_consistency (which scans the whole page and flags
# BOTH directions — too noisy or too clean). Treat this as a complementary,
# narrower signal, not a replacement: it only fires deep inside a flat
# background region, where _analyze_noise_consistency's blocks would mostly
# sit near its own mean and not cross NOISE_Z_THRESHOLD.
ERASURE_BLOCK_SIZE       = 15
ERASURE_STRIDE           = 8
ERASURE_BG_MIN_BRIGHTNESS = 180   # only check background-colored pixels, not text/photos
ERASURE_RATIO_THRESHOLD  = 0.2    # block std below this fraction of the page's median std = "too clean"
ERASURE_MIN_MEDIAN_STD   = 3.0    # a page already this flat overall isn't a real scan — skip it
ERASURE_CLUSTER_DIST_PT  = 30
ERASURE_SCORE_PER_REGION = 15
ERASURE_SCORE_CAP        = 45
ERASURE_MAX_REGIONS      = 10

# Flat/pasted-patch detection on scanned/mixed raster pages — the SAME
# flat-zone + boundary-step algorithm the image pipeline's Check 1/2 uses
# (utils/flat_zone_detection), applied to each raster page render. Catches
# the pattern neither existing image-doc check can: a stamp/seal sitting on
# a flat, texture-less rectangle starkly different from the surrounding
# scan grain. _analyze_noise_consistency can't fire on it (a low-variance
# block can never get 9.5 sigma BELOW the page mean) and
# _detect_erased_regions only examines near-white blocks (>=180 brightness),
# so a DARK pasted patch behind a seal passed both. Gated to non-native_text
# documents (see the run condition in analyze()) plus the same born-digital
# noise-floor gate the image pipeline applies — a render with no scan noise
# makes flat regions normal, not evidence. Scored into this layer's
# anomaly_score (rides WEIGHTS["ela"], no new weighted layer), same pattern
# as the noise/erasure checks above.
FLAT_ZONE_SCORE_PER_REGION   = 15    # × the region's 0-1 confidence
FLAT_ZONE_SCORE_CAP          = 45
FLAT_ZONE_STAMP_OVERLAP_MIN  = 0.5   # fraction of a stamp's bbox inside the
                                     # flat patch for "pasted stamp" labeling
FLAT_ZONE_MAX_PAGE_FRACTION  = 0.5   # a "flat region" covering more than half
                                     # the page is the page background itself
                                     # (e.g. uniform colored form stock), not
                                     # a pasted patch

# Cross-page noise-consistency check (possible whole-page substitution).
CROSS_PAGE_MIN_PAGES     = 3     # need at least this many pages to compare
CROSS_PAGE_Z_THRESHOLD   = 2.5
CROSS_PAGE_SCORE_PER_PAGE = 20
CROSS_PAGE_SCORE_CAP      = 60
CROSS_PAGE_MERGE_DIVISOR  = 2    # how much this sub-score contributes to the final score

# PDF object-fingerprinting score weights (incremental updates, deleted/
# reused objects, FreeText/Redact annotations, Form XObjects).
EOF_SCORE_PER_REVISION       = 15
EOF_SCORE_CAP                = 40
HIGH_GEN_SCORE_PER_OBJECT    = 10
HIGH_GEN_SCORE_CAP           = 35
FREETEXT_SCORE_PER_ANNOT     = 15
FREETEXT_SCORE_CAP           = 40

# Form XObjects are a standard PDF mechanism for reusable content (logos,
# letterhead headers/footers, repeated stamps) — not inherently a sign of
# paste-over editing. A template component shows up on most pages, sits in
# the header/footer band, is small relative to the page, or is image-only
# (no text inside it to "paste over" anything with). Only an XObject that
# is rare, body-positioned, and contains its own text content resembles an
# injected paste-over rather than a reused template element.
FORM_XOBJECT_MAX_TEMPLATE_FREQUENCY = 0.5   # appears on >=50% of pages = template, not injected
FORM_XOBJECT_HEADER_ZONE_FRACTION   = 0.15  # top 15% of page = header/branding
FORM_XOBJECT_FOOTER_ZONE_FRACTION   = 0.10  # bottom 10% of page = footer/branding
FORM_XOBJECT_MIN_AREA_FRACTION      = 0.05  # below this = small logo/stamp
FORM_XOBJECT_SCORE_PER_ITEM  = 10
FORM_XOBJECT_SCORE_CAP       = 30
OBJECT_MERGE_DIVISOR         = 3   # how much the object-fingerprint sub-score contributes to the final score

# Incremental-update / old-object-recovery scoring. A conformant PDF reader
# (pikepdf included) only ever resolves the MOST RECENT xref entry for a
# given object id — a prior revision's bytes are never reachable through the
# normal object API, just shadowed in the file. We find them with a raw
# byte scan for repeated "<id> <gen> obj" definitions instead.
INCREMENTAL_EOF_SCORE_PER_REVISION = 20
INCREMENTAL_EOF_SCORE_CAP          = 50
INCREMENTAL_XREF_MISMATCH_SCORE    = 25
INCREMENTAL_OLD_OBJECTS_SCORE      = 20
INCREMENTAL_MERGE_DIVISOR          = 2  # how much this sub-score contributes to the final score
OLD_OBJECT_PREVIEW_BYTES           = 200
OLD_OBJECT_MAX_REPORTED            = 5

# DCT coefficient analysis (8x8 JPEG blocks).
DCT_BLOCK_SIZE       = 8
DCT_MIN_BLOCKS       = 10   # need at least this many 8x8 blocks to compute stats
DCT_Z_THRESHOLD      = 3.5  # higher than Z_THRESHOLD — DCT energy is noisier than ELA error
DCT_SCORE_PER_REGION = 3
DCT_SCORE_CAP        = 30
DCT_MERGE_DIVISOR    = 4    # how much the DCT sub-score contributes to the final score

# Shadow attack detection: new content appended after a digital signature
# via PDF incremental updates. The signature still cryptographically
# validates (it only covers the bytes present when it was applied) but the
# visible content has changed since signing.
SHADOW_EOF_SIG_SCORE            = 50  # incremental updates + a signature present
SHADOW_BYTERANGE_GAP_SCORE      = 60  # signature's ByteRange doesn't cover the whole file
SHADOW_OBJECTS_AFTER_SIG_SCORE  = 40  # bytes exist after the signed range
SHADOW_ATTACK_SCORE_DIVISOR     = 2   # how much this sub-score contributes to the final score

# Digital signature validation.
SIG_BYTERANGE_GAP_SCORE           = 70  # ByteRange doesn't cover the entire file
SIG_MODIFIED_AFTER_SIGNING_SCORE  = 50  # document ModDate is after the signing date
SIGNATURE_SCORE_DIVISOR           = 2   # how much this sub-score contributes to the final score

