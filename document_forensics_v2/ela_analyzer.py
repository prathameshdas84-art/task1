"""
Error Level Analysis (ELA) — Layer 5
Detects localized image-editing artifacts by re-compressing the page as
JPEG and measuring the per-block difference against the original render.
Blocks with abnormally high recompression error (relative to the page's
own block-error distribution) indicate a region that was likely edited
or pasted in after the rest of the page was finalized.
"""

import io
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime

import fitz
import numpy as np
from PIL import Image

from pdf_utils import get_qr_zones, bbox_overlaps_qr_zone


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


@dataclass
class ELARegion:
    page: int
    bbox: tuple        # (x0, y0, x1, y1) in PDF points — resolution-independent
    mean_error: float
    z_score: float
    render_dpi: float = RENDER_DPI  # DPI this region's block was measured at

    # Multi-scale / multi-signal confirmation metadata (high-resolution
    # analysis — see RENDER_SCALES). Populated only for ELA-derived regions;
    # noise-consistency regions set noise_anomaly directly without going
    # through scale confirmation.
    confirmed_scales: list = field(default_factory=list)  # e.g. ["low","medium","high"]
    sharpness_anomaly: bool = False
    noise_anomaly: bool = False
    erasure_anomaly: bool = False
    score_weight: float = 1.0  # scanned-doc header/footer zones count for less (see SCANNED_* constants)


@dataclass
class ELAReport:
    pdf_type: str
    anomaly_score: int
    regions: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    incremental_updates: dict = field(default_factory=dict)


class ELAAnalyzer:

    def analyze(self, pdf_path: str, pdf_type: str = "native_text") -> ELAReport:
        doc = fitz.open(pdf_path)

        # Use higher DPI for vector PDFs (no compression artifacts at low
        # DPI, need more pixels for meaningful ELA).
        from content_analyzer import ContentAnalyzer
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
            if r.erasure_anomaly:
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
        ela_confirmed_regions = [r for r in all_regions if not r.noise_anomaly and not r.erasure_anomaly]
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

    def _collect_xobject_placements(self, pdf_path: str) -> dict:
        """
        Map each Form XObject xref to where/how it's actually used across
        the document: how many pages invoke it, its bbox + page geometry
        (from the first invocation), and whether it carries its own text
        content (fonts / text-showing operators). Used to tell a reused
        template element (logo/letterhead) apart from an injected paste-over.
        """
        placements = {}
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc[page_num]
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                for xref, name, invoker, bbox in page.get_xobjects():
                    entry = placements.setdefault(xref, {
                        "pages": set(),
                        "bbox": bbox,
                        "page_height": page_rect.height,
                        "page_area": page_area,
                    })
                    entry["pages"].add(page_num)

            for xref, entry in placements.items():
                try:
                    xobj_dict = doc.xref_object(xref)
                    has_font = "/Font" in xobj_dict
                    stream = doc.xref_stream(xref) or b""
                    has_text_ops = b"Tj" in stream or b"TJ" in stream
                except Exception:
                    has_font, has_text_ops = False, False
                entry["has_text_content"] = bool(has_font or has_text_ops)
                entry["page_frequency"] = len(entry["pages"]) / total_pages if total_pages else 0

            doc.close()
        except Exception:
            return {}
        return placements

    def _is_injected_xobject(self, xref: int, placements: dict) -> bool:
        """
        True only for Form XObjects that look like an injected paste-over
        rather than a reused template component. See FORM_XOBJECT_* constants.
        """
        entry = placements.get(xref)
        if entry is None:
            return True  # no placement info — fall back to flagging (old behavior)

        if entry["page_frequency"] >= FORM_XOBJECT_MAX_TEMPLATE_FREQUENCY:
            return False  # appears on most pages = template, not injected

        bbox = fitz.Rect(entry["bbox"])
        page_height = entry["page_height"]
        page_area = entry["page_area"]

        if page_height > 0:
            top_frac = bbox.y0 / page_height
            bottom_frac = (page_height - bbox.y1) / page_height
            if top_frac <= FORM_XOBJECT_HEADER_ZONE_FRACTION or bottom_frac <= FORM_XOBJECT_FOOTER_ZONE_FRACTION:
                return False  # header/footer branding zone

        area_frac = (bbox.width * bbox.height) / page_area if page_area > 0 else 0
        if area_frac < FORM_XOBJECT_MIN_AREA_FRACTION:
            return False  # small logo/stamp

        if not entry.get("has_text_content"):
            return False  # image-only content, nothing to "paste over"

        return True

    def _pdf_object_fingerprint(self, pdf_path: str) -> tuple[list[str], int]:
        """
        Analyze PDF object structure for signs of post-creation editing.

        Signals:
        1. Objects with generation number > 0 (deleted and reused = edit)
        2. FreeText/Redact annotations (Acrobat edit vectors)
        3. Form XObjects covering significant page area (paste-over)
        4. Incremental updates (%%EOF count > 1)
        5. Mixed producer fingerprints in object streams
        """
        signals = []
        score   = 0

        try:
            import pikepdf

            # Signal 1: Count %%EOF markers (incremental updates)
            with open(pdf_path, 'rb') as f:
                content = f.read()
            eof_count = content.count(b'%%EOF')
            if eof_count > 1:
                signals.append(
                    f"PDF has {eof_count} revision layers (incremental updates) — "
                    f"document was saved multiple times after creation"
                )
                score += min(EOF_SCORE_CAP, (eof_count - 1) * EOF_SCORE_PER_REVISION)

            xobject_placements = self._collect_xobject_placements(pdf_path)

            with pikepdf.open(pdf_path) as pdf:

                high_gen_objects = []
                freetext_annots  = []
                form_xobjects    = []

                for objid in range(1, len(pdf.objects) + 1):
                    try:
                        obj = pdf.get_object(objid, 0)
                    except Exception:
                        continue

                    # Signal 2: High generation number = object was deleted and reused
                    try:
                        gen = obj.objgen[1] if hasattr(obj, 'objgen') else 0
                        if gen > 0:
                            high_gen_objects.append(objid)
                    except Exception:
                        pass

                    # Signal 3: FreeText/Redact annotations
                    try:
                        if (hasattr(obj, 'get') and
                            obj.get('/Type') == pikepdf.Name('/Annot') and
                            obj.get('/Subtype') in (
                                pikepdf.Name('/FreeText'),
                                pikepdf.Name('/Redact')
                            )):
                            freetext_annots.append(objid)
                    except Exception:
                        pass

                    # Signal 4: Form XObjects (paste-over content) — only
                    # ones that look injected, not reused template elements
                    try:
                        if (hasattr(obj, 'get') and
                            obj.get('/Type') == pikepdf.Name('/XObject') and
                            obj.get('/Subtype') == pikepdf.Name('/Form') and
                            self._is_injected_xobject(objid, xobject_placements)):
                            form_xobjects.append(objid)
                    except Exception:
                        pass

                if high_gen_objects:
                    signals.append(
                        f"{len(high_gen_objects)} PDF object(s) have generation "
                        f"number > 0 — objects were deleted and recreated, "
                        f"indicating direct object-level editing"
                    )
                    score += min(HIGH_GEN_SCORE_CAP, len(high_gen_objects) * HIGH_GEN_SCORE_PER_OBJECT)

                if freetext_annots:
                    signals.append(
                        f"{len(freetext_annots)} FreeText/Redact annotation(s) found — "
                        f"classic Acrobat/Foxit text overlay edit pattern"
                    )
                    score += min(FREETEXT_SCORE_CAP, len(freetext_annots) * FREETEXT_SCORE_PER_ANNOT)

                if form_xobjects:
                    signals.append(
                        f"{len(form_xobjects)} Form XObject(s) found — "
                        f"content may have been pasted over original"
                    )
                    score += min(FORM_XOBJECT_SCORE_CAP, len(form_xobjects) * FORM_XOBJECT_SCORE_PER_ITEM)

        except Exception:
            pass

        return signals, min(100, score)

    def _detect_incremental_updates(self, pdf_path: str) -> dict:
        """
        Detect incremental-update structure and attempt to recover shadowed
        prior object versions.

        A PDF incremental update appends a NEW xref section + trailer to the
        end of the file rather than rewriting it; the old bytes are still
        physically present but a conformant reader (pikepdf included) only
        ever resolves the MOST RECENT xref entry for a given object id, so
        `pdf.get_object()` can never surface a shadowed earlier version no
        matter what generation number is requested — that would require
        re-implementing xref-chain resolution by hand. We get the same
        result more directly and more reliably with a raw byte scan: any
        object id appearing in more than one "<id> <gen> obj" definition in
        the file has at least one shadowed earlier version, and the FIRST
        occurrence is that pre-edit content.
        """
        result = {
            "has_incremental_updates": False,
            "update_count": 0,
            "eof_count": 0,
            "xref_count": 0,
            "startxref_count": 0,
            "prev_trailer_offset": None,
            "old_objects_found": [],
            "signals": [],
            "score": 0,
        }
        try:
            with open(pdf_path, "rb") as f:
                raw = f.read()

            eof_count = raw.count(b"%%EOF")
            xref_count = raw.count(b"\nxref")
            startxref_count = raw.count(b"startxref")
            result["eof_count"] = eof_count
            result["xref_count"] = xref_count
            result["startxref_count"] = startxref_count

            if eof_count > 1:
                result["has_incremental_updates"] = True
                result["update_count"] = eof_count - 1
                result["signals"].append(
                    f"{eof_count} %%EOF markers found — document has "
                    f"{eof_count - 1} incremental update(s) layered on top "
                    f"of the original save"
                )
                result["score"] += min(
                    INCREMENTAL_EOF_SCORE_CAP,
                    (eof_count - 1) * INCREMENTAL_EOF_SCORE_PER_REVISION,
                )

            if xref_count > 1 and xref_count != startxref_count:
                result["signals"].append(
                    f"{xref_count} xref section(s) vs {startxref_count} "
                    f"startxref marker(s) — cross-reference table structure "
                    f"is consistent with chained incremental updates"
                )
                result["score"] += INCREMENTAL_XREF_MISMATCH_SCORE

            try:
                import pikepdf
                with pikepdf.open(pdf_path) as pdf:
                    prev = pdf.trailer.get("/Prev")
                    if prev is not None:
                        result["has_incremental_updates"] = True
                        result["prev_trailer_offset"] = int(prev)
                        result["signals"].append(
                            f"PDF trailer contains a /Prev pointer to byte "
                            f"offset {int(prev)} — structurally confirms an "
                            f"earlier revision's xref table still exists in "
                            f"the file"
                        )
            except Exception:
                pass

            # Raw byte scan for shadowed object versions.
            obj_def_re = re.compile(rb"(\d+)[ \t]+(\d+)[ \t]+obj\b")
            occurrences: dict[int, list[int]] = {}
            for m in obj_def_re.finditer(raw):
                objid = int(m.group(1))
                occurrences.setdefault(objid, []).append(m.start())

            old_versions = []
            for objid, offsets in occurrences.items():
                if len(offsets) <= 1:
                    continue
                start = offsets[0]  # earliest = pre-edit version
                snippet = raw[start:start + OLD_OBJECT_PREVIEW_BYTES]
                preview = snippet.decode("latin-1", errors="replace")
                preview = preview.split("endobj")[0].strip()
                old_versions.append({
                    "objid": objid,
                    "version_count": len(offsets),
                    "preview": preview[:160],
                })

            if old_versions:
                old_versions.sort(key=lambda v: -v["version_count"])
                result["old_objects_found"] = old_versions[:OLD_OBJECT_MAX_REPORTED]
                result["signals"].append(
                    f"{len(old_versions)} object(s) have a shadowed earlier "
                    f"version still present in the raw file — pre-edit "
                    f"content recovered where possible, see old_objects_found"
                )
                result["score"] += INCREMENTAL_OLD_OBJECTS_SCORE

        except Exception as e:
            result["signals"].append(f"Could not analyze incremental updates: {e}")

        result["score"] = min(100, result["score"])
        return result

    @staticmethod
    def _parse_pdf_date(date_obj) -> "datetime | None":
        """Parse a PDF date value ('D:YYYYMMDDHHmmSS...') to a datetime."""
        if date_obj is None:
            return None
        date_str = str(date_obj).strip()
        if date_str.startswith("D:"):
            date_str = date_str[2:]
        match = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", date_str)
        if not match:
            return None
        try:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), int(match.group(5)), int(match.group(6))
            )
        except ValueError:
            return None

    @staticmethod
    def _find_signature_objects(pdf) -> list:
        """Return all /Type /Sig objects (signature dictionaries) in the PDF."""
        import pikepdf
        sigs = []
        for objid in range(1, len(pdf.objects) + 1):
            try:
                obj = pdf.get_object(objid, 0)
            except Exception:
                continue
            try:
                if hasattr(obj, 'get') and obj.get('/Type') == pikepdf.Name('/Sig'):
                    sigs.append(obj)
            except Exception:
                continue
        return sigs

    def _detect_shadow_attack(self, pdf_path: str) -> tuple[list[str], int]:
        """
        Shadow attacks append new content AFTER a digital signature using
        PDF incremental updates. The signature still cryptographically
        validates — it only ever covered the bytes that existed at signing
        time — but the visible content has changed since then.

        Detection logic:
        1. Count %%EOF markers — more than one means incremental updates exist.
        2. Find all /Sig signature dictionary objects.
        3. For each signature's /ByteRange [0, offset1, offset2, end],
           check whether it covers the entire file.
        4. Check whether any bytes exist after the signed range (content
           added after the last signature).
        """
        signals = []
        score   = 0

        try:
            import pikepdf

            with open(pdf_path, 'rb') as f:
                content = f.read()
            file_size = len(content)
            eof_count = content.count(b'%%EOF')

            with pikepdf.open(pdf_path) as pdf:
                sigs = self._find_signature_objects(pdf)

                if eof_count > 1 and sigs:
                    signals.append(
                        f"PDF has {eof_count} revision layers AND a digital "
                        f"signature is present — incremental updates after "
                        f"signing are the mechanism shadow attacks use"
                    )
                    score += SHADOW_EOF_SIG_SCORE

                for sig in sigs:
                    try:
                        byte_range = sig.get('/ByteRange')
                        if byte_range is None or len(byte_range) < 4:
                            continue
                        offset1 = int(byte_range[1])
                        offset2 = int(byte_range[2])
                        end     = int(byte_range[3])

                        # Does the signed range cover the whole file?
                        if offset1 + (end - offset2) != file_size:
                            signals.append(
                                f"Signature ByteRange does not cover the entire "
                                f"file ({offset1 + (end - offset2)} bytes signed "
                                f"vs {file_size} byte file) — gap exists between "
                                f"signed content and end of file"
                            )
                            score += SHADOW_BYTERANGE_GAP_SCORE

                        # Were bytes added after the signed range?
                        signed_end = offset2 + end
                        if file_size - signed_end > 0:
                            signals.append(
                                f"{file_size - signed_end} byte(s) exist after "
                                f"the signed range — objects were added to the "
                                f"file after this signature was applied"
                            )
                            score += SHADOW_OBJECTS_AFTER_SIG_SCORE
                    except Exception:
                        continue

        except Exception:
            pass

        return signals, min(100, score)

    def _validate_digital_signature(self, pdf_path: str) -> tuple[list[str], int]:
        """
        Validate the structural integrity of any digital signature present
        (this is a forensic ByteRange/date check, not a cryptographic
        signature verification — it checks for shadow-attack patterns and
        post-signing modification, not whether the signature itself is
        cryptographically authentic).
        """
        signals = []
        score   = 0

        try:
            import pikepdf

            with open(pdf_path, 'rb') as f:
                file_size = len(f.read())

            with pikepdf.open(pdf_path) as pdf:
                sigs = self._find_signature_objects(pdf)

                if not sigs:
                    return ["No digital signature present"], 0

                mod_date = self._parse_pdf_date(pdf.docinfo.get('/ModDate'))

                for sig in sigs:
                    sub_filter   = str(sig.get('/SubFilter', '')) or None
                    reason       = str(sig.get('/Reason', '')) or None
                    contact_info = str(sig.get('/ContactInfo', '')) or None
                    name         = str(sig.get('/Name', '')) or None
                    sign_date    = self._parse_pdf_date(sig.get('/M'))

                    detail_bits = [b for b in (
                        f"algorithm={sub_filter}" if sub_filter else None,
                        f"signer={name}" if name else None,
                        f"reason={reason}" if reason else None,
                        f"contact={contact_info}" if contact_info else None,
                    ) if b]
                    detail = f" ({', '.join(detail_bits)})" if detail_bits else ""

                    byte_range = sig.get('/ByteRange')
                    covers_full_file = False
                    if byte_range is not None and len(byte_range) >= 4:
                        offset1 = int(byte_range[1])
                        offset2 = int(byte_range[2])
                        end     = int(byte_range[3])
                        covers_full_file = (offset1 + (end - offset2) == file_size)

                    if not covers_full_file:
                        signals.append(
                            f"Digital signature does not cover entire file — "
                            f"shadow attack pattern{detail}"
                        )
                        score = max(score, SIG_BYTERANGE_GAP_SCORE)
                        continue

                    if sign_date and mod_date and mod_date > sign_date:
                        signals.append(
                            f"Document was modified after digital signature "
                            f"was applied (signed {sign_date}, modified {mod_date}){detail}"
                        )
                        score = max(score, SIG_MODIFIED_AFTER_SIGNING_SCORE)
                        continue

                    signals.append(f"Digital signature valid{detail}")

        except Exception:
            pass

        return signals, min(100, score)

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

    # ── Multi-scale helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _bbox_overlaps(a: tuple, b: tuple, padding: float = SCALE_MATCH_PADDING_PT) -> bool:
        """Rectangle-intersection test with a small tolerance — block grids
        from different DPIs/crops don't land on identical point boundaries,
        so an exact-coordinate match would miss the same physical block."""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (ax0 - padding < bx1 and ax1 + padding > bx0 and
                ay0 - padding < by1 and ay1 + padding > by0)

    def _region_confirmed_by(self, bbox: tuple, other_regions: list) -> bool:
        """Does `bbox` (a phase-N candidate) overlap any region found in a
        different scale's independent pass over the same page?"""
        return any(self._bbox_overlaps(bbox, r.bbox) for r in other_regions)

    @staticmethod
    def _get_image_block_bboxes(page) -> list:
        """Bounding boxes (PDF points) of raster image blocks on this page."""
        try:
            blocks = page.get_text("dict").get("blocks", [])
            return [tuple(b["bbox"]) for b in blocks if b.get("type") == 1]
        except Exception:
            return []

    def _restrict_to_raster_content(self, page, regions: list) -> list:
        """
        ELA's JPEG-recompression model only means something for RASTER
        content — vector text is rasterized fresh at whatever DPI it's
        rendered at, so it has no real prior "compression history" to
        violate. A block landing on dense vector text (a bold header, a
        busy paragraph) reads as a recompression outlier purely because
        it's denser than the surrounding blank page — and that gap gets
        WORSE, not better, at higher DPI (crisper edges read as more
        anomalous), which is the opposite of what multi-scale confirmation
        assumes for a genuine logo/raster artifact. So: keep a flagged
        block only if it overlaps actual embedded image content, unless
        the page has no extractable text at all (a fully scanned/photo
        page, where the entire page IS raster and ELA fully applies).
        """
        has_text = bool((page.get_text("text") or "").strip())
        if not has_text:
            return regions
        image_bboxes = self._get_image_block_bboxes(page)
        if not image_bboxes:
            return []
        return [r for r in regions if any(
            self._bbox_overlaps(r.bbox, b) for b in image_bboxes
        )]

    def _is_image_based_document(self, doc, pdf_type: str) -> bool:
        """
        True if this document is a photograph/scan of pages rather than a
        native-text PDF — ELA's JPEG-recompression model doesn't apply to
        such documents the same way; noise-pattern analysis does instead.
        """
        if pdf_type == "scanned":
            return True
        if len(doc) == 0:
            return False
        image_pages = 0
        for page in doc:
            text = page.get_text("text") or ""
            if len(text.strip()) < 20 and len(page.get_images()) >= 1:
                image_pages += 1
        return image_pages >= max(1, len(doc) // 2)

    def _is_compiled_document(self, pdf_path: str, pdf_type: str) -> bool:
        """
        True when a multi-page scanned PDF is likely a portfolio of several
        independently-scanned source documents merged together.  Each source
        page carries its own JPEG compression baseline; ELA sees page-boundary
        differences as "edits" and produces false positives on every page.
        Detection heuristic: scanned type + at least COMPILED_MIN_PAGES pages.
        """
        if pdf_type not in ("scanned", "scanned_native"):
            return False
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
        except Exception:
            return False
        return total_pages >= COMPILED_MIN_PAGES

    def _filter_to_significant_clusters(self, regions: list, min_size: int = None) -> list:
        """
        Group regions into connected components by bbox proximity (touching
        or near-touching), then keep only blocks belonging to a component
        with >= min_size members. See the SCANNED_* constant comment above
        for why this separates a real edit from scan noise.
        """
        if min_size is None:
            min_size = SCANNED_MIN_CLUSTER_SIZE
        n = len(regions)
        if n == 0:
            return []
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        tol = SCANNED_CLUSTER_GAP_TOLERANCE_PT
        for i in range(n):
            xi0, yi0, xi1, yi1 = regions[i].bbox
            for j in range(i + 1, n):
                xj0, yj0, xj1, yj1 = regions[j].bbox
                gap_x = max(xi0, xj0) - min(xi1, xj1)
                gap_y = max(yi0, yj0) - min(yi1, yj1)
                if gap_x <= tol and gap_y <= tol:
                    union(i, j)

        component_sizes = {}
        for i in range(n):
            root = find(i)
            component_sizes[root] = component_sizes.get(root, 0) + 1

        return [
            regions[i] for i in range(n)
            if component_sizes[find(i)] >= min_size
        ]

    def _refine_region_at_high_dpi(self, page, bbox_pts: tuple, high_dpi: float):
        """
        Render ONLY a padded crop around a phase-2-confirmed block at
        high_dpi (not the whole page) and re-run the block-grid check inside
        that crop. This gets an exact-location refinement and a third,
        independent confirmation at high resolution for the price of a
        small render instead of a full-page 600 DPI pass.

        Returns (refined_bbox_or_None, confirmed_at_high_dpi: bool).
        """
        try:
            x0, y0, x1, y1 = bbox_pts
            pad = BLOCK_SIZE_PT * HIGH_DPI_CROP_PADDING_BLOCKS
            page_rect = page.rect
            crop = fitz.Rect(
                max(page_rect.x0, x0 - pad), max(page_rect.y0, y0 - pad),
                min(page_rect.x1, x1 + pad), min(page_rect.y1, y1 + pad),
            )
            if crop.width < 8 or crop.height < 8:
                return None, False

            mat = fitz.Matrix(high_dpi / 72, high_dpi / 72)
            pix = page.get_pixmap(matrix=mat, clip=crop, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
            gray = np.asarray(img.convert("L"), dtype=np.float64)

            block_px = self._block_px_for_dpi(high_dpi)
            z, _errors, n_rows, n_cols = self._ela_block_grid(img, gray, ELA_QUALITY, block_px)
            if z is None:
                return None, False

            flagged_idx = [i for i in range(z.size) if z[i] > Z_THRESHOLD]
            if not flagged_idx:
                return None, False

            pts_scale = 72 / high_dpi
            xs0 = [(i % n_cols) * block_px * pts_scale for i in flagged_idx]
            ys0 = [(i // n_cols) * block_px * pts_scale for i in flagged_idx]
            refined_bbox = (
                crop.x0 + min(xs0),
                crop.y0 + min(ys0),
                crop.x0 + max(xs0) + block_px * pts_scale,
                crop.y0 + max(ys0) + block_px * pts_scale,
            )
            return refined_bbox, True
        except Exception:
            return None, False

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
