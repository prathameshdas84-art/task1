"""
Test loop for the image-document forensic pipeline (Part 5 of the spec).

Builds synthetic test images in test_images/ modeled on the actual attack
(AI inpainting removes content + a phone app overlays new text), runs
ImageDocumentAnalyzer on each, and asserts per-case expectations —
including the false-positive cases (clean photo, genuine glare,
born-digital render) that the honesty requirement makes non-negotiable.

Canonical cases (spec Part 5):
  1  inpainted patch          → Check 1 must flag it
  2  crisp overlay text       → Check 5 must flag it
  3  both combined            → both fire, co-located, fusion cross-validates
  4  clean noisy photo        → NOTHING may fire
  5  genuine camera glare     → Check 2's discrimination must NOT flag it
  6  born-digital flat render → Check 1 gates out, zero false positives
  7a organic stamp            → Checks 7-9: detected, NOT flagged
  7b pasted stamp             → Checks 7-9: detected AND flagged (8 and/or 9)

Supplementary cases (every implemented check needs a positive proof, per
Part 0 — the canonical 7 exercise only negatives for these):
  S1 cloned region            → Check 6 offset-consensus must fire
  S2 double JPEG save         → Check 4 must not report "single_compression"
  S3 JPEG history in a PNG    → Check 3 must detect it through the container

Born-digital gate robustness (regression for the /analyze-image false
gate: low-quality re-compression erases a genuine capture's sensor noise
and used to trip the born-digital gate, suppressing all detection):
  S4 recompressed capture     → genuine stamped photo re-saved at low JPEG
                                quality (noise floor below the gate) must
                                NOT be classified born-digital — blocking-
                                grid residuals prove compression erased the
                                noise; stamp still detected; no false hits
  S5 born-digital as JPEG     → a true vector render exported AS JPEG must
                                STILL gate (container format alone is not
                                evidence of a capture pipeline)
  S6 scanned form w/ stamp    → police-verification-style scanned form,
                                re-compressed: not gated, stamp detected

Run:  ..\\.venv\\Scripts\\python test_image_pipeline.py
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.image_document_analyzer import (
    ImageDocumentAnalyzer, normalize_for_fusion,
)
from fusion.signal_fusion import SignalFusion
from utils.flat_zone_detection import BORN_DIGITAL_STD_FLOOR

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
W, H = 800, 600
BLUR_SIGMA = 1.8      # print ink-spread + lens blur
NOISE_SIGMA = 6.0     # sensor noise
JPEG_QUALITY = 90     # phone-camera-ish save

PATCH = (500, 150, 160, 110)     # x, y, w, h — the inpainted region
OVERLAY_XY = (510, 185)          # overlay text lands inside PATCH for case 3
GLARE_CENTER = (560, 210)
GLARE_SIGMA = 70
GLARE_AMP = 95


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_base(seed, organic_stamp=False):
    """Simulated photographed ID-card-like document: printed text →
    ink/lens blur → sensor noise. Returns float64 RGB array (not yet saved)."""
    rng = np.random.default_rng(seed)
    pil = Image.new("RGB", (W, H), (208, 205, 200))
    d = ImageDraw.Draw(pil)
    d.text((40, 30), "GOVERNMENT OF EXAMPLIA", font=_font(30), fill=(40, 40, 60))
    d.text((40, 80), "IDENTITY CARD", font=_font(24), fill=(60, 60, 80))
    rows = ["Name:      RAHUL K SHARMA", "DOB:       14/03/1987",
            "ID No:     EXP-2214-8871-05", "Issued:    02/11/2021",
            "Valid to:  02/11/2031", "Address:   44 Lake View Road"]
    for i, t in enumerate(rows):
        d.text((40, 150 + i * 44), t, font=_font(22), fill=(30, 30, 30))
    d.rectangle([560, 330, 720, 520], outline=(50, 50, 50), width=3)
    d.text((585, 410), "PHOTO", font=_font(20), fill=(120, 120, 120))
    d.line([40, 545, 760, 545], fill=(80, 80, 80), width=2)

    img = np.asarray(pil, dtype=np.float64)

    if organic_stamp:
        img = draw_organic_stamp(img, 640, 180, rng)

    for ch in range(3):
        img[:, :, ch] = cv2.GaussianBlur(img[:, :, ch], (0, 0), BLUR_SIGMA)
    img += rng.normal(0, NOISE_SIGMA, img.shape)
    return np.clip(img, 0, 255)


def draw_organic_stamp(img, cx, cy, rng, color=(185.0, 25.0, 35.0), strength=1.0):
    """Wet-ink stamp: elliptical ring + center bar, per-pixel density
    modulated by a smooth random field (uneven pressure), feathered
    boundary (ink bleed). Composited BEFORE the photo blur+noise —
    it is part of the photographed scene."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt(((xx - cx) / 90.0) ** 2 + ((yy - cy) / 60.0) ** 2)
    ring = (np.abs(r - 1.0) < 0.13).astype(np.float64)
    bar = ((np.abs(yy - cy) < 9) & (np.abs(xx - cx) < 55)).astype(np.float64)
    alpha = np.clip(ring + bar, 0, 1)
    field = rng.random((H // 8, W // 8))
    field = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), 4)
    density = 0.45 + 0.55 * (field - field.min()) / (np.ptp(field) + 1e-9)
    alpha *= density * strength
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)   # ink bleed feather
    color = np.array(color, dtype=np.float64)
    return img * (1 - alpha[..., None]) + color[None, None, :] * alpha[..., None]


def draw_pasted_stamp(img, cx, cy):
    """Digitally pasted stamp: uniform flat fill, hard cutout edge,
    composited AFTER blur+noise (it never went through the camera)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt(((xx - cx) / 90.0) ** 2 + ((yy - cy) / 60.0) ** 2)
    ring = np.abs(r - 1.0) < 0.13
    bar = (np.abs(yy - cy) < 9) & (np.abs(xx - cx) < 55)
    mask = ring | bar
    out = img.copy()
    out[mask] = np.array([185.0, 25.0, 35.0])
    return out


def apply_inpaint_patch(img):
    """Simulated AI removal: the patch region's pixels get smoothed hard,
    destroying the sensor-noise texture there."""
    x, y, w, h = PATCH
    out = img.copy()
    region = out[y:y + h, x:x + w]
    for ch in range(3):
        region[:, :, ch] = cv2.GaussianBlur(region[:, :, ch], (0, 0), 4.0)
    out[y:y + h, x:x + w] = region
    return out


def apply_overlay_text(img, text="PAID 12/06/2026"):
    """Simulated Instagram-style text tool: crisp antialiased glyphs drawn
    straight onto the finished photo."""
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    d.text(OVERLAY_XY, text, font=_font(30), fill=(10, 10, 10))
    return np.asarray(pil, dtype=np.float64)


def apply_glare(img):
    """Genuine camera glare: soft gradient bright spot, clipped core."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = GLARE_CENTER
    blob = GLARE_AMP * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * GLARE_SIGMA ** 2))
    return np.clip(img + blob[..., None], 0, 255)


def apply_clone(img):
    """Copy a textured region elsewhere (same pixels, one offset)."""
    out = img.copy()
    src = out[150:230, 40:220].copy()      # a text block — feature-rich
    out[380:460, 300:480] = src
    return out


def make_police_form(seed=67):
    """Scanned police-verification-style form: white paper, printed form
    fields with rule lines, and a faint blue-purple round office stamp —
    photographed/scanned (blur + sensor noise), like the real-world false
    gate case."""
    rng = np.random.default_rng(seed)
    pil = Image.new("RGB", (W, H), (246, 245, 242))
    d = ImageDraw.Draw(pil)
    d.text((170, 25), "POLICE VERIFICATION CERTIFICATE", font=_font(26), fill=(25, 25, 35))
    d.text((40, 80), "Office of the Superintendent of Police", font=_font(18), fill=(60, 60, 70))
    # Varied row pitch, underline extents, and font sizes on purpose —
    # perfectly periodic identical rules read as a clone lattice to
    # Check 6, which is a separate (pre-existing) concern from the
    # born-digital gate this fixture exercises.
    rows = [("Applicant Name: SUNIL R MEHTA", 20, 470),
            ("Father's Name: RAMESH MEHTA", 21, 430),
            ("Address: 12 Station Road, Ward 4, Exampli City", 19, 545),
            ("Purpose of Verification: Passport", 22, 385),
            ("Police Station: Central Division No. 3", 20, 505),
            ("Result: NO ADVERSE RECORD FOUND", 23, 460)]
    y = 140
    for t, fs, lw in rows:
        d.text((40, y), t, font=_font(fs), fill=(35, 35, 40))
        d.line([40, y + 30, 40 + lw, y + 30], fill=(150, 150, 155), width=1)
        y += 38 + fs
    d.text((40, 540), "Date: 02/05/2026", font=_font(18), fill=(35, 35, 40))
    d.text((560, 540), "Signature", font=_font(18), fill=(35, 35, 40))

    img = np.asarray(pil, dtype=np.float64)
    img = draw_organic_stamp(img, 640, 430, rng,
                             color=(70.0, 55.0, 160.0), strength=0.8)
    for ch in range(3):
        img[:, :, ch] = cv2.GaussianBlur(img[:, :, ch], (0, 0), BLUR_SIGMA)
    img += rng.normal(0, NOISE_SIGMA, img.shape)
    return np.clip(img, 0, 255)


def make_born_digital():
    """Pure flat digital render: no noise, no blur, crisp everything."""
    pil = Image.new("RGB", (W, H), (235, 235, 238))
    d = ImageDraw.Draw(pil)
    d.text((40, 40), "INVOICE #2214", font=_font(34), fill=(20, 20, 30))
    for i in range(5):
        d.text((40, 140 + i * 50), f"Line item {i + 1}    ...    $ {120 + i * 37}.00",
               font=_font(22), fill=(40, 40, 40))
    d.rectangle([500, 400, 740, 540], fill=(70, 120, 200))
    d.text((520, 450), "TOTAL $606", font=_font(26), fill=(255, 255, 255))
    return np.asarray(pil, dtype=np.float64)


def save(img, name, fmt="JPEG"):
    os.makedirs(TEST_DIR, exist_ok=True)
    path = os.path.join(TEST_DIR, name)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if fmt == "JPEG":
        pil.save(path, "JPEG", quality=JPEG_QUALITY)
    else:
        pil.save(path, "PNG")
    return path


def bboxes_overlap(b1, b2, pad=40):
    """(x,y,w,h) overlap with tolerance — detector bboxes are grid-quantized."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return (x1 < x2 + w2 + pad and x1 + w1 + pad > x2 and
            y1 < y2 + h2 + pad and y1 + h1 + pad > y2)


def hits(report, check, near=None):
    out = [a for a in report.anomalies if a.evidence_check == check]
    if near is not None:
        out = [a for a in out if bboxes_overlap(a.bbox, near)]
    return out


def build_all():
    base = make_base(seed=7)
    paths = {}
    paths["1_inpaint"] = save(apply_inpaint_patch(base), "1_inpaint_patch.jpg")
    paths["2_overlay"] = save(apply_overlay_text(base), "2_overlay_text.jpg")
    paths["3_combined"] = save(apply_overlay_text(apply_inpaint_patch(base)),
                               "3_combined_attack.jpg")
    paths["4_clean"] = save(base, "4_clean.jpg")
    paths["5_glare"] = save(apply_glare(make_base(seed=11)), "5_glare.jpg")
    paths["6_born_digital"] = save(make_born_digital(), "6_born_digital.png", "PNG")
    paths["7a_organic"] = save(make_base(seed=23, organic_stamp=True),
                               "7a_stamp_organic.jpg")
    paths["7b_pasted"] = save(draw_pasted_stamp(make_base(seed=23), 640, 180),
                              "7b_stamp_pasted.jpg")
    paths["s1_clone"] = save(apply_clone(make_base(seed=31)), "s1_clone.jpg")

    # S2: double JPEG save — q75 first, then q95 (different quality = the
    # envelope-periodicity case Check 4 targets).
    p_first = save(make_base(seed=41), "s2_tmp_first.jpg")
    first = Image.open(p_first)
    p2 = os.path.join(TEST_DIR, "s2_double_saved.jpg")
    first.convert("RGB").save(p2, "JPEG", quality=95)
    os.unlink(p_first)
    # overwrite the first save at q75 for a stronger primary comb
    pil = Image.fromarray(np.clip(make_base(seed=41), 0, 255).astype(np.uint8))
    tmp = os.path.join(TEST_DIR, "s2_tmp.jpg")
    pil.save(tmp, "JPEG", quality=75)
    Image.open(tmp).convert("RGB").save(p2, "JPEG", quality=95)
    os.unlink(tmp)
    paths["s2_double"] = p2

    # S3: JPEG history hiding inside a PNG container.
    tmp = os.path.join(TEST_DIR, "s3_tmp.jpg")
    Image.fromarray(np.clip(make_base(seed=51), 0, 255).astype(np.uint8)).save(
        tmp, "JPEG", quality=75)
    p3 = os.path.join(TEST_DIR, "s3_jpeg_in_png.png")
    Image.open(tmp).convert("RGB").save(p3, "PNG")
    os.unlink(tmp)
    paths["s3_png_history"] = p3

    # S4: genuine stamped capture re-saved at low JPEG quality — the
    # quantization erases the sensor-noise floor (baseline drops below
    # BORN_DIGITAL_STD_FLOOR), which used to falsely trip the gate.
    p4 = os.path.join(TEST_DIR, "s4_recompressed_capture.jpg")
    Image.open(paths["7a_organic"]).convert("RGB").save(p4, "JPEG", quality=30)
    paths["s4_recompressed"] = p4

    # S5: a TRUE born-digital render exported as JPEG — must still gate.
    p5 = os.path.join(TEST_DIR, "s5_born_digital.jpg")
    Image.fromarray(np.clip(make_born_digital(), 0, 255).astype(np.uint8)).save(
        p5, "JPEG", quality=85)
    paths["s5_bd_jpeg"] = p5

    # S6: scanned police-verification-style form with a stamp, then run
    # through a messaging-app-style transform (downscale + mid-quality
    # re-save) — the real-world false-gate report: noise floor pushed
    # below BORN_DIGITAL_STD_FLOOR on a genuine capture.
    form = Image.fromarray(np.clip(make_police_form(), 0, 255).astype(np.uint8))
    small = form.resize((int(form.width * 0.75), int(form.height * 0.75)),
                        Image.BILINEAR)
    p6 = os.path.join(TEST_DIR, "s6_police_form_recompressed.jpg")
    small.save(p6, "JPEG", quality=60)
    paths["s6_police_form"] = p6
    return paths


def run():
    paths = build_all()
    analyzer = ImageDocumentAnalyzer()
    reports = {k: analyzer.analyze(p) for k, p in paths.items()}
    results = []   # (case, ok, detail)

    def case(name, ok, detail):
        results.append((name, ok, detail))

    # ── 1: inpainted patch → Check 1 fires on it ─────────────────────────
    r = reports["1_inpaint"]
    got = hits(r, "check1_local_variance", near=PATCH)
    stray = [a for a in r.anomalies
             if a.evidence_check == "check1_local_variance" and not bboxes_overlap(a.bbox, PATCH)]
    case("1 inpaint patch", bool(got) and not stray,
         f"check1 hits on patch={len(got)}, stray check1={len(stray)}, "
         f"all anomalies={[(a.evidence_check, a.bbox) for a in r.anomalies]}")

    # ── 2: overlay text → Check 5 fires on it ────────────────────────────
    r = reports["2_overlay"]
    ov_box = (OVERLAY_XY[0], OVERLAY_XY[1], 260, 40)
    got = hits(r, "check5_edge_sharpness", near=ov_box)
    stray = [a for a in r.anomalies
             if a.evidence_check == "check5_edge_sharpness" and not bboxes_overlap(a.bbox, ov_box)]
    case("2 overlay text", bool(got) and not stray,
         f"check5 hits on overlay={len(got)}, stray check5={len(stray)}, "
         f"baseline={r.metrics.get('edge_sharpness', {}).get('edge_sharpness_baseline')}, "
         f"anomalies={[(a.evidence_check, a.bbox, a.confidence) for a in r.anomalies]}")

    # ── 3: combined → both fire, co-located, fusion cross-validates ──────
    r = reports["3_combined"]
    c1 = hits(r, "check1_local_variance", near=PATCH)
    c5 = hits(r, "check5_edge_sharpness", near=PATCH)
    co_located = any(bboxes_overlap(a.bbox, b.bbox, pad=60) for a in c1 for b in c5)
    fused, stats = SignalFusion().fuse(extra_findings=normalize_for_fusion(r))
    cross = [f for f in fused if len(set(f.confirming_layers)) >= 2]
    case("3 combined attack", bool(c1) and bool(c5) and co_located and bool(cross),
         f"check1={len(c1)}, check5={len(c5)}, co_located={co_located}, "
         f"fused(2+ layers)={[(f.confirming_layers, f.confidence) for f in cross]}")

    # ── 4: clean photo → NOTHING fires (score 0) ─────────────────────────
    r = reports["4_clean"]
    case("4 clean photo", len(r.anomalies) == 0 and r.anomaly_score == 0
         and r.compression_history != "double_compression_suspected",
         f"anomalies={[(a.evidence_check, a.bbox, a.confidence) for a in r.anomalies]}, "
         f"score={r.anomaly_score}, compression={r.compression_history}")

    # ── 5: genuine glare → Check 2 discrimination keeps it out ───────────
    r = reports["5_glare"]
    case("5 camera glare", len(r.anomalies) == 0 and r.anomaly_score == 0,
         f"anomalies={[(a.evidence_check, a.bbox, a.detail) for a in r.anomalies]}, "
         f"glare_excluded={r.metrics.get('glare_regions_excluded')}")

    # ── 6: born-digital → gate, zero false positives ─────────────────────
    r = reports["6_born_digital"]
    case("6 born-digital", r.is_born_digital and len(r.anomalies) == 0
         and r.compression_history == "not_applicable"
         and not r.jpeg_history_detected,
         f"is_born_digital={r.is_born_digital}, "
         f"baseline={r.metrics.get('noise_baseline_std')}, "
         f"jpeg_history={r.jpeg_history_detected}, "
         f"anomalies={[(a.evidence_check, a.bbox) for a in r.anomalies]}")

    # ── 7a: organic stamp → detected, NOT flagged ────────────────────────
    r = reports["7a_organic"]
    flagged = (hits(r, "check8_stamp_texture") + hits(r, "check9_stamp_boundary"))
    case("7a organic stamp", r.stamp_detected and not flagged,
         f"stamp_detected={r.stamp_detected}, ink_components={r.metrics.get('ink_components')}, "
         f"flagged={[(a.evidence_check, a.detail) for a in flagged]}")

    # ── 7b: pasted stamp → detected AND flagged by 8 and/or 9 ────────────
    r = reports["7b_pasted"]
    flagged = (hits(r, "check8_stamp_texture") + hits(r, "check9_stamp_boundary"))
    case("7b pasted stamp", r.stamp_detected and bool(flagged),
         f"stamp_detected={r.stamp_detected}, ink_components={r.metrics.get('ink_components')}, "
         f"flagged={[(a.evidence_check, a.confidence, a.detail) for a in flagged]}")

    # ── S1: clone → Check 6 consensus fires ──────────────────────────────
    r = reports["s1_clone"]
    got = hits(r, "check6_copy_move")
    case("S1 clone (supp.)", bool(got),
         f"copy_move hits={[(a.bbox, a.detail) for a in got]}, "
         f"metrics={r.metrics.get('copy_move')}")
    # …and Check 6 must have stayed silent on every canonical case
    cm_fp = {k: len(hits(rep, 'check6_copy_move'))
             for k, rep in reports.items() if k.startswith(('1', '2', '3', '4', '5', '6', '7'))}
    case("S1b clone negatives", all(v == 0 for v in cm_fp.values()), f"per-case={cm_fp}")

    # ── S2: double save → not reported as single ─────────────────────────
    r = reports["s2_double"]
    case("S2 double save (supp.)", r.compression_history != "single_compression",
         f"compression_history={r.compression_history}, "
         f"metrics={r.metrics.get('double_compression')}")

    # ── S3: JPEG history inside PNG container ────────────────────────────
    r = reports["s3_png_history"]
    case("S3 png w/ jpeg hist (supp.)",
         r.jpeg_history_detected and r.metrics["container_format"] == "PNG",
         f"jpeg_history={r.jpeg_history_detected}, "
         f"blockiness={r.metrics.get('blockiness')}")

    # ── S4: recompressed genuine capture → NOT born-digital ──────────────
    r = reports["s4_recompressed"]
    bl = r.metrics.get("noise_baseline_std", 99)
    case("S4 recompressed capture",
         bl < BORN_DIGITAL_STD_FLOOR          # still reproduces the failure mode
         and not r.is_born_digital            # ...but no longer gates
         and r.jpeg_history_detected
         and r.stamp_detected                 # stamp isolation still works
         and len(r.anomalies) == 0,           # and no false positives appear
         f"baseline={bl} (< floor {BORN_DIGITAL_STD_FLOOR}), "
         f"is_born_digital={r.is_born_digital}, stamp={r.stamp_detected}, "
         f"grid_z={r.metrics.get('blockiness', {}).get('grid_phase_z')}, "
         f"anomalies={[(a.evidence_check, a.bbox) for a in r.anomalies]}")

    # ── S5: born-digital exported AS JPEG → must STILL gate ──────────────
    r = reports["s5_bd_jpeg"]
    case("S5 born-digital as JPEG",
         r.is_born_digital and len(r.anomalies) == 0,
         f"is_born_digital={r.is_born_digital}, "
         f"baseline={r.metrics.get('noise_baseline_std')}, "
         f"grid_z={r.metrics.get('blockiness', {}).get('grid_phase_z')}, "
         f"anomalies={[(a.evidence_check, a.bbox) for a in r.anomalies]}")

    # ── S6: recompressed scanned form w/ stamp → not gated, stamp found ──
    r = reports["s6_police_form"]
    bl = r.metrics.get("noise_baseline_std", 99)
    case("S6 scanned form w/ stamp",
         bl < BORN_DIGITAL_STD_FLOOR          # reproduces the failure mode
         and not r.is_born_digital
         and r.stamp_detected
         and len(r.anomalies) == 0,
         f"baseline={bl} (< floor {BORN_DIGITAL_STD_FLOOR}), "
         f"is_born_digital={r.is_born_digital}, stamp={r.stamp_detected}, "
         f"grid_z={r.metrics.get('blockiness', {}).get('grid_phase_z')}, "
         f"anomalies={[(a.evidence_check, a.bbox) for a in r.anomalies]}")

    # ── report ────────────────────────────────────────────────────────────
    print()
    print(f"{'CASE':<28} {'RESULT':<6} DETAIL")
    print("-" * 110)
    for name, ok, detail in results:
        print(f"{name:<28} {'PASS' if ok else 'FAIL':<6} {detail}")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("-" * 110)
    print(f"{len(results) - n_fail}/{len(results)} passed")
    # single-compression sanity across canonical JPEG cases
    for k in ("1_inpaint", "2_overlay", "3_combined", "4_clean", "5_glare",
              "7a_organic", "7b_pasted"):
        print(f"  {k}: compression_history={reports[k].compression_history}, "
              f"score={reports[k].anomaly_score}")
    return n_fail


if __name__ == "__main__":
    sys.exit(min(1, run()))
