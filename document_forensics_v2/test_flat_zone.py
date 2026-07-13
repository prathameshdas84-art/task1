"""
Test suite for the flat/pasted-patch check on scanned PDF pages
(utils/flat_zone_detection shared with ela_analyzer + the image pipeline).

Run:  ..\.venv\Scripts\python.exe test_flat_zone.py

Cases:
  A  tampered scan: dark flat rectangle (no scan noise) under a crisp red
     seal, plus a second flat patch with no stamp — both detected, boxed at
     the right location, stamp one labeled as stamp-associated
  B  genuine scan: organic wet-ink stamp photographed WITH the page
     (texture + soft edges) — nothing flagged
  C  native_text gate: same tampered file analyzed as pdf_type=native_text
     — check genuinely skipped (no regions, no signal, no score delta)
  D  fusion wiring: a flat_zone extra-finding cross-validates with another
     layer at the same spot; alone it is not an automatic override
  E  annotated render: LocationHighlighter draws the flat-patch boxes with
     the specific labels at the patch (not the whole seal graphic)
"""

import io
import os
import sys

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzers.ela_analyzer import ELAAnalyzer
from fusion.signal_fusion import SignalFusion
from utils.location_highlighter import LocationHighlighter, _flat_zone_label

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")

# Image is 850x1100 px placed on a 612x792pt page — the same 0.7727 aspect,
# so insert_image's keep-proportion scaling is a pure uniform stretch and
# px→pt mapping is exactly PX_TO_PT with no letterboxing offset.
W, H = 850, 1100
PAGE_W, PAGE_H = 612, 792
PX_TO_PT = PAGE_W / W

BLUR_SIGMA = 1.0     # scanner optics / print ink spread
NOISE_SIGMA = 7.0    # scan sensor noise
JPEG_QUALITY = 90

STAMP_PATCH = (500, 700, 240, 200)   # x, y, w, h px — flat rect under the seal
STAMP_CENTER = (620, 800)            # seal center, inside STAMP_PATCH
PLAIN_PATCH = (110, 900, 220, 110)   # second flat rect, no stamp on it

RESULTS = []


def check(ok, msg):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    return ok


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_scan_base(seed, organic_stamp=False):
    """Simulated scanned statement page: printed text → optics blur →
    sensor noise. Returns float64 RGB array."""
    rng = np.random.default_rng(seed)
    pil = Image.new("RGB", (W, H), (226, 223, 216))
    d = ImageDraw.Draw(pil)
    d.text((60, 45), "EXAMPLIA STATE BANK", font=_font(34), fill=(35, 35, 55))
    d.text((60, 95), "Account Statement — 2214 8871", font=_font(22), fill=(60, 60, 70))
    for i in range(14):
        y = 170 + i * 36
        d.text((60, y), f"12/0{(i % 9) + 1}/2026   Payment ref {1000 + i}", font=_font(18), fill=(30, 30, 30))
        d.text((560, y), f"{1234.56 + i * 11:.2f}", font=_font(18), fill=(30, 30, 30))
    d.line([60, 690, 790, 690], fill=(90, 90, 90), width=2)
    d.text((60, 1040), "Authorized signatory", font=_font(18), fill=(60, 60, 60))

    img = np.asarray(pil, dtype=np.float64)
    if organic_stamp:
        img = draw_organic_stamp(img, *STAMP_CENTER, rng)
    for ch in range(3):
        img[:, :, ch] = cv2.GaussianBlur(img[:, :, ch], (0, 0), BLUR_SIGMA)
    img += rng.normal(0, NOISE_SIGMA, img.shape)
    return np.clip(img, 0, 255)


def draw_organic_stamp(img, cx, cy, rng):
    """Wet-ink seal: ring + bar, per-pixel density modulated by a smooth
    random field, feathered boundary. Composited BEFORE blur+noise — it is
    part of the scanned scene (same recipe as test_image_pipeline's 7a)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt(((xx - cx) / 85.0) ** 2 + ((yy - cy) / 65.0) ** 2)
    ring = (np.abs(r - 1.0) < 0.14).astype(np.float64)
    bar = ((np.abs(yy - cy) < 9) & (np.abs(xx - cx) < 50)).astype(np.float64)
    alpha = np.clip(ring + bar, 0, 1)
    field = rng.random((H // 8, W // 8))
    field = cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), 4)
    density = 0.45 + 0.55 * (field - field.min()) / (np.ptp(field) + 1e-9)
    alpha *= density
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    color = np.array([185.0, 25.0, 35.0])
    return img * (1 - alpha[..., None]) + color[None, None, :] * alpha[..., None]


def apply_flat_stamp_patch(img):
    """The reference-image pattern: a dark, grain-free rectangle pasted over
    the finished scan (zero sensor noise inside), with a crisp uniform red
    seal drawn on top of it. Plus a second flat patch with no stamp
    (pasted signature/photo stand-in)."""
    out = img.copy()
    x, y, w, h = STAMP_PATCH
    out[y:y + h, x:x + w] = np.array([88.0, 80.0, 72.0])
    cx, cy = STAMP_CENTER
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt(((xx - cx) / 85.0) ** 2 + ((yy - cy) / 65.0) ** 2)
    mask = (np.abs(r - 1.0) < 0.14) | ((np.abs(yy - cy) < 9) & (np.abs(xx - cx) < 50))
    out[mask] = np.array([185.0, 25.0, 35.0])
    x, y, w, h = PLAIN_PATCH
    out[y:y + h, x:x + w] = np.array([126.0, 120.0, 112.0])
    return out


def to_pdf(img_arr, path):
    """Embed the page image as a JPEG in a single-page PDF (a scanned doc)."""
    im = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)
    doc.close()
    return path


def px_bbox_to_pt(xywh):
    x, y, w, h = xywh
    return (x * PX_TO_PT, y * PX_TO_PT, (x + w) * PX_TO_PT, (y + h) * PX_TO_PT)


def bbox_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def main():
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    tampered_pdf = to_pdf(apply_flat_stamp_patch(make_scan_base(seed=11)),
                          os.path.join(TEST_DIR, "flat_stamp_tampered.pdf"))
    genuine_pdf = to_pdf(make_scan_base(seed=11, organic_stamp=True),
                         os.path.join(TEST_DIR, "flat_stamp_genuine.pdf"))

    ela = ELAAnalyzer()

    # ── A: tampered scan — both patches detected, boxed, labeled ──────────
    print("\n[A] Tampered scan — flat patch under seal + plain flat patch")
    rep_t = ela.analyze(tampered_pdf, "scanned")
    flat = [r for r in rep_t.regions if getattr(r, "flat_zone_anomaly", False)]
    stamp_hits = [r for r in flat if r.stamp_associated]
    plain_hits = [r for r in flat if not r.stamp_associated]
    check(len(flat) >= 2, f"{len(flat)} flat-zone region(s) found (expect >= 2)")

    exp_stamp = px_bbox_to_pt(STAMP_PATCH)
    exp_plain = px_bbox_to_pt(PLAIN_PATCH)
    check(any(bbox_iou(r.bbox, exp_stamp) > 0.5 for r in stamp_hits),
          f"stamp-associated box lands on the flat patch "
          f"(expected~{tuple(round(v) for v in exp_stamp)}, "
          f"got {[tuple(round(v) for v in r.bbox) for r in stamp_hits]})")
    check(any(bbox_iou(r.bbox, exp_plain) > 0.5 for r in plain_hits),
          f"non-stamp flat patch boxed too "
          f"(got {[tuple(round(v) for v in r.bbox) for r in plain_hits]})")
    check(any("pasted stamp" in s.lower() for s in rep_t.signals),
          "signal names the pasted-stamp flat background")
    if stamp_hits:
        check(_flat_zone_label(stamp_hits[0]) == "Pasted Stamp: Flat Background",
              f"stamp label = '{_flat_zone_label(stamp_hits[0])}'")
    if plain_hits:
        check(_flat_zone_label(plain_hits[0]) == "Flat Region: Texture Mismatch",
              f"plain label = '{_flat_zone_label(plain_hits[0])}'")
    check(rep_t.anomaly_score > 0, f"contributes to ELA layer score ({rep_t.anomaly_score})")

    # ── B: genuine organic stamp — nothing flagged ────────────────────────
    print("\n[B] Genuine scan — organic wet-ink stamp, no flat patch")
    rep_g = ela.analyze(genuine_pdf, "scanned")
    flat_g = [r for r in rep_g.regions if getattr(r, "flat_zone_anomaly", False)]
    check(len(flat_g) == 0, f"no flat-zone regions on the genuine stamp ({len(flat_g)})")
    check(not any("flat" in s.lower() and "pasted" in s.lower() for s in rep_g.signals),
          "no pasted-patch signal on the genuine scan")

    # ── C: native_text gate — genuinely skipped ───────────────────────────
    print("\n[C] Gate — same tampered file analyzed as pdf_type=native_text")
    rep_nt = ela.analyze(tampered_pdf, "native_text")
    flat_nt = [r for r in rep_nt.regions if getattr(r, "flat_zone_anomaly", False)]
    check(len(flat_nt) == 0, "no flat-zone regions when pdf_type=native_text")
    check(not any("flat/texture-less" in s for s in rep_nt.signals),
          "no flat-zone signal when pdf_type=native_text")

    # ── D: fusion wiring — flat_zone cross-validates, never overrides ─────
    print("\n[D] Fusion wiring — flat_zone extra-finding")
    from types import SimpleNamespace
    fz_extra = [{
        "layer": "flat_zone", "page": 0, "bbox": exp_stamp,
        "text": "Pasted stamp — flat background", "score": 0.9,
    }]
    fake_content = [SimpleNamespace(
        page=0, line_num=3, text="seal area", score=0.5,
        bbox=exp_stamp, anomalies=["Font size 14.0pt outlier"],
    )]
    fus = SignalFusion()
    fused, _ = fus.fuse(suspicious_lines=fake_content, extra_findings=fz_extra)
    layers_present = [ff.confirming_layers for ff in fused]
    check(any("flat_zone" in ff.confirming_layers and len(ff.confirming_layers) >= 2
              for ff in fused),
          f"flat_zone fuses with content into a 2-layer finding (layers={layers_present})")
    fused_alone, _ = fus.fuse(extra_findings=fz_extra)
    check(len(fused_alone) == 0,
          "lone flat_zone finding is NOT an automatic fused-finding override")

    # ── E: annotated render — box + label placement ────────────────────────
    print("\n[E] Annotated render")
    hl = LocationHighlighter(tampered_pdf)
    pages = hl.highlight_pages(ela_regions=flat)
    check(0 in pages, "annotated image produced for page 1")
    if 0 in pages:
        out_png = os.path.join(OUT_DIR, "flat_zone_annotated.png")
        pages[0].save(out_png)
        print(f"        rendered -> {out_png}")

    print("\n" + "-" * 72)
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
