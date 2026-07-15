"""
Test suite for embedded-image forensics (utils/embedded_image_forensics):
the standalone image pipeline's checks applied to raster image XObjects
extracted from a PDF's own structure, with findings mapped into the
parent page's coordinate space.

Run:  ..\.venv\Scripts\python.exe test_embedded_images.py

Cases:
  A  native-text PDF with an embedded photo containing a pasted (flat,
     cutout-edge) stamp — detected, and the finding bboxes map onto the
     image's placement rect on the page, not just anywhere
  B  same PDF with a clean, unmanipulated photo — zero findings
  C  metadata independence — /analyze metadata output is IDENTICAL with
     the embedded-image check active vs stubbed out
  D  full route — verdict/signals/annotated image on the tampered PDF
"""

import io
import os
import sys

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embedded_image_forensics import analyze_embedded_images, MIN_DIMENSION_PX

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")

IMG_W, IMG_H = 640, 480
BLUR_SIGMA = 1.8
NOISE_SIGMA = 6.0
STAMP_CENTER = (450, 160)          # in image px
# Placement rect on the page — same 4:3 aspect as the image so fitz's
# keep-proportion insert is a pure scale (no letterboxing) and the
# ground-truth px->pt mapping below is exact.
PLACEMENT = fitz.Rect(280, 470, 580, 695)

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


def make_photo(seed, pasted_stamp=False):
    """Photographed-document image (same recipe as test_image_pipeline):
    printed content -> blur -> sensor noise; optionally a digitally pasted
    flat stamp composited AFTER blur+noise (never went through the camera)."""
    rng = np.random.default_rng(seed)
    pil = Image.new("RGB", (IMG_W, IMG_H), (209, 206, 200))
    d = ImageDraw.Draw(pil)
    d.text((30, 25), "RECEIPT — EXAMPLIA SUPPLIES", font=_font(24), fill=(40, 40, 60))
    for i, t in enumerate(["Item 1042  Widget assembly     240.00",
                           "Item 1055  Fastener pack        18.50",
                           "Item 1101  Bearing set          96.25",
                           "Subtotal                       354.75",
                           "Tax (8%)                        28.38",
                           "TOTAL                          383.13"]):
        d.text((30, 90 + i * 40), t, font=_font(19), fill=(30, 30, 30))
    img = np.asarray(pil, dtype=np.float64)
    for ch in range(3):
        img[:, :, ch] = cv2.GaussianBlur(img[:, :, ch], (0, 0), BLUR_SIGMA)
    img += rng.normal(0, NOISE_SIGMA, img.shape)
    img = np.clip(img, 0, 255)
    if pasted_stamp:
        cx, cy = STAMP_CENTER
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)
        r = np.sqrt(((xx - cx) / 90.0) ** 2 + ((yy - cy) / 60.0) ** 2)
        mask = (np.abs(r - 1.0) < 0.13) | ((np.abs(yy - cy) < 9) & (np.abs(xx - cx) < 55))
        img[mask] = np.array([185.0, 25.0, 35.0])
    return img


def make_pdf(photo_arr, path):
    """A NATIVE-TEXT page (real vector text) with the photo embedded as an
    image XObject at PLACEMENT — the 'image pasted into a text PDF' case."""
    buf = io.BytesIO()
    Image.fromarray(np.clip(photo_arr, 0, 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=90
    )
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    text = [
        "EXAMPLIA SUPPLIES LTD",
        "Expense reimbursement claim — July 2026",
        "",
        "Claimant: R. Sharma          Department: Engineering",
        "Claim ID: EXP-2026-0713      Amount claimed: 383.13",
        "",
        "The attached receipt photograph supports this claim.",
        "Approved subject to standard verification of the receipt below.",
    ]
    for i, line in enumerate(text):
        page.insert_text((72, 90 + i * 22), line, fontsize=12)
    page.insert_image(PLACEMENT, stream=buf.getvalue())
    page.insert_text((72, 730), "Finance use only: batch 44-A, verified against ledger.", fontsize=10)
    doc.save(path)
    doc.close()
    return path


def px_to_pt(x, y):
    return (PLACEMENT.x0 + x * PLACEMENT.width / IMG_W,
            PLACEMENT.y0 + y * PLACEMENT.height / IMG_H)


def main():
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    tampered_pdf = make_pdf(make_photo(seed=5, pasted_stamp=True),
                            os.path.join(TEST_DIR, "embedded_tampered.pdf"))
    clean_pdf = make_pdf(make_photo(seed=5),
                         os.path.join(TEST_DIR, "embedded_clean.pdf"))

    # ── A: tampered embedded image — detected and correctly mapped ────────
    print("\n[A] Tampered embedded image — detection + page-space mapping")
    res_t = analyze_embedded_images(tampered_pdf)
    check(res_t["images_analyzed"] == 1,
          f"exactly the one qualifying image analyzed ({res_t['images_analyzed']}, "
          f"skipped={res_t['images_skipped']}, min side gate {MIN_DIMENSION_PX}px)")
    check(len(res_t["findings"]) >= 1, f"{len(res_t['findings'])} finding(s) produced")
    check(res_t["fold_score"] > 0, f"ELA fold contribution = {res_t['fold_score']:.1f}")

    # Every finding must land INSIDE the image's placement rect on the page
    inside = all(
        f["bbox"][0] >= PLACEMENT.x0 - 2 and f["bbox"][1] >= PLACEMENT.y0 - 2 and
        f["bbox"][2] <= PLACEMENT.x1 + 2 and f["bbox"][3] <= PLACEMENT.y1 + 2
        for f in res_t["findings"]
    )
    check(inside, "all finding bboxes lie inside the placement rect "
                  f"({tuple(round(v) for v in PLACEMENT)})")

    # And at least one must sit on the stamp itself (ground truth known)
    scx, scy = px_to_pt(*STAMP_CENTER)
    on_stamp = [
        f for f in res_t["findings"]
        if f["bbox"][0] <= scx <= f["bbox"][2] and f["bbox"][1] <= scy <= f["bbox"][3]
    ]
    check(len(on_stamp) >= 1,
          f"finding box contains the stamp center ({scx:.0f},{scy:.0f})pt — "
          f"labels: {sorted({f['label'] for f in on_stamp})}")
    check(all(f["layer"] == "embedded_image" for f in res_t["findings"]),
          "findings carry layer='embedded_image'")
    check(all(f["label"].startswith("Embedded Image:") for f in res_t["findings"]),
          "labels use the 'Embedded Image:' convention")

    # ── B: clean embedded image — zero false positives ────────────────────
    print("\n[B] Clean embedded image")
    res_c = analyze_embedded_images(clean_pdf)
    check(res_c["images_analyzed"] == 1, "clean image analyzed")
    check(len(res_c["findings"]) == 0, f"zero findings ({len(res_c['findings'])})")
    check(res_c["fold_score"] == 0, f"zero fold score ({res_c['fold_score']})")

    # ── C: metadata independence ──────────────────────────────────────────
    print("\n[C] Metadata output identical with the check active vs stubbed")
    from fastapi.testclient import TestClient
    from main import app
    import utils.embedded_image_forensics as eif
    client = TestClient(app)

    def run(path):
        with open(path, "rb") as f:
            r = client.post("/analyze", files={"file": (os.path.basename(path), f, "application/pdf")})
        assert r.status_code == 200, r.text
        return r.json()

    real = run(tampered_pdf)
    orig_fn = eif.analyze_embedded_images
    eif.analyze_embedded_images = lambda p: {
        "findings": [], "signals": [], "fold_score": 0,
        "images_analyzed": 0, "images_skipped": 0,
    }
    try:
        stubbed = run(tampered_pdf)
    finally:
        eif.analyze_embedded_images = orig_fn
    # metadata.authenticity is BY DESIGN recomputed from the final combined
    # verdict (analysis_routes: "Recompute authenticity now that the
    # cross-layer verdict exists") — so it legitimately reflects the score
    # this check contributed. Everything else in the metadata output — the
    # actual metadata ANALYSIS — must be byte-identical.
    m_real = {k: v for k, v in real["metadata"].items() if k != "authenticity"}
    m_stub = {k: v for k, v in stubbed["metadata"].items() if k != "authenticity"}
    diff_keys = sorted(
        set(k for k in real["metadata"] if real["metadata"][k] != stubbed["metadata"].get(k))
    )
    check(m_real == m_stub,
          "metadata analysis identical (all fields except verdict-derived authenticity)")
    check(diff_keys in ([], ["authenticity"]),
          f"only the verdict-derived authenticity block may differ (diff={diff_keys})")
    check(real["layers"]["metadata"] == stubbed["layers"]["metadata"],
          f"metadata layer score identical ({real['layers']['metadata']})")
    check(real["layers"]["ela"] > stubbed["layers"]["ela"],
          f"ELA layer carries the fold ({stubbed['layers']['ela']} -> {real['layers']['ela']})")

    # ── D: full route — verdict, signals, annotated image ─────────────────
    print("\n[D] Full /analyze route on the tampered PDF")
    check(real["pdf_type"] == "native_text", f"pdf_type={real['pdf_type']}")
    check(real["verdict"] in ("MODIFIED", "UNCERTAIN"),
          f"verdict={real['verdict']} combined={real['combined_score']}")
    emb_signals = [s for s in real["signals"] if s.startswith("[EMBEDDED_IMG]")]
    check(len(emb_signals) >= 2, f"{len(emb_signals)} [EMBEDDED_IMG] signal(s)")
    check(len(real.get("embedded_image_findings", [])) >= 1,
          "embedded_image_findings present in response payload")
    img = client.get(f"/annotated-image/{real['analysis_id']}?page=1")
    ok_img = img.status_code == 200 and len(img.content) > 10000
    check(ok_img, f"annotated image rendered ({img.status_code}, {len(img.content)} bytes)")
    if ok_img:
        out_png = os.path.join(OUT_DIR, "embedded_image_annotated.png")
        with open(out_png, "wb") as f:
            f.write(img.content)
        print(f"        rendered -> {out_png}")

    print("\n" + "-" * 72)
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
