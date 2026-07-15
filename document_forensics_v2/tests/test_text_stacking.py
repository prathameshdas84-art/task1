"""
Test suite for the NEW coordinate-collision text-stacking check
(utils/hidden_text_extractor.detect_stacked_text) and its fusion wiring.

Run:  ..\.venv\Scripts\python.exe test_text_stacking.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import fitz

from utils.hidden_text_extractor import HiddenTextExtractor, TextStackingFinding
from fusion.signal_fusion import SignalFusion

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_pdfs_stacking")
os.makedirs(OUT, exist_ok=True)

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
IDFC = os.path.join(DOWNLOADS, "IDFCFIRSTBankstatement_10109986314_181639948.pdf")

FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        FAILURES.append(msg)


# ── Synthetic PDF builders ──────────────────────────────────────────────────

def _base_page(doc):
    """A page with normal, non-overlapping body text (so the detector has a
    realistic amount of legitimate text to NOT flag)."""
    page = doc.new_page(width=420, height=300)
    page.insert_text((60, 60), "ACME TECHNOLOGIES PVT LTD", fontsize=13)
    page.insert_text((60, 90), "Annual CTC:", fontsize=11)          # label
    page.insert_text((200, 90), "INR 8,50,000 per annum", fontsize=11)  # value (adjacent)
    page.insert_text((60, 120), "Designation: Senior Engineer", fontsize=11)
    page.insert_text((60, 150), "Date of Joining: 01 July 2024", fontsize=11)
    return page


def build_stacked(path):
    """#1 — genuine stacked edit: original '8,50,000' covered by a white rect,
    '18,50,000' typed on top at the same coordinates. MUST be flagged."""
    doc = fitz.open()
    page = _base_page(doc)
    p = fitz.Point(200, 200)
    page.insert_text(p, "8,50,000", fontsize=12, color=(0, 0, 0))       # original (hidden)
    page.draw_rect(fitz.Rect(198, 188, 260, 204), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(p, "18,50,000", fontsize=12, color=(0, 0, 0))      # overlay
    doc.save(path)
    doc.close()


def build_lone_hidden(path):
    """#2 — a single hidden run under a white rect, NO visible counterpart at
    that spot. MUST NOT be flagged by text-stacking (still the recovery
    methods' job)."""
    doc = fitz.open()
    page = _base_page(doc)
    page.insert_text((200, 200), "8,50,000", fontsize=12, color=(0, 0, 0))
    page.draw_rect(fitz.Rect(198, 188, 260, 204), color=(1, 1, 1), fill=(1, 1, 1))
    # no overlay text
    doc.save(path)
    doc.close()


def build_adjacent(path):
    """#3 — a label next to its value, distinct and non-overlapping. MUST NOT
    be flagged."""
    doc = fitz.open()
    page = _base_page(doc)
    page.insert_text((60, 200), "Net Pay:", fontsize=12)
    page.insert_text((160, 200), "72,910", fontsize=12)   # clearly to the right, no overlap
    doc.save(path)
    doc.close()


def build_identical_dup(path):
    """#4 — the SAME text drawn twice at the SAME position (a rendering
    duplicate / font-fallback style artifact). MUST NOT be flagged."""
    doc = fitz.open()
    page = _base_page(doc)
    p = fitz.Point(200, 200)
    page.insert_text(p, "8,50,000", fontsize=12, color=(0, 0, 0))
    page.insert_text(p, "8,50,000 ", fontsize=12, color=(0, 0, 0))   # identical + trailing space
    doc.save(path)
    doc.close()


def build_currency_reformat(path):
    """#4b — same numeric VALUE formatted differently at the same spot
    ('8,50,000' vs '8,50,000.00'). MUST NOT be flagged (formatting, not a
    value change). Uses a trailing-decimals difference rather than a currency
    symbol so it renders with the base font (the ₹ case is asserted directly
    at the unit level below, where no glyph rendering is involved)."""
    doc = fitz.open()
    page = _base_page(doc)
    p = fitz.Point(200, 200)
    page.insert_text(p, "8,50,000", fontsize=12)
    page.insert_text(p, "8,50,000.00", fontsize=12)
    doc.save(path)
    doc.close()


# ── Tests ───────────────────────────────────────────────────────────────────

def run():
    ext = HiddenTextExtractor()

    print("\n[A] Detection unit tests (synthetic PDFs)")

    p1 = os.path.join(OUT, "stacked.pdf");            build_stacked(p1)
    p2 = os.path.join(OUT, "lone_hidden.pdf");        build_lone_hidden(p2)
    p3 = os.path.join(OUT, "adjacent.pdf");           build_adjacent(p3)
    p4 = os.path.join(OUT, "identical_dup.pdf");      build_identical_dup(p4)
    p4b = os.path.join(OUT, "currency_reformat.pdf"); build_currency_reformat(p4b)

    f1 = ext.detect_stacked_text(p1)
    check(len(f1) == 1, f"#1 stacked → exactly 1 finding (got {len(f1)})")
    if f1:
        vals = {ext._stacking_normalize(t) for t in f1[0].texts}
        check(vals == {"8,50,000", "18,50,000"},
              f"#1 reports BOTH values (got {f1[0].texts})")
        check(f1[0].confidence == "HIGH", "#1 confidence HIGH")
        check(f1[0].page == 0, "#1 page is 0-indexed")

    f2 = ext.detect_stacked_text(p2)
    check(len(f2) == 0, f"#2 lone hidden run → NOT flagged (got {len(f2)})")
    # sanity: the existing recovery methods still see the covered text
    rep2 = ext.analyze(p2)
    check(rep2.total_found >= 1, "#2 still caught by existing hidden-text recovery (unchanged)")

    f3 = ext.detect_stacked_text(p3)
    check(len(f3) == 0, f"#3 adjacent label+value → NOT flagged (got {len(f3)})")

    f4 = ext.detect_stacked_text(p4)
    check(len(f4) == 0, f"#4 identical duplicate → NOT flagged (got {len(f4)})")

    f4b = ext.detect_stacked_text(p4b)
    check(len(f4b) == 0, f"#4b same value reformatted → NOT flagged (got {len(f4b)})")

    # Direct unit checks of the content-difference rule (no glyph rendering).
    check(ext._stacking_texts_differ("₹8,50,000", "8,50,000") is False,
          "currency-symbol reformat ('₹8,50,000' vs '8,50,000') → not different")
    check(ext._stacking_texts_differ("8,50,000", "8,50,000 ") is False,
          "trailing-whitespace diff → not different")
    check(ext._stacking_texts_differ("8,50,000", "18,50,000") is True,
          "real value change ('8,50,000' vs '18,50,000') → different")
    check(ext._stacking_texts_differ("A1", "B1") is True,
          "alphanumeric labels ('A1' vs 'B1') → different (not treated as same number)")

    print("\n[B] Regression — existing clean/tampered PDFs + IDFC must yield ZERO stacking findings")
    regression = [
        os.path.join(HERE, "test_pdfs", "clean.pdf"),
        os.path.join(HERE, "test_pdfs", "tampered_font.pdf"),
        os.path.join(HERE, "test_pdfs", "tampered_salary.pdf"),
        os.path.join(HERE, "test_pdfs", "tampered_date.pdf"),
        IDFC,
    ]
    for path in regression:
        name = os.path.basename(path)
        if not os.path.exists(path):
            check(False, f"{name} MISSING at {path}")
            continue
        f = ext.detect_stacked_text(path)
        check(len(f) == 0,
              f"{name} → 0 stacking findings (byte-identical score preserved) (got {len(f)})")
        if f:
            for x in f:
                print(f"        FP: page {x.page} {x.texts} overlap={x.overlap_fraction:.2f}")

    print("\n[C] Fusion wiring — a text_stacking extra_finding cross-validates with another layer")
    fus = SignalFusion()
    # A stacking finding co-located with a content finding on the same spot
    # should form a 2-layer fused finding (cross-validation, not override).
    ts_extra = [{
        "layer": "text_stacking", "page": 0, "bbox": (198.0, 188.0, 260.0, 204.0),
        "text": "8,50,000 | 18,50,000", "score": 0.95,
    }]
    fake_content = [{
        "page": 0, "bbox": (200.0, 188.0, 255.0, 204.0),
        "line_num": 5, "text": "18,50,000", "score": 0.7,
    }]
    fused, stats = fus.fuse(suspicious_lines=fake_content, extra_findings=ts_extra)
    layers_present = set()
    for ff in fused:
        layers_present |= set(ff.confirming_layers)
    check(any("text_stacking" in ff.confirming_layers and len(ff.confirming_layers) >= 2
              for ff in fused),
          f"text_stacking fuses with content into a 2-layer finding (layers={layers_present})")

    # Isolation: a lone text_stacking finding with nothing to corroborate is
    # NOT auto-promoted to a high-confidence fused finding (no override).
    fused_alone, _ = fus.fuse(extra_findings=ts_extra)
    check(len(fused_alone) == 0,
          "lone text_stacking finding is NOT an automatic fused-finding override")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for m in FAILURES:
            print("  - " + m)
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    run()
