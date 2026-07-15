"""
Tests for Part 1 (missing vs replaced hidden-text classification) and
Part 2 (authoritative-box display priority / "also flagged by" folding).

Run:  ..\.venv\Scripts\python.exe test_hidden_text_display.py
"""

import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import fitz
from PIL import Image

from utils.hidden_text_extractor import HiddenTextExtractor, HiddenTextFinding
from utils.location_highlighter import LocationHighlighter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_pdfs_stacking")   # reuse the folder
os.makedirs(OUT, exist_ok=True)
SCRATCH = r"C:\Users\PRATHA~1\AppData\Local\Temp\claude\D--task1\a365ca46-cf7e-426e-9925-78feb1c3f17d\scratchpad"
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
IDFC = os.path.join(DOWNLOADS, "IDFCFIRSTBankstatement_10109986314_181639948.pdf")

FAILURES = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILURES.append(msg)


def magenta_px(img):
    a = np.array(img.convert("RGB"))
    return int(((a[:, :, 0] > 200) & (a[:, :, 1] < 80) & (a[:, :, 2] > 200)).sum())


def red_px(img):
    a = np.array(img.convert("RGB"))
    return int(((a[:, :, 0] > 200) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)).sum())


# ── Synthetic PDFs ──────────────────────────────────────────────────────────

def build_whiteout_nothing(path):
    """White rectangle over original text, nothing typed on top → 'missing'."""
    d = fitz.open(); p = d.new_page(width=420, height=240)
    p.insert_text((60, 60), "ACME TECHNOLOGIES PVT LTD", fontsize=13)
    p.insert_text((60, 120), "Annual CTC: 8,50,000 per annum", fontsize=11)
    p.insert_text((60, 160), "CONFIDENTIAL REMARKS: pending review", fontsize=11)
    p.draw_rect(fitz.Rect(58, 150, 330, 166), color=(1, 1, 1), fill=(1, 1, 1))  # cover the remarks
    d.save(path); d.close()


def build_whiteout_replaced(path):
    """White-out + different visible text on top → 'replaced'."""
    d = fitz.open(); p = d.new_page(width=420, height=240)
    p.insert_text((60, 60), "ACME TECHNOLOGIES PVT LTD", fontsize=13)
    p.insert_text((60, 120), "Designation: Senior Engineer", fontsize=11)
    p.insert_text((200, 160), "8,50,000", fontsize=12, color=(0, 0, 0))
    p.draw_rect(fitz.Rect(198, 150, 262, 166), color=(1, 1, 1), fill=(1, 1, 1))
    p.insert_text((200, 160), "18,50,000", fontsize=12, color=(0.15, 0.15, 0.85))
    d.save(path); d.close()


# ── Tests ───────────────────────────────────────────────────────────────────

def run():
    ext = HiddenTextExtractor()

    print("\n[1] Part 1 — white-out with NOTHING over it → detected + 'missing'")
    p_miss = os.path.join(OUT, "whiteout_nothing.pdf"); build_whiteout_nothing(p_miss)
    rep = ext.analyze(p_miss)
    miss = [f for f in rep.findings if f.replacement_type == "missing"]
    check(len(miss) >= 1, f"missing finding detected (got {len(miss)} of {rep.total_found})")
    if miss:
        f = miss[0]
        check(f.covering_text.strip() == "", f"covering_text empty (got {f.covering_text!r})")
        check("no replacement text visible" in f.description.lower(),
              f"description reads as missing: {f.description!r}")

    print("\n[2] Part 1 — white-out + different text → 'replaced'")
    p_rep = os.path.join(OUT, "whiteout_replaced.pdf"); build_whiteout_replaced(p_rep)
    rep2 = ext.analyze(p_rep)
    repl = [f for f in rep2.findings if f.replacement_type == "replaced"]
    check(len(repl) >= 1, f"replaced finding detected (got {len(repl)} of {rep2.total_found})")
    if repl:
        check("replaced with different visible text" in repl[0].description.lower(),
              f"description reads as replaced: {repl[0].description!r}")

    print("\n[3] Part 2 — content finding overlapping a hidden box folds in (ONE box)")
    # Deterministic highlighter-level test: one hidden 'replaced' finding + a
    # content SuspiciousLine overlapping it on the same page.
    class _SL:  # minimal stand-in for content SuspiciousLine
        def __init__(s, page, bbox, score, line_num, anomalies):
            s.page, s.bbox, s.score, s.line_num, s.anomalies = page, bbox, score, line_num, anomalies
    hidden = HiddenTextFinding(
        page=1, method="white_rectangle_cover", original_text="8,50,000",
        covering_text="18,50,000", bbox=(198, 150, 262, 166), confidence="HIGH",
        description="", replacement_type="replaced",
    )
    content_same = _SL(0, (196, 149, 300, 167), 0.9, 4, ["Font mismatch on CTC line"])
    hl = LocationHighlighter(p_rep)
    imgs = hl.highlight_pages(suspicious_lines=[content_same], hidden_text_findings=[hidden])
    check(0 in imgs, "page 0 rendered")
    if 0 in imgs:
        img = imgs[0]
        img.save(os.path.join(SCRATCH, "part2_case2.png"))
        rp, mp = red_px(img), magenta_px(img)
        # Content red box must be absorbed (no separate red box), magenta present.
        check(mp > 0, f"authoritative magenta box drawn (magenta px={mp})")
        check(rp == 0, f"redundant red content box NOT drawn — folded in (red px={rp})")

    print("\n[4] Part 2 — content finding NOT overlapping any hidden box draws normally")
    content_far = _SL(0, (60, 40, 300, 58), 0.9, 0, ["Font mismatch header"])
    hl2 = LocationHighlighter(p_rep)
    imgs2 = hl2.highlight_pages(suspicious_lines=[content_far], hidden_text_findings=[hidden])
    if 0 in imgs2:
        rp2 = red_px(imgs2[0])
        check(rp2 > 0, f"non-overlapping content box still drawn normally (red px={rp2})")

    print("\n[5] Regression — /hidden-text response carries replacement_type")
    # (exercised via analyze findings already; the route just forwards the field)
    check(all(hasattr(f, "replacement_type") for f in rep.findings + rep2.findings),
          "all findings carry replacement_type")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for m in FAILURES:
            print("  - " + m)
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    run()
