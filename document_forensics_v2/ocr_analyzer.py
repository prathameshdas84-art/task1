"""
OCR Consistency Analyzer — Layer 3
Renders PDF pages to images, runs Tesseract with confidence scores,
finds regions where OCR confidence drops below page average.
Low confidence regions = pasted content, image overlay, or print+scan edit.

Also cross-checks OCR text vs embedded PDF text for native PDFs.
"""

import os
import shutil
import statistics
from dataclasses import dataclass, field

import fitz
import numpy as np
import pytesseract
from PIL import Image

from content_analyzer import (
    ContentAnalyzer,
    SCANNER_KEYWORDS,
    NATIVE_TEXT_RATIO_THRESHOLD,
    NATIVE_TEXT_MIN_CHARS,
)


def _resolve_tesseract_cmd() -> str:
    """
    Locate the Tesseract binary cross-platform instead of hardcoding the
    Windows install path: an explicit TESSERACT_CMD env var wins, then a
    PATH lookup (works out of the box on Linux/Mac where Tesseract is
    typically installed via a package manager), then the common Windows
    install location as a last-resort default for this deployment.
    """
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path:
        return env_path
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    return r'C:\Program Files\Tesseract-OCR\tesseract.exe'


pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract_cmd()

RENDER_DPI = 200   # higher DPI for better OCR accuracy

# Embedded-vs-OCR mismatch and overall-confidence thresholds.
LOW_OCR_CONFIDENCE_THRESHOLD      = 70    # avg confidence below this = unreliable/low quality
EMBEDDED_OCR_MISMATCH_THRESHOLD   = 0.35  # word-overlap dissimilarity above this = mismatch
SEVERE_DROP_THRESHOLD             = 30    # confidence-drop above this = "severe" in signal text


@dataclass
class SuspiciousRegion:
    page: int
    text: str                  # OCR text in this region
    bbox: tuple                # (x0, y0, x1, y1) in PDF points
    confidence: float          # OCR confidence of this word/region
    page_avg_confidence: float # average confidence of whole page
    drop: float                # how much below average (page_avg - confidence)
    reason: str


@dataclass
class OCRReport:
    pages_analyzed: int
    suspicious_regions: list[SuspiciousRegion]
    ocr_vs_embedded_mismatch: bool   # True if OCR text differs from embedded text
    mismatch_ratio: float            # 0.0-1.0 how different
    avg_confidence: float            # overall document OCR confidence
    anomaly_score: int               # 0-100
    signals: list[str]
    pdf_type_detected: str           # "has_text" | "image_only"


class OCRAnalyzer:

    CONFIDENCE_DROP_THRESHOLD = 20   # points below page avg = suspicious
    MIN_CONFIDENCE_ABSOLUTE   = 50   # any word below this = always suspicious
    MIN_WORD_LENGTH           = 2    # ignore single chars

    def analyze(self, pdf_path: str) -> OCRReport:
        # For native text PDFs OCR cross-check is unreliable
        # embedded text already exists — skip OCR scoring
        # If producer/creator indicates a scanner — always run OCR
        # regardless of embedded text presence
        # Vector PDFs (text outlined to paths — Canva/Figma/Illustrator
        # exports) have no usable text layer either, so they also need
        # full OCR. ContentAnalyzer's _detect_pdf_type() reports "scanned"
        # for these, but that result isn't threaded into this method — this
        # class does its own independent native-text detection below, so
        # the vector check has to be repeated here too, not just there.
        try:
            is_vector = ContentAnalyzer()._is_vector_pdf(pdf_path)
        except Exception:
            is_vector = False

        try:
            import pikepdf
            with pikepdf.open(pdf_path) as pdf:
                info = pdf.docinfo
                producer = str(info.get("/Producer", "")).lower()
                creator  = str(info.get("/Creator", "")).lower()
                is_scanner = any(
                    kw in producer or kw in creator
                    for kw in SCANNER_KEYWORDS
                )
                if is_scanner or is_vector:
                    # Skip the native text early return
                    # Fall through to full OCR analysis
                    pass
                else:
                    # existing native text check here — uses the same
                    # ratio/char-count thresholds as content_analyzer's
                    # _detect_pdf_type() so both files make the identical
                    # native-text decision (previously this used a
                    # different minimum char count than that file, so the
                    # two layers could disagree about the same PDF).
                    import pdfplumber
                    with pdfplumber.open(pdf_path) as pdf2:
                        total = len(pdf2.pages)
                        with_text = sum(
                            1 for p in pdf2.pages
                            if p.extract_text() and len(p.extract_text().strip()) > NATIVE_TEXT_MIN_CHARS
                        )
                    if with_text / total >= NATIVE_TEXT_RATIO_THRESHOLD:
                        return OCRReport(
                            pages_analyzed=0,
                            suspicious_regions=[],
                            ocr_vs_embedded_mismatch=False,
                            mismatch_ratio=0.0,
                            avg_confidence=0.0,
                            anomaly_score=0,
                            signals=["Native text PDF — OCR layer skipped (embedded text used instead)"],
                            pdf_type_detected="has_text",
                        )
        except Exception:
            pass

        doc         = fitz.open(pdf_path)
        scale       = RENDER_DPI / 72

        all_suspicious   = []
        all_confidences  = []
        pages_analyzed   = 0
        has_embedded_text = False

        # Check if PDF has embedded text
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as plumber_pdf:
                embedded_texts = []
                for page in plumber_pdf.pages:
                    t = page.extract_text() or ""
                    embedded_texts.append(t)
                    if len(t.strip()) > 30:
                        has_embedded_text = True
        except Exception:
            embedded_texts = [""] * len(doc)
            has_embedded_text = False

        ocr_texts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            mat  = fitz.Matrix(scale, scale)
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img  = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)

            # Run Tesseract — get per-word data with confidence
            try:
                data = pytesseract.image_to_data(
                    img,
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )
            except Exception:
                continue

            pages_analyzed += 1

            # Extract words with confidence > 0
            page_words = []
            for i in range(len(data["text"])):
                word = data["text"][i].strip()
                conf = int(data["conf"][i])
                if conf <= 0 or len(word) < self.MIN_WORD_LENGTH:
                    continue
                import re
                # English/Latin-alphabet-specific noise filter: a word with
                # no vowel is almost always OCR gibberish in English text,
                # but this heuristic will incorrectly discard real words in
                # languages where consonant clusters without latin vowels
                # are normal (e.g. transliterated text) — only valid for
                # English-language documents.
                vowels = set('aeiouAEIOU')
                if len(word) > 1 and not any(c in vowels for c in word):
                    continue
                if re.match(r'^[BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz]+$', word):
                    continue
                # Skip reference codes and OCR noise patterns
                word_clean = word.strip()

                # Pattern 1: UPI/reference codes (digits + separators, case insensitive)
                if re.match(r'^[A-Za-z0-9~\-\.\@\:\_�]+$', word_clean) and \
                   any(c.isdigit() for c in word_clean) and \
                   len(word_clean) > 6 and \
                   any(c in '~-@:_' for c in word_clean):
                    continue

                # Pattern 2: Pure noise — 2-3 char lowercase gibberish (ee, eee, Ce., wo)
                if len(word_clean) <= 3 and \
                   not word_clean.isdigit() and \
                   word_clean.lower() == word_clean:
                    continue

                # Pattern 3: ALL CAPS short words that are document labels not content
                # (ATE, IFSC, etc.) — only flag if very short AND all caps
                if len(word_clean) <= 4 and word_clean.isupper() and \
                   not any(c.isdigit() for c in word_clean):
                    continue
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                page_words.append({
                    "text": word,
                    "conf": conf,
                    "bbox_px": (x, y, x+w, y+h)
                })
                all_confidences.append(conf)

            if not page_words:
                continue

            # Page-level confidence stats
            page_confs    = [w["conf"] for w in page_words]
            page_avg_conf = statistics.mean(page_confs)
            page_std_conf = statistics.stdev(page_confs) if len(page_confs) > 1 else 10

            # Find suspicious words on this page
            for word_data in page_words:
                conf = word_data["conf"]
                drop = page_avg_conf - conf

                is_suspicious = (
                    drop > self.CONFIDENCE_DROP_THRESHOLD or
                    conf < self.MIN_CONFIDENCE_ABSOLUTE
                )

                if is_suspicious:
                    # Convert pixel bbox → PDF points
                    px0, py0, px1, py1 = word_data["bbox_px"]
                    pts_scale = 72 / RENDER_DPI
                    bbox_pts  = (
                        px0 * pts_scale,
                        py0 * pts_scale,
                        px1 * pts_scale,
                        py1 * pts_scale,
                    )

                    reason = (
                        f"OCR confidence {conf}% vs page avg {page_avg_conf:.0f}% "
                        f"(drop of {drop:.0f} points)"
                        if drop > self.CONFIDENCE_DROP_THRESHOLD
                        else f"Very low OCR confidence: {conf}%"
                    )

                    all_suspicious.append(SuspiciousRegion(
                        page=page_num,
                        text=word_data["text"],
                        bbox=bbox_pts,
                        confidence=conf,
                        page_avg_confidence=page_avg_conf,
                        drop=drop,
                        reason=reason,
                    ))

            # Collect OCR text for cross-check
            page_ocr_text = " ".join(w["text"] for w in page_words)
            ocr_texts.append(page_ocr_text)

        doc.close()

        # Sort suspicious regions by drop (worst first)
        all_suspicious.sort(key=lambda r: r.drop, reverse=True)

        # Cross-check OCR vs embedded text
        mismatch_ratio = 0.0
        ocr_vs_embedded_mismatch = False
        if has_embedded_text:
            mismatch_ratio = self._compute_mismatch(
                embedded_texts, ocr_texts
            )
            if mismatch_ratio > EMBEDDED_OCR_MISMATCH_THRESHOLD:
                ocr_vs_embedded_mismatch = True

        avg_confidence = (
            statistics.mean(all_confidences) if all_confidences else 0.0
        )

        # Image quality too poor for reliable confidence-drop analysis
        if avg_confidence < LOW_OCR_CONFIDENCE_THRESHOLD and pages_analyzed > 0:
            return OCRReport(
                pages_analyzed=pages_analyzed,
                suspicious_regions=[],
                ocr_vs_embedded_mismatch=False,
                mismatch_ratio=0.0,
                avg_confidence=avg_confidence,
                anomaly_score=0,
                signals=[
                    f"OCR avg confidence too low ({avg_confidence:.0f}%) "
                    f"for reliable analysis — image quality insufficient"
                ],
                pdf_type_detected="image_only",
            )

        pdf_type_detected = "has_text" if has_embedded_text else "image_only"

        # Build signals and score
        signals, score = self._build_signals(
            all_suspicious,
            ocr_vs_embedded_mismatch,
            mismatch_ratio,
            avg_confidence,
            pages_analyzed,
        )

        return OCRReport(
            pages_analyzed=pages_analyzed,
            suspicious_regions=all_suspicious,  # return ALL regions
            ocr_vs_embedded_mismatch=ocr_vs_embedded_mismatch,
            mismatch_ratio=mismatch_ratio,
            avg_confidence=avg_confidence,
            anomaly_score=score,
            signals=signals,
            pdf_type_detected=pdf_type_detected,
        )

    def _compute_mismatch(
        self,
        embedded_texts: list[str],
        ocr_texts: list[str]
    ) -> float:
        """
        Compute word-level mismatch ratio between embedded and OCR text.
        Returns 0.0 (identical) to 1.0 (completely different).
        """
        total_overlap = 0
        total_union   = 0

        for emb, ocr in zip(embedded_texts, ocr_texts):
            emb_words = set(emb.lower().split())
            ocr_words = set(ocr.lower().split())
            if not emb_words and not ocr_words:
                continue
            overlap    = len(emb_words & ocr_words)
            union      = len(emb_words | ocr_words)
            total_overlap += overlap
            total_union   += union

        if total_union == 0:
            return 0.0
        similarity    = total_overlap / total_union
        return round(1.0 - similarity, 3)

    def _build_signals(
        self,
        suspicious: list[SuspiciousRegion],
        mismatch: bool,
        mismatch_ratio: float,
        avg_conf: float,
        pages: int,
    ) -> tuple[list[str], int]:
        signals = []
        score   = 0

        if pages == 0:
            signals.append("OCR could not process any pages")
            return signals, 0

        # Overall confidence signal
        if avg_conf < LOW_OCR_CONFIDENCE_THRESHOLD:
            signals.append(
                f"Low overall OCR confidence: {avg_conf:.0f}% — "
                f"document may be low quality scan or contain pasted regions"
            )
            score += 15
        else:
            signals.append(
                f"Overall OCR confidence: {avg_conf:.0f}% across {pages} page(s)"
            )

        # Suspicious word regions
        high_drop = [r for r in suspicious if r.drop > SEVERE_DROP_THRESHOLD]
        med_drop  = [r for r in suspicious if self.CONFIDENCE_DROP_THRESHOLD < r.drop <= SEVERE_DROP_THRESHOLD]

        if high_drop:
            signals.append(
                f"{len(high_drop)} word(s) with severely low OCR confidence — "
                f"possible pasted text, image overlay, or print+scan edit"
            )
            score += min(50, len(high_drop) * 8)

        if med_drop:
            signals.append(
                f"{len(med_drop)} word(s) with moderately low OCR confidence"
            )
            score += min(20, len(med_drop) * 4)

        # Embedded vs OCR mismatch
        if mismatch:
            signals.append(
                f"Embedded text vs OCR text mismatch: {mismatch_ratio:.0%} difference — "
                f"visible content differs from embedded PDF text layer "
                f"(possible image overlay hiding original text)"
            )
            score += 35

        if not high_drop and not med_drop and not mismatch:
            signals.append(
                "OCR consistency check passed — "
                "no confidence anomalies or text layer mismatches detected"
            )

        return signals, min(100, score)
