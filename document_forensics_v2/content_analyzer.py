"""
Content Consistency Analyzer — Layer 2
Extracts per-line features from PDF.
Builds document statistical profile.
Finds outlier lines that break consistency.
No training data. No ML. Pure statistics.
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

import cv2
import fitz
import numpy as np
import pdfplumber


# ── Shared constants ────────────────────────────────────────────────────────────
# Scanner brand/keyword fingerprints used to detect "scanned_native" PDFs
# (a native-text PDF whose producer/creator metadata identifies a physical
# scanner). Also imported by ocr_analyzer.py — kept in one place so the two
# files can never independently drift to different scanner keyword sets.
SCANNER_KEYWORDS = [
    "scan", "canon", "epson", "hp", "fujitsu", "brother", "xerox",
    "ricoh", "sharp", "kodak", "ij scan", "scansnap", "twain", "wia",
]

# Thresholds for classifying a PDF as native_text / mixed / scanned, based
# on the fraction of pages with a substantial embedded text layer. Also
# imported by ocr_analyzer.py so both files make the identical decision
# about whether a PDF "has text" (previously ocr_analyzer used a different
# minimum character count than this file for the same check).
NATIVE_TEXT_RATIO_THRESHOLD = 0.7   # >=70% of pages have text -> native_text
MIXED_TEXT_RATIO_THRESHOLD  = 0.3   # >=30% of pages have text -> mixed
NATIVE_TEXT_MIN_CHARS       = 30    # min chars extracted to count a page as "has text"

# Z-score cutoff for flagging a line's font size / spacing / visual feature
# as an outlier relative to the rest of the document. 2.5 sigma is the same
# cutoff used by every per-line statistical check in _score_lines() below.
Z_OUTLIER_THRESHOLD = 2.5

# A font used on more than this fraction of lines is treated as deliberate
# document styling (e.g. a Canva template mixing two or three font
# families) rather than a tamper signal — see _build_profile().
DESIGN_FONT_RATIO_THRESHOLD = 0.15

# Keywords that mark a value line as high-stakes (payroll/identity fields).
# Lines matching these are never treated as "structural" (always scored),
# and a font mismatch on one of the CRITICAL subset scores higher.
# NOTE: this list is English-language and includes India-specific payroll/
# identity terms (ctc, pan, aadhaar) — extend with local-language/regional
# equivalents before using this analyzer on non-Indian or non-English
# documents.
ALWAYS_CHECK_KEYWORDS = [
    "salary", "amount", "balance", "total",
    "net pay", "gross", "income", "compensation",
    "remuneration", "stipend", "payment",
    "account number", "aadhaar",
    "date of birth",
]
# Short keywords collide with ordinary words when matched as bare
# substrings — "pan" matches "company"/"Japan"/"expand", "dob" matches
# "Adobe", "wage" matches "sewage", "ctc" is short enough to risk the
# same. Require word boundaries so they only match the actual abbreviation.
ALWAYS_CHECK_KEYWORDS_WORD_BOUNDARY = [r"\bctc\b", r"\bpan\b", r"\bdob\b", r"\bwage\b"]
CRITICAL_VALUE_KEYWORDS = [
    "salary", "ctc", "amount", "balance", "total",
    "net pay", "gross", "income", "compensation",
]

# Payslip table-header rows (e.g. "Total Days in Month: 31.00 Days Paid:
# 31.00 LWP/Absent: Arrears Days Paid:") naturally have irregular spacing
# because they span multiple table columns — that's a layout artifact, not
# evidence of editing. English/India-payroll-specific terms; extend for
# other locales' payslip formats.
NEVER_FLAG_PATTERNS = [
    "total days",
    "days paid",
    "lwp",
    "lop",
    "absent",
    "arrears days",
    "working days",
]

# Regex patterns identifying letterhead/address/contact lines, which
# legitimately use a different font from the document body.
# India-specific: the city/state name list and "taluk"/"mandal" (Indian
# administrative-division terms) only match Indian addresses — extend this
# list before relying on it for documents from other countries.
ADDRESS_PATTERNS = [
    r'\d+[\/\-]\d+',           # address numbers like "A/1" "7-32"
    r'\b\d{6}\b',              # 6-digit pincode (India-specific format)
    r'@\w+\.\w+',              # email address
    r'\+?\d[\d\s\-]{8,}',     # phone number
    r'\b(road|street|nagar|colony|compound|post|village|district|'
    r'taluk|mandal|state|india|maharashtra|karnataka|gujarat|'
    r'delhi|mumbai|bangalore|bengaluru|hyderabad|chennai|pune)\b',
]

# _is_structural_line() heuristic thresholds (see that method for context).
LETTERHEAD_LINE_COUNT        = 3     # first N lines of page 0 = letterhead
ALL_CAPS_RATIO_THRESHOLD     = 0.85  # fraction of uppercase alpha chars = header
SHORT_LINE_MAX_WORDS         = 3     # lines with <= N words = field label
NUMERIC_LINE_RATIO_THRESHOLD = 0.7   # fraction of digits = purely numeric/date line
LABEL_PATTERN_MAX_WORDS      = 8     # "Label: Value" lines up to N words
RULE7_MAX_WORDS              = 5     # short lines with measured line height
SEPARATOR_MIN_LENGTH         = 5     # min length for a "----"/"====" divider line

# Character-spacing uniformity check (_score_lines()): genuine typed text
# has natural variation in per-character width; retyped/edited text often
# has unnaturally uniform spacing because it was set with fixed character
# advances rather than the original font's natural kerning.
CHAR_SPACING_CV_THRESHOLD  = 0.05  # coefficient of variation below this = too uniform
CHAR_SPACING_CV_MIN_CHARS  = 8     # only evaluate lines with more than this many chars
CHAR_SPACING_CV_SCORE      = 0.25  # anomaly score contribution when flagged

# Unicode replacement/placeholder glyphs that show up when a font can't
# render a character it was asked to — e.g. a currency symbol (₹, €, $)
# typed in a font/encoding that doesn't have that glyph after editing.
# Always checked regardless of line type: an encoding-failure glyph isn't
# something a letterhead/label can legitimately contain.
REPLACEMENT_CHARS = [
    '■',  # ■ BLACK SQUARE
    '□',  # □ WHITE SQUARE
    '▪',  # ▪ BLACK SMALL SQUARE
    '▫',  # ▫ WHITE SMALL SQUARE
    '●',  # ● BLACK CIRCLE
    '○',  # ○ WHITE CIRCLE
    '�',  # � UNICODE REPLACEMENT CHARACTER
]
REPLACEMENT_CHAR_SCORE = 0.60

# ── Upgrade 4: glyph consistency filter ─────────────────────────────────────────
# Canva/Figma/InDesign/Puppeteer/wkhtmltopdf all subset-embed fonts with
# custom glyph IDs ("AAAAAA+Helvetica") — when PyMuPDF can't map one of those
# IDs back to a real character it reads as U+FFFD/"?"/NBSP. That's the
# EXPORT TOOL's behavior, repeated identically everywhere that glyph is used
# in that subset (e.g. every Rs./currency symbol on a Canva payslip) — not a
# one-off edit. GLYPH_WATCH_CHARS is tracked for the registry/ratio
# computation; only chars also in REPLACEMENT_CHARS above actually get
# flagged in _score_lines — a bare "?" is far too common in legitimate text
# (questions, "N/A?", etc.) to safely treat as a tamper signal on its own,
# so it's tracked for ratio purposes but never itself score-flagged.
GLYPH_WATCH_CHARS = ('�', '?', ' ')

# -- Upgrade 5: form field suppression -----------------------------------------
# Form lines (Date:____, Sign:____, table cells separated by tabs) have wide,
# deliberately irregular spacing by design -- not an edit. Suppresses ONLY the
# spacing-related checks (char/word spacing, line height) for these lines;
# font size and color checks still run, since a genuine edit on a form field
# would still show up there.
FORM_FIELD_PATTERNS = [
    r'_{3,}',                  # three or more underscores
    r'\t{2,}',                 # multiple tabs
    r':\s{5,}',                # colon followed by 5+ spaces
    r'^(date|sign|signature|name|place|witness|designation|stamp|seal)\s*:?\s*$',
]
FORM_FIELD_SHORT_LINE_MAX_WORDS = 2
FORM_FIELD_SHORT_LINE_MIN_LEN   = 30  # 1-2 words spanning this much horizontal width = signature block, not a sentence
GLYPH_CONSISTENCY_RATIO_THRESHOLD = 0.02  # char recurring on >2% of a subset font's chars = platform behavior, not an edit

# _score_lines() per-anomaly score contributions. Each "outlier" check adds
# min(CAP, z * MULT) so a borderline z-score (just above Z_OUTLIER_THRESHOLD)
# contributes little while a very extreme one saturates at CAP.
FONT_SIZE_SCORE_CAP,    FONT_SIZE_SCORE_MULT    = 0.25, 0.05
CHAR_SPACING_SCORE_CAP, CHAR_SPACING_SCORE_MULT = 0.20, 0.04
WORD_SPACING_SCORE_CAP, WORD_SPACING_SCORE_MULT = 0.20, 0.04
LINE_HEIGHT_SCORE_CAP,  LINE_HEIGHT_SCORE_MULT  = 0.15, 0.03
NOISE_SCORE_CAP,        NOISE_SCORE_MULT        = 0.25, 0.05
SHARPNESS_SCORE_CAP,    SHARPNESS_SCORE_MULT    = 0.20, 0.04

# Font-mismatch score tiers — a CIDFont-subset mismatch (different
# embedded-subset fonts claiming the same role) is the strongest signal
# since it indicates two separate editing sessions; a mismatch on a
# critical value line (salary/total/etc.) is next; a mismatch on the
# letterhead is weighted lowest since letterheads legitimately use
# different fonts from the body even in unmodified documents.
FONT_MISMATCH_CIDFONT_SCORE   = 0.90
FONT_MISMATCH_CRITICAL_SCORE  = 0.70
FONT_MISMATCH_LETTERHEAD_SCORE = 0.15
FONT_MISMATCH_DEFAULT_SCORE   = 0.40

# Document-level signal (_build_signals()): the same base font family
# present both embedded and non-embedded (different subset tags) means the
# document went through two separate save/edit sessions with different font
# handling — one session embedded its subset, the other didn't. On the
# document's 0-100 anomaly_score scale, not the 0.0-1.0 per-line scale.
MIXED_FONT_EMBEDDING_SCORE = 25

# Per-line font-color consistency: spans within one PyMuPDF text LINE were
# drawn by the same renderer in the same pass, so they should share the
# exact RGB. Raising COLOR_DIFF_MIN alone can't separate a sloppy edit
# (ink color that's close-but-not-quite a match) from deliberate "Label:
# Value" styling (gray label + black filled-in value on one visual row —
# extremely common on payslips, bank statements, certificates), because
# the latter often has an even BIGGER RGB distance than a careless forgery.
# The actual distinguishing fact is repetition: a label/value color pair
# recurs on many lines throughout the document (it's the template's
# style), while an edited span's slightly-off color appears once. So a
# color is only a candidate anomaly if it's RARE document-wide — same
# frequency-clustering principle ocr_analyzer.py already uses for OCR
# word size/color outliers (_common_value_clusters/SIZE_CLUSTER_MIN_SHARE).
COLOR_DIFF_MIN                    = 15   # filters anti-aliasing/rounding noise, not real styling
COLOR_CLUSTER_MIN_SHARE           = 0.03  # a color on >=3% of spans is a deliberate document style
COLOR_CONSISTENCY_SCORE_PER_SPAN  = 10
COLOR_CONSISTENCY_SCORE_CAP       = 40

# Government ID cards (Aadhaar, PAN, driving licence, passport, voter ID)
# intentionally mix multiple ink colors on the same visual line by template
# design (e.g. Aadhaar's blue/black/orange) — the per-line color-consistency
# check below would otherwise flag that as tampering on every single one.
ID_CARD_KEYWORDS = [
    "aadhaar", "aadhar", "uid", "uidai", "unique identification",
    "permanent account number", "pan card",
    "driving licence", "driving license",
    "passport", "voter id", "epic no",
    "date of birth", "dob", "s/o", "d/o",
    "government of india",
]

# ── Upgrade 1: vertical line-gap density check ──────────────────────────────────
# Text injected into empty page space keeps the surrounding font/color, so the
# font/spacing checks above miss it — but it almost always breaks the page's
# vertical rhythm (the gap above/below it doesn't match the rest of the page).
# Computed PER PAGE (not document-wide) since different pages can legitimately
# have different line spacing.
LINE_GAP_MIN_LINES_PER_PAGE = 5     # need this many lines on a page for meaningful gap stats
LINE_GAP_LARGE_MULTIPLIER   = 3.0   # gaps > this many x the median = section/paragraph break, excluded from baseline
LINE_GAP_Z_THRESHOLD        = 3.5
LINE_GAP_MIN_WORDS          = 3     # don't flag short lines (labels/headers)
LINE_GAP_REPEAT_EXCLUDE     = 3     # a gap size recurring this many+ times = deliberate paragraph spacing
LINE_GAP_SCORE_PER_ANOMALY  = 20
LINE_GAP_SCORE_CAP          = 60
LINE_GAP_SCORE_WEIGHT       = 0.5

# Step 5 form-field exclusion for the gap check specifically — a line that's
# just a form field (date/signature line, underscores) legitimately sits in
# wide empty space and isn't an injection target.
LINE_GAP_FORM_FIELD_PATTERNS = [
    r'_{3,}',                                              # "____"
    r'^\s*(date|sign|signature)\s*:?\s*$',                 # "Date:", "Sign:", "Signature" alone
    r'\t{2,}',                                             # tab-separated single words (form fields)
    r'^\s*for\s+[\w\s.,&]+,\s*$',                          # "For Acme Technologies Pvt Ltd," — letter closing/signature block opener, always preceded by a deliberate gap reserved for the signature
]

# ── Upgrade 3: page-level baseline isolation ────────────────────────────────────
# A merged/compiled PDF legitimately has different fonts/sizes/colors per
# source page — scoring page 2 against page 1's document-wide baseline is
# what causes false positives on compilations. Below this many lines, a
# page's own stats are too unstable to score against (falls back to the
# document-wide profile instead).
MIN_LINES_FOR_PAGE_PROFILE = 8


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LineProfile:
    page: int
    line_num: int
    text: str
    font_name: str
    font_size: float
    char_spacing: float
    word_spacing: float
    line_height: float
    bbox: tuple            # (x0, y0, x1, y1) PDF points
    noise: float           # visual noise of line region
    sharpness: float       # visual sharpness of line region
    char_widths: list = field(default_factory=list)  # per-word avg char width samples, for CV check


@dataclass
class SuspiciousLine:
    page: int
    line_num: int
    text: str
    bbox: tuple
    anomalies: list[str]   # what specifically is wrong
    score: float           # 0.0 - 1.0


@dataclass
class ContentReport:
    total_lines: int
    suspicious_lines: list[SuspiciousLine]
    dominant_font: str
    dominant_font_ratio: float   # 0.0 - 1.0
    font_count: int              # how many unique fonts
    anomaly_score: int           # 0-100
    signals: list[str]           # human readable summary signals
    pdf_type: str                # native_text | scanned | mixed


# ── Feature extraction ─────────────────────────────────────────────────────────

class ContentAnalyzer:

    RENDER_DPI = 150

    def analyze(self, pdf_path: str, fonts: list = None) -> ContentReport:
        """
        fonts: optional list of {'name', 'embedded', ...} dicts from
        MetadataExtractor's MetadataReport.fonts — passed in by main.py
        (already extracted for Layer 1) rather than re-extracted here, so
        the same pikepdf font table isn't read twice per analysis.
        """
        pdf_type = self._detect_pdf_type(pdf_path)
        lines    = self._extract_lines(pdf_path)

        if not lines:
            return ContentReport(
                total_lines=0,
                suspicious_lines=[],
                dominant_font="",
                dominant_font_ratio=0.0,
                font_count=0,
                anomaly_score=0,
                signals=["No extractable text found — document may be image-based"],
                pdf_type=pdf_type,
            )

        profile          = self._build_profile(lines)
        try:
            per_page_profiles = self._build_per_page_profiles(lines, profile)
        except Exception:
            per_page_profiles = None

        # Upgrade 4 — first pass over subset-embedded fonts before scoring,
        # so platform-generated missing-glyph placeholders (Canva/Figma/
        # InDesign/etc.) can be told apart from a one-off injected/edited
        # occurrence of the same character.
        try:
            glyph_registry = self._build_glyph_registry(pdf_path)
        except Exception:
            glyph_registry = {}

        suspicious_lines = self._score_lines(lines, profile, per_page_profiles, glyph_registry)

        # Upgrade 1 — vertical line-gap density: catches text injected into
        # empty page space that font/spacing checks above miss (it inherits
        # the surrounding font/color, but breaks the page's line rhythm).
        # Computed from `lines` (not `suspicious_lines`) and scored
        # separately in _build_signals, so these don't also get double-
        # counted into the generic high/med anomaly-count buckets there.
        try:
            gap_findings = self._check_line_gap_density(lines)
        except Exception:
            gap_findings = []

        # ID cards (Aadhaar/PAN/driving licence/passport/voter ID)
        # legitimately mix ink colors on one line by template design — the
        # per-line color-consistency check is suppressed for them entirely
        # rather than threshold-tuned, since there's no single tolerance
        # that fits both "blue/black/orange on one Aadhaar line" and "one
        # tampered span" at the same time.
        is_id_card = self._is_id_card_document(lines)
        color_issues = [] if is_id_card else self._check_color_consistency_per_line(pdf_path)
        signals, score   = self._build_signals(lines, suspicious_lines, profile, fonts or [], color_issues, gap_findings)

        # Merged into the report's suspicious_lines AFTER scoring above, so
        # they're visible/highlightable without affecting the high/med
        # anomaly-count buckets _build_signals already used to score them.
        for g in gap_findings:
            suspicious_lines.append(SuspiciousLine(
                page=g["page"],
                line_num=g["line_num"],
                text=g["text"],
                bbox=g["bbox"],
                anomalies=[f"[LINE_GAP] {g['reason']}"],
                score=0.5,
            ))
        suspicious_lines.sort(key=lambda x: x.score, reverse=True)

        return ContentReport(
            total_lines=len(lines),
            suspicious_lines=suspicious_lines,  # return ALL suspicious lines, no cap
            dominant_font=profile["dominant_font"],
            dominant_font_ratio=profile["dominant_font_ratio"],
            font_count=profile["font_count"],
            anomaly_score=score,
            signals=signals,
            pdf_type=pdf_type,
        )

    # ── PDF type detection ─────────────────────────────────────────────────────

    def _detect_pdf_type(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total = len(pdf.pages)
                with_text = sum(
                    1 for p in pdf.pages
                    if p.extract_text() and len(p.extract_text().strip()) > NATIVE_TEXT_MIN_CHARS
                )
            ratio = with_text / total if total > 0 else 0
            if ratio >= NATIVE_TEXT_RATIO_THRESHOLD:
                pdf_type = "native_text"
            elif ratio >= MIXED_TEXT_RATIO_THRESHOLD:
                pdf_type = "mixed"
            else:
                pdf_type = "scanned"
        except Exception:
            pdf_type = "native_text"

        try:
            import pikepdf
            with pikepdf.open(pdf_path) as pdf:
                info = pdf.docinfo
                producer = str(info.get("/Producer","")).lower()
                creator = str(info.get("/Creator","")).lower()
                if any(kw in producer or kw in creator for kw in SCANNER_KEYWORDS):
                    return "scanned_native"
        except Exception:
            pass

        # Override: vector PDFs (text outlined to paths, e.g. Canva/Figma/
        # Illustrator exports) have no usable text layer despite rendering
        # visible content — treat as "scanned" so the OCR layer runs on them.
        if self._is_vector_pdf(pdf_path):
            return "scanned"

        return pdf_type

    def _is_vector_pdf(self, pdf_path: str) -> bool:
        """
        Returns True if this PDF has no usable text layer — text is stored
        as vector paths, not text objects. Happens with Canva, Figma,
        Illustrator, and any tool that exports PDF with outlined/converted
        text. Not tool-specific: detects the underlying condition (no text
        objects despite visible page content) rather than matching producer/
        creator strings, so it generalizes to any vector-export pipeline.

        Detection: pdfplumber finds (near) no words across all pages AND the
        page still renders visible drawings — distinguishing "no text
        because vector-only" from "no text because blank page".
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_words = sum(
                    len(p.extract_words() or [])
                    for p in pdf.pages
                )
            if total_words > 10:
                return False  # has real text layer

            # Check if page has visual content despite no text
            doc = fitz.open(pdf_path)
            has_visual = False
            for page in doc:
                # If page has paths/drawings but no text = vector PDF
                paths = page.get_drawings()
                text = page.get_text("text").strip()
                if len(paths) > 10 and len(text) < 20:
                    has_visual = True
                    break
            doc.close()
            return has_visual
        except Exception:
            return False

    # ── Line extraction ────────────────────────────────────────────────────────

    def _extract_lines(self, pdf_path: str) -> list[LineProfile]:
        """
        Extract per-line features using pdfplumber + PyMuPDF for visuals.

        Word extraction falls back through three methods per page, for
        corporate PDFs with custom/non-standard font encodings that defeat
        the primary extractor:
          1. pdfplumber extract_words() — handles the vast majority of PDFs.
          2. PyMuPDF (fitz) text extraction — has its own independent
             encoding/CMap handling and recovers text pdfplumber misses.
          3. pikepdf raw content-stream decode — last resort, recovers
             literal Tj/TJ string operands directly with no font/encoding
             interpretation at all (see _extract_words_pikepdf_fallback).
          4. If all three find nothing on a page, that page contributes no
             lines — the OCR layer (Layer 3) picks up the slack since it
             works from the rendered image, not the text layer.
        """
        page_images = self._render_pages(pdf_path)
        all_lines   = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    words = page.extract_words(
                        extra_attrs=["fontname", "size"],
                        keep_blank_chars=False
                    )
                except Exception:
                    words = []

                if not words:
                    words = self._extract_words_fitz_fallback(pdf_path, page_num)
                if not words:
                    words = self._extract_words_pikepdf_fallback(pdf_path, page_num)
                if not words:
                    continue

                grouped = self._group_into_lines(words)
                img     = page_images.get(page_num)

                for line_num, line_words in enumerate(grouped):
                    if not line_words:
                        continue

                    text = " ".join(w["text"] for w in line_words).strip()
                    if not text or len(text) < 2:
                        continue

                    fonts = [w.get("fontname", "Unknown") for w in line_words]
                    sizes = [float(w.get("size", 11)) for w in line_words]
                    font_name = Counter(fonts).most_common(1)[0][0]
                    font_size = statistics.median(sizes)

                    x0 = min(w["x0"]     for w in line_words)
                    y0 = min(w["top"]    for w in line_words)
                    x1 = max(w["x1"]     for w in line_words)
                    y1 = max(w["bottom"] for w in line_words)

                    noise, sharpness = self._visual_features(
                        img, (x0, y0, x1, y1),
                        scale=self.RENDER_DPI / 72
                    )

                    all_lines.append(LineProfile(
                        page=page_num,
                        line_num=line_num,
                        text=text,
                        font_name=font_name,
                        font_size=font_size,
                        char_spacing=self._char_spacing(line_words),
                        word_spacing=self._word_spacing(line_words),
                        line_height=y1 - y0,
                        bbox=(x0, y0, x1, y1),
                        noise=noise,
                        sharpness=sharpness,
                        char_widths=self._char_widths(line_words),
                    ))

        return all_lines

    def _extract_words_fitz_fallback(self, pdf_path: str, page_num: int) -> list[dict]:
        """
        Fallback step 2: PyMuPDF text extraction, converted to the same
        word dict shape pdfplumber's extract_words() produces. PyMuPDF has
        its own font/encoding handling independent of pdfplumber's, so it
        can recover text from PDFs whose custom encoding defeats pdfplumber
        (no per-character font/size info is available this way, so
        fontname/size are filled with placeholders — downstream font-based
        checks simply won't fire on these lines, which is acceptable since
        the alternative is no text at all).
        """
        words = []
        try:
            doc = fitz.open(pdf_path)
            if page_num < len(doc):
                for x0, y0, x1, y1, word_text, *_ in doc[page_num].get_text("words"):
                    if word_text.strip():
                        words.append({
                            "text": word_text,
                            "x0": x0, "top": y0, "x1": x1, "bottom": y1,
                            "fontname": "Unknown",
                            "size": 11.0,
                        })
            doc.close()
        except Exception:
            pass
        return words

    def _extract_words_pikepdf_fallback(self, pdf_path: str, page_num: int) -> list[dict]:
        """
        Fallback step 3 (last resort): decode the page's raw content stream
        with pikepdf and recover literal strings straight from Tj/TJ
        text-showing operators. No font/encoding interpretation and no real
        text-positioning state machine is applied (that would require
        reimplementing the PDF rendering model), so recovered strings are
        placed on synthetic sequential lines spanning the page width rather
        than at their true coordinates. This keeps at least the document's
        text content available to the content/numeric layers when both
        pdfplumber and PyMuPDF fail to decode the encoding at all.
        """
        words = []
        try:
            import pikepdf
            with pikepdf.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return []
                page = pdf.pages[page_num]
                page_width = float(page.MediaBox[2] - page.MediaBox[0]) if "/MediaBox" in page else 612.0

                row = 0
                for operands, operator in pikepdf.parse_content_stream(page):
                    op = str(operator)
                    strings = []
                    if op == "Tj" and operands:
                        strings.append(operands[0])
                    elif op == "TJ" and operands:
                        for item in operands[0]:
                            if not isinstance(item, (int, float)):
                                strings.append(item)

                    for s in strings:
                        try:
                            text = bytes(s).decode("latin-1", errors="replace").strip()
                        except Exception:
                            continue
                        if not text:
                            continue
                        y = 20.0 * row
                        words.append({
                            "text": text,
                            "x0": 20.0, "top": y, "x1": min(page_width - 20.0, 20.0 + len(text) * 6.0), "bottom": y + 14.0,
                            "fontname": "Unknown",
                            "size": 11.0,
                        })
                        row += 1
        except Exception:
            pass
        return words

    def _render_pages(self, pdf_path: str) -> dict:
        images = {}
        try:
            doc   = fitz.open(pdf_path)
            scale = self.RENDER_DPI / 72
            mat   = fitz.Matrix(scale, scale)
            for i, page in enumerate(doc):
                try:
                    pix       = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    images[i] = np.frombuffer(
                        pix.samples, dtype=np.uint8
                    ).reshape(pix.h, pix.w, 3)
                except Exception:
                    continue
            doc.close()
        except Exception:
            pass
        return images

    def _group_into_lines(self, words: list) -> list[list]:
        if not words:
            return []
        sorted_words = sorted(words, key=lambda w: (round(w["top"] / 4) * 4, w["x0"]))
        lines, current, current_y = [], [sorted_words[0]], sorted_words[0]["top"]
        for w in sorted_words[1:]:
            if abs(w["top"] - current_y) <= 5:
                current.append(w)
            else:
                lines.append(sorted(current, key=lambda x: x["x0"]))
                current, current_y = [w], w["top"]
        if current:
            lines.append(sorted(current, key=lambda x: x["x0"]))
        return lines

    def _char_widths(self, words: list) -> list[float]:
        """
        Per-word average character width samples: (x1-x0)/len(text) for
        each word with more than one character. pdfplumber's
        extract_words() doesn't expose individual character bounding
        boxes, so each word's average stands in as one "character width"
        sample — used both for the document-wide char_spacing mean and
        for the per-line coefficient-of-variation uniformity check.
        """
        widths = []
        for w in words:
            if len(w["text"]) > 1:
                widths.append((w["x1"] - w["x0"]) / len(w["text"]))
        return widths

    def _char_spacing(self, words: list) -> float:
        widths = self._char_widths(words)
        return statistics.mean(widths) if widths else 0.0

    def _char_width_cv(self, line: LineProfile) -> float:
        """
        Coefficient of variation (std/mean) of this line's character-width
        samples. Returns None when there isn't enough data (fewer than 2
        samples, or a degenerate zero mean) to compute a meaningful CV.
        """
        widths = line.char_widths
        if len(widths) < 2:
            return None
        mean = statistics.mean(widths)
        if mean <= 0:
            return None
        return statistics.stdev(widths) / mean

    def _word_spacing(self, words: list) -> float:
        if len(words) < 2:
            return 0.0
        sw   = sorted(words, key=lambda w: w["x0"])
        gaps = [sw[i+1]["x0"] - sw[i]["x1"] for i in range(len(sw)-1)
                if 0 < sw[i+1]["x0"] - sw[i]["x1"] < 50]
        return statistics.mean(gaps) if gaps else 0.0

    def _visual_features(self, img, bbox, scale) -> tuple[float, float]:
        if img is None:
            return 0.0, 0.0
        x0, y0, x1, y1 = bbox
        px0 = max(0, int(x0 * scale))
        py0 = max(0, int(y0 * scale))
        px1 = min(img.shape[1], int(x1 * scale))
        py1 = min(img.shape[0], int(y1 * scale))
        if px1 <= px0 or py1 <= py0:
            return 0.0, 0.0
        region = img[py0:py1, px0:px1]
        if region.size == 0:
            return 0.0, 0.0
        gray      = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        noise     = float(np.std(gray))
        sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        return noise, sharpness

    # ── Document profile ───────────────────────────────────────────────────────

    def _build_profile(self, lines: list[LineProfile]) -> dict:
        def safe_stats(values):
            vals = [v for v in values if v and v > 0]
            if len(vals) < 2:
                return {"mean": vals[0] if vals else 0, "std": 1e-9, "median": vals[0] if vals else 0}
            return {
                "mean":   statistics.mean(vals),
                "std":    max(statistics.stdev(vals), 1e-9),
                "median": statistics.median(vals),
            }

        def trimmed_stats(values, trim_pct=0.10):
            """
            Same shape as safe_stats, but excludes the top/bottom trim_pct
            of values before computing mean/std — threshold saturation:
            one injected 36pt line among fifty 11pt lines otherwise
            inflates `std` enough that a separate, smaller font-size/
            spacing edit elsewhere in the document no longer clears
            Z_OUTLIER_THRESHOLD. Falls back to safe_stats when there
            aren't enough values left to trim meaningfully.
            """
            vals = [v for v in values if v and v > 0]
            if len(vals) < 4:
                return safe_stats(vals)
            sorted_vals = sorted(vals)
            trim = max(1, int(len(sorted_vals) * trim_pct))
            trimmed = sorted_vals[trim:-trim]
            if len(trimmed) < 2:
                return safe_stats(vals)
            return {
                "mean":   statistics.mean(trimmed),
                "std":    max(statistics.stdev(trimmed), 1e-9),
                "median": statistics.median(trimmed),
            }

        font_counts = Counter(l.font_name for l in lines)
        dominant    = font_counts.most_common(1)[0][0]

        total = len(lines)
        # Fonts appearing on >15% of lines are "design fonts" —
        # part of intentional document styling, not anomalies
        design_fonts = {
            font for font, count in font_counts.items()
            if count / total > DESIGN_FONT_RATIO_THRESHOLD
        }

        return {
            "dominant_font":       dominant,
            "dominant_font_ratio": font_counts[dominant] / len(lines),
            "font_count":          len(font_counts),
            "design_fonts":        design_fonts,
            "font_size":           trimmed_stats([l.font_size     for l in lines]),
            "char_spacing":        trimmed_stats([l.char_spacing   for l in lines]),
            "word_spacing":        trimmed_stats([l.word_spacing   for l in lines]),
            "line_height":         safe_stats([l.line_height    for l in lines]),
            "noise":               safe_stats([l.noise          for l in lines]),
            "sharpness":           safe_stats([l.sharpness      for l in lines]),
        }

    def _build_per_page_profiles(self, lines: list[LineProfile], global_profile: dict) -> dict:
        """
        One profile per page, used instead of the document-wide profile
        when scoring that page's lines — see Upgrade 3 in _score_lines().
        A page with too few lines falls back to the global profile since
        its own mean/std would be too unstable to score against reliably.
        """
        by_page: dict = {}
        for l in lines:
            by_page.setdefault(l.page, []).append(l)

        per_page = {}
        for page_num, page_lines in by_page.items():
            if len(page_lines) >= MIN_LINES_FOR_PAGE_PROFILE:
                per_page[page_num] = self._build_profile(page_lines)
            else:
                per_page[page_num] = global_profile
        return per_page

    # ── Line classification ────────────────────────────────────────────────────

    def _is_structural_line(self, line: LineProfile, all_lines: list) -> bool:
        """
        Returns True if this line is structural (header/footer/label)
        and should NOT be flagged for font mismatch.

        Structural lines:
        1. ALL CAPS text (section headers)
        2. Short lines under 4 words (field labels like "Name:", "Date:")
        3. Lines repeated on multiple pages (page headers/footers)
        4. Lines that are just numbers or dates
        """
        text = line.text.strip()
        words = text.split()

        # OVERRIDE: Never skip these lines regardless of any rule
        # These are the most common tamper targets
        text_lower = text.lower()
        if any(kw in text_lower for kw in ALWAYS_CHECK_KEYWORDS):
            return False  # never structural — always check
        if any(re.search(kw, text_lower) for kw in ALWAYS_CHECK_KEYWORDS_WORD_BOUNDARY):
            return False  # never structural — always check

        # Rule 1: ALL CAPS line = header
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > ALL_CAPS_RATIO_THRESHOLD:
            return True

        # Rule 2: Very short line = label (under 4 words)
        if len(words) <= SHORT_LINE_MAX_WORDS:
            return True

        # Rule 3: Repeated on multiple pages = header/footer
        same_text_pages = set(
            l.page for l in all_lines
            if l.text.strip() == text and l != line
        )
        if len(same_text_pages) >= 1:  # appears on at least one other page
            return True

        # Rule 4: Line is purely numeric/date (amounts, dates, IDs)
        non_space = text.replace(" ", "").replace("-", "").replace("/", "").replace(".", "")
        if non_space and sum(1 for c in non_space if c.isdigit()) / len(non_space) > NUMERIC_LINE_RATIO_THRESHOLD:
            return True

        # Rule 6: Universal structural patterns
        # Lines that are purely field label + value pairs

        # Pattern: "Label : Value" or "Label: Value" (field pairs)
        if re.match(r'^[A-Za-z\s]+\s*:\s*.+$', text) and len(words) <= LABEL_PATTERN_MAX_WORDS:
            return True

        # Pattern: Line starts with bullet/number (list items)
        if re.match(r'^[\-\•\*\d]+[\.\)]\s', text):
            return True

        # Pattern: Line is a separator/divider
        if re.match(r'^[\*\-\_\=\#]{5,}', text.strip()):
            return True

        # Rule 7
        if len(words) <= RULE7_MAX_WORDS and line.line_height > 0:
            return True

        # Rule 8: Colon anywhere in line = field label (fix space-colon pattern)
        if ' : ' in text or text.endswith(':'):
            return True

        # Rule 9: Separator lines (asterisks, dashes, equals)
        stripped = text.strip()
        unique_chars = set(stripped.replace(' ', ''))
        if len(unique_chars) <= 2 and len(stripped) > SEPARATOR_MIN_LENGTH:
            return True

        # Rule 11: First N lines of document = letterhead/header area
        # Company name, address, contact info always use different fonts
        if line.line_num < LETTERHEAD_LINE_COUNT and line.page == 0:
            return True

        # Rule 12: Line contains typical address/contact patterns
        text_lower_addr = text.lower()
        for pat in ADDRESS_PATTERNS:
            if re.search(pat, text_lower_addr if '@' in pat or 'road' in pat else text):
                return True

        return False

    def _same_font_family(self, font_a: str, font_b: str) -> bool:
        """
        Returns True if two fonts are from the same family.
        Times-Roman and Times-Bold = same family → not suspicious
        Helvetica and Courier = different family → suspicious
        """
        def base_family(font_name: str) -> str:
            name = font_name.lower()
            # Strip common suffixes
            for suffix in [
                "-bold", "-italic", "-bolditalic", "-regular",
                "-medium", "-light", "-heavy", "-black",
                "-roman", "-narrow", "-condensed", "-extended",
                "-oblique",
                "bold", "italic", "oblique", "regular", "roman", "mt", "ps",
                "bolditalicmt", "boldmt", "italicmt",
            ]:
                name = name.replace(suffix, "")
            # Strip AAAAAA+ prefix (embedded subset prefix)
            if "+" in name:
                name = name.split("+", 1)[1]
            return name.strip("-_ ")

        return base_family(font_a) == base_family(font_b)

    # ── Line scoring ───────────────────────────────────────────────────────────

    def _score_lines(
        self, lines: list[LineProfile], profile: dict,
        per_page_profiles: dict = None, glyph_registry: dict = None,
    ) -> list[SuspiciousLine]:
        suspicious = []
        global_profile = profile
        glyph_registry = glyph_registry or {}

        for line in lines:
            text_lower_check = line.text.lower()
            if any(p in text_lower_check for p in NEVER_FLAG_PATTERNS):
                continue  # skip this line entirely — it's a payslip header row

            # Upgrade 5 — form fields (Date:____, Sign:____, tab-separated
            # table cells) have wide, deliberately irregular spacing by
            # design. Suppresses ONLY the spacing-related checks below
            # (char/word spacing, line height); font size and color checks
            # still run, since a genuine edit on a form field would still
            # show up there.
            is_form_field = self._is_form_field_line(line.text)

            # Upgrade 3 — score this line against its OWN PAGE's profile,
            # not the document-wide one. A multi-document compilation
            # legitimately has different fonts/sizes/colors per source
            # page; comparing page 2 against page 1's baseline is what
            # produced false positives on merged PDFs. Falls back to the
            # global profile when the page has too few lines for its own
            # stats to be reliable (see _build_per_page_profiles).
            profile = (per_page_profiles or {}).get(line.page, global_profile)

            anomalies = []
            score     = 0.0

            # Replacement/placeholder glyph — a font/encoding failure, often
            # from a currency symbol (₹, €, $) typed in a font that lacks
            # that glyph after editing. Always checked, never gated by
            # _is_structural_line — an encoding-failure glyph is suspicious
            # on any line, structural or not.
            #
            # Upgrade 4 — but only if it's NOT consistent platform-subsetting
            # behavior: Canva/Figma/InDesign/Puppeteer/wkhtmltopdf all subset
            # fonts with custom glyph IDs, and a missing-glyph placeholder
            # that recurs throughout that font's usage (e.g. every Rs. symbol
            # on a Canva payslip) is the export tool's own behavior, not an
            # edit — see _is_glyph_consistent.
            found_chars = [ch for ch in REPLACEMENT_CHARS if ch in line.text]
            flagged_chars = [
                ch for ch in found_chars
                if not self._is_glyph_consistent(line.font_name, ch, glyph_registry)
            ]
            if flagged_chars:
                anomalies.append(
                    "Replacement character found in line — "
                    "possible font encoding mismatch from editing"
                )
                score += REPLACEMENT_CHAR_SCORE

            # Font mismatch — skip structural lines (headers, labels, repeated)
            if line.font_name != profile["dominant_font"]:
                if not self._is_structural_line(line, lines):
                    # Skip if same font family (Bold vs Regular = same family)
                    if not self._same_font_family(line.font_name, profile["dominant_font"]):
                        # Skip if this font is a design font (appears on >15% of lines)
                        # CIDFont mismatches are ALWAYS checked — they indicate
                        # different editing sessions, never intentional design choices
                        is_cidfont = "cidfont" in line.font_name.lower()
                        if is_cidfont or line.font_name not in profile.get("design_fonts", set()):
                            # Upgrade 4 — a CIDFont "mismatch" whose only
                            # distinguishing content is a consistently-
                            # subsetted placeholder glyph (not a real
                            # second editing session) is downgraded out of
                            # the highest-severity tier rather than fully
                            # suppressed, since the font name mismatch
                            # itself is still mildly informative.
                            line_replacement_chars = [ch for ch in REPLACEMENT_CHARS if ch in line.text]
                            is_glyph_subsetting_artifact = bool(line_replacement_chars) and all(
                                self._is_glyph_consistent(line.font_name, ch, glyph_registry)
                                for ch in line_replacement_chars
                            )
                            is_cidfont_mismatch = (
                                "cidfont" in line.font_name.lower() and
                                "cidfont" in profile["dominant_font"].lower() and
                                line.font_name != profile["dominant_font"] and
                                not is_glyph_subsetting_artifact
                            )
                            text_lower = line.text.lower()
                            is_critical = any(kw in text_lower for kw in CRITICAL_VALUE_KEYWORDS)
                            is_letterhead = (line.line_num < LETTERHEAD_LINE_COUNT and line.page == 0)
                            anomalies.append(
                                f"Font: '{line.font_name}' != dominant '{profile['dominant_font']}'"
                            )
                            score += FONT_MISMATCH_CIDFONT_SCORE if is_cidfont_mismatch else \
                                     (FONT_MISMATCH_CRITICAL_SCORE if is_critical else \
                                     (FONT_MISMATCH_LETTERHEAD_SCORE if is_letterhead else FONT_MISMATCH_DEFAULT_SCORE))

            # Font size outlier — only flag if NOT a structural line
            z = self._z(line.font_size, profile["font_size"])
            if z > Z_OUTLIER_THRESHOLD and not self._is_structural_line(line, lines):
                anomalies.append(
                    f"Font size {line.font_size:.1f}pt outlier "
                    f"(doc avg {profile['font_size']['mean']:.1f}pt, z={z:.1f})"
                )
                score += min(FONT_SIZE_SCORE_CAP, z * FONT_SIZE_SCORE_MULT)

            # Character spacing outlier
            if line.char_spacing > 0 and not self._is_structural_line(line, lines) and not is_form_field:
                z = self._z(line.char_spacing, profile["char_spacing"])
                if z > Z_OUTLIER_THRESHOLD:
                    anomalies.append(
                        f"Char spacing outlier (z={z:.1f})"
                    )
                    score += min(CHAR_SPACING_SCORE_CAP, z * CHAR_SPACING_SCORE_MULT)

            # Word spacing outlier
            if line.word_spacing > 0 and not self._is_structural_line(line, lines) and not is_form_field:
                z = self._z(line.word_spacing, profile["word_spacing"])
                if z > Z_OUTLIER_THRESHOLD:
                    anomalies.append(
                        f"Word spacing outlier (z={z:.1f})"
                    )
                    score += min(WORD_SPACING_SCORE_CAP, z * WORD_SPACING_SCORE_MULT)

            # Line height outlier
            if not self._is_structural_line(line, lines) and not is_form_field:
                z = self._z(line.line_height, profile["line_height"])
                if z > Z_OUTLIER_THRESHOLD:
                    anomalies.append(
                        f"Line height outlier (z={z:.1f})"
                    )
                    score += min(LINE_HEIGHT_SCORE_CAP, z * LINE_HEIGHT_SCORE_MULT)

            # Visual noise outlier
            if line.noise > 0 and not self._is_structural_line(line, lines):
                z = self._z(line.noise, profile["noise"])
                if z > Z_OUTLIER_THRESHOLD:
                    anomalies.append(
                        f"Visual noise outlier (z={z:.1f})"
                    )
                    score += min(NOISE_SCORE_CAP, z * NOISE_SCORE_MULT)

            # Visual sharpness outlier
            if line.sharpness > 0 and not self._is_structural_line(line, lines):
                z = self._z(line.sharpness, profile["sharpness"])
                if z > Z_OUTLIER_THRESHOLD:
                    anomalies.append(
                        f"Sharpness outlier (z={z:.1f})"
                    )
                    score += min(SHARPNESS_SCORE_CAP, z * SHARPNESS_SCORE_MULT)

            # Character-spacing uniformity (TASK 3): genuine typed text has
            # natural per-character width variation; retyped/edited text is
            # often unnaturally uniform.
            if not self._is_structural_line(line, lines) and len(line.text) > CHAR_SPACING_CV_MIN_CHARS:
                cv = self._char_width_cv(line)
                if cv is not None and cv < CHAR_SPACING_CV_THRESHOLD:
                    anomalies.append(
                        f"Unnaturally uniform character spacing (CV={cv:.3f})"
                    )
                    score += CHAR_SPACING_CV_SCORE

            if anomalies:
                suspicious.append(SuspiciousLine(
                    page=line.page,
                    line_num=line.line_num,
                    text=line.text[:80],
                    bbox=line.bbox,
                    anomalies=anomalies,
                    score=min(1.0, score),
                ))

        suspicious.sort(key=lambda x: x.score, reverse=True)
        return suspicious

    def _z(self, value: float, stats: dict) -> float:
        return abs(value - stats["mean"]) / stats["std"]

    def _trimmed_mean_std(self, values: list[float], trim_percent: int = 10) -> tuple[float, float]:
        """
        Mean/std excluding the top/bottom trim_percent of values — avoids
        threshold saturation where one extreme value inflates std enough
        that other real outliers no longer clear a z-score threshold.
        """
        vals = sorted(v for v in values if v is not None)
        if len(vals) < 4:
            if not vals:
                return 0.0, 1e-9
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) >= 2 else 1e-9
            return mean, max(std, 1e-9)
        trim = max(1, int(len(vals) * trim_percent / 100))
        trimmed = vals[trim:-trim]
        if len(trimmed) < 2:
            trimmed = vals
        return statistics.mean(trimmed), max(statistics.stdev(trimmed), 1e-9)

    def _is_line_gap_form_field(self, text: str) -> bool:
        text_lower = text.lower().strip()
        return any(re.search(p, text_lower) for p in LINE_GAP_FORM_FIELD_PATTERNS)

    def _check_line_gap_density(self, lines: list[LineProfile]) -> list[dict]:
        """
        Flags a line preceded by an abnormally large vertical gap relative
        to the rest of ITS OWN page — the signature of text dropped into
        empty page space rather than typed in the normal flow (which would
        push surrounding lines apart consistently, not just leave one gap).
        """
        findings = []
        by_page: dict = {}
        for l in lines:
            by_page.setdefault(l.page, []).append(l)

        for page_num, page_lines in by_page.items():
            page_lines = sorted(page_lines, key=lambda l: l.bbox[1])
            if len(page_lines) < LINE_GAP_MIN_LINES_PER_PAGE:
                continue

            # gap[i] = vertical space between line i and the line below it
            pairs = [
                (page_lines[i + 1].bbox[1] - page_lines[i].bbox[3], page_lines[i + 1])
                for i in range(len(page_lines) - 1)
            ]
            positive_gaps = [g for g, _ in pairs if g > 0]
            if len(positive_gaps) < LINE_GAP_MIN_LINES_PER_PAGE:
                continue
            median_gap = statistics.median(positive_gaps)

            # Body-text baseline: exclude negative (overlapping lines —
            # headers/tables) and very large (section/paragraph breaks,
            # which are expected) gaps before computing the baseline.
            body_gaps = [g for g in positive_gaps if g <= median_gap * LINE_GAP_LARGE_MULTIPLIER]
            if len(body_gaps) < LINE_GAP_MIN_LINES_PER_PAGE:
                continue
            mean, std = self._trimmed_mean_std(body_gaps)

            # Uniform spacing (std near zero) means there is no anomalous gap
            # to find — every gap is the same, so z-scores are meaningless and
            # dividing by std approaches infinity.  Skip the page entirely.
            if std < 0.5:
                continue

            # A gap value recurring 3+ times on the page is the page's
            # deliberate paragraph-spacing rhythm, not a one-off injection.
            gap_value_counts = Counter(round(g, 1) for g, _ in pairs if g > 0)

            page_candidates = []
            for gap, line_below in pairs:
                if gap <= 0:
                    continue
                z = min(abs(gap - mean) / max(std, 0.5), 100.0)
                if z <= LINE_GAP_Z_THRESHOLD:
                    continue
                if gap <= mean:
                    continue  # smaller gap = compressed text, not injection
                if gap_value_counts[round(gap, 1)] >= LINE_GAP_REPEAT_EXCLUDE:
                    continue  # identical gap value recurring = paragraph spacing
                if len(line_below.text.split()) < LINE_GAP_MIN_WORDS:
                    continue  # short line = header/label, not the injected content
                if self._is_line_gap_form_field(line_below.text):
                    continue
                page_candidates.append((gap, line_below, z))

            # 3+ DIFFERENT oversized gaps on one page (even at different
            # exact values) means the page mixes a dense table/list region
            # with normal section breaks — a single per-page baseline can't
            # tell a real injection from "the gap before/after the table"
            # in that layout, so the whole page's candidates are dropped
            # rather than risk flagging every section break in a payslip/
            # invoice template. A genuine isolated injection stays a lone
            # candidate and still fires.
            if len(page_candidates) >= LINE_GAP_REPEAT_EXCLUDE:
                continue

            for gap, line_below, z in page_candidates:
                findings.append({
                    "page": line_below.page,
                    "line_num": line_below.line_num,
                    "bbox": line_below.bbox,
                    "text": line_below.text[:80],
                    "gap": round(gap, 1),
                    "expected_gap": round(mean, 1),
                    "z_score": round(z, 2),
                    "reason": (
                        f"Line preceded by abnormal vertical gap "
                        f"({gap:.1f}pt vs page baseline {mean:.1f}±{std:.1f}pt, "
                        f"z={z:.1f}) — possible text inserted into empty space"
                    ),
                })

        return findings

    def _is_id_card_document(self, lines: list) -> bool:
        text_lower = " ".join(line.text.lower() for line in lines)
        return any(kw in text_lower for kw in ID_CARD_KEYWORDS)

    def _is_form_field_line(self, text: str) -> bool:
        """
        True for form-field lines (Date:____, Sign:____, table cells
        separated by tabs) — these have wide, deliberately irregular
        spacing by design, not from an edit.
        """
        text_lower = text.lower().strip()
        for pattern in FORM_FIELD_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        # 1-2 words but a long character count = wide gaps/padding between
        # them ("___________  Authorized Signatory"), not a real sentence.
        words = text_lower.split()
        if len(words) <= FORM_FIELD_SHORT_LINE_MAX_WORDS and len(text) > FORM_FIELD_SHORT_LINE_MIN_LEN:
            return True
        return False

    def _build_glyph_registry(self, pdf_path: str) -> dict:
        """
        First pass over every subset-embedded font ("AAAAAA+Helvetica") in
        the document: counts how often each watched placeholder/replacement
        glyph appears, against that font's total character count. A glyph
        that recurs throughout a subset font's usage is the export tool's
        own missing-glyph behavior (Canva/Figma/InDesign/Puppeteer/
        wkhtmltopdf all do this) — see _is_glyph_consistent.
        """
        registry: dict = {}
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return registry
        try:
            for page in doc:
                rawdict = page.get_text("rawdict")
                for block in rawdict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font = span.get("font", "")
                            if "+" not in font:
                                continue
                            text = "".join(ch.get("c", "") for ch in span.get("chars", []))
                            if not text:
                                continue
                            entry = registry.setdefault(font, {})
                            for char in text:
                                entry["__total__"] = entry.get("__total__", 0) + 1
                                if char in GLYPH_WATCH_CHARS:
                                    entry[char] = entry.get(char, 0) + 1
        finally:
            doc.close()
        return registry

    def _is_glyph_consistent(self, font_name: str, char: str, glyph_registry: dict) -> bool:
        """
        True if `char` appears on more than GLYPH_CONSISTENCY_RATIO_THRESHOLD
        of `font_name`'s subset characters document-wide — platform-
        generated missing-glyph behavior, not a one-off injected/edited
        occurrence. Only subset fonts ("+" in name) are tracked at all, so
        a non-subset font always returns False (never suppressed here).
        """
        if "+" not in font_name or font_name not in glyph_registry:
            return False
        entry = glyph_registry[font_name]
        total = entry.get("__total__", 1)
        char_count = entry.get(char, 0)
        return (char_count / total) > GLYPH_CONSISTENCY_RATIO_THRESHOLD

    def _check_color_consistency_per_line(self, pdf_path: str) -> list[dict]:
        """
        Within each text line, flag a span whose RGB color both (a)
        differs meaningfully from the line's dominant color and (b) is
        RARE across the whole document — see COLOR_CLUSTER_MIN_SHARE above
        for why frequency, not raw color distance, is what separates a
        real edit from deliberate label/value styling.
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return []

        line_spans = []  # list of list[span_color_dict], one per line
        all_colors = []

        try:
            for page_num in range(len(doc)):
                rawdict = doc[page_num].get_text("rawdict")
                for block in rawdict.get("blocks", []):
                    for line in block.get("lines", []):
                        spans = []
                        for span in line.get("spans", []):
                            # rawdict spans have no "text" field, only a
                            # per-character "chars" list (see
                            # ocr_analyzer._split_span_into_words) — same
                            # reconstruction needed here.
                            text = "".join(ch.get("c", "") for ch in span.get("chars", [])).strip()
                            if not text:
                                continue
                            color_int = span.get("color", 0)
                            rgb = (
                                (color_int >> 16) & 0xFF,
                                (color_int >> 8) & 0xFF,
                                color_int & 0xFF,
                            )
                            spans.append({
                                "text": text, "rgb": rgb,
                                "bbox": span.get("bbox", (0, 0, 0, 0)),
                                "page": page_num,
                            })
                            all_colors.append(rgb)
                        if len(spans) >= 2:
                            line_spans.append(spans)
        finally:
            doc.close()

        if not all_colors:
            return []

        color_counts = Counter(all_colors)
        threshold = max(2, round(len(all_colors) * COLOR_CLUSTER_MIN_SHARE))
        common_colors = {c for c, n in color_counts.items() if n >= threshold}

        candidates = []
        for spans in line_spans:
            counts = Counter(s["rgb"] for s in spans)
            dominant_color = counts.most_common(1)[0][0]
            if len(counts) <= 1:
                continue
            for s in spans:
                if s["rgb"] == dominant_color or s["rgb"] in common_colors:
                    continue
                diff = sum(abs(a - b) for a, b in zip(s["rgb"], dominant_color))
                if diff < COLOR_DIFF_MIN:
                    continue
                candidates.append((s, dominant_color, diff))

        # A real edit happens once, in one place. The exact same (text,
        # color, dominant_color) combination recurring on 2+ DIFFERENT
        # pages is a repeated letterhead/header/footer element rendered
        # with a slightly different anti-aliased near-black than the body
        # text -- not an edit replicated identically across pages.
        pages_per_combo = {}
        for s, dominant_color, _ in candidates:
            key = (s["text"], s["rgb"], dominant_color)
            pages_per_combo.setdefault(key, set()).add(s["page"])

        anomalies = []
        for s, dominant_color, diff in candidates:
            key = (s["text"], s["rgb"], dominant_color)
            if len(pages_per_combo[key]) >= 2:
                continue
            anomalies.append({
                    "page": s["page"],
                    "text": s["text"],
                    "bbox": s["bbox"],
                    "color": s["rgb"],
                    "dominant_color": dominant_color,
                    "color_diff": diff,
                    "reason": (
                        f"Color mismatch within same line: span '{s['text'][:20]}' "
                        f"uses RGB{s['rgb']} while the rest of the line uses "
                        f"RGB{dominant_color} (diff={diff}) — possible text "
                        f"edited with a different tool"
                    ),
                })
        return anomalies

    # ── Signals + score ────────────────────────────────────────────────────────

    def _mixed_font_embedding_families(self, fonts: list) -> set:
        """
        Base font families that appear both embedded and non-embedded
        (under different subset-prefix tags) — see MIXED_FONT_EMBEDDING_SCORE.
        """
        embedded_families = set()
        unembedded_families = set()

        for font in fonts:
            name = font.get("name", "")
            base = name.lstrip("/")
            base = re.sub(r'^[A-Z]{6}\+', '', base)
            base = base.lower().split('-')[0]
            if base.endswith('mt'):
                base = base[:-2]
            if not base:
                continue
            if font.get("embedded"):
                embedded_families.add(base)
            else:
                unembedded_families.add(base)

        return embedded_families & unembedded_families

    def _build_signals(
        self, lines, suspicious_lines, profile, fonts: list = None,
        color_issues: list = None, gap_findings: list = None,
    ) -> tuple[list[str], int]:
        signals = []
        score   = 0

        # Mixed embedded/non-embedded subsets of the same font family —
        # indicates two separate edit sessions with different font handling.
        mixed = self._mixed_font_embedding_families(fonts or [])
        if mixed:
            signals.append(
                f"Font family '{', '.join(sorted(mixed))}' has both embedded "
                f"and non-embedded subsets — indicates multiple edit "
                f"sessions with different font rendering"
            )
            score += MIXED_FONT_EMBEDDING_SCORE

        # Font diversity — only flag if dominant font covers less than 60% of lines
        if profile["font_count"] > 3 and profile["dominant_font_ratio"] < 0.60:
            signals.append(
                f"{profile['font_count']} different fonts detected "
                f"(dominant: '{profile['dominant_font']}' "
                f"in {profile['dominant_font_ratio']:.0%} of lines) — "
                f"unusual font diversity for a single document"
            )
            score += 15

        # High-confidence suspicious lines
        high = [l for l in suspicious_lines if l.score > 0.5]
        med  = [l for l in suspicious_lines if 0.3 < l.score <= 0.5]

        if high:
            signals.append(
                f"{len(high)} line(s) with strong anomaly — "
                f"font/spacing/visual breaks consistency"
            )
            score += min(50, len(high) * 12)

        if med:
            signals.append(
                f"{len(med)} line(s) with moderate anomaly"
            )
            score += min(20, len(med) * 5)

        if color_issues:
            signals.append(
                f"{len(color_issues)} span(s) with color inconsistency within "
                f"the same text line — text color doesn't match the rest of "
                f"the line, possible edit with a different tool"
            )
            score += min(COLOR_CONSISTENCY_SCORE_CAP,
                         len(color_issues) * COLOR_CONSISTENCY_SCORE_PER_SPAN)

        if gap_findings:
            for g in gap_findings:
                bbox = g["bbox"]
                signals.append(
                    f"[LINE_GAP] Page {g['page']+1}: line preceded by abnormal "
                    f"vertical gap ({g['gap']:.1f}pt vs page baseline "
                    f"{g['expected_gap']:.1f}pt, z={g['z_score']:.1f}) — "
                    f"possible text inserted into empty space "
                    f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
                )
            gap_score = min(LINE_GAP_SCORE_CAP, len(gap_findings) * LINE_GAP_SCORE_PER_ANOMALY)
            score += gap_score * LINE_GAP_SCORE_WEIGHT

        if not signals:
            signals.append(
                "Content is internally consistent — "
                "no font, spacing, or visual anomalies found"
            )

        return signals, min(100, score)
