"""
OCR Word-Level Analyzer — Layer 3
Extracts every word/span with its bounding box, confidence, estimated font
size, and sampled text color, then flags words that are statistical outliers
against the document's own baseline.

This makes OCR useful for IMAGE-based document analysis (scanned/photographed
pages), not just confidence-drop checking: a word pasted in from a different
source tends to differ in size or color from the surrounding text, even when
OCR confidence on it looks fine.

For native-text PDFs, the same size/color/position extraction runs against
PyMuPDF's span data instead of Tesseract (faster and exact, since there's
no OCR uncertainty on embedded text).
"""

import os
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field

import fitz
import numpy as np
import pytesseract
from PIL import Image

from pdf_utils import get_qr_zones, bbox_overlaps_qr_zone

# Pixel profiling imports — graceful degradation if missing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


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

RENDER_DPI         = 300    # higher DPI for better OCR accuracy on images
CONFIDENCE_MIN     = 30     # skip words below this (likely noise)
CONFIDENCE_LOW     = 70     # used for reporting/labeling only — NOT the flag cutoff
# A flat global "confidence < 70%" cutoff misfires on any inherently
# low-quality scan, where ordinary OCR noise puts much of the page below 70%
# with nothing actually edited. What's actually suspicious is a word reading
# meaningfully WORSE than the rest of ITS OWN page — that's the original
# signal this layer was built around, and it survives here as the primary
# confidence check. CONFIDENCE_ABSOLUTE_FLOOR is a much lower backstop for
# words so low they're suspicious no matter how bad the page already is.
CONFIDENCE_DROP_THRESHOLD  = 20   # points below this word's own page average
CONFIDENCE_ABSOLUTE_FLOOR  = 50
# If at least this fraction of a page's words already read below
# CONFIDENCE_ABSOLUTE_FLOOR, the floor stops being a meaningful per-word
# outlier signal for that page — see the comment above page_below_floor_rate.
PAGE_NOISY_FLOOR_RATE      = 0.25

# Per-anomaly-type score weights — size/color anomalies are rare and
# specific (a real visual edit), confidence anomalies are common and noisy
# (ordinary low-quality-scan OCR misreads), so they're weighted and capped
# very differently rather than contributing equally per word.
SIZE_SCORE_PER_WORD       = 8
SIZE_SCORE_CAP            = 40
COLOR_SCORE_PER_WORD      = 8
COLOR_SCORE_CAP           = 40
CONFIDENCE_SCORE_PER_WORD = 3
CONFIDENCE_SCORE_CAP      = 30
SIZE_Z_THRESHOLD   = 3.0    # z-score to flag font size anomaly
COLOR_Z_THRESHOLD  = 3.0    # z-score to flag color anomaly
MIN_WORDS          = 10     # need at least this many words for stats

# Baseline-alignment check: words on the same line (same Tesseract block/
# par/line_num, or same PyMuPDF line_id for native text) should share a
# common bottom-edge y-coordinate -- they were rendered by the same engine
# in the same pass. A manually inserted/replaced word rarely lands on the
# exact same baseline as its neighbors. Threshold is in PDF points (bbox
# values here are already converted to points, not pixels) and is applied
# together with a z-score so a globally skewed/rotated scan (every word
# on the line shifted by the same amount) doesn't fire -- only a word
# that's the outlier WITHIN its own line counts.
BASELINE_DEVIATION_PT      = 1.2
# A z-score from a line's OWN (small-n) stdev has a hard mathematical
# ceiling: for a line of 3 words where 2 match and 1 differs, z = sqrt(3)/2
# = 1.73 no matter how large the actual deviation is -- the outlier itself
# inflates the stdev it's being measured against. 2.5 would never fire on a
# 3-word line regardless of how badly misaligned a word is. 1.6 sits just
# above that ceiling instead, leaning on BASELINE_DEVIATION_PT as the
# primary real-world filter.
BASELINE_Z_THRESHOLD       = 1.6
MIN_WORDS_PER_LINE_BASELINE = 3
# Below this OCR confidence, the bbox itself is unreliable (Tesseract
# barely knows what it read), so it's excluded from both the line's
# reference baseline and the flaggable set -- see _check_baseline_alignment.
BASELINE_MIN_CONFIDENCE    = 75
BASELINE_SCORE_PER_WORD    = 10
BASELINE_SCORE_CAP         = 40

# A document's body text is the dominant, most-repeated size — headings,
# titles, and table headers are legitimate, INTENTIONAL size outliers, not
# tampering, and would otherwise get flagged on every normal document that
# has so much as a title line. The size baseline is therefore built from
# the single most common size bucket (rounded to SIZE_BUCKET_PT), not a
# flat mean/std over every span — only words that don't belong to ANY
# common, repeated size cluster are candidates for a size anomaly.
SIZE_BUCKET_PT          = 0.5   # rounding granularity for clustering sizes
SIZE_CLUSTER_MIN_SHARE  = 0.05  # a size used by >=5% of words counts as "common", not anomalous

# Native-text PDFs legitimately mix colors (headers, links, brand colors),
# so a color anomaly there only matters if it's RARE — a color used by many
# words (e.g. an entire colored letterhead block) is a deliberate design
# choice, not an edit. Same clustering logic as size, applied to brightness.
COLOR_CLUSTER_MIN_SHARE = 0.05

# Tesseract noise filter — single stray characters/symbols are common OCR
# garbage on real scans and would otherwise pollute the size/color baseline.
MIN_OCR_WORD_LEN = 2

# Reference codes/IDs (UPI handles, transaction refs, account numbers,
# bracket-enclosed fragments) are intrinsically hard for Tesseract to read
# confidently — a long alphanumeric code or punctuation-heavy token reads
# low-confidence on EVERY scan, tampered or not. Counting them in the
# confidence baseline just adds noise that swamps the real signal, so
# they're excluded from extraction entirely (mirrors the dedicated filter
# the previous confidence-only version of this analyzer had). See
# _looks_like_reference_code for the exact rule.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UPGRADE 1: UNIVERSAL OCR PIXEL PROFILING (Layer 3, second pass)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Pixel-level profiling scores and thresholds
PIXEL_SIZE_ANOMALY_SCORE   = 15
PIXEL_COLOR_ANOMALY_SCORE  = 20
PIXEL_DIGITAL_PASTE_SCORE  = 30
PIXEL_PHRASE_BONUS         = 15
PIXEL_SCORE_CAP            = 100

# Font height profiling thresholds
PIXEL_SIZE_DEVIATION_RATIO = 0.20     # 20% deviation from median
PIXEL_SIZE_DEVIATION_MIN   = 5        # minimum 5px difference to flag

# Dominant-cluster exemption: a height/brightness that recurs across the
# document (company header, title, table row) is a legitimate layout
# element, not an edited word, even though it differs from the median of
# whatever line/page it happens to sit on. Mirrors the same frequency-
# clustering principle as _common_value_clusters/_line_dominant_buckets
# (used by the native-text path) — without it, every document with a
# header line scores 100 purely from its own header.
PIXEL_DOMINANT_HEIGHT_BUCKET_PX = 2     # round heights to nearest 2px before clustering
PIXEL_DOMINANT_TOP_N            = 3     # the 3 most common heights/brightnesses are always exempt
PIXEL_DOMINANT_MIN_SHARE        = 0.05  # plus anything else used by >=5% of words/page

# Per-line brightness tolerance: a short word ("LTD") naturally averages a
# slightly different brightness than a long word ("TECHNOLOGIES") in the
# SAME ink/font due to anti-aliasing — a couple of brightness units of
# spread among words on the same physical line is normal rendering noise,
# not a different word matching a different dominant bucket exactly.
PIXEL_LINE_BRIGHTNESS_TOLERANCE = 6

# Color profiling thresholds
PIXEL_COLOR_MAD_THRESHOLD  = 3.0      # MAD-based z-score > 3
PIXEL_DIGITAL_BLACK_THRESHOLD = 8     # pure black on scanned document
PIXEL_SCANNED_BRIGHTNESS_MIN  = 25    # threshold above which document is "scanned"

# Government ID cards (Aadhaar, PAN, driving licence, passport, voter ID)
# intentionally print 3-4 ink colors by template design, and a colored
# header/banner band across the top of the card — both legitimate, so the
# pixel-color check is loosened (not disabled) for documents that look like
# an ID card, and skips the header band entirely.
ID_CARD_KEYWORDS = [
    "aadhaar", "aadhar", "uid", "uidai", "unique identification",
    "permanent account number", "pan card",
    "driving licence", "driving license",
    "passport", "voter id", "epic no",
    "date of birth", "dob", "s/o", "d/o",
    "government of india",
]
PIXEL_COLOR_MAD_THRESHOLD_ID_CARD     = 5.5
PIXEL_DIGITAL_BLACK_THRESHOLD_ID_CARD = 3
ID_CARD_HEADER_ZONE_FRAC              = 0.2  # top 20% of the page is excluded
ID_CARD_MIN_ANOMALIES_FOR_SCORE       = 3    # 1-2 hits on an ID card are noise, not a tamper signal

# Rendering for pixel profiling
PIXEL_PROFILE_DPI = 144  # 2x standard 72 DPI is enough for pixel analysis
PIXEL_MIN_WORD_PIXELS = 15  # need at least this many pixels to be reliable
PIXEL_MIN_TEXT_PIXELS_IN_BBOX = 15  # threshold for text pixel detection


@dataclass
class WordData:
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    estimated_size: float    # font size estimated from bbox height
    color_r: int              # sampled text color
    color_g: int
    color_b: int
    color_brightness: float  # 0=black, 255=white


@dataclass
class OCRWordAnomaly:
    page: int
    word: str
    bbox: tuple
    anomaly_types: list       # "size", "color", "confidence", "baseline"
    size_z: float = 0.0
    color_z: float = 0.0
    baseline_z: float = 0.0
    confidence: float = 0.0
    reason: str = ""


@dataclass
class PixelAnomaly:
    """Anomaly detected by pixel profiling (Upgrade 1)."""
    page: int
    word: str
    bbox: tuple
    anomaly_type: str  # "size", "color", or "digital_paste"
    severity: str      # "low", "medium", "high"
    confidence: float  # 0-100
    reason: str = ""


@dataclass
class OCRReport:
    pages_analyzed: int
    pdf_type: str
    word_count: int
    avg_confidence: float
    avg_font_size: float
    avg_color_brightness: float
    word_anomalies: list = field(default_factory=list)
    anomaly_score: int = 0
    signals: list = field(default_factory=list)
    pixel_anomalies: list = field(default_factory=list)  # Upgrade 1
    pixel_score: int = 0  # Upgrade 1


class OCRAnalyzer:

    def analyze(self, pdf_path: str) -> OCRReport:
        doc = fitz.open(pdf_path)

        pdf_type = self._detect_pdf_type(doc, pdf_path)

        # For native text PDFs with a real text layer, use PyMuPDF's span
        # data (faster + exact — no OCR uncertainty on embedded text). For
        # image/scanned/vector-outlined PDFs, fall back to Tesseract OCR.
        if pdf_type == "native_text":
            report = self._analyze_native_text(doc)
        else:
            report = self._analyze_with_ocr(doc)

        # UPGRADE 1: Run pixel profiling on all document types
        if HAS_CV2:
            # _analyze_native_text()/_analyze_with_ocr() each close `doc`
            # internally before returning, so it's no longer usable here —
            # pixel profiling opens its own fresh copy from pdf_path instead
            # of reusing this (already-closed) doc object.
            pixel_report = self._profile_pixels_all_pages(pdf_path)
            report.pixel_anomalies = pixel_report.get("anomalies", [])
            report.pixel_score = pixel_report.get("score", 0)
            # Combine scores using max (they may detect the same edit)
            original_score = report.anomaly_score
            report.anomaly_score = max(original_score, report.pixel_score)
            if report.pixel_score > 0:
                report.signals.append(
                    f"Pixel profiling: {len(report.pixel_anomalies)} anomalies "
                    f"(score={report.pixel_score})"
                )

        # _analyze_native_text()/_analyze_with_ocr() already close `doc`
        # unconditionally before returning — closing it again here raises
        # ValueError("document closed"), uncaught, which is what actually
        # turns into the 500 (the pixel-profiling warning is a side effect
        # of the same already-closed doc, not the thing that crashes the
        # request).
        return report

    def _is_id_card_document(self, doc) -> bool:
        try:
            text_lower = " ".join(page.get_text().lower() for page in doc)
        except Exception:
            return False
        return any(kw in text_lower for kw in ID_CARD_KEYWORDS)

    def _detect_pdf_type(self, doc, pdf_path: str) -> str:
        """
        chars-per-page > 100 is the primary signal — vector PDFs (text
        outlined to paths) naturally fail this since there's nothing to
        extract, which already routes them to OCR correctly. A PDF
        produced by a scanner is forced to "scanned" even if it carries a
        thin/inaccurate embedded text layer (some scanners bake one in) —
        trusting that layer's span data would mean trusting bad font-size/
        color readings from a layer that was never meant to be authoritative.
        """
        total_chars = sum(len(p.get_text()) for p in doc)
        total_pages = len(doc)
        if total_chars / max(total_pages, 1) <= 100:
            return "scanned"
        if self._is_scanner_or_vector(pdf_path):
            return "scanned"
        return "native_text"

    @staticmethod
    def _is_scanner_or_vector(pdf_path: str) -> bool:
        try:
            from content_analyzer import ContentAnalyzer, SCANNER_KEYWORDS
            if ContentAnalyzer()._is_vector_pdf(pdf_path):
                return True
            import pikepdf
            with pikepdf.open(pdf_path) as pdf:
                info = pdf.docinfo
                producer = str(info.get("/Producer", "")).lower()
                creator = str(info.get("/Creator", "")).lower()
                return any(kw in producer or kw in creator for kw in SCANNER_KEYWORDS)
        except Exception:
            return False

    @staticmethod
    def _looks_like_reference_code(word: str) -> bool:
        """True for UPI handles, transaction refs, account numbers, and
        bracket/pipe-corrupted number fragments — see the module-level
        comment above _REFERENCE_CODE_RE for why these are excluded from
        OCR stats."""
        if len(word) <= 6:
            return False
        if any(c in '[]|' for c in word):
            return True
        digit_count = sum(c.isdigit() for c in word)
        if digit_count == 0:
            return False
        # Majority-digit tokens (account numbers, amounts) are intrinsically
        # low-confidence reads regardless of tampering.
        if digit_count / len(word) >= 0.5:
            return True
        # Digit + separator (UPI-1234567-DR-NAME style) is code-like even
        # when letters outnumber digits.
        return any(c in '~-@:_/' for c in word)

    @staticmethod
    def _is_ocr_gibberish(word: str) -> bool:
        """
        English-language-specific OCR noise filter (ported from the
        confidence-only version of this analyzer, which measurably produced
        a cleaner baseline than reference-code filtering alone): an
        all-consonant run is almost always a misread, not a real word, and
        a 2-3 character all-lowercase token ("ee", "wo", "Ce.") is the
        characteristic shape of OCR garbage on noisy scan regions. Only
        valid for English-language documents — will incorrectly discard
        real words in languages where consonant clusters without Latin
        vowels are normal.
        """
        vowels = set('aeiouAEIOU')
        if len(word) > 1 and not any(c in vowels for c in word):
            return True
        if len(word) <= 3 and not word.isdigit() and word.lower() == word:
            return True
        return False

    @staticmethod
    def _split_span_into_words(span: dict) -> list:
        """
        rawdict spans have NO "text" field — PyMuPDF only exposes a
        per-character "chars" list there (size/color/font are span-level,
        since a span is a run of identically-styled characters, but a span
        commonly covers an entire drawString() line, not a single word).
        Reconstruct word-level text + bbox by splitting that char list on
        whitespace, so position/size/color are tracked per WORD as intended,
        not per multi-word run.
        """
        words = []
        current = []
        for ch in span.get("chars", []):
            c = ch.get("c", "")
            if c.isspace():
                if current:
                    words.append(current)
                    current = []
                continue
            current.append(ch)
        if current:
            words.append(current)

        result = []
        for word_chars in words:
            text = "".join(ch["c"] for ch in word_chars)
            bboxes = [ch["bbox"] for ch in word_chars]
            bbox = (
                min(bb[0] for bb in bboxes), min(bb[1] for bb in bboxes),
                max(bb[2] for bb in bboxes), max(bb[3] for bb in bboxes),
            )
            result.append((text, bbox))
        return result

    def _analyze_native_text(self, doc) -> OCRReport:
        """
        For native text PDFs: use PyMuPDF rawdict to get per-character data,
        regrouped into words (see _split_span_into_words). Extract font size
        and color for every word and flag words that differ from the
        document's baseline.
        """
        all_spans = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            rawdict = page.get_text("rawdict")

            for block in rawdict.get("blocks", []):
                for line_idx, line in enumerate(block.get("lines", [])):
                    line_id = (page_num, block.get("number", 0), line_idx)
                    for span in line.get("spans", []):
                        size = float(span.get("size", 0))
                        color_int = span.get("color", 0)

                        r = (color_int >> 16) & 0xFF
                        g = (color_int >> 8) & 0xFF
                        b = color_int & 0xFF
                        brightness = 0.299 * r + 0.587 * g + 0.114 * b

                        for text, bbox in self._split_span_into_words(span):
                            if len(text) < 2:
                                continue

                            all_spans.append({
                                "text": text,
                                "page": page_num,
                                "bbox": bbox,
                                "size": size,
                                "color_r": r,
                                "color_g": g,
                                "color_b": b,
                                "brightness": brightness,
                                "confidence": 100,  # native text = perfect confidence
                                "line_id": line_id,
                            })

        doc.close()

        if len(all_spans) < MIN_WORDS:
            return OCRReport(
                pages_analyzed=0, pdf_type="native_text",
                word_count=0, avg_confidence=100,
                avg_font_size=0, avg_color_brightness=0,
                word_anomalies=[], anomaly_score=0,
                signals=["Native text PDF — insufficient spans for analysis"],
            )

        return self._compute_anomalies(all_spans, "native_text")

    def _extract_words_from_image(
        self, img: Image.Image, img_array: np.ndarray, page_num: int, scale: float
    ) -> list:
        """
        Run Tesseract on `img` and extract word-level text/position/size/
        color. `scale` is pixels-per-PDF-point (RENDER_DPI/72 when `img`
        came from rendering a PDF page at RENDER_DPI; 1.0 for a directly
        uploaded image with no PDF page geometry to map onto, in which case
        bbox/size values are in the image's native pixel space instead of
        points — fine since callers only ever compare values within the
        same report, never across unit systems).
        """
        all_words = []
        try:
            tsv_data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config="--psm 6"
            )
        except Exception:
            return all_words

        n_boxes = len(tsv_data["text"])
        for i in range(n_boxes):
            word_text = tsv_data["text"][i].strip()
            if not word_text or len(word_text) < MIN_OCR_WORD_LEN:
                continue
            if self._looks_like_reference_code(word_text) or self._is_ocr_gibberish(word_text):
                continue

            conf = float(tsv_data["conf"][i])
            if conf < CONFIDENCE_MIN:
                continue

            x = tsv_data["left"][i]
            y = tsv_data["top"][i]
            w = tsv_data["width"][i]
            h = tsv_data["height"][i]

            if w < 3 or h < 3:
                continue

            # Font size estimate: bbox height converted to PDF points
            estimated_size = h / scale

            # Sample text color: take the center region of the word
            # bbox and average the darkest pixels (text strokes, not
            # background) within it.
            cx = x + w // 2
            cy = y + h // 2
            sample_w = max(1, w // 4)
            sample_h = max(1, h // 2)

            y1_s = max(0, cy - sample_h // 2)
            y2_s = min(img_array.shape[0], cy + sample_h // 2)
            x1_s = max(0, cx - sample_w // 2)
            x2_s = min(img_array.shape[1], cx + sample_w // 2)

            region = img_array[y1_s:y2_s, x1_s:x2_s]
            if region.size == 0:
                continue

            flat = region.reshape(-1, 3)
            brightness_vals = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]

            dark_threshold = np.percentile(brightness_vals, 20)
            dark_pixels = flat[brightness_vals <= dark_threshold]

            if len(dark_pixels) == 0:
                color_r, color_g, color_b = 0, 0, 0
            else:
                color_r = int(np.mean(dark_pixels[:, 0]))
                color_g = int(np.mean(dark_pixels[:, 1]))
                color_b = int(np.mean(dark_pixels[:, 2]))

            brightness = 0.299 * color_r + 0.587 * color_g + 0.114 * color_b

            pts_scale = 1 / scale
            x0_pt = x * pts_scale
            y0_pt = y * pts_scale
            x1_pt = (x + w) * pts_scale
            y1_pt = (y + h) * pts_scale

            line_id = (
                page_num,
                tsv_data["block_num"][i],
                tsv_data["par_num"][i],
                tsv_data["line_num"][i],
            )

            all_words.append({
                "text": word_text,
                "page": page_num,
                "bbox": (x0_pt, y0_pt, x1_pt, y1_pt),
                "size": estimated_size,
                "color_r": color_r,
                "color_g": color_g,
                "color_b": color_b,
                "brightness": brightness,
                "confidence": conf,
                "line_id": line_id,
            })

        return all_words

    def _analyze_with_ocr(self, doc) -> OCRReport:
        """
        For scanned/image PDFs: render each page at 300 DPI, run Tesseract
        with word-level output, sample pixel color at each word's bbox,
        and compute statistics.
        """
        all_words = []
        scale = RENDER_DPI / 72

        for page_num in range(len(doc)):
            page = doc[page_num]

            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
            img_array = np.array(img)

            all_words.extend(self._extract_words_from_image(img, img_array, page_num, scale))

        doc.close()

        if len(all_words) < MIN_WORDS:
            return OCRReport(
                pages_analyzed=0, pdf_type="scanned",
                word_count=len(all_words),
                avg_confidence=0, avg_font_size=0,
                avg_color_brightness=0,
                word_anomalies=[],
                anomaly_score=0,
                signals=["Insufficient words extracted for statistical analysis"],
            )

        return self._compute_anomalies(all_words, "scanned")

    def analyze_image(self, image_path: str) -> OCRReport:
        """
        Run OCR word-level analysis directly on an uploaded image (JPG/PNG)
        instead of routing it through convert_to_pdf() first. That
        conversion stretches the image into a fixed-size PDF page and
        re-rasterizes it at a different DPI, which can blur fine detail —
        analyzing the original pixels directly avoids that round-trip and
        gives Tesseract (and the size/color sampling) the native resolution.

        Bounding boxes in the returned report are in the image's native
        pixel space, not PDF points — there's no PDF page to map onto for a
        raw image upload.
        """
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        all_words = self._extract_words_from_image(img, img_array, page_num=0, scale=1.0)

        if len(all_words) < MIN_WORDS:
            return OCRReport(
                pages_analyzed=0, pdf_type="scanned",
                word_count=len(all_words),
                avg_confidence=0, avg_font_size=0,
                avg_color_brightness=0,
                word_anomalies=[],
                anomaly_score=0,
                signals=["Insufficient words extracted for statistical analysis"],
            )

        return self._compute_anomalies(all_words, "scanned")

    # ── Baseline clustering ──────────────────────────────────────────────────

    @staticmethod
    def _common_value_clusters(values: list, bucket: float, min_share: float) -> set:
        """
        Bucket `values` to the nearest `bucket` increment and return the set
        of bucketed values that each account for at least `min_share` of the
        population — the document's "common, repeated" sizes/brightnesses.
        A heading that recurs on every page (or a whole colored letterhead
        block) clusters into its own common bucket and is therefore NOT an
        outlier, even though it differs from the single global mean/std.
        """
        if not values:
            return set()
        buckets = [round(v / bucket) * bucket for v in values]
        counts = {}
        for b in buckets:
            counts[b] = counts.get(b, 0) + 1
        # A floor of 1 would make ANY single occurrence "common" on a small
        # document (round(N * min_share) rounds down to 1 whenever N is
        # small) — exactly defeating the point of clustering, since a lone
        # edited word would then trivially count as its own "common"
        # cluster. Require at least 2 occurrences before "repeated" applies.
        threshold = max(2, round(len(values) * min_share))
        return {b for b, c in counts.items() if c >= threshold}

    @staticmethod
    def _line_dominant_buckets(words: list, key: str, bucket: float) -> dict:
        """
        For each line_id, the bucketed value (size or brightness) shared by
        the MAJORITY of words on that line — e.g. an entire heading line at
        16pt has a dominant size bucket of 16, even though only 4 of the
        document's 131 words are that size. A heading/title line is
        internally uniform with its own neighbors, so it's not an outlier
        in its own context — only a word that disagrees with its own line's
        neighbors (the dominant bucket) is a real candidate for "edited."
        Lines with a single word are excluded: one word can't establish
        consistency with itself.
        """
        line_groups = {}
        for w in words:
            line_groups.setdefault(w.get("line_id"), []).append(w)

        dominant = {}
        for line_id, line_words in line_groups.items():
            if line_id is None or len(line_words) < 2:
                continue
            buckets = [round(w[key] / bucket) * bucket for w in line_words]
            counts = {}
            for b in buckets:
                counts[b] = counts.get(b, 0) + 1
            dominant[line_id] = max(counts, key=counts.get)
        return dominant

    @staticmethod
    def _check_baseline_alignment(words: list) -> dict:
        """
        For each line_id, flag a word whose bbox bottom-edge (its baseline)
        deviates from the line's median bottom-edge by more than
        BASELINE_DEVIATION_PT AND is a z-score outlier among its own line's
        neighbors. Returns {id(word_dict): (deviation_pt, z)} so the caller
        can fold this into the existing per-word anomaly loop without a
        second pass over `words`.

        Words containing a descender character (g, j, p, q, y) are excluded
        from BOTH the reference baseline and the flagged set: a tight OCR/
        glyph bbox naturally extends below the true typographic baseline
        for these letters, so "Pay" sitting a few points lower than "Net"
        on the same, untampered line is normal letterform shape, not an
        edit -- including them would make this check fire on ordinary text.

        Low-confidence words are excluded too: empirically, on a page with a
        textured/security-printed background (ID cards), Tesseract's bbox
        for a garbled low-confidence read is itself unreliable -- the
        "misalignment" is noise in the box-fitting, not a real position
        signal, the same root cause as the OCR confidence layer's own
        page-noise problem. A word OCR isn't confident about isn't a
        reliable reference point for "is this aligned" either.
        """
        descenders = set("gjpqy")
        line_groups = {}
        for w in words:
            if any(c in descenders for c in w.get("text", "")):
                continue
            if w.get("confidence", 100) < BASELINE_MIN_CONFIDENCE:
                continue
            line_groups.setdefault(w.get("line_id"), []).append(w)

        flagged = {}
        for line_id, line_words in line_groups.items():
            if line_id is None or len(line_words) < MIN_WORDS_PER_LINE_BASELINE:
                continue
            bottoms = [w["bbox"][3] for w in line_words]
            median_bottom = statistics.median(bottoms)
            if len(bottoms) < 2:
                continue
            std_bottom = statistics.stdev(bottoms)
            if std_bottom <= 0:
                continue
            for w, bottom in zip(line_words, bottoms):
                deviation = abs(bottom - median_bottom)
                z = deviation / std_bottom
                if deviation > BASELINE_DEVIATION_PT and z >= BASELINE_Z_THRESHOLD:
                    flagged[id(w)] = (deviation, z)
        return flagged

    @staticmethod
    def _trimmed_mean_std(values: list, trim_pct: float = 0.10) -> tuple:
        """
        Mean/std with the top/bottom trim_pct of values excluded first —
        threshold saturation guard. A single injected huge value (one 36pt
        word among hundreds of 11pt words) otherwise inflates `std` enough
        that a smaller, separate size/color anomaly elsewhere in the
        document no longer clears its z-score threshold. Falls back to the
        plain mean/std when there aren't enough values to trim meaningfully.
        """
        if len(values) < 4:
            mean = statistics.mean(values) if values else 0
            std = statistics.stdev(values) if len(values) > 1 else 1
            return mean, std
        sorted_vals = sorted(values)
        trim = max(1, int(len(sorted_vals) * trim_pct))
        trimmed = sorted_vals[trim:-trim]
        if len(trimmed) < 2:
            return statistics.mean(values), 1
        return statistics.mean(trimmed), statistics.stdev(trimmed)

    def _compute_anomalies(self, words: list, pdf_type: str) -> OCRReport:
        """Compute statistical baseline and flag outliers."""
        sizes      = [w["size"] for w in words if w["size"] > 4]
        brightness = [w["brightness"] for w in words]
        confs      = [w["confidence"] for w in words]

        if not sizes:
            return OCRReport(
                pages_analyzed=0, pdf_type=pdf_type,
                word_count=len(words), avg_confidence=0,
                avg_font_size=0, avg_color_brightness=0,
                word_anomalies=[], anomaly_score=0,
                signals=["No valid word sizes found"],
            )

        avg_size, std_size = self._trimmed_mean_std(sizes)
        avg_brightness, std_brightness = self._trimmed_mean_std(brightness)
        avg_conf       = statistics.mean(confs)

        # Per-page confidence baseline — see CONFIDENCE_DROP_THRESHOLD above.
        page_confs = {}
        for w in words:
            page_confs.setdefault(w["page"], []).append(w["confidence"])
        page_avg_conf = {p: statistics.mean(c) for p, c in page_confs.items()}

        # Per-page "already noisy" rate — see PAGE_NOISY_FLOOR_RATE below.
        # A page with a textured/security-printed background (ID cards,
        # certificates with seals/holograms/watermarks) makes Tesseract
        # misread background texture as low-confidence garbage "words"
        # across a large chunk of the page, with zero tampering involved.
        # CONFIDENCE_ABSOLUTE_FLOOR is meant to catch a word that reads
        # suspiciously low even on an otherwise-bad page — but if a quarter
        # of the page is already below that floor, it's not an outlier
        # anymore, it's just what this page's background looks like. The
        # relative drop-vs-page-average check still applies on these pages
        # and still catches a word reading worse than its noisy peers.
        page_below_floor_rate = {
            p: sum(1 for v in c if v < CONFIDENCE_ABSOLUTE_FLOOR) / len(c)
            for p, c in page_confs.items()
        }

        # Common-size/common-color clusters — see _common_value_clusters.
        # Multiple legitimate sizes/colors (body text, a heading, a footer)
        # coexist on real documents; only a word matching NONE of them is a
        # genuine candidate for a size/color anomaly.
        common_sizes = self._common_value_clusters(sizes, SIZE_BUCKET_PT, SIZE_CLUSTER_MIN_SHARE)
        common_colors = self._common_value_clusters(brightness, 1.0, COLOR_CLUSTER_MIN_SHARE)

        # Per-line dominant size/color — see _line_dominant_buckets. Catches
        # the case a global frequency cluster misses: a short heading/title
        # line that's too rare document-wide to clear SIZE_CLUSTER_MIN_SHARE
        # but is perfectly uniform with its own neighboring words.
        line_dominant_size = self._line_dominant_buckets(words, "size", SIZE_BUCKET_PT)
        line_dominant_color = self._line_dominant_buckets(words, "brightness", 1.0)
        # Restricted to native_text: vector-rendered glyph positions are
        # geometrically exact, so any real deviation is meaningful. On
        # scanned/OCR content, Tesseract's bbox-fitting has its own jitter
        # on dense or unusually laid-out text (e.g. small justified fine
        # print) independent of confidence — empirically still produced
        # false positives on real scanned content even after filtering
        # descenders and low-confidence words, so this check doesn't get
        # applied there rather than ship a known-unreliable signal.
        baseline_flags = self._check_baseline_alignment(words) if pdf_type == "native_text" else {}

        anomalies = []

        for w in words:
            anomaly_types = []
            size_z  = 0.0
            color_z = 0.0
            baseline_z = 0.0
            line_id = w.get("line_id")

            # SIZE anomaly — statistical outlier, not part of any common
            # document-wide size cluster, AND inconsistent with its own
            # line's neighbors (so a uniformly-styled heading line doesn't
            # fire just because it's rare relative to the whole document).
            if w["size"] > 4 and std_size > 0:
                size_z = abs(w["size"] - avg_size) / std_size
                size_bucket = round(w["size"] / SIZE_BUCKET_PT) * SIZE_BUCKET_PT
                matches_line = line_dominant_size.get(line_id) == size_bucket
                if (size_z >= SIZE_Z_THRESHOLD and size_bucket not in common_sizes
                        and not matches_line):
                    anomaly_types.append("size")

            # COLOR anomaly — same clustering + same-line consistency logic.
            if std_brightness > 0:
                color_z = abs(w["brightness"] - avg_brightness) / std_brightness
                color_bucket = round(w["brightness"])
                matches_line = line_dominant_color.get(line_id) == color_bucket
                if (color_z >= COLOR_Z_THRESHOLD and color_bucket not in common_colors
                        and not matches_line):
                    anomaly_types.append("color")

            # CONFIDENCE anomaly (OCR mode only — native text is always 100%).
            # Relative to THIS word's own page average, not a flat global
            # cutoff — an inherently low-quality scan reads low everywhere,
            # which isn't evidence of editing; a word reading meaningfully
            # worse than its own page's peers is.
            if pdf_type == "scanned":
                page_avg = page_avg_conf.get(w["page"], avg_conf)
                drop = page_avg - w["confidence"]
                page_is_noisy = page_below_floor_rate.get(w["page"], 0) >= PAGE_NOISY_FLOOR_RATE
                below_floor = w["confidence"] < CONFIDENCE_ABSOLUTE_FLOOR and not page_is_noisy
                if drop > CONFIDENCE_DROP_THRESHOLD or below_floor:
                    anomaly_types.append("confidence")

            # BASELINE anomaly — see _check_baseline_alignment. Flags a word
            # whose bottom edge doesn't sit on the same line as its
            # neighbors, independent of size/color/confidence.
            baseline_hit = baseline_flags.get(id(w))
            if baseline_hit is not None:
                baseline_z = baseline_hit[1]
                anomaly_types.append("baseline")

            if anomaly_types:
                reasons = []
                if "size" in anomaly_types:
                    reasons.append(
                        f"Font size {w['size']:.1f}pt vs avg {avg_size:.1f}pt "
                        f"(z={size_z:.1f})"
                    )
                if "color" in anomaly_types:
                    reasons.append(
                        f"Text brightness {w['brightness']:.0f} vs avg "
                        f"{avg_brightness:.0f} (z={color_z:.1f}) — "
                        f"color RGB({w['color_r']},{w['color_g']},{w['color_b']})"
                    )
                if "confidence" in anomaly_types:
                    reasons.append(
                        f"Low OCR confidence {w['confidence']:.0f}% — "
                        f"possible distortion from editing"
                    )
                if "baseline" in anomaly_types:
                    dev_pt = baseline_flags[id(w)][0]
                    reasons.append(
                        f"Baseline deviation {dev_pt:.1f}pt from line median "
                        f"(z={baseline_z:.1f}) — word not aligned with "
                        f"surrounding text on this line"
                    )

                anomalies.append(OCRWordAnomaly(
                    page=w["page"],
                    word=w["text"],
                    bbox=w["bbox"],
                    anomaly_types=anomaly_types,
                    size_z=size_z,
                    color_z=color_z,
                    baseline_z=baseline_z,
                    confidence=w["confidence"],
                    reason="; ".join(reasons),
                ))
        signals = []
        size_anomalies  = [a for a in anomalies if "size" in a.anomaly_types]
        color_anomalies = [a for a in anomalies if "color" in a.anomaly_types]
        conf_anomalies  = [a for a in anomalies if "confidence" in a.anomaly_types]
        baseline_anomalies = [a for a in anomalies if "baseline" in a.anomaly_types]

        # Score per anomaly TYPE, each independently capped, rather than one
        # unbounded per-word accumulator — confidence anomalies are by far
        # the noisiest signal (an inherently low-quality scan reads low
        # everywhere, not just where it was edited), so they're weighted
        # and capped much lower than size/color, which are rarer and far
        # more specific to an actual visual edit.
        score = (
            min(SIZE_SCORE_CAP, len(size_anomalies) * SIZE_SCORE_PER_WORD) +
            min(COLOR_SCORE_CAP, len(color_anomalies) * COLOR_SCORE_PER_WORD) +
            min(CONFIDENCE_SCORE_CAP, len(conf_anomalies) * CONFIDENCE_SCORE_PER_WORD) +
            min(BASELINE_SCORE_CAP, len(baseline_anomalies) * BASELINE_SCORE_PER_WORD)
        )

        if size_anomalies:
            signals.append(
                f"{len(size_anomalies)} word(s) with abnormal font size "
                f"vs document average ({avg_size:.1f}pt) — "
                f"possible text replacement with a different tool"
            )
        if color_anomalies:
            signals.append(
                f"{len(color_anomalies)} word(s) with abnormal text color "
                f"vs document average — possible edit with a different color setting"
            )
        if conf_anomalies:
            signals.append(
                f"{len(conf_anomalies)} word(s) with low OCR confidence "
                f"— possible distortion around edited regions"
            )
        if baseline_anomalies:
            signals.append(
                f"{len(baseline_anomalies)} word(s) with baseline misalignment "
                f"vs line median — text not aligned to the surrounding line's "
                f"grid, possible manual text insertion"
            )
        if not signals:
            unit = "words" if pdf_type == "scanned" else "text spans"
            signals.append(
                f"{len(words)} {unit} analyzed — "
                f"font size and color consistent throughout document"
            )

        return OCRReport(
            pages_analyzed=len(set(w["page"] for w in words)),
            pdf_type=pdf_type,
            word_count=len(words),
            avg_confidence=round(avg_conf, 1),
            avg_font_size=round(avg_size, 1),
            avg_color_brightness=round(avg_brightness, 1),
            word_anomalies=anomalies[:50],  # cap at 50
            anomaly_score=min(100, score),
            signals=signals,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UPGRADE 1 — UNIVERSAL OCR PIXEL PROFILING (Layer 3 second pass)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _profile_pixels_all_pages(self, pdf_path: str):
        """
        UPGRADE 1 — UNIVERSAL OCR PIXEL PROFILING (Layer 3 second pass)

        Render every page to pixels and profile word characteristics:
        - Font height per line (detect size anomalies)
        - Pixel brightness per word (detect color anomalies)
        - Digital black detector (pure RGB 0,0,0 on scanned docs)

        Opens its OWN fitz.Document from pdf_path rather than accepting one
        from the caller — analyze()'s doc is already closed by
        _analyze_native_text()/_analyze_with_ocr() before this runs.

        Returns dict with "anomalies" list and "score" (0-100). Any failure
        degrades gracefully to an empty result instead of raising, so a
        pixel-profiling problem never crashes the /analyze endpoint.
        """
        if not HAS_CV2:
            return {"anomalies": [], "score": 0}

        doc = None
        try:
            doc = fitz.open(pdf_path)

            is_id_card = self._is_id_card_document(doc)

            all_anomalies = []
            all_pixel_words = []
            scale = PIXEL_PROFILE_DPI / 72

            # Extract all words with pixel-level profiling
            for page_num in range(len(doc)):
                page = doc[page_num]
                qr_zones = get_qr_zones(page, doc)

                # Render page at lower DPI (144 is enough for pixel analysis)
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                # Copy the pixel data so it persists after `pix`/`doc` are released
                img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
                # PyMuPDF returns RGB, convert to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_RGB2BGR)
                page_height_pix = pix.height

                # OCR to get word-level bboxes
                try:
                    data = pytesseract.image_to_data(
                        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                        output_type=pytesseract.Output.DICT,
                        lang='eng',
                        config='--psm 6 --oem 3'
                    )
                except Exception:
                    continue

                page_words = []
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if not text or len(text) < 2:
                        continue

                    conf = float(data['conf'][i])
                    if conf < 30:  # Too low confidence
                        continue

                    left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                    if width < 5 or height < 5:  # Too small
                        continue

                    # Tesseract's line_num is only unique WITHIN a given
                    # (block_num, par_num) — it resets to 0 for every new
                    # paragraph, so two unrelated visual lines in different
                    # paragraphs can collide on line_num alone. Combine all
                    # three so per-line grouping below actually identifies
                    # one physical line, not an arbitrary cross-paragraph mix.
                    line_key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])

                    # Convert pixel coords to PDF points
                    x0_pt = left / scale
                    y0_pt = top / scale
                    x1_pt = (left + width) / scale
                    y1_pt = (top + height) / scale
                    bbox_pt = (x0_pt, y0_pt, x1_pt, y1_pt)

                    # QR codes are high-frequency B&W pixel regions that read
                    # as a tamper signal to every pixel-profiling check below
                    # (font height, color, digital-paste) — skip entirely.
                    if bbox_overlaps_qr_zone(bbox_pt, qr_zones):
                        continue

                    page_words.append({
                        'text': text,
                        'page': page_num,
                        'bbox_pt': bbox_pt,
                        'bbox_pix': (left, top, left + width, top + height),
                        'height_pix': height,
                        'line_num': line_key,
                        'conf': conf,
                        'img': img_bgr,  # Keep reference to image for later pixel extraction
                        'page_height_pix': page_height_pix,
                    })

                all_pixel_words.extend(page_words)

            if not all_pixel_words:
                return {"anomalies": [], "score": 0}

            # Step 1: Font HEIGHT profiling per line
            size_anomalies = self._profile_font_heights(all_pixel_words)
            all_anomalies.extend(size_anomalies)

            # Step 2: Pixel COLOR profiling
            color_anomalies = self._profile_pixel_colors(all_pixel_words, is_id_card=is_id_card)
            all_anomalies.extend(color_anomalies)

            # Step 3: Cross-word consistency check (phrase anomalies)
            all_anomalies = self._boost_phrase_anomalies(all_anomalies, all_pixel_words)

            # Step 4: Calculate score
            score = 0
            # ID cards legitimately produce 1-2 incidental pixel hits (a
            # banner edge, a watermark) — only count toward score once
            # enough anomalies accumulate to look like a real edit rather
            # than printing noise. Findings are still returned either way.
            if not (is_id_card and len(all_anomalies) < ID_CARD_MIN_ANOMALIES_FOR_SCORE):
                for anom in all_anomalies:
                    if anom['anomaly_type'] == 'size':
                        score += PIXEL_SIZE_ANOMALY_SCORE
                    elif anom['anomaly_type'] == 'color':
                        score += PIXEL_COLOR_ANOMALY_SCORE
                    elif anom['anomaly_type'] == 'digital_paste':
                        score += PIXEL_DIGITAL_PASTE_SCORE

                # Bonus for phrase anomalies (multiple consecutive words)
                phrase_count = sum(1 for a in all_anomalies if a.get('is_phrase_member'))
                score += (phrase_count // 2) * PIXEL_PHRASE_BONUS

            score = min(PIXEL_SCORE_CAP, int(score))

            return {
                "anomalies": all_anomalies[:50],  # Cap at 50
                "score": score,
            }
        except Exception as e:
            import logging
            logging.warning(f"Pixel profiling failed: {e}")
            return {"anomalies": [], "score": 0}
        finally:
            if doc is not None:
                doc.close()

    def _profile_font_heights(self, words):
        """Font HEIGHT profiling per line.

        Dominant-cluster detection is scoped per-page so compiled/merged
        documents (multiple source PDFs merged into one file) don't have
        their per-page dominant font sizes compete across pages.  A 12-page
        portfolio where each page has its own dominant size would see no
        single height clear the document-wide threshold, flagging everything.
        """
        anomalies = []

        if not words:
            return []

        # Group words by page — each page in a compiled doc is its own
        # independent layout with its own dominant font size(s).
        words_by_page = {}
        for w in words:
            words_by_page.setdefault(w['page'], []).append(w)

        for page_num, page_words in words_by_page.items():
            page_heights = [w['height_pix'] for w in page_words if w.get('height_pix', 0) > 0]
            if not page_heights:
                continue

            # Dominant height clusters FOR THIS PAGE only — see PIXEL_DOMINANT_*
            # constants.  A header/title that recurs on every line of this page is
            # a legitimate layout element, not an isolated edited word.
            rounded_heights = [
                round(h / PIXEL_DOMINANT_HEIGHT_BUCKET_PX) * PIXEL_DOMINANT_HEIGHT_BUCKET_PX
                for h in page_heights
            ]
            height_counts = Counter(rounded_heights)
            dominant_heights = set(h for h, _ in height_counts.most_common(PIXEL_DOMINANT_TOP_N))
            total_page_words = len(page_heights)
            dominant_heights.update(
                h for h, count in height_counts.items()
                if count / total_page_words >= PIXEL_DOMINANT_MIN_SHARE
            )

            # Group by line number within this page only
            lines = {}
            for w in page_words:
                lines.setdefault(w['line_num'], []).append(w)

            for line_key, line_words in lines.items():
                if len(line_words) < 3:  # Need at least 3 words per line
                    continue

                heights = [w['height_pix'] for w in line_words]
                median_h = statistics.median(heights)

                # Calculate MAD (Median Absolute Deviation)
                try:
                    mad = self._median_absolute_deviation(heights)
                except Exception:
                    mad = 0

                for w in line_words:
                    if len(w.get('text', '').strip()) < 3:
                        continue  # single chars and 2-char words
                                  # have unreliable bbox heights

                    if w.get('conf', 100) < 70:
                        continue  # low-confidence OCR word —
                                  # bbox measurement unreliable

                    # Height matches this page's dominant cluster — a
                    # legitimate, recurring layout element, not an isolated
                    # edited word. Skip before flagging.
                    word_height_rounded = round(w['height_pix'] / PIXEL_DOMINANT_HEIGHT_BUCKET_PX) * PIXEL_DOMINANT_HEIGHT_BUCKET_PX
                    if word_height_rounded in dominant_heights:
                        continue

                    deviation = abs(w['height_pix'] - median_h)

                    # Check if height is anomalous — must be both absolutely
                    # and proportionally significant to avoid Tesseract noise.
                    if (deviation > PIXEL_SIZE_DEVIATION_MIN
                            and deviation > median_h * PIXEL_SIZE_DEVIATION_RATIO):
                        anomalies.append({
                            'page': w['page'],
                            'word': w['text'],
                            'bbox': w['bbox_pt'],
                            'anomaly_type': 'size',
                            'severity': 'high' if deviation > median_h * 0.3 else 'medium',
                            'confidence': 75,
                            'reason': f"Font height {w['height_pix']}px vs line median {median_h:.0f}px ({deviation/median_h*100:.0f}% deviation)",
                        })

        return anomalies

    def _profile_pixel_colors(self, words, is_id_card: bool = False):
        """Pixel COLOR profiling using MAD z-score."""
        anomalies = []
        mad_threshold   = PIXEL_COLOR_MAD_THRESHOLD_ID_CARD if is_id_card else PIXEL_COLOR_MAD_THRESHOLD
        black_threshold = PIXEL_DIGITAL_BLACK_THRESHOLD_ID_CARD if is_id_card else PIXEL_DIGITAL_BLACK_THRESHOLD

        # Group by page for page-level statistics
        by_page = {}
        for w in words:
            if w['page'] not in by_page:
                by_page[w['page']] = []
            by_page[w['page']].append(w)

        for page_num, page_words in by_page.items():
            brightnesses = []

            # Extract pixel brightness for each word
            for w in page_words:
                left, top, right, bottom = w['bbox_pix']
                if top >= w['img'].shape[0] or left >= w['img'].shape[1]:
                    continue

                # ID cards print a colored header/banner band by template
                # design (Aadhaar logo strip, etc.) — skip color profiling
                # for words there entirely rather than threshold-tune it.
                if is_id_card and w.get('page_height_pix') and top < w['page_height_pix'] * ID_CARD_HEADER_ZONE_FRAC:
                    continue

                # Crop word region
                crop = w['img'][max(0, top):min(w['img'].shape[0], bottom),
                                max(0, left):min(w['img'].shape[1], right)]
                
                if crop.size == 0:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                # Adaptive thresholding to separate text from background
                binary = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Get text pixels
                text_pixels = gray[binary > 0]
                
                if len(text_pixels) < PIXEL_MIN_TEXT_PIXELS_IN_BBOX:
                    continue  # Too few pixels
                
                word_brightness = float(np.mean(text_pixels))
                word_brightness_std = float(np.std(text_pixels))
                
                w['brightness'] = word_brightness
                w['brightness_std'] = word_brightness_std
                brightnesses.append((word_brightness, w))
            
            if len(brightnesses) < 3:
                continue
            
            # Page-level statistics using MAD (robust to outliers)
            brightness_values = [b[0] for b in brightnesses]
            page_median_brightness = statistics.median(brightness_values)
            
            try:
                page_mad = self._median_absolute_deviation(brightness_values)
            except:
                page_mad = 1
            
            # Check for digital paste (pure black on scanned document)
            if page_median_brightness > PIXEL_SCANNED_BRIGHTNESS_MIN:
                # This is a scanned/printed document
                for brightness, w in brightnesses:
                    if brightness < black_threshold:
                        anomalies.append({
                            'page': w['page'],
                            'word': w['text'],
                            'bbox': w['bbox_pt'],
                            'anomaly_type': 'digital_paste',
                            'severity': 'high',
                            'confidence': 85,
                            'reason': f"Pure digital black (brightness {brightness:.0f}) on scanned document (median {page_median_brightness:.0f})",
                        })
                        continue

            # Dominant brightness clusters on this page — a header/title
            # rendered in a different (but consistent, recurring) ink
            # shade is a legitimate layout element, not an edited word.
            # Same exemption as _profile_font_heights, scoped per-page to
            # match this check's existing per-page statistics.
            rounded_brightness = [round(b) for b in brightness_values]
            brightness_counts = Counter(rounded_brightness)
            dominant_brightness = set(b for b, _ in brightness_counts.most_common(PIXEL_DOMINANT_TOP_N))
            total_words_page = len(brightness_values)
            dominant_brightness.update(
                b for b, count in brightness_counts.items()
                if count / total_words_page >= PIXEL_DOMINANT_MIN_SHARE
            )

            # Per-line brightness consistency — catches the case the
            # page-wide cluster check above misses: a short header/title
            # line (e.g. a 4-word company name) is too rare document-wide
            # to clear PIXEL_DOMINANT_MIN_SHARE on a page with 100+ body
            # words, but is internally consistent with its own neighboring
            # words. Mirrors _line_dominant_buckets' logic for the
            # native-text path, using a tolerance rather than exact-bucket
            # equality since a short word ("LTD") legitimately averages a
            # slightly different brightness than its longer line-mates.
            line_groups = {}
            for brightness, w in brightnesses:
                line_groups.setdefault(w.get('line_num'), []).append(brightness)
            line_median_brightness = {
                line_id: statistics.median(vals)
                for line_id, vals in line_groups.items()
                if line_id is not None and len(vals) >= 2
            }

            # MAD-based z-score for color anomalies
            for brightness, w in brightnesses:
                rounded = round(brightness)
                if rounded in dominant_brightness:
                    continue
                line_median = line_median_brightness.get(w.get('line_num'))
                if line_median is not None and abs(brightness - line_median) <= PIXEL_LINE_BRIGHTNESS_TOLERANCE:
                    continue

                if page_mad > 0:
                    mad_z = abs(brightness - page_median_brightness) / (page_mad * 1.4826 + 0.01)
                else:
                    mad_z = 0

                if mad_z > mad_threshold:
                    anomalies.append({
                        'page': w['page'],
                        'word': w['text'],
                        'bbox': w['bbox_pt'],
                        'anomaly_type': 'color',
                        'severity': 'high' if mad_z > 5 else 'medium',
                        'confidence': min(95, 70 + mad_z * 5),
                        'reason': f"Pixel brightness {brightness:.0f} vs page median {page_median_brightness:.0f} (MAD z-score {mad_z:.1f})",
                    })
        
        return anomalies

    def _boost_phrase_anomalies(self, anomalies, words):
        """Cross-word consistency: boost confidence for adjacent flagged words."""
        # Sort anomalies by page and position
        anomalies_by_page = {}
        for anom in anomalies:
            page = anom['page']
            if page not in anomalies_by_page:
                anomalies_by_page[page] = []
            anomalies_by_page[page].append(anom)
        
        # For each page, find adjacent anomalies
        for page, anoms in anomalies_by_page.items():
            page_words = [w for w in words if w['page'] == page]
            
            for i, anom1 in enumerate(anoms):
                # Find this word in page_words
                for j, w1 in enumerate(page_words):
                    if w1['text'] == anom1['word'] and w1['bbox_pt'] == anom1['bbox']:
                        # Check next words on same line
                        for k in range(j + 1, min(j + 3, len(page_words))):
                            w2 = page_words[k]
                            if w2['line_num'] != w1['line_num']:
                                break
                            # Check if w2 is also anomalous
                            for anom2 in anoms:
                                if anom2['word'] == w2['text'] and anom2['bbox'] == w2['bbox_pt']:
                                    # Mark as part of phrase
                                    anom1['is_phrase_member'] = True
                                    anom2['is_phrase_member'] = True
                                    anom1['confidence'] = min(100, anom1['confidence'] + 10)
                                    anom2['confidence'] = min(100, anom2['confidence'] + 10)
                                    break
        
        return anomalies

    @staticmethod
    def _median_absolute_deviation(values):
        """Calculate Median Absolute Deviation."""
        if not values:
            return 0
        median = statistics.median(values)
        deviations = [abs(v - median) for v in values]
        return statistics.median(deviations)
