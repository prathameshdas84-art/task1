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
from dataclasses import dataclass, field

import fitz
import numpy as np
import pytesseract
from PIL import Image


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
    anomaly_types: list       # "size", "color", "confidence"
    size_z: float = 0.0
    color_z: float = 0.0
    confidence: float = 0.0
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


class OCRAnalyzer:

    def analyze(self, pdf_path: str) -> OCRReport:
        doc = fitz.open(pdf_path)

        pdf_type = self._detect_pdf_type(doc, pdf_path)

        # For native text PDFs with a real text layer, use PyMuPDF's span
        # data (faster + exact — no OCR uncertainty on embedded text). For
        # image/scanned/vector-outlined PDFs, fall back to Tesseract OCR.
        if pdf_type == "native_text":
            return self._analyze_native_text(doc)
        else:
            return self._analyze_with_ocr(doc)

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

        avg_size       = statistics.mean(sizes)
        std_size       = statistics.stdev(sizes) if len(sizes) > 1 else 1
        avg_brightness = statistics.mean(brightness)
        std_brightness = statistics.stdev(brightness) if len(brightness) > 1 else 1
        avg_conf       = statistics.mean(confs)

        # Per-page confidence baseline — see CONFIDENCE_DROP_THRESHOLD above.
        page_confs = {}
        for w in words:
            page_confs.setdefault(w["page"], []).append(w["confidence"])
        page_avg_conf = {p: statistics.mean(c) for p, c in page_confs.items()}

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

        anomalies = []

        for w in words:
            anomaly_types = []
            size_z  = 0.0
            color_z = 0.0
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
                if drop > CONFIDENCE_DROP_THRESHOLD or w["confidence"] < CONFIDENCE_ABSOLUTE_FLOOR:
                    anomaly_types.append("confidence")

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

                anomalies.append(OCRWordAnomaly(
                    page=w["page"],
                    word=w["text"],
                    bbox=w["bbox"],
                    anomaly_types=anomaly_types,
                    size_z=size_z,
                    color_z=color_z,
                    confidence=w["confidence"],
                    reason="; ".join(reasons),
                ))
        signals = []
        size_anomalies  = [a for a in anomalies if "size" in a.anomaly_types]
        color_anomalies = [a for a in anomalies if "color" in a.anomaly_types]
        conf_anomalies  = [a for a in anomalies if "confidence" in a.anomaly_types]

        # Score per anomaly TYPE, each independently capped, rather than one
        # unbounded per-word accumulator — confidence anomalies are by far
        # the noisiest signal (an inherently low-quality scan reads low
        # everywhere, not just where it was edited), so they're weighted
        # and capped much lower than size/color, which are rarer and far
        # more specific to an actual visual edit.
        score = (
            min(SIZE_SCORE_CAP, len(size_anomalies) * SIZE_SCORE_PER_WORD) +
            min(COLOR_SCORE_CAP, len(color_anomalies) * COLOR_SCORE_PER_WORD) +
            min(CONFIDENCE_SCORE_CAP, len(conf_anomalies) * CONFIDENCE_SCORE_PER_WORD)
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
