"""Per-line feature math (char/word spacing, visual density), the
document-wide statistical profile, and structural-line / font-family
heuristics."""

import re
import statistics
from collections import Counter

import cv2
import numpy as np

from .constants import *
from .models import LineProfile


class LineFeaturesMixin:
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

