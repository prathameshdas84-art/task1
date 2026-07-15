"""Line anomaly scoring: z-score accumulation per feature, line-gap
density, ID-card/form-field exclusions, and the glyph registry."""

import re
import statistics
from collections import Counter

import fitz

from .constants import *
from .models import LineProfile, SuspiciousLine


class LineScoringMixin:
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

