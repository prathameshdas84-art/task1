"""Number extraction: pdfplumber word walk, ID-number exclusion,
clean-number parsing, and line grouping."""

import re

import pdfplumber

from .models import NUMBER_RE, _parse_number


class NumberExtractionMixin:
    def _extract_lines_with_numbers(self, pdf_path: str) -> list[dict]:
        """
        Extract all lines that contain numbers with position info.

        Numbers are extracted per-WORD (not by re-scanning the joined line
        text) so each number keeps its own x-position. A bank-statement row
        like "01/01/26 REF123 1,000.00 500.00 45,230.00" has 3+ numeric
        columns at very different x-positions — collapsing them to the
        line's overall x-center would treat unrelated columns (small
        transaction amounts vs. a much larger running balance) as the same
        statistical group.
        """
        result = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    words = page.extract_words(
                        extra_attrs=["fontname", "size"],
                        keep_blank_chars=False
                    )
                    if not words:
                        continue

                    # Group into lines
                    lines = self._group_into_lines(words)

                    for line_num, line_words in enumerate(lines):
                        if not line_words:
                            continue
                        text = " ".join(w["text"] for w in line_words)

                        x0 = min(w["x0"]     for w in line_words)
                        y0 = min(w["top"]    for w in line_words)
                        x1 = max(w["x1"]     for w in line_words)
                        y1 = max(w["bottom"] for w in line_words)
                        line_bbox = (x0, y0, x1, y1)

                        # Find numbers per-word so each keeps its own x-position
                        # for column clustering — but use the full LINE bbox for
                        # highlighting so the drawn box is wide enough to see,
                        # not just a tiny box around the single number.
                        number_entries = []
                        for w in line_words:
                            for val in self._extract_clean_numbers(w["text"]):
                                if val <= 0:
                                    continue
                                if self._is_id_number(val, text):
                                    continue
                                number_entries.append({
                                    "value":     val,
                                    "x_center":  (w["x0"] + w["x1"]) / 2,
                                    "word_bbox": (w["x0"], w["top"], w["x1"], w["bottom"]),
                                    "bbox":      line_bbox,  # full line bbox for visibility
                                })

                        if not number_entries:
                            continue

                        result.append({
                            "page":            page_num,
                            "line_num":        line_num,
                            "text":            text,
                            "bbox":            line_bbox,
                            "numbers":         [e["value"] for e in number_entries],
                            "number_entries":  number_entries,
                            "x_center":        (x0 + x1) / 2,
                            "y_pos":           y0,
                        })
        except Exception:
            pass
        return result

    def _is_id_number(self, value: float, text: str) -> bool:
        """
        Returns True if this number is likely an ID/reference,
        not a value to compare statistically.
        ID signals:
        - More than 8 digits (account numbers, reference IDs)
        - Line contains keywords like account, ref, no., id, ifsc, upi
        - Number has no decimal point AND is very large (>8 digits)
        """
        text_lower = text.lower()

        # Long, specific keywords — safe to match as plain substrings
        substring_keywords = [
            "account", "upi", "ifsc", "aadhaar", "gstin",
            "mobile", "phone", "pincode", "tran", "number", "no.",
        ] + self.ID_CONTEXT_KEYWORDS_EXTRA
        if any(kw in text_lower for kw in substring_keywords):
            return True

        # Short keywords collide with ordinary words when matched as bare
        # substrings — "id" matches "Paid"/"Provided"/"Considered", "pan"
        # matches "company"/"Japan", "cin" matches "medicine"/"vaccine".
        # Require word boundaries so they only match the actual abbreviation.
        word_boundary_keywords = [r"\bid\b", r"\bref\b", r"\bpan\b", r"\bcin\b", r"\bzip\b"]
        if any(re.search(kw, text_lower) for kw in word_boundary_keywords):
            return True

        # Large integer with no decimal = likely ID
        is_integer = value == int(value)
        str_val = str(int(value)) if is_integer else ""
        if len(str_val) > self.ID_NUMBER_MIN_DIGITS:
            return True

        # PIN code: bare 6-digit integer on an address-shaped line
        if is_integer and len(str_val) == self.PIN_CODE_DIGITS:
            if any(kw in text_lower for kw in self.ADDRESS_KEYWORDS):
                return True
            if any(state in text_lower for state in self.INDIAN_STATE_KEYWORDS):
                return True
            has_currency_marker = any(m in text_lower for m in self.CURRENCY_MARKERS)
            if not has_currency_marker:
                caps_words = [w for w in re.findall(r"[A-Za-z]+", text) if w.isupper() and len(w) >= 4]
                if len(caps_words) >= 2:
                    return True  # "<CITY> <STATE> <PIN>" shaped line
                # "<...address...> - 400069" / "<...address...>, 400069" —
                # a PIN code is conventionally the last token of an address
                # line, set off by a dash or comma, regardless of letter case.
                if re.search(r"[-,]\s*" + str_val + r"\s*$", text):
                    return True

        # Year: bare 4-digit integer in a plausible year range, only when
        # the line also carries a date/year cue (otherwise a genuine small
        # amount like "2000" would be wrongly dropped from analysis).
        if is_integer and len(str_val) == 4 and self.YEAR_MIN <= value <= self.YEAR_MAX:
            if any(kw in text_lower for kw in self.YEAR_CONTEXT_KEYWORDS):
                return True
            if re.search(r"[/\-]\s*" + str_val + r"\b|\b" + str_val + r"\s*[/\-]", text):
                return True  # part of a d/m/yyyy or yyyy-mm style date

        return False

    def _extract_clean_numbers(self, word_text: str) -> list[float]:
        """
        Returns numbers in word_text that represent standalone values,
        rejecting digit runs that are fragments embedded inside a longer
        alphanumeric/reference token — e.g. "WAYKAR-SAHILWAYKAR7-1@" (a UPI
        handle), "EPR2605001539" (a narration code), "GeneratedBy:301085849"
        (a label glued to its value), "18002600/18001600" (two phone
        numbers joined by a slash), or "05/01/26" (a slash-separated date,
        where naive matching would also yield a spurious "01" -> 1.0).
        Without this, NUMBER_RE happily matches any digit run regardless of
        what surrounds it, so a single PDF "word" containing letters,
        digits, and symbols spills unrelated tiny numbers into the
        statistical groups.

        A short (<=3 char) prefix/suffix is tolerated only if it contains
        no other digits and no letters before the match (currency symbols,
        a colon, or a short "Cr"/"Dr" suffix are fine) — any digit nearby
        means this is one fragment of a multi-segment code or date, not a
        standalone number.
        """
        values = []
        for m in NUMBER_RE.finditer(word_text):
            before = word_text[:m.start()]
            after  = word_text[m.end():]
            if len(before) > self.FRAGMENT_CONTEXT_MAX_LEN or len(after) > self.FRAGMENT_CONTEXT_MAX_LEN:
                continue
            if any(c.isalnum() for c in before):
                continue
            if any(c.isdigit() for c in after):
                continue
            val = _parse_number(m.group())
            if val is not None:
                values.append(val)
        return values

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

    # ── Grouping strategy ──────────────────────────────────────────────────────

