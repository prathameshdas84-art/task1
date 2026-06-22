"""
Numeric Consistency Analyzer — Layer 4
Extracts all numbers from document, groups them by context,
flags statistical outliers using z-score analysis.
Works on any document type — universal approach.
No training data. No ML. Pure statistics.
"""

import re
import statistics
from dataclasses import dataclass, field
from collections import defaultdict

import pdfplumber


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class NumericAnomaly:
    page: int
    line_num: int
    text: str               # full line text
    bbox: tuple             # (x0, y0, x1, y1) PDF points
    value: float            # the anomalous number
    group_mean: float       # mean of the group this number belongs to
    group_std: float        # std of the group
    z_score: float          # how many std deviations from mean
    context: str            # what kind of number group (column/label)
    reason: str


@dataclass
class NumericReport:
    anomalies: list[NumericAnomaly]
    groups_analyzed: int
    total_numbers: int
    anomaly_score: int      # 0-100
    signals: list[str]


# ── Number extractor ───────────────────────────────────────────────────────────

# Regex to find numbers including decimals, commas
NUMBER_RE = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b')


def _parse_number(text: str) -> float:
    """Parse number string to float, handling commas."""
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


class NumericAnalyzer:

    # Only flag extreme outliers
    Z_SCORE_THRESHOLD = 3.0

    # Minimum group size to analyze (need at least 4 numbers to compute stats)
    MIN_GROUP_SIZE = 4

    # Minimum std to avoid division issues
    MIN_STD = 0.01

    # Magnitude-ratio fallback (for groups too small for z-score): flag a
    # value that is this many times larger than every other value in its
    # group. Requires at least this many items — a ratio against a single
    # other value isn't statistically meaningful and false-positives on
    # coincidental same-x-position pairs from unrelated one-off lines.
    MAGNITUDE_RATIO_THRESHOLD  = 10.0
    MIN_MAGNITUDE_GROUP_SIZE   = 3

    # Column-clustering tolerance, in PDF points: numbers whose x-centers
    # fall within this distance are treated as the same table column.
    CLUSTER_X_TOLERANCE = 40

    # A number with no decimal point and more than this many digits is
    # treated as an ID/reference number (account number, transaction ID),
    # not a value to compare statistically.
    ID_NUMBER_MIN_DIGITS = 8

    # When checking whether a regex match is a standalone number vs. a
    # fragment embedded in a longer alphanumeric token, this is the max
    # length of surrounding prefix/suffix text tolerated before treating
    # the match as part of a larger token (see _extract_clean_numbers).
    FRAGMENT_CONTEXT_MAX_LEN = 3

    # _build_signals() z-score severity tiers — EXTREME_Z_SCORE is an
    # arbitrary "very confident" cutoff well above Z_SCORE_THRESHOLD, used
    # only to weight the signal text/score, not to decide whether to flag.
    EXTREME_Z_SCORE_THRESHOLD = 5.0

    def analyze(self, pdf_path: str) -> NumericReport:
        lines = self._extract_lines_with_numbers(pdf_path)

        if not lines:
            return NumericReport(
                anomalies=[],
                groups_analyzed=0,
                total_numbers=0,
                anomaly_score=0,
                signals=["No numeric data found in document"],
            )

        # Group numbers by context
        groups = self._group_numbers(lines)

        # Find outliers in each group
        all_anomalies = []
        for context, items in groups.items():
            if len(items) < 2:
                continue
            anomalies = self._find_outliers(items, context)
            all_anomalies.extend(anomalies)

        # Sort by z-score descending
        all_anomalies.sort(key=lambda x: x.z_score, reverse=True)

        total_numbers = sum(len(items) for items in groups.values())
        signals, score = self._build_signals(all_anomalies, len(groups), total_numbers)

        return NumericReport(
            anomalies=all_anomalies,
            groups_analyzed=len(groups),
            total_numbers=total_numbers,
            anomaly_score=score,
            signals=signals,
        )

    # ── Line extraction ────────────────────────────────────────────────────────

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
        ]
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
        str_val = str(int(value)) if value == int(value) else ""
        if len(str_val) > self.ID_NUMBER_MIN_DIGITS:
            return True
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

    def _group_numbers(self, lines: list[dict]) -> dict:
        """
        Group numbers by context so we only compare similar numbers.

        Strategy 1 — Column grouping:
        Numbers at similar x-positions across multiple lines = same column.
        Bank statement balance column, invoice amount column, etc.

        Strategy 2 — Label grouping:
        Numbers following the same label keyword = same field type.
        "Salary: X" across months, "Marks: X" per subject, etc.
        """
        groups = defaultdict(list)

        # Strategy 1: Column grouping — cluster each NUMBER's own x-position
        # (not the line's overall x-center, which spans every column in the
        # row and would merge unrelated columns like amounts and balances).
        page_entries = defaultdict(list)
        for line in lines:
            for entry in line["number_entries"]:
                page_entries[line["page"]].append({
                    "value":    entry["value"],
                    "page":     line["page"],
                    "line_num": line["line_num"],
                    "text":     line["text"],
                    "bbox":     entry["bbox"],
                    "x_center": entry["x_center"],
                })

        for page_num, entries in page_entries.items():
            if len(entries) < self.MIN_GROUP_SIZE:
                continue

            # Single-pass clustering — each number belongs to exactly one
            # column, eliminating the overlap/double-counting that a
            # separate "match any cluster within tolerance" pass allows.
            clusters = self._cluster_entries(entries, tolerance=self.CLUSTER_X_TOLERANCE)

            for cluster_id, cluster_items in enumerate(clusters):
                col_key = f"page{page_num}_col{cluster_id}"
                for item in cluster_items:
                    groups[col_key].append({
                        "value":    item["value"],
                        "page":     item["page"],
                        "line_num": item["line_num"],
                        "text":     item["text"],
                        "bbox":     item["bbox"],
                        "context":  f"column group (page {page_num+1})",
                    })

        # Strategy 2: Label-based grouping across pages.
        # English-language label keywords only — a non-English payslip or
        # statement (e.g. labels in Hindi/regional scripts) won't match any
        # of these patterns and will fall back to column-grouping alone.
        label_patterns = [
            (r'balance\s*:?\s*[\d,\.]+',   "balance"),
            (r'salary\s*:?\s*[\d,\.]+',    "salary"),
            (r'amount\s*:?\s*[\d,\.]+',    "amount"),
            (r'total\s*:?\s*[\d,\.]+',     "total"),
            (r'marks\s*:?\s*[\d,\.]+',     "marks"),
            (r'income\s*:?\s*[\d,\.]+',    "income"),
            (r'debit\s*:?\s*[\d,\.]+',     "debit"),
            (r'credit\s*:?\s*[\d,\.]+',    "credit"),
            (r'arrears\s*:?\s*[\d,\.]+',    "arrears"),
            (r'deduction\s*:?\s*[\d,\.]+',  "deduction"),
            (r'net\s*pay\s*:?\s*[\d,\.]+',  "net_pay"),
            (r'performance\s*:?\s*[\d,\.]+', "performance"),
            (r'bonus\s*:?\s*[\d,\.]+',      "bonus"),
        ]

        for line in lines:
            text_lower = line["text"].lower()
            for pattern, label in label_patterns:
                if re.search(pattern, text_lower):
                    for num in line["numbers"]:
                        if num > 0:
                            groups[f"label_{label}"].append({
                                "value":    num,
                                "page":     line["page"],
                                "line_num": line["line_num"],
                                "text":     line["text"],
                                "bbox":     line["bbox"],
                                "context":  f"{label} field",
                            })

        return dict(groups)

    def _cluster_entries(self, entries: list[dict], tolerance: float) -> list[list[dict]]:
        """
        Cluster entries by x_center into mutually exclusive column groups.

        Single pass over x-sorted entries: each entry is compared only to
        the x-position that started its current cluster and is assigned to
        exactly one cluster. This avoids the double-counting bug of
        matching every entry against every cluster independently within
        `tolerance`, which can put the same entry in two adjacent clusters
        whenever their seed positions are more than `tolerance` but less
        than `2 * tolerance` apart.
        """
        if not entries:
            return []
        sorted_entries = sorted(entries, key=lambda e: e["x_center"])
        clusters = [[sorted_entries[0]]]
        cluster_start_x = sorted_entries[0]["x_center"]
        for entry in sorted_entries[1:]:
            if entry["x_center"] - cluster_start_x > tolerance:
                clusters.append([])
                cluster_start_x = entry["x_center"]
            clusters[-1].append(entry)
        return clusters

    # ── Outlier detection ──────────────────────────────────────────────────────

    def _find_outliers(self, items: list[dict], context: str) -> list[NumericAnomaly]:
        """
        Find statistically anomalous values in a group using a leave-one-out
        (jackknife) z-score: each candidate value is compared against the
        mean/std of the OTHER values in the group, excluding itself.

        Computing mean/std from a sample that includes the candidate value
        caps the maximum reachable z-score at sqrt(n-1) (a single point can
        never be more than sqrt(n-1) standard deviations from a mean/std
        that it itself inflated). For MIN_GROUP_SIZE=4 that bound is
        sqrt(3) ≈ 1.73 — always below Z_SCORE_THRESHOLD=3.0, so the
        in-sample formula could never flag an outlier in a small group no
        matter how extreme the tampered value was. Excluding the candidate
        from its own baseline removes this self-masking effect.
        """
        anomalies = []
        seen = set()  # avoid duplicate anomalies for same line

        if len(items) >= self.MIN_GROUP_SIZE:
            for i, item in enumerate(items):
                others = [items[j]["value"] for j in range(len(items)) if j != i]
                if len(others) < 2:
                    continue
                mean = statistics.mean(others)
                std  = max(statistics.stdev(others), self.MIN_STD)

                val      = item["value"]
                z        = abs(val - mean) / std
                line_key = (item["page"], item["line_num"])

                if z >= self.Z_SCORE_THRESHOLD and line_key not in seen:
                    seen.add(line_key)
                    reason = (
                        f"Value {val:,.2f} is {z:.1f} standard deviations "
                        f"from group mean {mean:,.2f} (excluding this value) "
                        f"(std={std:,.2f}) in {item['context']}"
                    )
                    anomalies.append(NumericAnomaly(
                        page=item["page"],
                        line_num=item["line_num"],
                        text=item["text"][:80],
                        bbox=item["bbox"],
                        value=val,
                        group_mean=round(mean, 2),
                        group_std=round(std, 2),
                        z_score=round(z, 2),
                        context=item["context"],
                        reason=reason,
                    ))

        # Magnitude ratio fallback for small groups
        # If group too small for reliable z-score,
        # flag values that are >10x larger than all others
        # Require >=3 items — a ratio against a single other value is
        # not meaningful (coincidental same-x-position pairs from
        # unrelated one-off lines would otherwise false-positive).
        if len(items) >= self.MIN_MAGNITUDE_GROUP_SIZE:
            for i, item in enumerate(items):
                others = [items[j]["value"] for j in range(len(items)) if j != i]
                if not others:
                    continue
                max_other = max(others)
                if max_other > 0:
                    ratio = item["value"] / max_other
                    line_key = (item["page"], item["line_num"])
                    if ratio >= self.MAGNITUDE_RATIO_THRESHOLD and line_key not in seen:
                        seen.add(line_key)
                        reason = (
                            f"Value {item['value']:,.2f} is {ratio:.0f}x larger "
                            f"than other values in same group "
                            f"(max other: {max_other:,.2f}) — "
                            f"possible digit insertion or decimal shift"
                        )
                        anomalies.append(NumericAnomaly(
                            page=item["page"],
                            line_num=item["line_num"],
                            text=item["text"][:80],
                            bbox=item["bbox"],
                            value=item["value"],
                            group_mean=max_other,
                            group_std=0.0,
                            z_score=round(ratio, 2),
                            context=item["context"] + " (magnitude ratio)",
                            reason=reason,
                        ))

        return anomalies

    # ── Signals ────────────────────────────────────────────────────────────────

    def _build_signals(
        self,
        anomalies: list,
        groups: int,
        total: int,
    ) -> tuple[list[str], int]:
        signals = []
        score   = 0

        if not anomalies:
            signals.append(
                f"Numeric consistency check passed — "
                f"{total} numbers across {groups} groups, no outliers detected"
            )
            return signals, 0

        # High z-score anomalies
        extreme = [a for a in anomalies if a.z_score >= self.EXTREME_Z_SCORE_THRESHOLD]
        high    = [a for a in anomalies if self.Z_SCORE_THRESHOLD <= a.z_score < self.EXTREME_Z_SCORE_THRESHOLD]

        if extreme:
            signals.append(
                f"{len(extreme)} number(s) with extreme statistical anomaly "
                f"(z-score ≥ 5.0) — value is highly inconsistent with "
                f"surrounding numbers in same column/context"
            )
            score += min(60, len(extreme) * 20)

        if high:
            signals.append(
                f"{len(high)} number(s) with significant statistical anomaly "
                f"(z-score 3.0-5.0) — value deviates from group pattern"
            )
            score += min(30, len(high) * 10)

        return signals, min(100, score)
