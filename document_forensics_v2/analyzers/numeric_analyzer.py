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

    # PIN codes are 6 bare digits with no currency framing — "MUMBAI
    # MAHARASHTRA 400012" parses 400012 as a plain number with nothing in
    # the existing ID keyword list to catch it (no "account"/"ifsc"/etc.
    # nearby), so it lands in a column group next to real currency values
    # and produces an absurd z-score. We only exclude a 6-digit bare
    # integer when its line also looks like an address line: it contains
    # an explicit postal keyword, OR a known state/UT name (Indian postal
    # addresses always end "<City> <State> <PIN>"), OR is structurally
    # address-shaped (mostly ALL-CAPS words, no currency marker). A
    # hardcoded state list can't cover every locale, hence the structural
    # fallback.
    PIN_CODE_DIGITS = 6
    ADDRESS_KEYWORDS = [
        "pin", "pincode", "pin code", "postal", "address", "district",
        "village", "taluka", "tehsil",
    ]
    INDIAN_STATE_KEYWORDS = [
        "maharashtra", "karnataka", "tamil nadu", "west bengal", "telangana",
        "gujarat", "rajasthan", "punjab", "haryana", "kerala", "odisha",
        "bihar", "jharkhand", "assam", "uttar pradesh", "madhya pradesh",
        "chhattisgarh", "uttarakhand", "himachal pradesh", "goa",
        "andhra pradesh", "delhi", "chandigarh", "puducherry",
    ]
    CURRENCY_MARKERS = ["rs.", "rs ", "inr", "₹", "$", "amount", "total",
                         "balance", "salary", "payable", "due", "fee"]

    # Years (publication dates, "FY 2025-26", DOB) are bare 4-digit numbers
    # that read as plausible currency amounts in isolation. We only treat
    # one as a year — not a value — when the line also carries an actual
    # date/year cue, so a genuine small salary figure like "2000" isn't
    # silently dropped from analysis.
    YEAR_MIN, YEAR_MAX = 1950, 2099
    YEAR_CONTEXT_KEYWORDS = ["date", "year", "fy", "dob", "born", "since", "w.e.f", "financial year"]

    # Numbers tied to a reference/identifier field, even when long enough
    # to escape ID_NUMBER_MIN_DIGITS or short enough to look like a normal
    # value — phone/fax/MICR/UAN/PF/ESI numbers are never meant to be
    # compared statistically against currency figures.
    ID_CONTEXT_KEYWORDS_EXTRA = [
        "phone", "mobile", "tel", "fax", "micr", "uan", "esi", "pf no",
        "employee id", "emp id", "a/c", "a/c no",
    ]

    # Grand-total / net-pay / closing-balance rows are mathematically
    # larger than the line items they sum — expected arithmetic, not
    # tampering. We only suppress the z-score flag when the total
    # genuinely reconciles with its components (within this tolerance);
    # a total that DOESN'T reconcile is exactly the tamper signature this
    # layer exists to catch, so it must still be flagged.
    TOTAL_KEYWORDS = [
        "total", "net pay", "net salary", "gross total", "grand total",
        "subtotal", "sub total", "amount payable", "net payable",
        "net amount", "total earnings", "total deductions", "net wage",
        "closing balance", "opening balance", "total deposits",
        "total withdrawals", "carry forward", "balance c/f", "balance b/f",
    ]
    ARITHMETIC_VALID_PCT_TOLERANCE = 2.0  # ±2%

    # When checking whether a regex match is a standalone number vs. a
    # fragment embedded in a longer alphanumeric token, this is the max
    # length of surrounding prefix/suffix text tolerated before treating
    # the match as part of a larger token (see _extract_clean_numbers).
    FRAGMENT_CONTEXT_MAX_LEN = 3

    # _build_signals() z-score severity tiers — EXTREME_Z_SCORE is an
    # arbitrary "very confident" cutoff well above Z_SCORE_THRESHOLD, used
    # only to weight the signal text/score, not to decide whether to flag.
    EXTREME_Z_SCORE_THRESHOLD = 5.0

    # Arithmetic cross-validation (_arithmetic_validation): catches
    # tampering where MULTIPLE numbers were edited together — e.g. Basic
    # 25000->45000 AND Net Pay 30000->50000 moved together, so neither is
    # a statistical outlier on its own — but the document's internal math
    # no longer balances. Document-type keyword sets used to decide which
    # equation(s) are even applicable to a given document.
    PAYSLIP_TYPE_KEYWORDS = [
        "basic salary", "basic", "hra", "gross", "net pay", "net salary",
        "deductions", "allowance", "earnings", "pf", "tds",
        "professional tax", "take home",
    ]
    BANK_STATEMENT_TYPE_KEYWORDS = [
        "opening balance", "closing balance", "total deposits",
        "total withdrawals", "withdrawal", "deposit", "balance",
    ]

    # Canonical field keys this check reasons about, and every label
    # phrasing recognized as referring to that field. Order doesn't matter
    # for correctness — multiple phrasings mapping to the same key is the
    # point (fuzzy/partial matching: "net salary payable" and "take home"
    # both mean net_pay).
    EARNINGS_COMPONENT_KEYS = [
        "basic", "hra", "da", "conveyance", "allowance", "bonus", "incentive", "other",
    ]
    ARITHMETIC_LABEL_MAP = [
        ("basic salary", "basic"), ("basic", "basic"),
        ("house rent allowance", "hra"), ("hra", "hra"),
        ("dearness allowance", "da"), ("da", "da"),
        ("conveyance", "conveyance"),
        ("other allowance", "other"),
        ("special allowance", "allowance"), ("allowance", "allowance"),
        ("bonus", "bonus"),
        ("incentive", "incentive"),
        ("gross salary", "gross"), ("gross pay", "gross"), ("gross earnings", "gross"),
        ("total earnings", "gross"), ("gross", "gross"),
        ("total deductions", "total_deductions"), ("total deduction", "total_deductions"),
        ("net salary payable", "net_pay"), ("net amount payable", "net_pay"),
        ("net pay", "net_pay"), ("net salary", "net_pay"), ("take home", "net_pay"),
        ("opening balance", "opening_balance"),
        ("closing balance", "closing_balance"),
        ("total deposits", "total_deposits"), ("total deposit", "total_deposits"),
        ("total withdrawals", "total_withdrawals"), ("total withdrawal", "total_withdrawals"),
    ]

    # Payslip math tolerates minor rounding/tax noise; bank arithmetic is
    # exact, so its tolerance is much tighter. Above HIGH_CONFIDENCE_DIFF_PCT
    # the mismatch is large enough to call HIGH confidence rather than MEDIUM.
    PAYSLIP_TOLERANCE_PCT    = 2.0
    BANK_TOLERANCE_PCT       = 0.5
    HIGH_CONFIDENCE_DIFF_PCT = 20.0

    # Per-check score contributions (uncapped sum, then capped at 100) —
    # net pay and bank balance are weighted higher than the earnings-sum
    # check since they're the terminal value an attacker actually wants
    # the reader to believe, not an intermediate component.
    EARNINGS_SUM_SCORE     = 40
    NET_PAY_EQUATION_SCORE = 50
    BANK_BALANCE_SCORE     = 60

    # Arithmetic findings are weighted at 40% of their raw score when
    # folded into the layer's overall anomaly_score, so this check alone
    # doesn't dominate the layer when the statistical (z-score) checks
    # above are already firing on the same document.
    ARITHMETIC_SCORE_WEIGHT = 0.4

    # Row-by-row running-balance validation (bank statements): a column
    # qualifies as the BALANCE column when at least this fraction of
    # candidate transaction rows have a value in it — balance appears on
    # nearly every row, while withdrawal/deposit columns are sparse (each
    # row is normally either a debit or a credit, not both).
    BALANCE_COLUMN_MIN_RATIO = 0.8

    # Below this many candidate rows, ratio-based column detection is
    # unreliable (a couple of incidental numeric lines can spuriously hit
    # 100%) — skip rather than risk misattributing columns.
    MIN_TRANSACTION_ROWS = 5

    RUNNING_BALANCE_TOLERANCE          = 1.0   # ±1 rupee for rounding
    RUNNING_BALANCE_HIGH_DIFF_RUPEES   = 1000  # absolute diff, not pct — a
                                                # tampered row is wrong by a
                                                # large rupee amount, not a
                                                # large percentage of itself
    RUNNING_BALANCE_SCORE_PER_ROW = 30
    RUNNING_BALANCE_SCORE_CAP     = 90
    RUNNING_BALANCE_SCORE_WEIGHT  = 0.4

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
            
            # UPGRADE 3: Rolling window analysis for large column groups
            if len(items) > 30:
                rolling_anomalies = self._rolling_window_outliers(items, context)
                all_anomalies.extend(rolling_anomalies)

        # UPGRADE 3: Cross-column anomaly boost
        all_anomalies = self._boost_cross_column_anomalies(all_anomalies)

        total_numbers = sum(len(items) for items in groups.values())
        signals, score = self._build_signals(all_anomalies, len(groups), total_numbers)

        # Cross-field arithmetic validation (payslip / bank statement):
        # catches tampering where multiple numbers were edited together so
        # none alone is a statistical outlier, but the document's internal
        # math no longer balances.
        arithmetic_findings, arithmetic_score = self._arithmetic_validation(lines)
        for finding in arithmetic_findings:
            signals.append(f"[ARITHMETIC] {finding['signal']}")
            all_anomalies.append(NumericAnomaly(
                page=finding["page"],
                line_num=finding["line_num"],
                text=finding["text"],
                bbox=tuple(finding["bbox"]),
                value=finding["stated"],
                group_mean=finding["expected"],
                group_std=0.0,
                z_score=finding["diff_pct"],
                context=f"arithmetic_{finding['check']}",
                reason=finding["signal"],
            ))
        if arithmetic_findings:
            score = min(100, score + int(round(arithmetic_score * self.ARITHMETIC_SCORE_WEIGHT)))

        # Row-by-row running-balance validation (bank statements): catches
        # a tampered statement whose SUMMARY totals were also adjusted to
        # balance, but whose individual transaction rows don't — a much
        # harder forgery to pull off than fixing four summary numbers.
        rb_findings, rb_score = self._validate_running_balance(pdf_path, lines)
        for finding in rb_findings:
            signals.append(f"[RUNNING_BALANCE] {finding['signal']}")
            all_anomalies.append(NumericAnomaly(
                page=finding["page"],
                line_num=finding["line_num"],
                text=finding["text"],
                bbox=tuple(finding["bbox"]),
                value=finding["stated"],
                group_mean=finding["expected"],
                group_std=0.0,
                z_score=finding["diff_pct"],
                context="running_balance",
                reason=finding["signal"],
            ))
        if rb_findings:
            score = min(100, score + int(round(rb_score * self.RUNNING_BALANCE_SCORE_WEIGHT)))

        # Sort by z-score descending
        all_anomalies.sort(key=lambda x: x.z_score, reverse=True)

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

    def _is_summary_row(self, text_lower: str) -> bool:
        return any(kw in text_lower for kw in self.TOTAL_KEYWORDS)

    def _summary_row_arithmetic_valid(self, item: dict, others: list[dict]) -> bool:
        """
        True if a "total"-shaped value approximately equals the sum of the
        OTHER (non-summary) values in its column group — i.e. it's a real,
        recalculated total, not a tampered figure. False (not suppressed)
        when there aren't enough non-summary siblings to check against, so
        an unreconciled total still gets evaluated normally rather than
        being given a free pass.
        """
        components = [o["value"] for o in others if not self._is_summary_row(o["text"].lower())]
        if len(components) < 2:
            return False
        component_sum = sum(components)
        diff_pct = abs(component_sum - item["value"]) / max(abs(item["value"]), 1) * 100
        return diff_pct <= self.ARITHMETIC_VALID_PCT_TOLERANCE

    def _find_outliers(self, items: list[dict], context: str) -> list[NumericAnomaly]:
        """
        Find statistically anomalous values in a group using a leave-one-out
        (jackknife) z-score with TRIMMED MEAN/STD: each candidate value is 
        compared against the trimmed mean/std of the OTHER values in the group.
        
        UPGRADE 2 FIX — Threshold saturation prevention:
        By excluding top/bottom 10% of values before calculating mean/std,
        a single extreme outlier (tampered value) cannot inflate the standard
        deviation enough to prevent other edits from being detected.
        
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
                other_items = [items[j] for j in range(len(items)) if j != i]
                if self._is_summary_row(item["text"].lower()) and self._summary_row_arithmetic_valid(item, other_items):
                    continue  # reconciles with its components — legitimate total, not tampering

                others = [o["value"] for o in other_items]
                if len(others) < 2:
                    continue
                # UPGRADE 2 FIX: Use trimmed mean/std (exclude top/bottom 10%)
                mean, std = self._trimmed_mean_std(others, trim_percent=10)
                std = max(std, self.MIN_STD)

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
                other_items = [items[j] for j in range(len(items)) if j != i]
                if self._is_summary_row(item["text"].lower()) and self._summary_row_arithmetic_valid(item, other_items):
                    continue  # reconciles with its components — legitimate total, not tampering

                others = [o["value"] for o in other_items]
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

    # ── Arithmetic validation ──────────────────────────────────────────────────

    def _label_matches(self, text_lower: str, keyword: str) -> bool:
        """
        Plain substring match for longer labels ("total earnings", "gross"),
        but word-boundary match for short abbreviations ("pf", "hra") —
        a bare substring check on "pf" would also match inside ordinary
        words like "helpful", silently mislabeling unrelated lines.
        """
        if len(keyword) <= 3:
            return re.search(rf"\b{re.escape(keyword)}\b", text_lower) is not None
        return keyword in text_lower

    def _arithmetic_validation(self, lines: list[dict]) -> tuple[list[dict], int]:
        """
        Cross-field arithmetic validation — catches tampering where
        MULTIPLE numbers were edited together (e.g. Basic 25000->45000 AND
        Net Pay 30000->50000 moved together) so neither alone is a
        statistical outlier, but the document's internal math no longer
        balances:
            Payslip:        Sum(Earnings) - Sum(Deductions) = Net Pay
            Bank statement: Opening + Deposits - Withdrawals = Closing

        Each labeled field is looked up independently across the document
        by fuzzy keyword match, so a check only fires when all the fields
        it needs are actually present and labeled — a document without a
        recognizable payslip/bank structure is left alone.

        Returns (findings, arithmetic_score) where each finding is a dict
        with check/expected/stated/diff_pct/confidence/bbox/signal (plus
        page/line_num/text for building a NumericAnomaly for the report).
        """
        text_blob = " ".join(line["text"].lower() for line in lines)
        is_payslip = any(self._label_matches(text_blob, kw) for kw in self.PAYSLIP_TYPE_KEYWORDS)
        is_bank_statement = any(self._label_matches(text_blob, kw) for kw in self.BANK_STATEMENT_TYPE_KEYWORDS)

        labeled = {}  # key -> {"value", "bbox", "page", "line_num", "text"}
        for line in lines:
            if not line["numbers"]:
                continue
            text_lower = line["text"].lower()
            for keyword, key in self.ARITHMETIC_LABEL_MAP:
                if self._label_matches(text_lower, keyword):
                    labeled[key] = {
                        "value":    line["numbers"][-1],
                        "bbox":     line["bbox"],
                        "page":     line["page"],
                        "line_num": line["line_num"],
                        "text":     line["text"][:80],
                    }
                    break  # most-specific keyword wins; a line is one field, not several

        findings = []
        arithmetic_score = 0

        if is_payslip:
            # CHECK 1 — earnings components should sum to the stated gross
            component_sum = sum(
                labeled[k]["value"] for k in self.EARNINGS_COMPONENT_KEYS if k in labeled
            )
            gross = labeled.get("gross")
            if component_sum > 0 and gross:
                diff_pct = abs(component_sum - gross["value"]) / max(gross["value"], 1) * 100
                if diff_pct > self.PAYSLIP_TOLERANCE_PCT:
                    findings.append({
                        "check":      "earnings_sum",
                        "expected":   round(component_sum, 2),
                        "stated":     gross["value"],
                        "diff_pct":   round(diff_pct, 2),
                        "confidence": "HIGH" if diff_pct > self.HIGH_CONFIDENCE_DIFF_PCT else "MEDIUM",
                        "bbox":       list(gross["bbox"]),
                        "page":       gross["page"],
                        "line_num":   gross["line_num"],
                        "text":       gross["text"],
                        "signal": (
                            f"Earnings components sum to {component_sum:,.2f} but "
                            f"gross shown as {gross['value']:,.2f} (diff: {diff_pct:.1f}%)"
                        ),
                    })
                    arithmetic_score += self.EARNINGS_SUM_SCORE

            # CHECK 2 — Gross - Deductions = Net Pay
            if "gross" in labeled and "total_deductions" in labeled and "net_pay" in labeled:
                gross_v = labeled["gross"]["value"]
                ded_v   = labeled["total_deductions"]["value"]
                net     = labeled["net_pay"]
                expected_net = gross_v - ded_v
                diff_pct = abs(expected_net - net["value"]) / max(net["value"], 1) * 100
                if diff_pct > self.PAYSLIP_TOLERANCE_PCT:
                    findings.append({
                        "check":      "net_pay_equation",
                        "expected":   round(expected_net, 2),
                        "stated":     net["value"],
                        "diff_pct":   round(diff_pct, 2),
                        "confidence": "HIGH" if diff_pct > self.HIGH_CONFIDENCE_DIFF_PCT else "MEDIUM",
                        "bbox":       list(net["bbox"]),  # likely the tampered value
                        "page":       net["page"],
                        "line_num":   net["line_num"],
                        "text":       net["text"],
                        "signal": (
                            f"Gross ({gross_v:,.2f}) - Deductions ({ded_v:,.2f}) = "
                            f"{expected_net:,.2f} but Net Pay shown as {net['value']:,.2f}"
                        ),
                    })
                    arithmetic_score += self.NET_PAY_EQUATION_SCORE

        if is_bank_statement:
            # CHECK 3 — Opening + Deposits - Withdrawals = Closing
            required = ("opening_balance", "total_deposits", "total_withdrawals", "closing_balance")
            if all(k in labeled for k in required):
                opening     = labeled["opening_balance"]["value"]
                deposits    = labeled["total_deposits"]["value"]
                withdrawals = labeled["total_withdrawals"]["value"]
                closing     = labeled["closing_balance"]
                expected_closing = opening + deposits - withdrawals
                diff_pct = abs(expected_closing - closing["value"]) / max(closing["value"], 1) * 100
                if diff_pct > self.BANK_TOLERANCE_PCT:
                    findings.append({
                        "check":      "bank_balance",
                        "expected":   round(expected_closing, 2),
                        "stated":     closing["value"],
                        "diff_pct":   round(diff_pct, 2),
                        "confidence": "HIGH" if diff_pct > self.HIGH_CONFIDENCE_DIFF_PCT else "MEDIUM",
                        "bbox":       list(closing["bbox"]),
                        "page":       closing["page"],
                        "line_num":   closing["line_num"],
                        "text":       closing["text"],
                        "signal": (
                            f"Opening ({opening:,.2f}) + Deposits ({deposits:,.2f}) - "
                            f"Withdrawals ({withdrawals:,.2f}) = {expected_closing:,.2f} "
                            f"but Closing Balance shown as {closing['value']:,.2f}"
                        ),
                    })
                    arithmetic_score += self.BANK_BALANCE_SCORE

        return findings, min(100, arithmetic_score)

    def _extract_transaction_rows(self, pdf_path: str) -> list[dict]:
        """
        Re-extracts every line directly from the PDF for row-by-row bank
        transaction validation, bypassing _is_id_number.

        _is_id_number flags a number as an ID/reference whenever its LINE
        contains a substring like "upi"/"account"/"tran" — but real bank
        statement rows are literally narrated "UPI~<ref>~DR~NAME", so that
        check would blank out the withdrawal/deposit/balance amounts on
        almost every transaction row, not just the actual reference number
        embedded in the narration. _extract_clean_numbers' own per-word
        fragment-context guard already strips dates, TRAN IDs, and
        embedded reference numbers, so no further ID filtering is needed
        or wanted here.

        Then assigns each row's numbers to a WITHDRAWAL / DEPOSIT /
        BALANCE column by x-position: the balance column is identified as
        whichever column (within CLUSTER_X_TOLERANCE) has a value on
        BALANCE_COLUMN_MIN_RATIO+ of candidate rows and the highest mean
        x-position among those that do (balance is rightmost and present
        on nearly every row; withdrawal/deposit are sparser and to its
        left — but in statements dominated by debits, the withdrawal
        column alone can also clear the ratio threshold, so "highest
        x-position" is what actually disambiguates it from balance).

        Returns rows sorted in reading order, each with withdrawal/deposit
        (either may be None) and a required balance.
        """
        raw_lines = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    words = page.extract_words(
                        extra_attrs=["fontname", "size"],
                        keep_blank_chars=False,
                    )
                    if not words:
                        continue
                    for line_num, line_words in enumerate(self._group_into_lines(words)):
                        if not line_words:
                            continue
                        text = " ".join(w["text"] for w in line_words)
                        x0 = min(w["x0"]     for w in line_words)
                        y0 = min(w["top"]    for w in line_words)
                        x1 = max(w["x1"]     for w in line_words)
                        y1 = max(w["bottom"] for w in line_words)

                        number_entries = []
                        for w in line_words:
                            for val in self._extract_clean_numbers(w["text"]):
                                if val <= 0:
                                    continue
                                number_entries.append({
                                    "value":    val,
                                    "x_center": (w["x0"] + w["x1"]) / 2,
                                })

                        raw_lines.append({
                            "page":           page_num,
                            "line_num":       line_num,
                            "text":           text,
                            "bbox":           (x0, y0, x1, y1),
                            "number_entries": number_entries,
                        })
        except Exception:
            return []

        # Candidate transaction rows: 1-3 numbers, matching DATE | TRAN_ID |
        # PARTICULARS | WITHDRAWAL | DEPOSIT | BALANCE rows (at most one of
        # withdrawal/deposit plus balance). Excludes header/summary rows
        # with 4+ numbers (e.g. a totals row spanning several columns).
        candidates = [l for l in raw_lines if 1 <= len(l["number_entries"]) <= 3]
        if len(candidates) < self.MIN_TRANSACTION_ROWS:
            return []

        all_entries = []
        for line in candidates:
            for entry in line["number_entries"]:
                all_entries.append({
                    "x_center": entry["x_center"],
                    "value":    entry["value"],
                    "line":     line,
                })

        clusters = self._cluster_entries(all_entries, self.CLUSTER_X_TOLERANCE)
        cluster_info = []
        for cluster in clusters:
            lines_in_cluster = {id(e["line"]) for e in cluster}
            cluster_info.append({
                "x_mean":  sum(e["x_center"] for e in cluster) / len(cluster),
                "ratio":   len(lines_in_cluster) / len(candidates),
                "entries": cluster,
            })

        balance_candidates = [c for c in cluster_info if c["ratio"] >= self.BALANCE_COLUMN_MIN_RATIO]
        if not balance_candidates:
            return []
        balance_cluster = max(balance_candidates, key=lambda c: c["x_mean"])

        # Whatever is left to the balance column's left, closest-first, is
        # deposit then withdrawal (column order DATE|...|WITHDRAWAL|DEPOSIT|
        # BALANCE — deposit sits immediately left of balance).
        left_clusters = [c for c in cluster_info if c is not balance_cluster and c["x_mean"] < balance_cluster["x_mean"]]
        left_clusters.sort(key=lambda c: -c["x_mean"])
        deposit_cluster    = left_clusters[0] if len(left_clusters) >= 1 else None
        withdrawal_cluster = left_clusters[1] if len(left_clusters) >= 2 else None
        if deposit_cluster is None or withdrawal_cluster is None:
            # Can't separate withdrawal from deposit with confidence —
            # don't guess which column a single leftover cluster is.
            return []

        balance_by_line    = {id(e["line"]): e["value"] for e in balance_cluster["entries"]}
        deposit_by_line    = {id(e["line"]): e["value"] for e in deposit_cluster["entries"]}
        withdrawal_by_line = {id(e["line"]): e["value"] for e in withdrawal_cluster["entries"]}

        rows = []
        for line in candidates:
            lid = id(line)
            if lid not in balance_by_line:
                continue
            balance = balance_by_line[lid]
            withdrawal = withdrawal_by_line.get(lid)
            deposit = deposit_by_line.get(lid)
            if balance <= 0 or (withdrawal is None and deposit is None):
                continue
            rows.append({
                "page":       line["page"],
                "line_num":   line["line_num"],
                "text":       line["text"][:90],
                "bbox":       line["bbox"],
                "withdrawal": withdrawal,
                "deposit":    deposit,
                "balance":    balance,
            })

        rows.sort(key=lambda r: (r["page"], r["line_num"]))
        return rows

    def _validate_running_balance(self, pdf_path: str, lines: list[dict]) -> tuple[list[dict], int]:
        """
        Row-by-row running-balance check for bank statements: every row
        must satisfy prev_balance - withdrawal + deposit = new_balance.

        This is independent of _arithmetic_validation's summary-total
        check — a tampered statement can have correct Opening/Closing/
        Total Deposits/Withdrawals (if those were adjusted to match) while
        individual transaction rows are still mathematically impossible,
        since editing every row's running balance consistently is far
        harder than editing the four summary numbers.

        Returns (findings, running_balance_score); findings have keys
        check/prev_balance/withdrawal/deposit/expected/stated/diff/
        diff_pct/confidence/bbox/signal (plus page/line_num/text for
        building a NumericAnomaly for the report).
        """
        text_blob = " ".join(line["text"].lower() for line in lines)
        is_bank_statement = any(self._label_matches(text_blob, kw) for kw in self.BANK_STATEMENT_TYPE_KEYWORDS)
        if not is_bank_statement:
            return [], 0

        rows = self._extract_transaction_rows(pdf_path)
        if len(rows) < 2:
            return [], 0

        findings = []
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]

            withdrawal = curr["withdrawal"] or 0
            deposit    = curr["deposit"] or 0
            expected_balance = prev["balance"] - withdrawal + deposit
            actual_balance    = curr["balance"]

            diff = abs(expected_balance - actual_balance)
            if diff > self.RUNNING_BALANCE_TOLERANCE:
                diff_pct = diff / max(actual_balance, 1) * 100
                findings.append({
                    "check":        "running_balance",
                    "page":         curr["page"],
                    "line_num":     curr["line_num"],
                    "bbox":         list(curr["bbox"]),
                    "text":         curr["text"],
                    "prev_balance": prev["balance"],
                    "withdrawal":   withdrawal,
                    "deposit":      deposit,
                    "expected":     round(expected_balance, 2),
                    "stated":       actual_balance,
                    "diff":         round(diff, 2),
                    "diff_pct":     round(diff_pct, 2),
                    "confidence":   "HIGH" if diff > self.RUNNING_BALANCE_HIGH_DIFF_RUPEES else "MEDIUM",
                    "signal": (
                        f"Balance mismatch on row: prev({prev['balance']:,.2f}) "
                        f"- withdrawal({withdrawal:,.2f}) + deposit({deposit:,.2f}) "
                        f"= {expected_balance:,.2f} but stated as "
                        f"{actual_balance:,.2f} (diff: {diff:,.2f})"
                    ),
                })

        score = min(self.RUNNING_BALANCE_SCORE_CAP, len(findings) * self.RUNNING_BALANCE_SCORE_PER_ROW)
        return findings, score

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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UPGRADE 3 — COLUMN-AWARE Z-SCORE (Layer 4 enhancement)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _rolling_window_outliers(self, items: list[dict], context: str) -> list[NumericAnomaly]:
        """
        UPGRADE 3 — Rolling window analysis for large column groups (>30 values).
        
        For a column with many rows (e.g., salary slips with 50 line items):
        - A single anomalous value can appear normal when compared against all 50
        - But it's highly anomalous when compared against its local neighbors
        
        Algorithm:
        1. Sort items by line_num (top to bottom on page)
        2. Use rolling window of size=10, sliding by step=5
        3. For each window, compute trimmed mean/std on OTHER items in window
        4. Flag values that are outliers within their local context
        5. Merge with global anomalies (don't double-flag)
        
        Returns list of NumericAnomaly items detected by rolling window.
        """
        # Sort by line number (reading order)
        sorted_items = sorted(items, key=lambda x: (x["page"], x["line_num"]))
        
        if len(sorted_items) < 10:
            return []  # Need at least window size
        
        anomalies = []
        seen_lines = set()
        window_size = 10
        step = 5
        
        for start_idx in range(0, len(sorted_items) - window_size + 1, step):
            window = sorted_items[start_idx:start_idx + window_size]
            
            for i, item in enumerate(window):
                other_items_in_window = [window[j] for j in range(len(window)) if j != i]
                
                if self._is_summary_row(item["text"].lower()) and self._summary_row_arithmetic_valid(item, other_items_in_window):
                    continue  # Reconciles with its components
                
                others = [o["value"] for o in other_items_in_window]
                if len(others) < 2:
                    continue
                
                # Compute trimmed mean/std for this local window
                mean, std = self._trimmed_mean_std(others, trim_percent=10)
                std = max(std, self.MIN_STD)
                
                val = item["value"]
                z = abs(val - mean) / std
                line_key = (item["page"], item["line_num"])
                
                # Lower threshold for rolling window (local context is more strict)
                # Flag if z >= 2.5 (lower than global 3.0) for local anomalies
                if z >= 2.5 and line_key not in seen_lines:
                    seen_lines.add(line_key)
                    reason = (
                        f"Value {val:,.2f} is {z:.1f} standard deviations "
                        f"from local window mean {mean:,.2f} "
                        f"(rolling window of 10 items, window-local z-score)"
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
                        context=context + " (rolling window)",
                        reason=reason,
                    ))
        
        return anomalies

    def _boost_cross_column_anomalies(self, anomalies: list[NumericAnomaly]) -> list[NumericAnomaly]:
        """
        UPGRADE 3 — Cross-column anomaly boost.
        
        If the same line has anomalies flagged in multiple columns,
        boost confidence by +15 for each pairing.
        
        Pattern: Multi-location tampering often edits the same row across
        multiple columns (e.g., editing both Basic and HRA salary in same row).
        Detecting this cross-column pattern increases confidence.
        """
        # Group anomalies by (page, line_num)
        anomalies_by_line = defaultdict(list)
        for anom in anomalies:
            key = (anom.page, anom.line_num)
            anomalies_by_line[key].append(anom)
        
        # Boost anomalies found in multiple columns on same line
        boosted = []
        for line_key, line_anomalies in anomalies_by_line.items():
            if len(line_anomalies) > 1:
                # Multiple anomalies on same line — cross-column pattern detected
                # Boost z-score for each by 15 points
                for anom in line_anomalies:
                    boosted_z = anom.z_score + 15  # Direct z-score boost
                    # Create a new anomaly with boosted score
                    boosted_anom = NumericAnomaly(
                        page=anom.page,
                        line_num=anom.line_num,
                        text=anom.text,
                        bbox=anom.bbox,
                        value=anom.value,
                        group_mean=anom.group_mean,
                        group_std=anom.group_std,
                        z_score=round(boosted_z, 2),
                        context=anom.context,
                        reason=anom.reason + " [CROSS-COLUMN BOOST: Multiple anomalies on same line]",
                    )
                    boosted.append(boosted_anom)
            else:
                # Single anomaly on this line
                boosted.extend(line_anomalies)
        
        return boosted
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UPGRADE 2 — MULTI-LOCATION DETECTION FIX
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _trimmed_mean_std(self, values: list[float], trim_percent: int = 10) -> tuple[float, float]:
        """
        UPGRADE 2 FIX — Threshold saturation prevention
        
        Compute mean and standard deviation after removing top/bottom 10%.
        This prevents a single extreme outlier from inflating the standard
        deviation so much that other anomalies can't be detected.
        
        Algorithm:
        1. Sort values
        2. Exclude top and bottom trim_percent% of values
        3. Compute mean and stdev on remaining values
        4. Return (mean, stdev)
        
        Example: If values = [100, 200, 300, 5000] (last is edited):
        - Normal mean = 1400, normal std = 1936 → ratio ≈ 1.4 (can't flag 300 as outlier)
        - Trimmed mean = 200, trimmed std ≈ 71 → z-score for 5000 = 68 (strongly flagged)
        """
        if not values:
            return 0.0, 0.0
        
        if len(values) <= 2:
            # Can't meaningfully trim with so few values
            return statistics.mean(values), max(statistics.stdev(values) if len(values) >= 2 else 0, self.MIN_STD)
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        # Calculate how many items to exclude from each end
        trim_count = max(1, int(n * trim_percent / 100))
        
        # Exclude top and bottom trim_count items
        trimmed = sorted_vals[trim_count:n - trim_count]
        
        if len(trimmed) < 2:
            # If trimming removed too many, fall back to original
            return statistics.mean(values), max(statistics.stdev(values), self.MIN_STD)
        
        mean = statistics.mean(trimmed)
        stdev = max(statistics.stdev(trimmed), self.MIN_STD)
        
        return mean, stdev
