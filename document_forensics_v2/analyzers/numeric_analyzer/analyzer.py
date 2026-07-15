"""NumericAnalyzer core — tuning constants (class attributes) and the
analyze() orchestration. Extraction, grouping, outlier, and arithmetic
internals live in the mixin modules."""

from .models import NumericAnomaly, NumericReport
from .extraction import NumberExtractionMixin
from .grouping import NumberGroupingMixin
from .outliers import OutlierDetectionMixin
from .arithmetic import ArithmeticValidationMixin


class NumericAnalyzer(NumberExtractionMixin, NumberGroupingMixin,
                      OutlierDetectionMixin, ArithmeticValidationMixin):

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

