"""Arithmetic validation: total/subtotal cross-checks, bank-statement
transaction-row extraction, and running-balance verification."""

import re

import pdfplumber

from .models import NumericAnomaly


class ArithmeticValidationMixin:
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

