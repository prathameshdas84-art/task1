"""Context grouping: label-based and magnitude clustering, summary-row
detection and arithmetic sanity."""

import re
from collections import defaultdict


class NumberGroupingMixin:
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

