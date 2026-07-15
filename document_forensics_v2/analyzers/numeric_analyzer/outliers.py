"""Outlier detection: z-score + magnitude-ratio flags, rolling-window
column analysis, cross-column boost, and layer signals."""

import re
import statistics
from collections import defaultdict

from .models import NumericAnomaly


class OutlierDetectionMixin:
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
