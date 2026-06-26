"""
Forensic Calculator — arithmetic running-balance verification layer.

Table region detection is PURELY GEOMETRIC — no hardcoded keywords,
no fixed page-percentage thresholds.  The transaction table is located
by the geometric signature that distinguishes it from every other part
of the document: multiple rows where numbers appear at the same
x-positions (columns) repeatedly, with 2+ distinct columns per row.

Algorithm: numeric-density analysis
  1. Extract every parseable number with (page, x, y) coordinates.
  2. Group numbers that share the same y (within 8 pt) into rows.
  3. Cluster x-positions across all rows on the page.
  4. Mark x-clusters that appear in 5+ distinct rows as "popular".
  5. Score each row by the fraction of its numbers that land in popular
     clusters.  Rows with score >= 0.5 are "table rows".
  6. Find the largest contiguous span of table rows (gaps allowed as
     long as consecutive high-scoring rows stay within 3× median
     inter-row spacing).
  7. Validate: at least MIN_TABLE_ROWS rows required.
  8. Extract numbers only from the validated table region.
  9. Cluster those numbers globally across pages → columns.
"""

import pdfplumber

# ── Tuning constants ──────────────────────────────────────────────────────────

X_CLUSTER_TOLERANCE   = 40.0   # pt; max x-distance to merge into the same column
Y_ROW_TOLERANCE       = 8.0    # pt; max y-delta to consider cells on the same row
MIN_TABLE_ROWS        = 5      # rows needed to qualify as a real table
POPULAR_COL_MIN_ROWS  = 5      # column must appear in this many rows to be "popular"
TABLE_SCORE_THRESHOLD = 0.5    # min fraction of numbers in popular columns
ROW_GAP_MULTIPLIER    = 3.0    # max_gap = this × median gap between high-score rows

OPENING_LABELS = [
    "opening balance", "op balance", "opening bal",
    "brought forward", "balance brought forward",
    "b/f", "opening", "ob",
]


# ── Module-level helpers ───────────────────────────────────────────────────────

def _parse_number(text: str):
    """Float if text is a number (commas, currency symbols, parenthetical
    negatives handled), else None."""
    t = (
        text.strip()
        .replace(",", "")
        .replace("₹", "")
        .replace("$", "")
        .replace("£", "")
        .replace("€", "")
    )
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    try:
        return float(t)
    except ValueError:
        return None


def _cluster_by_x(entries: list, tolerance: float = X_CLUSTER_TOLERANCE) -> list:
    """Single-pass left→right x-clustering.  Returns list-of-lists.
    The same dict objects are shared between the input list and the
    returned clusters (no copying), so callers may mutate entries after
    clustering and the mutations are visible in the original list."""
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


# ── Main class ────────────────────────────────────────────────────────────────

class ForensicCalculator:

    def __init__(self):
        self._pdf_path = None   # set during run_calculation for _find_summary_balance

    # ── Raw extraction (no filtering) ─────────────────────────────────────────

    def _extract_raw_numbers(self, pdf_path: str, page_filter=None):
        """
        Extract every parseable number from the PDF.
        Returns (entries_list, page_heights_dict).
        page_heights: {page_num (0-indexed): height_in_pts}
        """
        entries = []
        page_heights = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    if page_filter is not None and page_num != page_filter - 1:
                        continue
                    page_heights[page_num] = float(page.height or 842)
                    words = page.extract_words(x_tolerance=3, y_tolerance=3)
                    for w in words:
                        val = _parse_number(w["text"])
                        if val is None:
                            continue
                        x_center = (w["x0"] + w["x1"]) / 2
                        y_center = (w["top"] + w["bottom"]) / 2
                        entries.append({
                            "value":    val,
                            "x_center": x_center,
                            "x0":       w["x0"],
                            "x1":       w["x1"],
                            "y_center": y_center,
                            "bbox":     (w["x0"], w["top"], w["x1"], w["bottom"]),
                            "page":     page_num,
                        })
        except Exception:
            pass
        return entries, page_heights

    # ── Table-region detection ─────────────────────────────────────────────────

    def _find_table_region(self, entries_by_page: dict, page_heights: dict) -> dict:
        """
        Locate the transaction table on each page using pure numeric-density
        geometry.  No keywords, no fixed percentages.

        Returns {page_num: (y_top, y_bottom)} for each page that has a
        detectable table.  Pages with no valid table are omitted.

        Algorithm overview
        ──────────────────
        • Group numbers into horizontal rows (Y_ROW_TOLERANCE).
        • Cluster x-positions across all rows; mark x-clusters that appear
          in >= POPULAR_COL_MIN_ROWS distinct rows as "popular".
        • Score each row: fraction of its numbers in popular clusters.
        • Collect high-scoring rows (>= TABLE_SCORE_THRESHOLD).
        • Partition them into contiguous blocks (consecutive high-scoring rows
          whose y-distance stays within ROW_GAP_MULTIPLIER × median gap).
        • Pick the largest block.  Validate >= MIN_TABLE_ROWS rows.
        • Return y_top / y_bottom for the validated block.
        """
        result = {}

        for page_num, page_entries in entries_by_page.items():
            if not page_entries:
                continue

            # ── STEP 1: group numbers into rows ──────────────────────────────
            sorted_entries = sorted(page_entries, key=lambda e: e["y_center"])
            rows = []
            current_row   = [sorted_entries[0]]
            current_row_y = sorted_entries[0]["y_center"]

            for entry in sorted_entries[1:]:
                if abs(entry["y_center"] - current_row_y) <= Y_ROW_TOLERANCE:
                    current_row.append(entry)
                else:
                    rows.append(current_row)
                    current_row   = [entry]
                    current_row_y = entry["y_center"]
            rows.append(current_row)

            # Mean y-center for each row
            row_ys = [
                sum(e["y_center"] for e in row) / len(row)
                for row in rows
            ]
            n = len(rows)

            # ── STEP 2: cluster x-positions; score each row ───────────────────
            # Build a flat annotated list that tracks which row each number
            # belongs to.  _cluster_by_x shares the same dict objects, so
            # mutating xe["cluster_idx"] after clustering is safe.
            annotated = []
            for row_idx, row in enumerate(rows):
                for e in row:
                    annotated.append({
                        "x_center":  e["x_center"],
                        "row_idx":   row_idx,
                        # cluster_idx will be filled in below
                    })

            x_clusters = _cluster_by_x(annotated, tolerance=X_CLUSTER_TOLERANCE)

            # Tag each annotated entry with its cluster id
            for cidx, cluster in enumerate(x_clusters):
                for xe in cluster:
                    xe["cluster_idx"] = cidx

            # Count distinct rows per cluster
            cluster_rows: dict = {}
            for xe in annotated:
                cidx = xe["cluster_idx"]
                cluster_rows.setdefault(cidx, set()).add(xe["row_idx"])

            popular_clusters = {
                cidx
                for cidx, row_set in cluster_rows.items()
                if len(row_set) >= POPULAR_COL_MIN_ROWS
            }

            # Score each row: fraction of its numbers in popular clusters
            # Build a per-row lookup from annotated to avoid quadratic scan
            row_xs: dict = {}
            for xe in annotated:
                row_xs.setdefault(xe["row_idx"], []).append(xe)

            row_scores = []
            for row_idx in range(n):
                rxs = row_xs.get(row_idx, [])
                if not rxs:
                    row_scores.append(0.0)
                    continue
                in_popular = sum(
                    1 for xe in rxs
                    if xe.get("cluster_idx") in popular_clusters
                )
                row_scores.append(in_popular / len(rxs))

            # ── STEP 3: largest contiguous block of high-scoring rows ─────────
            # Work only with high-scoring rows; intermediate low-scoring rows
            # (narration lines with a reference number, etc.) are treated as
            # invisible — they do not break the block as long as consecutive
            # HIGH-SCORING rows remain within ROW_GAP_MULTIPLIER × median gap.

            high_score_ys = [
                (i, row_ys[i])
                for i in range(n)
                if row_scores[i] >= TABLE_SCORE_THRESHOLD
            ]

            if len(high_score_ys) < MIN_TABLE_ROWS:
                continue  # not enough scorable rows on this page

            # Median y-gap between consecutive high-scoring rows
            hs_gaps = [
                high_score_ys[j + 1][1] - high_score_ys[j][1]
                for j in range(len(high_score_ys) - 1)
            ]
            positive_gaps = sorted(g for g in hs_gaps if g > 0)
            if positive_gaps:
                median_gap = positive_gaps[len(positive_gaps) // 2]
            else:
                median_gap = 20.0  # fallback for very dense tables
            max_allowed_gap = max(ROW_GAP_MULTIPLIER * median_gap, 50.0)

            # Partition high-scoring rows into contiguous blocks
            blocks       = []
            current_block = [high_score_ys[0][0]]  # stores row index

            for j in range(1, len(high_score_ys)):
                gap = high_score_ys[j][1] - high_score_ys[j - 1][1]
                if gap < max_allowed_gap:
                    current_block.append(high_score_ys[j][0])
                else:
                    blocks.append(current_block)
                    current_block = [high_score_ys[j][0]]
            blocks.append(current_block)

            best_block = max(blocks, key=len)

            # ── STEP 4: validate ──────────────────────────────────────────────
            if len(best_block) < MIN_TABLE_ROWS:
                continue  # block too small — not a real table

            first_row_idx = best_block[0]
            last_row_idx  = best_block[-1]
            y_top         = row_ys[first_row_idx] - 10.0
            y_bottom      = row_ys[last_row_idx]  + 10.0

            # Clamp to actual page dimensions
            page_h   = page_heights.get(page_num, 842.0)
            y_top    = max(0.0,    y_top)
            y_bottom = min(page_h, y_bottom)

            result[page_num] = (y_top, y_bottom)

        return result

    # ── Table-filtered extraction ─────────────────────────────────────────────

    def _get_table_entries(self, pdf_path: str, page_filter=None) -> list:
        """
        Extract numbers then restrict to the detected table region(s).
        Returns an empty list if no table is detected (no fallback to
        arbitrary page percentages).
        """
        entries, page_heights = self._extract_raw_numbers(pdf_path, page_filter)
        if not entries:
            return []

        entries_by_page: dict = {}
        for e in entries:
            entries_by_page.setdefault(e["page"], []).append(e)

        table_regions = self._find_table_region(entries_by_page, page_heights)
        if not table_regions:
            return []

        return [
            e for e in entries
            if e["page"] in table_regions
            and table_regions[e["page"]][0] <= e["y_center"] <= table_regions[e["page"]][1]
        ]

    # ── Column detection (public API) ─────────────────────────────────────────

    def extract_columns(self, pdf_path: str) -> list:
        """
        Detect numeric columns within the transaction table.
        Returns descriptors sorted left→right:
          col_index, x_center, x_range, sample_values, value_count, likely_type
        """
        try:
            entries = self._get_table_entries(pdf_path)
            if not entries:
                return []

            clusters = _cluster_by_x(entries)
            if not clusters:
                return []

            max_count = max(len(c) for c in clusters)

            result = []
            for idx, cluster in enumerate(clusters):
                x_vals   = [e["x_center"] for e in cluster]
                x_center = sum(x_vals) / len(x_vals)
                x_min    = min(e["x0"] for e in cluster)
                x_max    = max(e["x1"] for e in cluster)
                values   = [e["value"] for e in cluster]
                count    = len(values)

                # Balance columns appear in almost every row (dense);
                # transaction (debit/credit) columns appear sparsely.
                density = count / max(max_count, 1)
                if density >= 0.80:
                    likely_type = "balance"
                elif density < 0.60:
                    likely_type = "transaction"
                else:
                    likely_type = "unknown"

                result.append({
                    "col_index":     idx,
                    "x_center":      round(x_center, 1),
                    "x_range":       [round(x_min, 1), round(x_max, 1)],
                    "sample_values": values[:5],
                    "value_count":   count,
                    "likely_type":   likely_type,
                })
            return result
        except Exception:
            return []

    def _get_column_entries(self, all_entries: list, col_index: int) -> list:
        """Return entries for the given column index (0-based), sorted page/y."""
        clusters = _cluster_by_x(all_entries)
        if col_index >= len(clusters):
            return []
        return sorted(clusters[col_index], key=lambda e: (e["page"], e["y_center"]))

    # ── Opening balance resolution ─────────────────────────────────────────────

    def _find_summary_balance(self, pdf_path: str):
        """Scan for a labelled opening balance figure. Returns float or None."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    words     = page.extract_words(x_tolerance=3, y_tolerance=3)
                    full_text = (page.extract_text() or "").lower()
                    for label in OPENING_LABELS:
                        if label not in full_text:
                            continue
                        label_words = label.split()
                        for wi, w in enumerate(words):
                            if w["text"].lower() != label_words[0]:
                                continue
                            if len(label_words) > 1:
                                segment = " ".join(
                                    ww["text"].lower()
                                    for ww in words[wi: wi + len(label_words)]
                                )
                                if segment != label:
                                    continue
                            label_y   = (w["top"] + w["bottom"]) / 2
                            label_x   = w["x1"]
                            best      = None
                            best_dist = float("inf")
                            for cw in words[wi:]:
                                val = _parse_number(cw["text"])
                                if val is None:
                                    continue
                                cw_y = (cw["top"] + cw["bottom"]) / 2
                                dx   = cw["x0"] - label_x
                                dy   = abs(cw_y - label_y)
                                if dx < -10 or dy > 150:
                                    continue
                                dist = dx + dy
                                if dist < best_dist:
                                    best_dist = dist
                                    best = val
                            if best is not None:
                                return best
        except Exception:
            pass
        return None

    def resolve_opening_balance(
        self,
        bal_entries: list,
        first_row_y: float,
        first_row_page: int,
        first_row_deposit: float,
        first_row_withdrawal: float,
        first_row_printed_balance,
        user_provided=None,
        operation: str = "+-",
    ) -> dict:
        """
        4-path opening-balance resolver:
        A) Lookback  — balance cell above the first data row
        B) Inverse   — back-calculate from row-1 printed balance
        C) User      — caller-supplied starting_balance
        D) Fallback  — assume opening == first-row balance (LOW confidence)
        """
        # PATH A
        candidates = [
            e for e in bal_entries
            if (e["page"] < first_row_page)
            or (e["page"] == first_row_page and e["y_center"] < first_row_y - 5)
        ]
        if candidates:
            return {
                "opening_balance": candidates[-1]["value"],
                "method":          "lookback",
                "confidence":      "HIGH",
                "anomaly":         False,
                "anomaly_reason":  None,
            }

        # PATH B
        if first_row_printed_balance is not None:
            if operation == "+-":
                ob = first_row_printed_balance - first_row_deposit + first_row_withdrawal
            elif operation == "+":
                ob = first_row_printed_balance - first_row_deposit - first_row_withdrawal
            elif operation == "-":
                ob = first_row_printed_balance + first_row_withdrawal
            else:
                ob = first_row_printed_balance

            doc_summary    = self._find_summary_balance(self._pdf_path) if self._pdf_path else None
            anomaly        = False
            anomaly_reason = None
            if doc_summary is not None:
                diff     = abs(ob - doc_summary)
                diff_pct = diff / max(abs(doc_summary), 1.0) * 100
                if diff_pct > 0.5:
                    anomaly = True
                    anomaly_reason = (
                        f"Row 1 implies opening balance of {ob:,.2f} but "
                        f"document summary shows {doc_summary:,.2f} "
                        f"(diff: {diff:,.2f}) — possible tampered first transaction"
                    )
            return {
                "opening_balance": ob,
                "method":          "inverse_equation",
                "confidence":      "MEDIUM",
                "anomaly":         anomaly,
                "anomaly_reason":  anomaly_reason,
            }

        # PATH C
        if user_provided is not None:
            return {
                "opening_balance": float(user_provided),
                "method":          "user_provided",
                "confidence":      "LOW",
                "anomaly":         False,
                "anomaly_reason":  None,
            }

        # PATH D
        return {
            "opening_balance": float(first_row_printed_balance or 0.0),
            "method":          "assumed_from_row1",
            "confidence":      "LOW",
            "anomaly":         False,
            "anomaly_reason":  "Could not auto-detect opening balance",
        }

    # ── Main calculation ───────────────────────────────────────────────────────

    def run_calculation(self, pdf_path: str, request) -> dict:
        """
        Full running-balance arithmetic check using table-region-filtered entries.
        request attributes: col_a_index, col_b_index, operation, balance_col_index,
                            starting_balance, tolerance, page_filter
        """
        self._pdf_path = pdf_path
        try:
            all_entries = self._get_table_entries(pdf_path)
            if not all_entries:
                return self._empty_result(
                    "No transaction table detected. "
                    "The document may have fewer than "
                    f"{MIN_TABLE_ROWS} data rows or no repeating numeric columns."
                )

            col_a_entries = self._get_column_entries(all_entries, request.col_a_index)
            col_b_entries = self._get_column_entries(all_entries, request.col_b_index)
            bal_entries   = self._get_column_entries(all_entries, request.balance_col_index)

            if not col_a_entries:
                return self._empty_result(
                    f"Column A (index {request.col_a_index}) has no data"
                )

            def find_match(row_page, row_y, entries):
                for e in entries:
                    if e["page"] == row_page and abs(e["y_center"] - row_y) <= Y_ROW_TOLERANCE:
                        return e
                return None

            aligned_rows = []
            for a_entry in col_a_entries:
                b_match  = find_match(a_entry["page"], a_entry["y_center"], col_b_entries)
                ba_match = find_match(a_entry["page"], a_entry["y_center"], bal_entries)
                aligned_rows.append({
                    "page":            a_entry["page"],
                    "y_center":        a_entry["y_center"],
                    "bbox":            a_entry["bbox"],
                    "val_a":           a_entry["value"],
                    "val_b":           b_match["value"] if b_match else 0.0,
                    "printed_balance": ba_match["value"] if ba_match else None,
                })
            aligned_rows.sort(key=lambda r: (r["page"], r["y_center"]))

            if not aligned_rows:
                return self._empty_result("Could not align columns into rows")

            first      = aligned_rows[0]
            resolution = self.resolve_opening_balance(
                bal_entries=bal_entries,
                first_row_y=first["y_center"],
                first_row_page=first["page"],
                first_row_deposit=first["val_a"],
                first_row_withdrawal=first["val_b"],
                first_row_printed_balance=first["printed_balance"],
                user_provided=request.starting_balance,
                operation=request.operation,
            )
            running_balance = resolution["opening_balance"]
            tolerance       = request.tolerance
            op              = request.operation

            rows = []
            for ar in aligned_rows:
                val_a = ar["val_a"]
                val_b = ar["val_b"]

                if op == "+":
                    delta = val_a + val_b
                elif op == "-":
                    delta = val_a - val_b
                elif op == "*":
                    delta = val_a * val_b if val_b != 0 else 0.0
                elif op == "/":
                    delta = val_a / val_b if val_b != 0 else 0.0
                else:   # "+-" : add A, subtract B
                    delta = val_a - val_b

                expected_balance = running_balance + delta
                printed_balance  = ar["printed_balance"]

                if printed_balance is not None:
                    diff        = abs(expected_balance - printed_balance)
                    is_mismatch = diff > tolerance
                    severity    = (
                        "HIGH"   if diff > 1000 else
                        "MEDIUM" if diff > 1    else
                        "OK"
                    )
                    running_balance = printed_balance
                else:
                    diff        = 0.0
                    is_mismatch = False
                    severity    = "OK"
                    running_balance = expected_balance

                rows.append({
                    "row_num":          len(rows) + 1,
                    "page":             ar["page"] + 1,   # 1-indexed for UI
                    "y_position":       round(ar["y_center"], 1),
                    "bbox":             list(ar["bbox"]),
                    "val_a":            round(val_a, 2),
                    "val_b":            round(val_b, 2),
                    "operation":        op,
                    "delta":            round(delta, 2),
                    "expected_balance": round(expected_balance, 2),
                    "printed_balance":  round(printed_balance, 2) if printed_balance is not None else None,
                    "difference":       round(diff, 2),
                    "is_mismatch":      is_mismatch,
                    "severity":         severity,
                })

            mismatch_rows = [r for r in rows if r["is_mismatch"]]
            last          = rows[-1] if rows else {}
            last_expected = last.get("expected_balance", 0.0)
            last_printed  = last.get("printed_balance")

            return {
                "rows":                           rows,
                "total_rows":                     len(rows),
                "mismatch_count":                 len(mismatch_rows),
                "mismatch_rows":                  mismatch_rows,
                "opening_balance_method":         resolution["method"],
                "opening_balance_confidence":     resolution["confidence"],
                "opening_balance_anomaly":        resolution.get("anomaly", False),
                "opening_balance_anomaly_reason": resolution.get("anomaly_reason"),
                "summary": {
                    "opening_balance":  round(resolution["opening_balance"], 2),
                    "total_col_a":      round(sum(r["val_a"] for r in rows), 2),
                    "total_col_b":      round(sum(r["val_b"] for r in rows), 2),
                    "expected_closing": round(last_expected, 2),
                    "printed_closing":  last_printed,
                    "closing_mismatch": round(
                        abs(last_expected - (last_printed or last_expected)), 2
                    ),
                },
            }
        except Exception as e:
            return self._empty_result(str(e))

    def _empty_result(self, error: str = None) -> dict:
        return {
            "rows":                           [],
            "total_rows":                     0,
            "mismatch_count":                 0,
            "mismatch_rows":                  [],
            "opening_balance_method":         "unknown",
            "opening_balance_confidence":     "LOW",
            "opening_balance_anomaly":        False,
            "opening_balance_anomaly_reason": None,
            "error":                          error,
            "summary": {
                "opening_balance":  0,
                "total_col_a":      0,
                "total_col_b":      0,
                "expected_closing": 0,
                "printed_closing":  None,
                "closing_mismatch": 0,
            },
        }
