"""Per-line color consistency, mixed font-embedding families, and the
human-readable layer signals."""

import re
from collections import Counter

import fitz

from .constants import *


class ColorSignalsMixin:
    def _check_color_consistency_per_line(self, pdf_path: str) -> list[dict]:
        """
        Within each text line, flag a span whose RGB color both (a)
        differs meaningfully from the line's dominant color and (b) is
        RARE across the whole document — see COLOR_CLUSTER_MIN_SHARE above
        for why frequency, not raw color distance, is what separates a
        real edit from deliberate label/value styling.
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return []

        line_spans = []  # list of list[span_color_dict], one per line
        all_colors = []

        try:
            for page_num in range(len(doc)):
                rawdict = doc[page_num].get_text("rawdict")
                for block in rawdict.get("blocks", []):
                    for line in block.get("lines", []):
                        spans = []
                        for span in line.get("spans", []):
                            # rawdict spans have no "text" field, only a
                            # per-character "chars" list — reconstruct the
                            # text by joining the chars.
                            text = "".join(ch.get("c", "") for ch in span.get("chars", [])).strip()
                            if not text:
                                continue
                            color_int = span.get("color", 0)
                            rgb = (
                                (color_int >> 16) & 0xFF,
                                (color_int >> 8) & 0xFF,
                                color_int & 0xFF,
                            )
                            spans.append({
                                "text": text, "rgb": rgb,
                                "bbox": span.get("bbox", (0, 0, 0, 0)),
                                "page": page_num,
                            })
                            all_colors.append(rgb)
                        if len(spans) >= 2:
                            line_spans.append(spans)
        finally:
            doc.close()

        if not all_colors:
            return []

        color_counts = Counter(all_colors)
        threshold = max(2, round(len(all_colors) * COLOR_CLUSTER_MIN_SHARE))
        common_colors = {c for c, n in color_counts.items() if n >= threshold}

        candidates = []
        for spans in line_spans:
            counts = Counter(s["rgb"] for s in spans)
            dominant_color = counts.most_common(1)[0][0]
            if len(counts) <= 1:
                continue
            for s in spans:
                if s["rgb"] == dominant_color or s["rgb"] in common_colors:
                    continue
                diff = sum(abs(a - b) for a, b in zip(s["rgb"], dominant_color))
                if diff < COLOR_DIFF_MIN:
                    continue
                candidates.append((s, dominant_color, diff))

        # A real edit happens once, in one place. The exact same (text,
        # color, dominant_color) combination recurring on 2+ DIFFERENT
        # pages is a repeated letterhead/header/footer element rendered
        # with a slightly different anti-aliased near-black than the body
        # text -- not an edit replicated identically across pages.
        pages_per_combo = {}
        for s, dominant_color, _ in candidates:
            key = (s["text"], s["rgb"], dominant_color)
            pages_per_combo.setdefault(key, set()).add(s["page"])

        anomalies = []
        for s, dominant_color, diff in candidates:
            key = (s["text"], s["rgb"], dominant_color)
            if len(pages_per_combo[key]) >= 2:
                continue
            anomalies.append({
                    "page": s["page"],
                    "text": s["text"],
                    "bbox": s["bbox"],
                    "color": s["rgb"],
                    "dominant_color": dominant_color,
                    "color_diff": diff,
                    "reason": (
                        f"Color mismatch within same line: span '{s['text'][:20]}' "
                        f"uses RGB{s['rgb']} while the rest of the line uses "
                        f"RGB{dominant_color} (diff={diff}) — possible text "
                        f"edited with a different tool"
                    ),
                })
        return anomalies

    # ── Signals + score ────────────────────────────────────────────────────────

    def _mixed_font_embedding_families(self, fonts: list) -> set:
        """
        Base font families that appear both embedded and non-embedded
        (under different subset-prefix tags) — see MIXED_FONT_EMBEDDING_SCORE.
        """
        embedded_families = set()
        unembedded_families = set()

        for font in fonts:
            name = font.get("name", "")
            base = name.lstrip("/")
            base = re.sub(r'^[A-Z]{6}\+', '', base)
            base = base.lower().split('-')[0]
            if base.endswith('mt'):
                base = base[:-2]
            if not base:
                continue
            if font.get("embedded"):
                embedded_families.add(base)
            else:
                unembedded_families.add(base)

        return embedded_families & unembedded_families

    def _build_signals(
        self, lines, suspicious_lines, profile, fonts: list = None,
        color_issues: list = None, gap_findings: list = None,
    ) -> tuple[list[str], int]:
        signals = []
        score   = 0

        # Mixed embedded/non-embedded subsets of the same font family —
        # indicates two separate edit sessions with different font handling.
        mixed = self._mixed_font_embedding_families(fonts or [])
        if mixed:
            signals.append(
                f"Font family '{', '.join(sorted(mixed))}' has both embedded "
                f"and non-embedded subsets — indicates multiple edit "
                f"sessions with different font rendering"
            )
            score += MIXED_FONT_EMBEDDING_SCORE

        # Font diversity — only flag if dominant font covers less than 60% of lines
        if profile["font_count"] > 3 and profile["dominant_font_ratio"] < 0.60:
            signals.append(
                f"{profile['font_count']} different fonts detected "
                f"(dominant: '{profile['dominant_font']}' "
                f"in {profile['dominant_font_ratio']:.0%} of lines) — "
                f"unusual font diversity for a single document"
            )
            score += 15

        # High-confidence suspicious lines
        high = [l for l in suspicious_lines if l.score > 0.5]
        med  = [l for l in suspicious_lines if 0.3 < l.score <= 0.5]

        if high:
            signals.append(
                f"{len(high)} line(s) with strong anomaly — "
                f"font/spacing/visual breaks consistency"
            )
            score += min(50, len(high) * 12)

        if med:
            signals.append(
                f"{len(med)} line(s) with moderate anomaly"
            )
            score += min(20, len(med) * 5)

        if color_issues:
            signals.append(
                f"{len(color_issues)} span(s) with color inconsistency within "
                f"the same text line — text color doesn't match the rest of "
                f"the line, possible edit with a different tool"
            )
            score += min(COLOR_CONSISTENCY_SCORE_CAP,
                         len(color_issues) * COLOR_CONSISTENCY_SCORE_PER_SPAN)

        if gap_findings:
            for g in gap_findings:
                bbox = g["bbox"]
                signals.append(
                    f"[LINE_GAP] Page {g['page']+1}: line preceded by abnormal "
                    f"vertical gap ({g['gap']:.1f}pt vs page baseline "
                    f"{g['expected_gap']:.1f}pt, z={g['z_score']:.1f}) — "
                    f"possible text inserted into empty space "
                    f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
                )
            gap_score = min(LINE_GAP_SCORE_CAP, len(gap_findings) * LINE_GAP_SCORE_PER_ANOMALY)
            score += gap_score * LINE_GAP_SCORE_WEIGHT

        if not signals:
            signals.append(
                "Content is internally consistent — "
                "no font, spacing, or visual anomalies found"
            )

        return signals, min(100, score)
