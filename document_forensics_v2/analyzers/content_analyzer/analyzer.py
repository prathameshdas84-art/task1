"""ContentAnalyzer core — the analyze() orchestration. Extraction,
feature math, scoring, and color/signal internals live in the mixin
modules."""

import re

from .constants import *
from .models import ContentReport, SuspiciousLine
from .extraction import TextExtractionMixin
from .features import LineFeaturesMixin
from .scoring import LineScoringMixin
from .color_signals import ColorSignalsMixin


class ContentAnalyzer(TextExtractionMixin, LineFeaturesMixin,
                      LineScoringMixin, ColorSignalsMixin):

    RENDER_DPI = 150

    def analyze(self, pdf_path: str, fonts: list = None) -> ContentReport:
        """
        fonts: optional list of {'name', 'embedded', ...} dicts from
        MetadataExtractor's MetadataReport.fonts — passed in by main.py
        (already extracted for Layer 1) rather than re-extracted here, so
        the same pikepdf font table isn't read twice per analysis.
        """
        pdf_type = self._detect_pdf_type(pdf_path)
        lines    = self._extract_lines(pdf_path)

        if not lines:
            return ContentReport(
                total_lines=0,
                suspicious_lines=[],
                dominant_font="",
                dominant_font_ratio=0.0,
                font_count=0,
                anomaly_score=0,
                signals=["No extractable text found — document may be image-based"],
                pdf_type=pdf_type,
            )

        profile          = self._build_profile(lines)
        try:
            per_page_profiles = self._build_per_page_profiles(lines, profile)
        except Exception:
            per_page_profiles = None

        # Upgrade 4 — first pass over subset-embedded fonts before scoring,
        # so platform-generated missing-glyph placeholders (Canva/Figma/
        # InDesign/etc.) can be told apart from a one-off injected/edited
        # occurrence of the same character.
        try:
            glyph_registry = self._build_glyph_registry(pdf_path)
        except Exception:
            glyph_registry = {}

        suspicious_lines = self._score_lines(lines, profile, per_page_profiles, glyph_registry)

        # Upgrade 1 — vertical line-gap density: catches text injected into
        # empty page space that font/spacing checks above miss (it inherits
        # the surrounding font/color, but breaks the page's line rhythm).
        # Computed from `lines` (not `suspicious_lines`) and scored
        # separately in _build_signals, so these don't also get double-
        # counted into the generic high/med anomaly-count buckets there.
        try:
            gap_findings = self._check_line_gap_density(lines)
        except Exception:
            gap_findings = []

        # ID cards (Aadhaar/PAN/driving licence/passport/voter ID)
        # legitimately mix ink colors on one line by template design — the
        # per-line color-consistency check is suppressed for them entirely
        # rather than threshold-tuned, since there's no single tolerance
        # that fits both "blue/black/orange on one Aadhaar line" and "one
        # tampered span" at the same time.
        is_id_card = self._is_id_card_document(lines)
        color_issues = [] if is_id_card else self._check_color_consistency_per_line(pdf_path)
        signals, score   = self._build_signals(lines, suspicious_lines, profile, fonts or [], color_issues, gap_findings)

        # Merged into the report's suspicious_lines AFTER scoring above, so
        # they're visible/highlightable without affecting the high/med
        # anomaly-count buckets _build_signals already used to score them.
        for g in gap_findings:
            suspicious_lines.append(SuspiciousLine(
                page=g["page"],
                line_num=g["line_num"],
                text=g["text"],
                bbox=g["bbox"],
                anomalies=[f"[LINE_GAP] {g['reason']}"],
                score=0.5,
            ))
        suspicious_lines.sort(key=lambda x: x.score, reverse=True)

        # Same _is_structural_line() classification the anomaly-detection
        # gates above already call — just also collected here, once, for
        # cross-layer fusion. Not a new heuristic, not a new logic path.
        try:
            structural_line_locations = [
                {"page": l.page, "bbox": list(l.bbox), "text": l.text}
                for l in lines
                if self._is_structural_line(l, lines)
            ]
        except Exception:
            structural_line_locations = []

        return ContentReport(
            total_lines=len(lines),
            suspicious_lines=suspicious_lines,  # return ALL suspicious lines, no cap
            dominant_font=profile["dominant_font"],
            dominant_font_ratio=profile["dominant_font_ratio"],
            font_count=profile["font_count"],
            anomaly_score=score,
            signals=signals,
            pdf_type=pdf_type,
            structural_line_locations=structural_line_locations,
        )

    # ── PDF type detection ─────────────────────────────────────────────────────

