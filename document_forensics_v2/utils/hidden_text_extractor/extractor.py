"""HiddenTextExtractor core — merges the three recovery methods into
one deduplicated HiddenTextReport. READ-ONLY, never modifies the PDF."""

from .models import HiddenTextReport
from .extraction import HiddenTextExtractionMixin
from .narrative import HiddenTextNarrativeMixin
from .stacking import TextStackingMixin


class HiddenTextExtractor(HiddenTextExtractionMixin, HiddenTextNarrativeMixin,
                          TextStackingMixin):

    # ── Main entry point ────────────────────────────────────────────────

    def analyze(self, pdf_path: str) -> HiddenTextReport:
        all_findings = []

        # Method 1 — white rectangle cover-ups
        try:
            covered = self._extract_covered_text(pdf_path)
            all_findings.extend(covered)
        except Exception:
            pass

        # Method 2 — z-order text overlaps
        try:
            overlapping = self._extract_overlapping_text(pdf_path)
            all_findings.extend(overlapping)
        except Exception:
            pass

        # Method 3 — incremental update recovery
        try:
            revisions = self._extract_revision_text(pdf_path)
            all_findings.extend(revisions)
        except Exception:
            pass

        # Deduplicate findings at the same location (methods 1 and 2 may
        # both catch the same cover-up).
        seen_locations = set()
        unique_findings = []
        for f in all_findings:
            key = (f.page, f.original_text[:20])
            if key not in seen_locations:
                seen_locations.add(key)
                unique_findings.append(f)

        # Classify field type + missing/replaced, and attach a clear
        # description and plain-English explanation for each case.
        for f in unique_findings:
            f.field_type = self._classify_field_type(f.original_text)
            f.replacement_type = self._classify_replacement_type(f.covering_text)
            f.description = self._compose_hidden_text_description(f)
            if f.replacement_type == "missing":
                f.plain_explanation = (
                    "The original content was hidden or removed with nothing "
                    "visible put in its place. It still exists in the file's "
                    "underlying data even though the page shows a blank or "
                    "covered area where it used to be."
                )
            else:
                f.plain_explanation = self._get_plain_explanation(
                    f.method, f.field_type
                )

        # Build signals for the main report
        signals = []
        for f in unique_findings:
            if f.replacement_type == "missing":
                signals.append(
                    f"[HIDDEN TEXT] Page {f.page} "
                    f"({f.method}): "
                    f"Original='{f.original_text[:50]}' "
                    f"— data removed, no replacement visible"
                )
            else:
                signals.append(
                    f"[HIDDEN TEXT] Page {f.page} "
                    f"({f.method}): "
                    f"Original='{f.original_text[:50]}' "
                    f"Replaced by='{f.covering_text[:50]}'"
                )

        if unique_findings:
            methods_used = set(f.method for f in unique_findings)
            summary = (
                f"Found {len(unique_findings)} hidden text "
                f"region(s) via: "
                f"{', '.join(methods_used)}"
            )
        else:
            summary = "No hidden original text detected"

        report = HiddenTextReport(
            findings=unique_findings,
            total_found=len(unique_findings),
            recovery_summary=summary,
            signals=signals,
        )
        report.conclusion = self._generate_conclusion(unique_findings)
        return report
