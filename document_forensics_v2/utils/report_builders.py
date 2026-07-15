"""
Report-building helpers used by the /analyze route — merged-document
detection, cross-layer timeline assertion, and the confidence/summary/
full-metadata builders. Relocated verbatim out of main.py (Phase 2 folder
reorganization) — no logic changes.
"""

import re

from models import ConfidenceDetail, FullMetadata, FontDetail, PageDetail


# ── Upgrade 3: merged-document detection ────────────────────────────────────────
# A multi-document compilation (merged PDF) legitimately has different fonts/
# sizes/colors per source page by design — the metadata layer's score is
# weighted down for these so a global anomaly score doesn't get attributed to
# "tampering" when it's really just several different original documents.
MERGE_PRODUCER_KEYWORDS = [
    "pdf24", "ilovepdf", "smallpdf", "adobe acrobat",
    "foxit", "pdfcreator", "ghostscript",
    "microsoft print to pdf", "cups-pdf",
]
MERGE_MIN_PAGES                 = 3     # only consider merge detection above this page count
MERGE_MOD_INTERVAL_MINUTES      = 60    # long creation->modification gap = likely re-assembled from parts
METADATA_MERGE_SCORE_MULTIPLIER = 0.4


def _detect_merged_document(meta_report, total_pages: int) -> bool:
    if not meta_report or total_pages <= MERGE_MIN_PAGES:
        return False
    producer = (meta_report.producer or "").lower()
    creator  = (meta_report.creator or "").lower()
    has_merge_producer = any(kw in producer or kw in creator for kw in MERGE_PRODUCER_KEYWORDS)
    long_mod_interval = (
        meta_report.time_delta_seconds is not None
        and meta_report.time_delta_seconds / 60 > MERGE_MOD_INTERVAL_MINUTES
    )
    return has_merge_producer or long_mod_interval


# ── Upgrade 2: cross-layer timeline assertor ────────────────────────────────────
# A document printed with "09 July 2024" but digitally created in 2026 is a
# backdating signal no existing layer catches — metadata only compares its
# OWN dates against each other, content never compares against metadata.
TIMELINE_DATE_PATTERNS = [
    # "09 July 2024", "9 Jan 2023"
    re.compile(
        r'\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
        r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|'
        r'nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b',
        re.IGNORECASE,
    ),
    # "09/07/2024", "09-07-2024"
    re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-](20\d{2})\b'),
    # "2024-07-09" (ISO format)
    re.compile(r'\b(20\d{2})-\d{2}-\d{2}\b'),
]
TIMELINE_YEAR_MIN        = 2000
TIMELINE_YEAR_MAX        = 2030
TIMELINE_TOLERANCE_YEARS = 1    # documents may legitimately be prepared slightly before/after their stated date
TIMELINE_BACKDATE_SCORE  = 35
TIMELINE_FUTURE_SCORE    = 10
TIMELINE_SCORE_CAP       = 70


def _cross_validate_timeline(pdf_path: str, meta_report) -> tuple[int, list[str]]:
    """
    Scans the document's visible text for printed dates and compares their
    years against the PDF's metadata creation year. Returns (timeline_score,
    signals) — (0, []) if there's no metadata creation date to compare
    against, or every printed date is within ±TIMELINE_TOLERANCE_YEARS of it.
    """
    if not meta_report or not meta_report.creation_date:
        return 0, []

    creation_year = meta_report.creation_date.year

    try:
        import fitz
        doc = fitz.open(pdf_path)
        full_text = " ".join(page.get_text() for page in doc)
        doc.close()
    except Exception:
        return 0, []

    printed_years = set()
    for pattern in TIMELINE_DATE_PATTERNS:
        for m in pattern.finditer(full_text):
            year = int(m.group(1))
            if TIMELINE_YEAR_MIN <= year <= TIMELINE_YEAR_MAX:
                printed_years.add(year)

    if not printed_years:
        return 0, []

    timeline_score = 0
    signals = []
    for printed_year in sorted(printed_years):
        diff = creation_year - printed_year
        if diff > TIMELINE_TOLERANCE_YEARS:
            signals.append(
                f"[TIMELINE] Printed date year {printed_year} predates digital "
                f"creation year {creation_year} by {diff} years — possible "
                f"backdated document"
            )
            timeline_score += TIMELINE_BACKDATE_SCORE
        elif -diff > TIMELINE_TOLERANCE_YEARS:
            signals.append(
                f"[TIMELINE] Printed date year {printed_year} is {-diff} years "
                f"AFTER the digital creation year {creation_year}"
            )
            timeline_score += TIMELINE_FUTURE_SCORE

    return min(TIMELINE_SCORE_CAP, timeline_score), signals


# ── Confidence builder ─────────────────────────────────────────────────────────

def build_confidence_detail(
    verdict: str,
    combined_score: float,
    layer_scores: dict,
    signals: list[str],
    suspicious_lines: int,
    numeric_anomalies: int,
    effective_threshold: float = 20.0,
) -> ConfidenceDetail:
    """
    Build a human-readable confidence explanation based on
    which layers fired and how strongly.
    """
    score = 0
    active_layers = []
    key_signals = []

    # Determine which layers contributed significantly
    if layer_scores["metadata"] >= 30:
        active_layers.append("metadata")
        key_signals.append(f"metadata anomaly score {layer_scores['metadata']}/100")

    if layer_scores["content"] >= 20:
        active_layers.append("content")
        key_signals.append(f"content anomaly score {layer_scores['content']}/100")


    if layer_scores["numeric"] >= 20:
        active_layers.append("numeric")
        key_signals.append(f"{numeric_anomalies} numeric outlier(s) detected")

    if layer_scores["ela"] >= 20:
        active_layers.append("ELA")
        key_signals.append(f"ELA anomaly score {layer_scores['ela']}/100")

    if layer_scores.get("pymupdf", 0) >= 20:
        active_layers.append("PyMuPDF")
        key_signals.append(f"PyMuPDF deep analysis score {layer_scores['pymupdf']}/100")

    if layer_scores.get("xref", 0) >= 20:
        active_layers.append("XREF")
        key_signals.append(f"XREF sequence anomaly score {layer_scores['xref']}/100")

    n_active = len(active_layers)

    # Uncertain band — score sits too close to the threshold to call either
    # way. Surface the active signals and ask for human review.
    if verdict == "UNCERTAIN":
        key_signals_text = "; ".join(key_signals) if key_signals else "none"
        return ConfidenceDetail(
            score=50,
            label="REVIEW NEEDED",
            explanation=(
                f"Combined score {combined_score:.1f} is close to the "
                f"detection threshold ({effective_threshold:g}). The document "
                f"shows some anomalies but they are not strong enough for "
                f"a confident verdict. Manual review by a human is recommended. "
                f"Active signals: {key_signals_text}"
            ),
        )

    # Confidence score based on number of agreeing layers + combined score
    if verdict == "ORIGINAL":
        if combined_score < 10:
            score = 90
            label = "VERY HIGH"
            explanation = (
                "Document shows strong consistency across all analysis layers. "
                "No significant anomalies detected in metadata, content, "
                "visual, or numeric analysis."
            )
        elif combined_score < 20:
            score = 75
            label = "HIGH"
            explanation = (
                "Document appears clean. Minor signals detected but below "
                "modification threshold. "
                + (f"Active signals: {'; '.join(key_signals)}." if key_signals else "")
            )
        else:
            score = 55
            label = "MEDIUM"
            explanation = (
                "Document is likely original but some signals were detected. "
                "Manual review recommended. "
                + (f"Active signals: {'; '.join(key_signals)}." if key_signals else "")
            )
    else:  # MODIFIED
        if n_active >= 3:
            score = 92
            label = "VERY HIGH"
            explanation = (
                f"Multiple independent layers agree: {', '.join(active_layers)} "
                f"all detected anomalies. "
                f"Combined score: {combined_score:.1f}/100. "
                f"Key signals: {'; '.join(key_signals)}."
            )
        elif n_active == 2:
            score = 78
            label = "HIGH"
            explanation = (
                f"Two independent layers detected anomalies: "
                f"{' and '.join(active_layers)}. "
                f"Combined score: {combined_score:.1f}/100. "
                f"Key signals: {'; '.join(key_signals)}."
            )
        elif n_active == 1:
            score = 60
            label = "MEDIUM"
            explanation = (
                f"Single layer detected anomaly: {active_layers[0]}. "
                f"Combined score: {combined_score:.1f}/100. "
                f"Consider additional verification. "
                f"Signal: {'; '.join(key_signals)}."
            )
        else:
            score = 50
            label = "LOW"
            explanation = (
                f"Modification threshold crossed but signals are weak. "
                f"Combined score: {combined_score:.1f}/100. "
                f"Manual review strongly recommended."
            )

    if suspicious_lines > 0 and verdict == "MODIFIED":
        explanation += (
            f" {suspicious_lines} suspicious line(s) identified in document."
        )

    return ConfidenceDetail(
        score=score,
        label=label,
        explanation=explanation.strip(),
    )


# ── Summary builder ────────────────────────────────────────────────────────────

def build_summary(
    verdict: str,
    combined_score: float,
    pdf_type: str,
    source: str,
    n_signals: int,
    n_suspicious_lines: int,
    n_numeric: int,
) -> str:
    if verdict == "UNCERTAIN":
        return (
            f"Document analyzed as {pdf_type.replace('_', ' ')} from {source}. "
            f"Combined forensic score {combined_score:.1f}/100 sits near the "
            f"modification threshold — evidence is inconclusive and a confident "
            f"verdict cannot be given. Manual review recommended. "
            f"{n_signals} forensic signal(s) detected across analysis layers."
        )
    if verdict == "ORIGINAL":
        return (
            f"Document analyzed as {pdf_type.replace('_', ' ')} from {source}. "
            f"Combined forensic score {combined_score:.1f}/100 — below modification "
            f"threshold. No significant tampering evidence found."
        )
    else:
        parts = []
        if n_suspicious_lines:
            parts.append(f"{n_suspicious_lines} suspicious line(s)")
        if n_numeric:
            parts.append(f"{n_numeric} numeric outlier(s)")
        evidence = ", ".join(parts) if parts else "metadata and visual signals"
        return (
            f"Document analyzed as {pdf_type.replace('_', ' ')} from {source}. "
            f"Combined forensic score {combined_score:.1f}/100 — above modification "
            f"threshold. Tampering evidence: {evidence}. "
            f"{n_signals} forensic signal(s) detected across analysis layers."
        )


# ── Full metadata builder ──────────────────────────────────────────────────────

def build_full_metadata(meta_report, total_pages: int, has_images: bool) -> FullMetadata:
    """Map the raw MetadataReport (Layer 1) onto the API's FullMetadata model."""
    was_modified = bool(
        meta_report.creation_date and meta_report.modification_date and
        meta_report.creation_date != meta_report.modification_date
    )

    modification_interval = None
    if meta_report.time_delta_seconds is not None:
        secs = meta_report.time_delta_seconds
        modification_interval = (
            f"{secs:.0f} seconds" if secs < 120 else f"{secs / 60:.1f} minutes"
        )

    return FullMetadata(
        producer=meta_report.producer or None,
        creator=meta_report.creator or None,
        author=meta_report.author or None,
        title=meta_report.title or None,
        subject=meta_report.subject or None,
        keywords=meta_report.keywords or None,
        created=meta_report.creation_date.isoformat() if meta_report.creation_date else None,
        modified=meta_report.modification_date.isoformat() if meta_report.modification_date else None,
        was_modified=was_modified,
        modification_interval=modification_interval,
        edit_age_days=meta_report.edit_age.get("days_ago"),
        edit_age_human=meta_report.edit_age.get("human_readable"),
        is_recent_edit=meta_report.edit_age.get("is_recent", False),
        is_very_recent_edit=meta_report.edit_age.get("is_very_recent", False),
        xmp_mismatch=meta_report.xmp_docinfo_mismatch,
        multiple_producers=meta_report.multiple_producers,
        source_risk=meta_report.source.suspicion_level,
        source_name=meta_report.source.identified_name,
        pdf_version=meta_report.pdf_version,
        total_pages=total_pages,
        document_id=meta_report.document_id,
        trapped=meta_report.trapped,
        has_javascript=meta_report.has_javascript,
        js_context=meta_report.js_context,
        has_open_action=meta_report.has_open_action,
        has_embedded_files=meta_report.has_embedded_files,
        has_images=has_images,
        is_encrypted=meta_report.is_encrypted,
        permissions=meta_report.permissions,
        font_count=len(meta_report.fonts),
        fonts=[FontDetail(**f) for f in meta_report.fonts],
        page_details=[PageDetail(**p) for p in meta_report.page_details],
        xmp_fields=meta_report.xmp_fields,
        icc_profiles=meta_report.icc_profiles,
        has_icc_profiles=meta_report.has_icc_profiles,
        icc_profile_details=meta_report.icc_profile_details,
        page_rotation=meta_report.page_rotation,
        raw=meta_report.raw_metadata,
        structure=meta_report.structure,
        suspicious_content=meta_report.suspicious_content,
        dimensions=meta_report.dimensions_full,
        dates=meta_report.dates_full,
        authenticity=meta_report.authenticity,
        xmp_mm=meta_report.xmp_mm,
        trailer_ids=meta_report.trailer_ids,
        object_level_dates=meta_report.object_level_dates,
        revision_info=meta_report.revision_info,
    )
