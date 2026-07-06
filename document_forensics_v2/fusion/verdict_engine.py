"""
Verdict Engine — combines metadata + content + OCR + numeric scores → final verdict.
Weights adjust automatically based on PDF type.
"""

from dataclasses import dataclass
from analyzers.metadata_extractor import MetadataReport
from analyzers.content_analyzer import ContentReport
from analyzers.ocr_analyzer import OCRReport
from analyzers.numeric_analyzer import NumericReport
from analyzers.ela_analyzer import ELAReport
from analyzers.pymupdf_analyzer import PyMuPDFReport


# Weights per PDF type — must sum to 1.0
WEIGHTS = {
    "native_text":    {"metadata": 0.20, "content": 0.35, "ocr": 0.00, "numeric": 0.20, "ela": 0.15, "pymupdf": 0.10},
    "scanned":        {"metadata": 0.15, "content": 0.05, "ocr": 0.45, "numeric": 0.10, "ela": 0.15, "pymupdf": 0.10},
    "mixed":          {"metadata": 0.20, "content": 0.15, "ocr": 0.20, "numeric": 0.15, "ela": 0.20, "pymupdf": 0.10},
    "scanned_native": {"metadata": 0.15, "content": 0.05, "ocr": 0.40, "numeric": 0.15, "ela": 0.15, "pymupdf": 0.10},
}

THRESHOLD = 20   # combined score >= 20 → MODIFIED (default for most pdf_types)
SCANNED_THRESHOLD = 25   # scanned documents need a higher bar — OCR noise inflates scores

# Combined scores within ±UNCERTAIN_BAND of the effective threshold are too
# close to call confidently → verdict becomes "UNCERTAIN" (human review).
UNCERTAIN_BAND = 5

# Confidence formula: confidence = min(CONFIDENCE_CAP, CONFIDENCE_BASE +
# int(distance_from_threshold * CONFIDENCE_DISTANCE_MULTIPLIER)). A verdict
# exactly at the threshold gets CONFIDENCE_BASE; confidence grows with how
# far the combined score is from the threshold in either direction.
CONFIDENCE_BASE                 = 50
CONFIDENCE_DISTANCE_MULTIPLIER  = 1.8
CONFIDENCE_CAP                  = 99

# Per-pdf_type threshold — falls back to THRESHOLD for any type not listed
THRESHOLDS = {
    "native_text":    THRESHOLD,
    "scanned":         SCANNED_THRESHOLD,
    "mixed":          THRESHOLD,
    "scanned_native": THRESHOLD,
}


@dataclass
class FinalVerdict:
    verdict: str           # "MODIFIED" | "ORIGINAL" | "UNCERTAIN"
    confidence: int        # 0-100
    combined_score: float
    metadata_score: int
    content_score: int
    ocr_score: int
    numeric_score: int = 0
    ela_score: int = 0
    pymupdf_score: int = 0
    xref_score: int = 0
    pdf_type: str = ""
    all_signals: list[str] = None
    effective_threshold: float = THRESHOLD  # threshold actually applied (after adjustments)


def combine(
    meta: MetadataReport,
    content: ContentReport,
    ocr: OCRReport,
    numeric: NumericReport = None,
    ela: ELAReport = None,
    pymupdf: PyMuPDFReport = None,
    xref: "XrefReport" = None,
) -> FinalVerdict:

    pdf_type = content.pdf_type
    w        = WEIGHTS.get(pdf_type, WEIGHTS["native_text"])

    numeric_score = numeric.anomaly_score if numeric else 0
    ela_score     = ela.anomaly_score if ela else 0
    pymupdf_score = pymupdf.anomaly_score if pymupdf else 0
    xref_score    = xref.xref_score if xref else 0
    combined = (
        meta.anomaly_score    * w["metadata"] +
        content.anomaly_score * w["content"]  +
        ocr.anomaly_score     * w["ocr"]      +
        numeric_score         * w.get("numeric", 0) +
        ela_score              * w.get("ela", 0) +
        pymupdf_score           * w.get("pymupdf", 0) +
        xref_score             * w.get("xref", 0)
    )

    effective_threshold = THRESHOLDS.get(pdf_type, THRESHOLD)

    # Strong tamper pattern: cleared metadata + UNKNOWN source + modified.
    # When the author was sanitized to anonymous/unknown, the source tool is
    # unrecognized, AND the document carries a real edit history, lower the
    # bar so this combination is caught even if the weighted score lands just
    # under the normal threshold.
    metadata_cleared = False
    if meta and meta.author:
        author_lower = (meta.author or "").lower()
        metadata_cleared = any(
            p in author_lower
            for p in ["anonymous", "unspecified", "unknown"]
        )

    was_modified = bool(meta and meta.creation_date and meta.modification_date
                        and meta.creation_date != meta.modification_date)

    if metadata_cleared and was_modified and meta.source.suspicion_level == "UNKNOWN":
        effective_threshold = max(12, effective_threshold - 8)

    verdict    = "MODIFIED" if combined >= effective_threshold else "ORIGINAL"
    distance   = abs(combined - effective_threshold)
    confidence = min(CONFIDENCE_CAP, CONFIDENCE_BASE + int(distance * CONFIDENCE_DISTANCE_MULTIPLIER))

    # Merge all signals
    all_signals = []
    for a in meta.anomalies:
        all_signals.append(f"[METADATA] {a}")
    for s in content.signals:
        all_signals.append(f"[CONTENT]  {s}")
    for s in ocr.signals:
        all_signals.append(f"[OCR]      {s}")
    if numeric:
        for s in numeric.signals:
            all_signals.append(f"[NUMERIC]  {s}")
    if ela:
        for s in ela.signals:
            all_signals.append(f"[ELA]      {s}")
    if pymupdf:
        for s in pymupdf.signals:
            all_signals.append(f"[PYMUPDF]  {s}")
    if xref:
        for s in xref.signals:
            all_signals.append(f"[XREF]     {s}")

    # Uncertain band: when the combined score sits within UNCERTAIN_BAND of
    # the (possibly adjusted) threshold, the evidence is too close to call
    # either way — surface it as UNCERTAIN so a human reviews it rather than
    # forcing a low-confidence MODIFIED/ORIGINAL.
    if abs(combined - effective_threshold) <= UNCERTAIN_BAND:
        verdict = "UNCERTAIN"

    return FinalVerdict(
        verdict=verdict,
        confidence=confidence,
        combined_score=round(combined, 1),
        metadata_score=meta.anomaly_score,
        content_score=content.anomaly_score,
        ocr_score=ocr.anomaly_score,
        numeric_score=numeric_score,
        ela_score=ela_score,
        pymupdf_score=pymupdf_score,
        xref_score=xref_score,
        pdf_type=pdf_type,
        all_signals=all_signals,
        effective_threshold=effective_threshold,
    )
