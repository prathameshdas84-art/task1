import os
import time
import tempfile
import fitz
from fastapi import UploadFile, HTTPException

from analyzers.metadata_extractor import MetadataExtractor
from analyzers.content_analyzer import ContentAnalyzer
from analyzers.numeric_analyzer import NumericAnalyzer
from analyzers.ela_analyzer import ELAAnalyzer
from analyzers.pymupdf_analyzer import PyMuPDFAnalyzer
from analyzers.xref_analyzer import XrefAnalyzer
from fusion.verdict_engine import (
    combine, WEIGHTS, UNCERTAIN_BAND,
    CONFIDENCE_BASE, CONFIDENCE_DISTANCE_MULTIPLIER, CONFIDENCE_CAP,
)
from utils.hidden_text_extractor import HiddenTextExtractor
from utils.pdf_conversion import convert_to_pdf
from utils.report_builders import (
    _detect_merged_document, METADATA_MERGE_SCORE_MULTIPLIER,
    _cross_validate_timeline,
)
from fusion.signal_fusion import SignalFusion

# Constants
FUSION_ESCALATION_POINTS = {"HIGH": 10, "MEDIUM": 5}
FUSION_ESCALATION_CAP = 15

TEXT_STACKING_SCORE_PER_FINDING = 40
TEXT_STACKING_SCORE_CAP = 80


async def save_upload_to_temp_file(file: UploadFile, ext: str) -> tuple[str, float, bytes]:
    """Save the uploaded file to a temporary file and return the path, size, and content."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)
    content = await file.read()
    file_size_kb = len(content) / 1024
    with open(tmp_path, "wb") as f:
        f.write(content)
    return tmp_path, file_size_kb, content


def run_core_analysis_layers(pdf_path: str) -> dict:
    """Run core forensic layers (Metadata, Content, Numeric, ELA, PyMuPDF, Xref)."""
    try:
        meta_report = MetadataExtractor().extract(pdf_path)
    except Exception:
        meta_report = None

    try:
        content_report = ContentAnalyzer().analyze(
            pdf_path,
            fonts=meta_report.fonts if meta_report else None,
        )
    except Exception:
        content_report = None

    # Determine total pages and image presence
    try:
        doc_tmp = fitz.open(pdf_path)
        total_pages = len(doc_tmp)
        try:
            has_images = any(len(p.get_images()) > 0 for p in doc_tmp)
        except Exception:
            has_images = False
        doc_tmp.close()
    except Exception:
        total_pages = 1
        has_images = False

    # Upgrade 3 — merged-document detection: reduce metadata_score's
    # contribution before combine() ever sees it.
    try:
        if meta_report and _detect_merged_document(meta_report, total_pages):
            meta_report.anomaly_score = int(meta_report.anomaly_score * METADATA_MERGE_SCORE_MULTIPLIER)
            meta_report.anomalies.append(
                "[INFO] Document appears to be a multi-file compilation — "
                "metadata anomalies weighted down accordingly"
            )
    except Exception:
        pass

    try:
        numeric_report = NumericAnalyzer().analyze(pdf_path)
    except Exception:
        numeric_report = None

    try:
        ela_report = ELAAnalyzer().analyze(
            pdf_path,
            content_report.pdf_type if content_report else "native_text"
        )
    except Exception:
        ela_report = None

    try:
        pymupdf_report = PyMuPDFAnalyzer().analyze(pdf_path)
    except Exception:
        pymupdf_report = None

    try:
        xref_report = XrefAnalyzer().analyze(pdf_path)
    except Exception:
        xref_report = None

    if not all([meta_report, content_report]):
        failed = [name for name, r in [("metadata", meta_report), ("content", content_report)] if not r]
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse the uploaded file — the following core layers failed: {', '.join(failed)}. "
                   f"The file may be corrupt or password-protected. Try re-saving the PDF and re-uploading."
        )

    return {
        "meta_report": meta_report,
        "content_report": content_report,
        "numeric_report": numeric_report,
        "ela_report": ela_report,
        "pymupdf_report": pymupdf_report,
        "xref_report": xref_report,
        "total_pages": total_pages,
        "has_images": has_images,
    }


def run_extra_forensic_checks(pdf_path: str, content_report, ela_report, pymupdf_report) -> dict:
    """Run extra checks: text stacking, embedded image forensics, scanned page forensics, hidden text."""
    # Coordinate-collision text stacking
    try:
        text_stacking_findings = HiddenTextExtractor().detect_stacked_text(pdf_path)
    except Exception:
        text_stacking_findings = []

    if text_stacking_findings and pymupdf_report is not None:
        ts_layer_score = min(
            TEXT_STACKING_SCORE_CAP,
            len(text_stacking_findings) * TEXT_STACKING_SCORE_PER_FINDING,
        )
        pymupdf_report.anomaly_score = min(
            100, pymupdf_report.anomaly_score + ts_layer_score
        )

    # Embedded raster image objects analysis
    try:
        from utils.embedded_image_forensics import analyze_embedded_images
        embedded_img_result = analyze_embedded_images(pdf_path)
    except Exception:
        embedded_img_result = {"findings": [], "signals": [], "fold_score": 0,
                               "images_analyzed": 0, "images_skipped": 0}
    embedded_image_findings = embedded_img_result["findings"]

    if embedded_img_result["fold_score"] > 0 and ela_report is not None:
        ela_report.anomaly_score = min(
            100, int(round(ela_report.anomaly_score + embedded_img_result["fold_score"]))
        )

    # Scanned/mixed page renders checks
    try:
        from utils.scanned_page_forensics import analyze_scanned_pages
        scanned_px_result = analyze_scanned_pages(
            pdf_path,
            content_report.pdf_type if content_report else "native_text",
        )
    except Exception:
        scanned_px_result = {"findings": [], "signals": [], "fold_score": 0,
                             "pages_analyzed": 0, "pages_skipped": 0}
    scanned_pixel_findings = scanned_px_result["findings"]

    if scanned_px_result["fold_score"] > 0 and ela_report is not None:
        ela_report.anomaly_score = min(
            100, int(round(ela_report.anomaly_score + scanned_px_result["fold_score"]))
        )

    # Hidden text recovery
    try:
        hidden_text_report = HiddenTextExtractor().analyze(pdf_path)
        hidden_text_findings = hidden_text_report.findings
    except Exception:
        hidden_text_report = None
        hidden_text_findings = []

    return {
        "text_stacking_findings": text_stacking_findings,
        "embedded_img_result": embedded_img_result,
        "embedded_image_findings": embedded_image_findings,
        "scanned_px_result": scanned_px_result,
        "scanned_pixel_findings": scanned_pixel_findings,
        "hidden_text_report": hidden_text_report,
        "hidden_text_findings": hidden_text_findings,
    }


def apply_timeline_adjustments(pdf_path: str, meta_report, verdict_obj):
    """Apply timeline validation and adjustments to verdict_obj."""
    try:
        timeline_score, timeline_signals = _cross_validate_timeline(pdf_path, meta_report)
    except Exception:
        timeline_score, timeline_signals = 0, []

    if timeline_score > 0:
        verdict_obj.all_signals = (verdict_obj.all_signals or []) + timeline_signals
        verdict_obj.metadata_score = min(100, verdict_obj.metadata_score + timeline_score)
        metadata_weight = WEIGHTS.get(verdict_obj.pdf_type, WEIGHTS["native_text"])["metadata"]
        verdict_obj.combined_score = round(
            min(100, verdict_obj.combined_score + timeline_score * metadata_weight), 1
        )
        distance = abs(verdict_obj.combined_score - verdict_obj.effective_threshold)
        verdict_obj.confidence = min(
            CONFIDENCE_CAP, CONFIDENCE_BASE + int(distance * CONFIDENCE_DISTANCE_MULTIPLIER)
        )
        new_verdict = "MODIFIED" if verdict_obj.combined_score >= verdict_obj.effective_threshold else "ORIGINAL"
        if abs(verdict_obj.combined_score - verdict_obj.effective_threshold) <= UNCERTAIN_BAND:
            new_verdict = "UNCERTAIN"
        verdict_obj.verdict = new_verdict

    return timeline_score, timeline_signals


def apply_contradiction_adjustments(content_report, suspicious_lines, numeric_anomalies,
                                   ela_regions, overlay_regions, verdict_obj, layer_scores):
    """Detect contradictions and adjust layer scores and verdict accordingly."""
    fusion_engine = SignalFusion()
    contradicted_findings, contradiction_stats = fusion_engine.detect_contradictions(
        structural_line_locations=content_report.structural_line_locations if content_report else [],
        suspicious_lines=suspicious_lines,
        numeric_anomalies=numeric_anomalies,
        ela_regions=ela_regions,
        overlay_regions=overlay_regions,
    )

    if contradicted_findings:
        points_by_layer = {}
        for c in contradicted_findings:
            points_by_layer[c.layer] = points_by_layer.get(c.layer, 0) + c.weight_reduction_points

        layer_score_attr = {
            "content": "content_score", "numeric": "numeric_score",
            "ela": "ela_score", "pymupdf": "pymupdf_score",
        }
        weights = WEIGHTS.get(verdict_obj.pdf_type, WEIGHTS["native_text"])
        score_delta = 0.0
        for layer, points in points_by_layer.items():
            attr = layer_score_attr.get(layer)
            if not attr:
                continue
            before = getattr(verdict_obj, attr)
            after = max(0, before - points)
            setattr(verdict_obj, attr, after)
            score_delta += (before - after) * weights.get(layer, 0)

        if score_delta > 0:
            verdict_obj.combined_score = round(max(0.0, verdict_obj.combined_score - score_delta), 1)
            distance = abs(verdict_obj.combined_score - verdict_obj.effective_threshold)
            verdict_obj.confidence = min(
                CONFIDENCE_CAP, CONFIDENCE_BASE + int(distance * CONFIDENCE_DISTANCE_MULTIPLIER)
            )
            new_verdict = "MODIFIED" if verdict_obj.combined_score >= verdict_obj.effective_threshold else "ORIGINAL"
            if abs(verdict_obj.combined_score - verdict_obj.effective_threshold) <= UNCERTAIN_BAND:
                new_verdict = "UNCERTAIN"
            verdict_obj.verdict = new_verdict

            # Refresh layer scores dict
            layer_scores["content"]  = verdict_obj.content_score
            layer_scores["numeric"]  = verdict_obj.numeric_score
            layer_scores["ela"]      = verdict_obj.ela_score
            layer_scores["pymupdf"]  = verdict_obj.pymupdf_score

    return contradicted_findings


def apply_fusion_escalation(verdict_obj, fused_findings):
    """Escalate combined score/verdict for clean verdicts with strong fused findings."""
    if verdict_obj.verdict == "ORIGINAL" and fused_findings:
        fusion_boost = min(
            FUSION_ESCALATION_CAP,
            sum(FUSION_ESCALATION_POINTS.get(f.confidence, 0) for f in fused_findings),
        )
        if fusion_boost > 0:
            verdict_obj.combined_score = round(
                min(100, verdict_obj.combined_score + fusion_boost), 1
            )
            distance = abs(verdict_obj.combined_score - verdict_obj.effective_threshold)
            verdict_obj.confidence = min(
                CONFIDENCE_CAP, CONFIDENCE_BASE + int(distance * CONFIDENCE_DISTANCE_MULTIPLIER)
            )
            new_verdict = "MODIFIED" if verdict_obj.combined_score >= verdict_obj.effective_threshold else "ORIGINAL"
            if abs(verdict_obj.combined_score - verdict_obj.effective_threshold) <= UNCERTAIN_BAND:
                new_verdict = "UNCERTAIN"
            verdict_obj.verdict = new_verdict
            verdict_obj.all_signals = (verdict_obj.all_signals or []) + [
                f"[FUSION]   {len(fused_findings)} region(s) independently flagged by 2+ layers "
                f"— cross-validated agreement added +{fusion_boost} to the combined score"
            ]


def suppress_details_if_clean(verdict_obj, suspicious_lines, numeric_anomalies, ela_regions,
                              overlay_regions, fused_findings, contradicted_findings,
                              text_stacking_findings, hidden_text_findings,
                              embedded_image_findings, scanned_pixel_findings):
    """If document is ORIGINAL, zero out all coordinate-bearing lists to avoid false overlays."""
    is_clean = verdict_obj.verdict == "ORIGINAL"
    if is_clean:
        return {
            "suspicious_lines": [],
            "numeric_anomalies": [],
            "ela_regions": [],
            "overlay_regions": [],
            "fused_findings": [],
            "contradicted_findings": [],
            "text_stacking_findings": [],
            "hidden_text_findings": [],
            "embedded_image_findings": [],
            "scanned_pixel_findings": [],
        }
    return {
        "suspicious_lines": suspicious_lines,
        "numeric_anomalies": numeric_anomalies,
        "ela_regions": ela_regions,
        "overlay_regions": overlay_regions,
        "fused_findings": fused_findings,
        "contradicted_findings": contradicted_findings,
        "text_stacking_findings": text_stacking_findings,
        "hidden_text_findings": hidden_text_findings,
        "embedded_image_findings": embedded_image_findings,
        "scanned_pixel_findings": scanned_pixel_findings,
    }
