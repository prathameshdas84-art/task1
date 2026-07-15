"""POST /analyze — the full 6-layer PDF pipeline: runs every analyzer,
fuses signals, computes the verdict, and caches the result for the
render/hidden-text endpoints."""

import os
import time
import uuid
from pathlib import Path

from fastapi import File, UploadFile, HTTPException

from analyzers.metadata_extractor import MetadataExtractor
from fusion.verdict_engine import combine
from models import (
    ForensicResponse,
    LayerScores, SuspiciousLine, NumericAnomaly, ConfidenceDetail,
    FusedFindingModel, FusionStats, ContradictedFindingModel,
    TextStackingFindingModel, EmbeddedImageFindingModel,
)
from fusion.signal_fusion import SignalFusion
from utils.pdf_conversion import convert_to_pdf
from utils.report_builders import (
    build_confidence_detail, build_summary, build_full_metadata,
)
from api.analysis_cache import _analysis_cache, MAX_CACHED_ANALYSES, persist_analysis

from .base import router
from .pipeline_steps import (
    save_upload_to_temp_file,
    run_core_analysis_layers,
    run_extra_forensic_checks,
    apply_timeline_adjustments,
    apply_contradiction_adjustments,
    apply_fusion_escalation,
    suppress_details_if_clean,
)

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".docx", ".doc"}


@router.post("/analyze", response_model=ForensicResponse, tags=["Forensics"])
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze a document for signs of tampering.

    Upload any PDF, image (JPG/PNG), or Word document (.docx).
    Returns a complete forensic analysis with verdict, confidence, and evidence.
    """
    start_time = time.time()

    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. "
                   f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Save upload to temp file
    tmp_path = ""
    cached_pdf_path = None
    try:
        tmp_path, file_size_kb, content = await save_upload_to_temp_file(file, ext)

        # Convert to PDF if needed
        pdf_path = convert_to_pdf(tmp_path, file.filename)

        # Run core forensic analysis layers (Metadata, Content, Numeric, ELA, PyMuPDF, Xref)
        core_results = run_core_analysis_layers(pdf_path)
        meta_report = core_results["meta_report"]
        content_report = core_results["content_report"]
        numeric_report = core_results["numeric_report"]
        ela_report = core_results["ela_report"]
        pymupdf_report = core_results["pymupdf_report"]
        xref_report = core_results["xref_report"]
        total_pages = core_results["total_pages"]
        has_images = core_results["has_images"]

        # Run extra forensic checks (Text stacking, Embedded images, Scanned page pixel forensics, Hidden text)
        extra_results = run_extra_forensic_checks(
            pdf_path, content_report, ela_report, pymupdf_report
        )
        text_stacking_findings = extra_results["text_stacking_findings"]
        embedded_img_result = extra_results["embedded_img_result"]
        embedded_image_findings = extra_results["embedded_image_findings"]
        scanned_px_result = extra_results["scanned_px_result"]
        scanned_pixel_findings = extra_results["scanned_pixel_findings"]
        hidden_text_report = extra_results["hidden_text_report"]
        hidden_text_findings = extra_results["hidden_text_findings"]

        # Combine verdict using the weighted scoring framework
        verdict_obj = combine(
            meta_report, content_report,
            numeric_report, ela_report, pymupdf_report, xref_report
        )

        # Apply timeline adjustments (Upgrade 2)
        apply_timeline_adjustments(pdf_path, meta_report, verdict_obj)

        # Recompute authenticity score now that the cross-layer verdict exists
        meta_report.authenticity = MetadataExtractor()._compute_authenticity_score(
            meta_report,
            combined_score=verdict_obj.combined_score,
            verdict=verdict_obj.verdict,
        )

        # Build signals listing
        if embedded_img_result["signals"]:
            verdict_obj.all_signals = (verdict_obj.all_signals or []) + [
                f"[EMBEDDED_IMG] {embedded_img_result['images_analyzed']} embedded image "
                f"object(s) analyzed with the image-forensics checks — "
                f"{len(embedded_image_findings)} anomaly finding(s)"
            ] + [f"[EMBEDDED_IMG] {s}" for s in embedded_img_result["signals"]]

        if scanned_px_result["signals"]:
            verdict_obj.all_signals = (verdict_obj.all_signals or []) + [
                f"[SCANNED_PIXEL] {scanned_px_result['pages_analyzed']} scanned/mixed "
                f"page render(s) routed through the image-pipeline pixel checks — "
                f"{len(scanned_pixel_findings)} anomaly finding(s)"
            ] + [f"[SCANNED_PIXEL] {s}" for s in scanned_px_result["signals"]]

        if text_stacking_findings:
            verdict_obj.all_signals = (verdict_obj.all_signals or []) + [
                f"[TEXT_STACKING] {len(text_stacking_findings)} location(s) where 2+ "
                f"different text runs occupy the same coordinates — new text placed "
                f"over original without removing it"
            ] + [
                f"[TEXT_STACKING] Page {f.page + 1} "
                f"bbox={tuple(round(v, 1) for v in f.bbox)} "
                f"({f.confidence}, {f.overlap_fraction*100:.0f}% overlap): "
                + " vs ".join(f"'{t[:40]}'" for t in f.texts)
                for f in text_stacking_findings
            ]

        processing_time = round(time.time() - start_time, 2)

        # Map initial layer scores
        layer_scores = {
            "metadata": verdict_obj.metadata_score,
            "content":  verdict_obj.content_score,
            "numeric":  verdict_obj.numeric_score,
            "ela":      verdict_obj.ela_score,
            "pymupdf":  verdict_obj.pymupdf_score,
            "xref":     verdict_obj.xref_score,
        }

        # Retrieve raw lists of anomalies prior to gating
        suspicious_lines = content_report.suspicious_lines if content_report else []
        numeric_anomalies = numeric_report.anomalies if numeric_report else []
        ela_regions = ela_report.regions if ela_report else []
        overlay_regions = pymupdf_report.overlay_regions if pymupdf_report else []

        # Prepare metadata pseudo-findings for spatial fusion
        metadata_findings = []
        if meta_report and meta_report.anomaly_score >= 30:
            metadata_findings.append({
                "layer": "metadata",
                "page": 0,
                "score": meta_report.anomaly_score / 100,
                "text": "; ".join(meta_report.anomalies[:3]),
            })

        # Add timeline-backdating pseudo-finding
        if getattr(verdict_obj, "timeline_score", 0) > 0 and meta_report and meta_report.anomaly_score > 0:
            metadata_findings.append({
                "layer": "metadata",
                "page": 0,
                "score": min(1.0, verdict_obj.timeline_score / 100),
                "text": "; ".join(verdict_obj.timeline_signals[:3]),
            })

        # Add text-stacking findings for spatial fusion
        text_stacking_extra = [
            {
                "layer": "text_stacking",
                "page": f.page,
                "bbox": tuple(f.bbox) if f.bbox else None,
                "text": " | ".join(f.texts)[:80],
                "score": f.score,
                "raw": f,
            }
            for f in text_stacking_findings
        ]

        # Separate flat/pasted-patch ELA regions for spatial fusion
        flat_zone_regions = [
            r for r in ela_regions if getattr(r, "flat_zone_anomaly", False)
        ]
        ela_fusion_regions = [
            r for r in ela_regions if not getattr(r, "flat_zone_anomaly", False)
        ]
        flat_zone_extra = [
            {
                "layer": "flat_zone",
                "page": r.page,
                "bbox": tuple(r.bbox),
                "text": ("Pasted stamp — flat background" if r.stamp_associated
                         else "Flat/uniform region — inconsistent with page texture"),
                "score": r.flat_confidence,
                "raw": r,
            }
            for r in flat_zone_regions
        ]

        # Spatial fusion engine execution
        fused_findings, fusion_stats = SignalFusion().fuse(
            suspicious_lines=suspicious_lines,
            numeric_anomalies=numeric_anomalies,
            ela_regions=ela_fusion_regions,
            overlay_regions=overlay_regions,
            metadata_findings=metadata_findings,
            extra_findings=(text_stacking_extra
                            + flat_zone_extra + embedded_image_findings
                            + scanned_pixel_findings),
        )

        # Adjust score for contradictions
        contradicted_findings = apply_contradiction_adjustments(
            content_report, suspicious_lines, numeric_anomalies, ela_regions, overlay_regions,
            verdict_obj, layer_scores
        )

        # Apply fusion escalation boost
        apply_fusion_escalation(verdict_obj, fused_findings)

        # Gatekeeper: Zero out coordinate details if the final verdict is ORIGINAL
        gated_data = suppress_details_if_clean(
            verdict_obj, suspicious_lines, numeric_anomalies, ela_regions, overlay_regions,
            fused_findings, contradicted_findings, text_stacking_findings,
            hidden_text_findings, embedded_image_findings, scanned_pixel_findings
        )
        suspicious_lines = gated_data["suspicious_lines"]
        numeric_anomalies = gated_data["numeric_anomalies"]
        ela_regions = gated_data["ela_regions"]
        overlay_regions = gated_data["overlay_regions"]
        fused_findings = gated_data["fused_findings"]
        contradicted_findings = gated_data["contradicted_findings"]
        text_stacking_findings = gated_data["text_stacking_findings"]
        hidden_text_findings = gated_data["hidden_text_findings"]
        embedded_image_findings = gated_data["embedded_image_findings"]
        scanned_pixel_findings = gated_data["scanned_pixel_findings"]

        confidence = build_confidence_detail(
            verdict=verdict_obj.verdict,
            combined_score=verdict_obj.combined_score,
            layer_scores=layer_scores,
            signals=verdict_obj.all_signals or [],
            suspicious_lines=len(suspicious_lines),
            numeric_anomalies=len(numeric_anomalies),
            effective_threshold=verdict_obj.effective_threshold,
        )

        # Build final response schema
        result = ForensicResponse(
            verdict=verdict_obj.verdict,
            combined_score=verdict_obj.combined_score,
            confidence=confidence,
            filename=file.filename,
            file_size_kb=round(file_size_kb, 1),
            pdf_type=verdict_obj.pdf_type,
            document_source=meta_report.source.identified_name if meta_report else "Unknown",
            processing_time_seconds=processing_time,
            layers=LayerScores(**layer_scores),
            signals=verdict_obj.all_signals or [],
            suspicious_lines=[
                SuspiciousLine(
                    page=sl.page + 1,
                    line_num=sl.line_num + 1,
                    text=sl.text,
                    anomaly_score_pct=int(sl.score * 100),
                    reasons=sl.anomalies[:3],
                    bbox=list(sl.bbox) if sl.bbox else None,
                )
                for sl in suspicious_lines[:15]
            ],
            numeric_anomalies=[
                NumericAnomaly(
                    page=a.page + 1,
                    line_num=a.line_num + 1,
                    text=a.text,
                    value=a.value,
                    z_score=a.z_score,
                    reason=a.reason,
                    bbox=list(a.bbox) if a.bbox else None,
                )
                for a in numeric_anomalies[:10]
            ],
            fused_findings=[
                FusedFindingModel(
                    page=f.page + 1,
                    bbox=[float(v) for v in f.bbox],
                    confirming_layers=f.confirming_layers,
                    confidence=f.confidence,
                    score=f.score,
                    description=f.description,
                )
                for f in fused_findings
            ],
            fusion_stats=FusionStats(**fusion_stats),
            contradicted_findings=[
                ContradictedFindingModel(
                    page=c.page + 1,
                    bbox=[float(v) for v in c.bbox],
                    layer=c.layer,
                    original_description=c.original_description,
                    contradiction_rule=c.contradiction_rule,
                    contradicting_evidence=c.contradicting_evidence,
                    weight_reduction_points=c.weight_reduction_points,
                )
                for c in contradicted_findings
            ],
            text_stacking_findings=[
                TextStackingFindingModel(
                    page=f.page + 1,
                    bbox=[float(v) for v in f.bbox],
                    texts=list(f.texts),
                    overlap_fraction=round(f.overlap_fraction, 3),
                    confidence=f.confidence,
                    description=f.description,
                )
                for f in text_stacking_findings
            ],
            embedded_image_findings=[
                EmbeddedImageFindingModel(
                    page=f["page"] + 1,
                    bbox=[float(v) for v in f["bbox"]],
                    label=f["label"],
                    detail=f["text"],
                    confidence=f["score"],
                    evidence_check=f["evidence_check"],
                )
                for f in embedded_image_findings
            ],
            scanned_pixel_findings=[
                EmbeddedImageFindingModel(
                    page=f["page"] + 1,
                    bbox=[float(v) for v in f["bbox"]],
                    label=f["label"],
                    detail=f["text"],
                    confidence=f["score"],
                    evidence_check=f["evidence_check"],
                )
                for f in scanned_pixel_findings
            ],
            summary=build_summary(
                verdict=verdict_obj.verdict,
                combined_score=verdict_obj.combined_score,
                pdf_type=verdict_obj.pdf_type,
                source=meta_report.source.identified_name if meta_report else "Unknown",
                n_signals=len(verdict_obj.all_signals or []),
                n_suspicious_lines=len(suspicious_lines),
                n_numeric=len(numeric_anomalies),
            ),
            metadata=build_full_metadata(meta_report, total_pages, has_images),
        )

        # Cache analysis results
        analysis_id = str(uuid.uuid4())
        _analysis_cache[analysis_id] = {
            "pdf_path": pdf_path,
            "response": result,
            "suspicious_lines": core_results["content_report"].suspicious_lines if core_results["content_report"] else [],
            "numeric_anomalies": core_results["numeric_report"].anomalies if core_results["numeric_report"] else [],
            "ela_regions": core_results["ela_report"].regions if core_results["ela_report"] else [],
            "overlay_regions": core_results["pymupdf_report"].overlay_regions if core_results["pymupdf_report"] else [],
            "fused_findings": fused_findings,
            "text_stacking_findings": text_stacking_findings,
            "hidden_text_findings": hidden_text_findings,
            "embedded_image_findings": embedded_image_findings,
            "scanned_pixel_findings": scanned_pixel_findings,
            "hidden_text_report": hidden_text_report,
        }
        cached_pdf_path = pdf_path

        # Persist to the disk spool so /annotated-image and /hidden-text
        # survive a server restart (uvicorn --reload wipes the in-memory
        # dict on every source-file change). The spool owns the analyzed
        # PDF's lifetime from here: its pruning deletes the PDF together
        # with the entry, so in-memory eviction below only drops the dict
        # slot and must NOT unlink the file.
        persist_analysis(analysis_id, _analysis_cache[analysis_id])

        # Handle eviction for the in-memory session cache
        if len(_analysis_cache) > MAX_CACHED_ANALYSES:
            _analysis_cache.popitem(last=False)

        result_dict = result.dict()
        result_dict["analysis_id"] = analysis_id
        result_dict["total_pages"] = total_pages
        result_dict["incremental_updates"] = ela_report.incremental_updates if ela_report else {}
        return result_dict

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected analysis error: {type(e).__name__}: {e}"
        )
    finally:
        # Cleanup temporary files
        for path in [tmp_path]:
            try:
                if os.path.exists(path) and path != cached_pdf_path:
                    os.unlink(path)
            except Exception:
                pass
