"""
Analysis routes — /analyze, /annotated-image, /hidden-text. Relocated
verbatim out of main.py (Phase 2 folder reorganization) — no logic
changes; only the @app.* decorators became APIRouter routes and imports
were adjusted for the new package layout.
"""

import io
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from analyzers.metadata_extractor import MetadataExtractor
from analyzers.content_analyzer import ContentAnalyzer
from analyzers.ocr_analyzer import OCRAnalyzer
from analyzers.numeric_analyzer import NumericAnalyzer
from analyzers.ela_analyzer import ELAAnalyzer
from analyzers.pymupdf_analyzer import PyMuPDFAnalyzer
from analyzers.xref_analyzer import XrefAnalyzer
from fusion.verdict_engine import (
    combine, WEIGHTS, UNCERTAIN_BAND,
    CONFIDENCE_BASE, CONFIDENCE_DISTANCE_MULTIPLIER, CONFIDENCE_CAP,
)
from models import (
    ForensicResponse,
    LayerScores, SuspiciousLine, NumericAnomaly, ConfidenceDetail,
    FullMetadata, FontDetail, PageDetail,
    FusedFindingModel, FusionStats, ContradictedFindingModel,
)
from fusion.signal_fusion import SignalFusion, FusedFinding
from utils.hidden_text_extractor import HiddenTextExtractor
from utils.pdf_conversion import convert_to_pdf
from utils.report_builders import (
    _detect_merged_document, METADATA_MERGE_SCORE_MULTIPLIER,
    _cross_validate_timeline,
    build_confidence_detail, build_summary, build_full_metadata,
)
from api.analysis_cache import _analysis_cache, MAX_CACHED_ANALYSES

router = APIRouter()

# ── Supported file types ───────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".docx", ".doc"}

# ── Routes ─────────────────────────────────────────────────────────────────────

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
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    cached_pdf_path = None  # set once the analyzed PDF is stored in _analysis_cache
    try:
        os.close(tmp_fd)
        content = await file.read()
        file_size_kb = len(content) / 1024
        with open(tmp_path, "wb") as f:
            f.write(content)

        # For a directly-uploaded image, OCR the ORIGINAL pixels before
        # convert_to_pdf() stretches the image into a fixed-size PDF page
        # and re-rasterizes it at a different DPI (and deletes the
        # original file) — that round-trip loses fine detail this layer
        # can otherwise use directly.
        direct_image_ocr = None
        if ext in (".jpg", ".jpeg", ".png"):
            try:
                direct_image_ocr = OCRAnalyzer().analyze_image(tmp_path)
            except Exception:
                direct_image_ocr = None

        # Convert to PDF if needed
        pdf_path = convert_to_pdf(tmp_path, file.filename)

        # Run all 5 layers
        try:
            meta_report    = MetadataExtractor().extract(pdf_path)
        except Exception as e:
            meta_report    = None

        try:
            content_report = ContentAnalyzer().analyze(
                pdf_path,
                fonts=meta_report.fonts if meta_report else None,
            )
        except Exception as e:
            content_report = None

        import fitz
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
        # contribution before combine() ever sees it, so a multi-file
        # compilation's per-page style differences don't get weighted as
        # heavily as a single inconsistent document's would be.
        try:
            if _detect_merged_document(meta_report, total_pages):
                meta_report.anomaly_score = int(meta_report.anomaly_score * METADATA_MERGE_SCORE_MULTIPLIER)
                meta_report.anomalies.append(
                    "[INFO] Document appears to be a multi-file compilation — "
                    "metadata anomalies weighted down accordingly"
                )
        except Exception:
            pass

        try:
            ocr_report     = direct_image_ocr if direct_image_ocr is not None else OCRAnalyzer().analyze(pdf_path)
        except Exception as e:
            ocr_report     = None

        try:
            numeric_report = NumericAnalyzer().analyze(pdf_path)
        except Exception as e:
            numeric_report = None

        try:
            ela_report     = ELAAnalyzer().analyze(
                pdf_path,
                content_report.pdf_type if content_report else "native_text"
            )
        except Exception as e:
            ela_report     = None

        try:
            pymupdf_report = PyMuPDFAnalyzer().analyze(pdf_path)
        except Exception as e:
            pymupdf_report = None

        try:
            xref_report = XrefAnalyzer().analyze(pdf_path)
        except Exception as e:
            xref_report = None

        if not all([meta_report, content_report, ocr_report]):
            failed = [name for name, r in [("metadata", meta_report), ("content", content_report), ("ocr", ocr_report)] if not r]
            raise HTTPException(
                status_code=422,
                detail=f"Could not parse the uploaded file — the following core layers failed: {', '.join(failed)}. "
                       f"The file may be corrupt or password-protected. Try re-saving the PDF and re-uploading."
            )

        # Combine verdict
        verdict_obj = combine(
            meta_report, content_report, ocr_report,
            numeric_report, ela_report, pymupdf_report, xref_report
        )

        # Upgrade 2 — cross-layer timeline assertion: a backdated printed
        # date is invisible to every existing layer (none compare printed
        # text dates against metadata creation date), so it's applied here
        # as a post-hoc adjustment to the already-combined verdict rather
        # than as a new weighted layer in verdict_engine.combine().
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

        # Gatekeeper — an ORIGINAL verdict means no layer accumulated enough
        # evidence to call this document modified. Surfacing per-region
        # location arrays anyway (a borderline ELA block, a lone OCR word
        # anomaly that didn't move the score) draws false highlight boxes on
        # a document the engine itself just called clean. Every coordinate-
        # bearing list below is zeroed at the source so it can't reach
        # either the JSON response or the annotation cache that
        # /annotated-image reads from.
        is_clean = verdict_obj.verdict == "ORIGINAL"

        # Recompute authenticity now that the cross-layer verdict exists — the
        # metadata-only score produced inside extract() can't see content/OCR/
        # numeric/ELA/PyMuPDF findings. Fold the combined score and verdict in.
        meta_report.authenticity = MetadataExtractor()._compute_authenticity_score(
            meta_report,
            combined_score=verdict_obj.combined_score,
            verdict=verdict_obj.verdict,
        )

        processing_time = round(time.time() - start_time, 2)

        # Build layer scores dict
        layer_scores = {
            "metadata": verdict_obj.metadata_score,
            "content":  verdict_obj.content_score,
            "ocr":      verdict_obj.ocr_score,
            "numeric":  verdict_obj.numeric_score,
            "ela":      verdict_obj.ela_score,
            "pymupdf":  verdict_obj.pymupdf_score,
            "xref":     verdict_obj.xref_score,
        }

        # Build confidence detail — all gated to [] when is_clean, so every
        # consumer below (fusion, the response payload, and the annotation
        # cache) sees the same suppressed state.
        suspicious_lines = [] if is_clean else (content_report.suspicious_lines if content_report else [])
        numeric_anomalies = [] if is_clean else (numeric_report.anomalies if numeric_report else [])
        ela_regions = [] if is_clean else (ela_report.regions if ela_report else [])
        ocr_word_anomalies = [] if is_clean else (ocr_report.word_anomalies if ocr_report else [])
        overlay_regions = [] if is_clean else (pymupdf_report.overlay_regions if pymupdf_report else [])

        # Metadata is a document-level signal with no bbox, so on its own it can
        # never fuse spatially. Represent a flagged metadata report as a global
        # page-1 pseudo-finding so it can cross-validate any location-based
        # anomaly (a metadata edit trace + a visual/content anomaly is strong).
        metadata_findings = []
        if not is_clean and meta_report and meta_report.anomaly_score >= 30:
            metadata_findings.append({
                "layer": "metadata",
                "page": 0,  # 0-indexed; surfaces as page 1 in the response
                "score": meta_report.anomaly_score / 100,
                "text": "; ".join(meta_report.anomalies[:3]),
            })

        # Upgrade 2 — surface a backdating signal as its own cross-validated
        # pseudo-finding too, but only when the metadata layer independently
        # has anomalies of its own — a timeline mismatch alone (clean
        # metadata otherwise) isn't corroborated by anything to fuse with.
        if not is_clean and timeline_score > 0 and meta_report and meta_report.anomaly_score > 0:
            metadata_findings.append({
                "layer": "metadata",
                "page": 0,
                "score": min(1.0, timeline_score / 100),
                "text": "; ".join(timeline_signals[:3]),
            })

        # Cross-layer signal fusion — surface regions confirmed by 2+ layers as
        # an ADDITIONAL high-confidence view. This does not suppress the
        # per-layer markings drawn on the annotated image.
        fusion_engine = SignalFusion()
        fused_findings, fusion_stats = fusion_engine.fuse(
            suspicious_lines=suspicious_lines,
            numeric_anomalies=numeric_anomalies,
            ela_regions=ela_regions,
            ocr_regions=ocr_word_anomalies,
            overlay_regions=overlay_regions,
            metadata_findings=metadata_findings,
        )

        # Contradiction-aware fusion (Phase 1, additive) — a finding from one
        # layer that independent structural evidence from ANOTHER layer
        # undermines (currently: overlaps content_analyzer's own structural/
        # repeated-page-furniture classification) gets its layer's score
        # reduced, never zeroed. Same post-hoc-adjustment pattern as the
        # timeline assertion above, so verdict/confidence/combined_score stay
        # internally consistent. Rule 1 (metadata vs. structural fingerprint)
        # is intentionally NOT implemented — see signal_fusion.py docstring.
        contradicted_findings, contradiction_stats = fusion_engine.detect_contradictions(
            structural_line_locations=content_report.structural_line_locations if content_report else [],
            suspicious_lines=suspicious_lines,
            numeric_anomalies=numeric_anomalies,
            ela_regions=ela_regions,
            ocr_regions=ocr_word_anomalies,
            overlay_regions=overlay_regions,
        )
        if contradicted_findings:
            points_by_layer = {}
            for c in contradicted_findings:
                points_by_layer[c.layer] = points_by_layer.get(c.layer, 0) + c.weight_reduction_points

            layer_score_attr = {
                "content": "content_score", "numeric": "numeric_score",
                "ela": "ela_score", "ocr": "ocr_score", "pymupdf": "pymupdf_score",
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

                # layer_scores was built BEFORE this adjustment — refresh it
                # so confidence/response below reflect the adjusted scores.
                layer_scores["content"]  = verdict_obj.content_score
                layer_scores["numeric"]  = verdict_obj.numeric_score
                layer_scores["ela"]      = verdict_obj.ela_score
                layer_scores["ocr"]      = verdict_obj.ocr_score
                layer_scores["pymupdf"]  = verdict_obj.pymupdf_score

        confidence = build_confidence_detail(
            verdict=verdict_obj.verdict,
            combined_score=verdict_obj.combined_score,
            layer_scores=layer_scores,
            signals=verdict_obj.all_signals or [],
            suspicious_lines=len(suspicious_lines),
            numeric_anomalies=len(numeric_anomalies),
            effective_threshold=verdict_obj.effective_threshold,
        )

        # Build response
        result = ForensicResponse(
            verdict=verdict_obj.verdict,
            combined_score=verdict_obj.combined_score,
            confidence=confidence,
            filename=file.filename,
            file_size_kb=round(file_size_kb, 1),
            pdf_type=verdict_obj.pdf_type,
            document_source=meta_report.source.identified_name,
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
                    page=f.page + 1,  # 1-indexed for display
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
                    page=c.page + 1,  # 1-indexed for display
                    bbox=[float(v) for v in c.bbox],
                    layer=c.layer,
                    original_description=c.original_description,
                    contradiction_rule=c.contradiction_rule,
                    contradicting_evidence=c.contradicting_evidence,
                    weight_reduction_points=c.weight_reduction_points,
                )
                for c in contradicted_findings
            ],
            summary=build_summary(
                verdict=verdict_obj.verdict,
                combined_score=verdict_obj.combined_score,
                pdf_type=verdict_obj.pdf_type,
                source=meta_report.source.identified_name,
                n_signals=len(verdict_obj.all_signals or []),
                n_suspicious_lines=len(suspicious_lines),
                n_numeric=len(numeric_anomalies),
            ),
            metadata=build_full_metadata(meta_report, total_pages, has_images),
        )

        # Store result for image retrieval
        analysis_id = str(uuid.uuid4())
        _analysis_cache[analysis_id] = {
            "pdf_path": pdf_path,        # path to analyzed PDF
            "response": result,           # ForensicResponse object
            "suspicious_lines": suspicious_lines,
            "numeric_anomalies": numeric_anomalies,
            "ela_regions": ela_regions,
            "ocr_word_anomalies": ocr_word_anomalies,
            "overlay_regions": overlay_regions,
            "fused_findings": fused_findings,   # raw FusedFinding objects (0-indexed pages)
            # Exposed for Layer 7 (api/ai_review_routes.py) so its AI-adjusted
            # verdict label uses the SAME effective threshold (including any
            # dynamic backdating adjustment combine() applied), not a
            # re-derived approximation.
            "effective_threshold": verdict_obj.effective_threshold,
        }
        # This path is now owned by the cache — the finally block below
        # must not delete it (see cached_pdf_path tracking there).
        cached_pdf_path = pdf_path

        # Evict oldest if cache full
        if len(_analysis_cache) > MAX_CACHED_ANALYSES:
            oldest_id, oldest = next(iter(_analysis_cache.items()))
            try:
                if os.path.exists(oldest["pdf_path"]):
                    os.unlink(oldest["pdf_path"])
            except Exception:
                pass
            _analysis_cache.popitem(last=False)

        result_dict = result.dict()
        result_dict["analysis_id"] = analysis_id
        result_dict["total_pages"] = total_pages
        result_dict["ocr_word_anomalies"] = [
            {
                "page": a.page + 1,
                "word": a.word,
                "bbox": list(a.bbox),
                "anomaly_types": a.anomaly_types,
                "size_z": round(a.size_z, 2),
                "color_z": round(a.color_z, 2),
                "reason": a.reason,
            }
            for a in ocr_word_anomalies
        ]
        result_dict["ocr_stats"] = {
            "word_count": ocr_report.word_count if ocr_report else 0,
            "avg_font_size": ocr_report.avg_font_size if ocr_report else 0,
            "avg_color_brightness": ocr_report.avg_color_brightness if ocr_report else 0,
            "avg_confidence": ocr_report.avg_confidence if ocr_report else 0,
        }
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
        # Cleanup — handle both original and converted paths. The PDF path
        # that was just stored in _analysis_cache (cached_pdf_path) must
        # survive past this request — it's released later when evicted
        # from the cache (see eviction logic above), not here.
        for path in [tmp_path]:
            try:
                if os.path.exists(path) and path != cached_pdf_path:
                    os.unlink(path)
            except Exception:
                pass


@router.get("/annotated-image/{analysis_id}", tags=["Forensics"])
async def get_annotated_image(analysis_id: str, page: int = 1):
    """
    Get an annotated page image for a previous analysis.

    - **analysis_id**: ID returned from /analyze
    - **page**: Page number (1-indexed, default: 1)

    Always draws every individual per-layer marking. Cross-validated fusion is
    surfaced separately in the UI Overview tab and never replaces these boxes.

    Returns PNG image with red boxes (font anomalies), orange boxes (OCR
    confidence drops), yellow boxes (numeric outliers), purple boxes (ELA
    outliers), cyan boxes (white-rect overlays), and magenta boxes (image
    overlays) drawn on suspicious regions.
    """
    if analysis_id not in _analysis_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found. "
                   f"Results are cached for the session only."
        )

    cached = _analysis_cache[analysis_id]
    pdf_path = cached["pdf_path"]

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=410,
            detail="Annotated image no longer available — PDF was cleaned up."
        )

    try:
        from utils.location_highlighter import LocationHighlighter

        page_idx = page - 1  # convert to 0-indexed

        # Document's last-modification age — drives age-based box coloring and
        # the top-right "Modified: …" badge on the annotated page.
        age_days = None
        if cached.get("response") and cached["response"].metadata:
            age_days = cached["response"].metadata.edit_age_days

        # highlight_pages() renders and annotates EVERY anomalous page in
        # one call (it has to — boxes are computed from the full set of
        # findings, not per-page), so calling it again per page request
        # would redundantly redo that work N times for an N-page document.
        # Cache the result on the analysis the first time it's needed and
        # reuse it for every subsequent page of the SAME analysis_id.
        highlighted = cached.get("highlighted_pages")
        if highlighted is None:
            highlighter = LocationHighlighter(pdf_path)
            highlighted = highlighter.highlight_pages(
                suspicious_lines=cached["suspicious_lines"],
                ocr_word_anomalies=cached["ocr_word_anomalies"],
                numeric_anomalies=cached["numeric_anomalies"],
                ela_regions=cached["ela_regions"],
                overlay_regions=cached.get("overlay_regions", []),
                age_days=age_days,
                fused_findings=cached.get("fused_findings", []),
            )
            cached["highlighted_pages"] = highlighted

        if page_idx not in highlighted:
            # Return clean page if no anomalies on this page
            import fitz
            from PIL import Image as PILImage
            doc = fitz.open(pdf_path)
            if page_idx >= len(doc):
                raise HTTPException(
                    status_code=400,
                    detail=f"Page {page} does not exist in this document."
                )
            pix = doc[page_idx].get_pixmap(
                matrix=fitz.Matrix(150/72, 150/72),
                colorspace=fitz.csRGB
            )
            img = PILImage.frombytes("RGB", [pix.w, pix.h], pix.samples)
            doc.close()
        else:
            img = highlighted[page_idx]

        # Convert PIL image to PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="page_{page}.png"',
                "X-Analysis-ID": analysis_id,
                "X-Page": str(page),
                "X-Verdict": cached["response"].verdict,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")


# ── Hidden Text Recovery endpoint ──────────────────────────────────────────────

@router.get("/hidden-text/{file_id}", tags=["Forensics"])
async def get_hidden_text(file_id: str):
    """
    Attempt to recover original text that was covered up by a later edit
    (white-out rectangles, layered text overlaps, or incremental-update
    revisions). Read-only — never modifies the analyzed PDF.
    """
    if file_id not in _analysis_cache:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    pdf_path = _analysis_cache[file_id]["pdf_path"]

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=410,
            detail="PDF no longer available"
        )

    try:
        report = HiddenTextExtractor().analyze(pdf_path)
        return {
            "file_id": file_id,
            "total_found": report.total_found,
            "summary": report.recovery_summary,
            "conclusion": report.conclusion,
            "findings": [
                {
                    "page": f.page,
                    "method": f.method,
                    "original_text": f.original_text,
                    "covering_text": f.covering_text,
                    "bbox": f.bbox,
                    "confidence": f.confidence,
                    "description": f.description,
                    "field_type": f.field_type,
                    "plain_explanation": f.plain_explanation
                }
                for f in report.findings
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Hidden text extraction failed: {e}"
        )


