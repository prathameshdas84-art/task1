"""
Document Forensics Engine — FastAPI Backend
Run: start.bat (uses ..\.venv — do not run with a global/system Python;
     PyMuPDF is only correctly installed in .venv and fails at import
     time otherwise, before the app object is even created)
Test: http://localhost:8000/docs
"""

from dotenv import load_dotenv
load_dotenv()

import io
import json
import os
import re
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from analyzers.metadata_extractor import MetadataExtractor, PRODUCER_DB, _DB_PATH as _PRODUCER_DB_PATH
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
    ForensicResponse, HealthResponse,
    LayerScores, SuspiciousLine, NumericAnomaly, ConfidenceDetail,
    FullMetadata, FontDetail, PageDetail,
    FusedFindingModel, FusionStats, ContradictedFindingModel,
)
from fusion.signal_fusion import SignalFusion, FusedFinding
from utils.hidden_text_extractor import HiddenTextExtractor
from ai_review.gemini_advisor import GeminiAdvisor, GeminiNotConfigured, GeminiRequestError
from utils.pdf_conversion import convert_to_pdf
from utils.report_builders import (
    _detect_merged_document, METADATA_MERGE_SCORE_MULTIPLIER,
    _cross_validate_timeline,
    build_confidence_detail, build_summary, build_full_metadata,
)

# In-memory cache — stores last 100 analysis results + pdf paths so the
# annotated-image endpoint can re-render a page without re-uploading.
# OrderedDict used to evict oldest entries when limit reached.
MAX_CACHED_ANALYSES = 100
_analysis_cache: OrderedDict = OrderedDict()

# Top-level metadata (version/description) from producer_database.json —
# metadata_extractor.PRODUCER_DB only exposes the flat "producers" list
# (that's the shape _identify_source() needs), so these are read separately
# for the /producers endpoint.
try:
    with open(_PRODUCER_DB_PATH, "r", encoding="utf-8") as _f:
        _producer_db_raw = json.load(_f)
    PRODUCER_DB_VERSION     = _producer_db_raw.get("version", "unknown")
    PRODUCER_DB_DESCRIPTION = _producer_db_raw.get("description", "")
except Exception:
    PRODUCER_DB_VERSION     = "unknown"
    PRODUCER_DB_DESCRIPTION = ""

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document Forensics Engine",
    description="""
Detect tampering and modifications in PDF documents.

## How it works
Upload any PDF, image, or Word document and receive a detailed forensic analysis:
- **Layer 1 — Metadata**: Who created/modified the document and when
- **Layer 2 — Content**: Font consistency, spacing anomalies, CIDFont edit detection
- **Layer 3 — OCR**: Embedded vs visible text comparison, confidence analysis
- **Layer 4 — Numeric**: Statistical outlier detection in number fields
- **Layer 5 — ELA**: Error Level Analysis, shadow attack detection, signature validation

## Supported formats
PDF, JPG, JPEG, PNG, DOCX, DOC

## Verdict
- **MODIFIED**: Evidence of tampering detected
- **ORIGINAL**: No significant tampering evidence found
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supported file types ───────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".docx", ".doc"}

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if the API is running."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        layers=["metadata", "content", "ocr", "numeric", "ela", "pymupdf"],
    )


@app.get("/producers", tags=["System"])
async def list_producers():
    """
    Return the full producer/creator fingerprint database (from
    producer_database.json) so callers can see what sources are recognized
    and at what suspicion level, without reading the JSON file directly.
    """
    return {
        "version": PRODUCER_DB_VERSION,
        "description": PRODUCER_DB_DESCRIPTION,
        "count": len(PRODUCER_DB),
        "producers": PRODUCER_DB,
    }


@app.post("/analyze", response_model=ForensicResponse, tags=["Forensics"])
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


@app.get("/annotated-image/{analysis_id}", tags=["Forensics"])
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

@app.get("/hidden-text/{file_id}", tags=["Forensics"])
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


# ── AI Review (Gemini) — opt-in, supplementary, never part of /analyze ────────
# Every region Job B reviews is something the 6-layer engine ALREADY flagged
# (fused findings, suspicious lines, numeric outliers, ELA regions, OCR word
# anomalies, PyMuPDF overlays). Job C (below) is the one exception — it scans
# full pages independently of what the 6 layers already found.

MAX_AI_REVIEW_REGIONS     = 8   # cap regions sent to Gemini's vision endpoint per click — bounds latency/cost
AI_REVIEW_CROP_PADDING_PT = 6   # PDF points of padding added around each flagged bbox when cropping

MAX_AI_REVIEW_PAGES = 5           # cap pages sent to Job C's independent page scan per click
JOB_C_RENDER_DPI     = 150        # matches the DPI Job B crops / /annotated-image already use
GEMINI_INTER_CALL_DELAY_SECONDS = 1.5  # pause between the sequential Job A / B / C calls, on top of
                                        # gemini_advisor's own per-call 429 retry/backoff, to further
                                        # reduce rate-limit collisions across the 3 calls in one click

# ── Layer 7 (Gemini) scoring — feeds a SEPARATE combined_score_with_ai; never
# mutates the deterministic combined_score/layers computed in /analyze.
# Per-finding down-weights: each Job B region reclassified as a
# template-element, OR each Job C per_finding_verification entry Gemini
# marks "contradicted", subtracts its own downweight constant from THAT
# finding's own source layer score (floored at 0) — a targeted correction
# for a specific false positive, not a blanket layer override.
# Layer 7's own 0-100 "AI anomaly score" (Job C's additional_findings,
# confidence-weighted; Job B/C corroboration or contradiction of existing
# findings) is added on top scaled by LAYER7_WEIGHT — kept low/conservative
# since it's a supplementary, non-deterministic signal, not a 7th vote of
# equal weight to the deterministic layers.
JOB_B_TEMPLATE_DOWNWEIGHT_POINTS   = 8    # points subtracted per Job B template-element reclassification
JOB_C_CONTRADICTION_DOWNWEIGHT_POINTS = 8 # points subtracted per Job C "contradicted" verification
LAYER7_WEIGHT                    = 0.10  # Layer 7 score's contribution to combined_score_with_ai
JOB_C_CONFIDENCE_POINTS = {"low": 8, "medium": 18, "high": 32}  # per Job C additional_finding
JOB_B_CORROBORATION_BONUS   = 5   # Job B confirms an existing finding as a real possible-edit
JOB_B_CONTRADICTION_PENALTY = 5   # Job B reclassifies an existing finding as a template element
JOB_C_SUPPORTED_BONUS       = 5   # Job C verification supports an existing finding
JOB_C_CONTRADICTED_PENALTY  = 5   # Job C verification contradicts an existing finding

_KNOWN_LAYER_KEYS = {"metadata", "content", "ocr", "numeric", "ela", "pymupdf", "xref"}


def _gather_flagged_regions(cached: dict, max_regions: int = MAX_AI_REVIEW_REGIONS) -> list:
    """
    Fused (cross-validated) findings are added first since they're the
    highest-confidence subset; the rest only fill out the remaining cap.
    De-duplicates by (page, rounded bbox) since a fused finding and its own
    source-layer finding would otherwise both add nearly the same region.
    """
    regions = []

    def add(page, bbox, layer, description):
        if bbox and len(bbox) == 4:
            regions.append({
                "page": page,
                "bbox": tuple(float(v) for v in bbox),
                "layer": layer,
                "description": (description or "")[:200],
            })

    for f in cached.get("fused_findings", []):
        add(f.page, f.bbox, "fusion (" + "+".join(f.confirming_layers) + ")", f.description)
    for sl in cached.get("suspicious_lines", []):
        add(sl.page, sl.bbox, "content", sl.text)
    for na in cached.get("numeric_anomalies", []):
        add(na.page, na.bbox, "numeric", na.text)
    for er in cached.get("ela_regions", []):
        add(er.page, er.bbox, "ela", f"ELA anomaly (z-score {er.z_score:.1f})")
    for oa in cached.get("ocr_word_anomalies", []):
        add(oa.page, oa.bbox, "ocr", oa.word)
    for ov in cached.get("overlay_regions", []):
        add(ov.page, ov.bbox, "pymupdf", ov.reason)

    seen = set()
    deduped = []
    for r in regions:
        key = (r["page"], tuple(round(v) for v in r["bbox"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    return deduped[:max_regions]


def _crop_region_image(pdf_path: str, page_idx: int, bbox: tuple,
                        dpi: int = 150, padding_pt: float = AI_REVIEW_CROP_PADDING_PT):
    """Render ONLY the flagged bbox (plus a small padding margin) — never
    the whole page — and return PNG bytes, or None if the region is unusable."""
    import fitz
    doc = fitz.open(pdf_path)
    try:
        if page_idx < 0 or page_idx >= len(doc):
            return None
        page = doc[page_idx]
        x0, y0, x1, y1 = bbox
        clip = fitz.Rect(x0 - padding_pt, y0 - padding_pt, x1 + padding_pt, y1 + padding_pt) & page.rect
        if clip.is_empty or clip.width <= 0 or clip.height <= 0:
            return None
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip, colorspace=fitz.csRGB)
        return pix.tobytes("png")
    finally:
        doc.close()


def _render_page_image(pdf_path: str, page_idx: int, dpi: int = JOB_C_RENDER_DPI):
    """Render a FULL page (no crop) for Job C's independent scan — reuses the
    same rasterization approach as /annotated-image and _crop_region_image."""
    import fitz
    doc = fitz.open(pdf_path)
    try:
        if page_idx < 0 or page_idx >= len(doc):
            return None
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csRGB)
        return pix.tobytes("png")
    finally:
        doc.close()


def _build_job_c_analysis_summary(response: ForensicResponse, cached: dict) -> dict:
    """The engine's own full analysis JSON, sent to Job C ALONGSIDE the
    rendered pages so Gemini can cross-examine each finding against what's
    actually visible — a superset of Job A's narrower explainer summary,
    since Job C also needs ELA findings and metadata to verify against."""
    return {
        "verdict": response.verdict,
        "combined_score": response.combined_score,
        "confidence": response.confidence.dict(),
        "layers": response.layers.dict(),
        "signals": response.signals,
        "fused_findings": [f.dict() for f in response.fused_findings],
        "suspicious_lines": [s.dict() for s in response.suspicious_lines],
        "numeric_anomalies": [n.dict() for n in response.numeric_anomalies],
        "ela_findings": [
            {
                "page": r.page + 1,
                "bbox": list(r.bbox) if r.bbox else None,
                "z_score": r.z_score,
            }
            for r in cached.get("ela_regions", [])
        ],
        "metadata": {} if not response.metadata else {
            "producer": response.metadata.producer,
            "creator": response.metadata.creator,
            "source_name": response.metadata.source_name,
            "source_risk": response.metadata.source_risk,
            "xmp_mismatch": response.metadata.xmp_mismatch,
            "multiple_producers": response.metadata.multiple_producers,
            "is_recent_edit": response.metadata.is_recent_edit,
            "is_very_recent_edit": response.metadata.is_very_recent_edit,
            "edit_age_human": response.metadata.edit_age_human,
            "has_javascript": response.metadata.has_javascript,
            "has_embedded_files": response.metadata.has_embedded_files,
        },
        "summary": response.summary,
    }


def _extract_layer_names(source_layer: str) -> list:
    """'fusion (content+ela)' -> ['content','ela']; 'ela' -> ['ela']."""
    text = (source_layer or "").lower()
    if "(" in text and ")" in text:
        inner = text[text.index("(") + 1: text.index(")")]
        parts = inner.split("+")
    else:
        parts = [text]
    return [p.strip() for p in parts if p.strip() in _KNOWN_LAYER_KEYS]


def _compute_layer7_score(additional_findings: list, job_c_verifications: list, job_b_regions: list) -> int:
    """Layer 7's own 0-100 'AI anomaly score' — weighted by Job C's
    additional_findings (confidence-weighted; these are by construction NOT
    already reflected in the deterministic layers, since Gemini was given
    the full engine output and told to only report what it missed), plus
    whether Job B/Job C corroborated or contradicted the engine's own
    findings."""
    score = 0.0
    for f in additional_findings:
        score += JOB_C_CONFIDENCE_POINTS.get(f["confidence"], 8)
    for r in job_b_regions:
        if r["label"] == "possible-edit":
            score += JOB_B_CORROBORATION_BONUS
        elif r["label"] == "template-element":
            score -= JOB_B_CONTRADICTION_PENALTY
    for v in job_c_verifications:
        if v["gemini_verdict"] == "supported":
            score += JOB_C_SUPPORTED_BONUS
        elif v["gemini_verdict"] == "contradicted":
            score -= JOB_C_CONTRADICTED_PENALTY
    return int(max(0, min(100, round(score))))


def _compute_combined_score_with_ai(response: ForensicResponse, job_b_regions: list,
                                     job_c_verifications: list, layer7_score: int) -> dict:
    """Recomputes a SEPARATE, AI-adjusted score. Never mutates response.layers
    or response.combined_score — those stay the untouched deterministic
    baseline used by every other endpoint/report. Uses the same WEIGHTS
    dict verdict_engine.combine() uses (imported read-only, never modified
    here) so the two scores are directly comparable. Two independent
    per-finding downweight sources feed the same `adjusted` layer scores:
    Job B template-element reclassifications, and Job C "contradicted"
    verifications — both floor at 0 per layer, neither is a blanket
    layer override."""
    adjusted = dict(response.layers.dict())
    downweight_applied = []

    for r in job_b_regions:
        if r["label"] != "template-element":
            continue
        for layer_name in _extract_layer_names(r["source_layer"]):
            before = adjusted.get(layer_name, 0)
            adjusted[layer_name] = max(0, before - JOB_B_TEMPLATE_DOWNWEIGHT_POINTS)
            downweight_applied.append({
                "source": "job_b",
                "layer": layer_name,
                "finding_description": r["engine_description"],
                "points": before - adjusted[layer_name],
            })

    for v in job_c_verifications:
        if v["gemini_verdict"] != "contradicted":
            continue
        for layer_name in _extract_layer_names(v["layer"]):
            before = adjusted.get(layer_name, 0)
            adjusted[layer_name] = max(0, before - JOB_C_CONTRADICTION_DOWNWEIGHT_POINTS)
            downweight_applied.append({
                "source": "job_c",
                "layer": layer_name,
                "finding_description": v["engine_finding"],
                "points": before - adjusted[layer_name],
            })

    weights = WEIGHTS.get(response.pdf_type, WEIGHTS["native_text"])
    adjusted_combined = sum(adjusted.get(l, 0) * weights.get(l, 0) for l in _KNOWN_LAYER_KEYS)
    combined_score_with_ai = adjusted_combined + layer7_score * LAYER7_WEIGHT
    combined_score_with_ai = round(max(0.0, min(100.0, combined_score_with_ai)), 1)

    return {
        "combined_score_with_ai": combined_score_with_ai,
        "downweight_applied": downweight_applied,
        "layer7_weight": LAYER7_WEIGHT,
        "job_b_template_downweight_points": JOB_B_TEMPLATE_DOWNWEIGHT_POINTS,
        "job_c_contradiction_downweight_points": JOB_C_CONTRADICTION_DOWNWEIGHT_POINTS,
    }


@app.post("/api/analysis/{analysis_id}/ai-review", tags=["AI Review"])
async def ai_review(analysis_id: str):
    """
    Opt-in supplementary AI review — ONLY invoked when the user clicks
    "Ask AI" in the UI. Never runs during /analyze, never mutates the cached
    ForensicResponse/combined_score/verdict. Reads the already-cached
    analysis result and runs:
      Job A — plain-English explanation of the existing verdict.
      Job B — template-vs-possible-edit labels for regions the engine
              already flagged (one batched Gemini call, not one per region).
      Job C — genuine cross-examination: Gemini gets BOTH the rendered page
              images AND the engine's own full analysis JSON in one call,
              independently verifies each engine finding as supported/
              contradicted/unverifiable against the actual document,
              surfaces anything the engine missed, and gives its own
              overall assessment (never auto-applied to the verdict —
              surfaced as a flagged disagreement for human review instead).
    Layer 7 (Gemini) score + a SEPARATE combined_score_with_ai are computed
    from Job B/C output — combined_score itself is never touched. Fails
    gracefully per-job (API key missing, network error, rate limit) without
    ever raising past this endpoint or affecting the cached verdict.

    Cached per analysis_id: a second call returns the exact same result
    instead of re-calling Gemini, so combined_score_with_ai stays
    reproducible for an already-reviewed document and no extra API calls/
    cost are incurred.
    """
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    cached = _analysis_cache[analysis_id]

    if "ai_review" in cached:
        return {**cached["ai_review"], "from_cache": True}

    pdf_path = cached["pdf_path"]
    response: ForensicResponse = cached["response"]

    try:
        advisor = GeminiAdvisor()
    except GeminiNotConfigured as e:
        return {
            "available": False,
            "reason": str(e),
            "explanation": None,
            "explanation_prompt": None,
            "regions": [],
            "per_finding_verification": [],
            "additional_findings": [],
            "overall_assessment": None,
            "ai_disagreement_flag": False,
            "ai_disagreement_message": None,
            "layer7_score": 0,
            "combined_score": response.combined_score,
            "combined_score_with_ai": None,
        }

    # Job A — narrow input: only the fields named in scope (layers, signals,
    # fused_findings, suspicious_lines, numeric_anomalies, summary), not the
    # full raw API response (metadata/ocr_stats/etc. aren't relevant to
    # "explain the verdict" and would just bloat the prompt).
    analysis_summary = {
        "verdict": response.verdict,
        "combined_score": response.combined_score,
        "confidence": response.confidence.dict(),
        "layers": response.layers.dict(),
        "signals": response.signals,
        "fused_findings": [f.dict() for f in response.fused_findings],
        "suspicious_lines": [s.dict() for s in response.suspicious_lines],
        "numeric_anomalies": [n.dict() for n in response.numeric_anomalies],
        "summary": response.summary,
    }

    explanation, explanation_prompt, explanation_error = None, None, None
    try:
        explanation, explanation_prompt = advisor.explain_report(analysis_summary)
    except GeminiRequestError as e:
        explanation_error = str(e)
    except Exception as e:
        explanation_error = f"Unexpected error generating explanation: {e}"

    time.sleep(GEMINI_INTER_CALL_DELAY_SECONDS)

    # Job B — crop every already-flagged region, then label them ALL in ONE
    # batched Gemini call (instead of one HTTP call per region).
    regions_out = []
    regions_error = None
    try:
        flagged_regions, crop_bytes_list = [], []
        for region in _gather_flagged_regions(cached):
            crop_bytes = _crop_region_image(pdf_path, region["page"], region["bbox"])
            if not crop_bytes:
                continue
            flagged_regions.append(region)
            crop_bytes_list.append(crop_bytes)

        if crop_bytes_list:
            label_results = advisor.label_regions_batch(crop_bytes_list)
            for region, label_result in zip(flagged_regions, label_results):
                regions_out.append({
                    "page": region["page"] + 1,  # 1-indexed for display
                    "bbox": list(region["bbox"]),
                    "source_layer": region["layer"],
                    "engine_description": region["description"],
                    "label": label_result["label"],
                    "reasoning": label_result["reasoning"],
                })
    except Exception as e:
        regions_error = f"Unexpected error labeling regions: {e}"

    time.sleep(GEMINI_INTER_CALL_DELAY_SECONDS)

    # Job C — cross-examine the engine's OWN findings against the actual
    # rendered pages in ONE combined call (full engine JSON + page images
    # together), rather than assessing pages in isolation. Deliberately one
    # call, not per-page: verifying claims like "this header repeats across
    # pages" needs more than one page in view — so it uses a longer timeout
    # (JOB_C_REQUEST_TIMEOUT_SECONDS) and the same retry/backoff as every
    # other call to manage that.
    per_finding_verification = []
    additional_findings_out = []
    overall_assessment = None
    job_c_error = None
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        doc.close()

        page_images = []
        for page_idx in range(min(n_pages, MAX_AI_REVIEW_PAGES)):
            img_bytes = _render_page_image(pdf_path, page_idx, dpi=JOB_C_RENDER_DPI)
            if img_bytes:
                page_images.append((page_idx + 1, img_bytes))

        if page_images:
            job_c_summary = _build_job_c_analysis_summary(response, cached)
            cross_exam = advisor.cross_examine_findings(page_images, job_c_summary)
            per_finding_verification = cross_exam["per_finding_verification"]
            overall_assessment = cross_exam["overall_assessment"]

            px_to_pt = 72.0 / JOB_C_RENDER_DPI
            for f in cross_exam["additional_findings"]:
                bbox_pt = None
                if f["bbox_px"]:
                    x0, y0, x1, y1 = f["bbox_px"]
                    bbox_pt = [round(v * px_to_pt, 1) for v in (x0, y0, x1, y1)]
                additional_findings_out.append({
                    "page": f["page"],
                    "bbox": bbox_pt,
                    "description": f["description"],
                    "confidence": f["confidence"],
                    "not_flagged_by_engine": True,
                })
    except GeminiRequestError as e:
        job_c_error = f"AI cross-examination unavailable — {e}"
    except Exception as e:
        job_c_error = f"Unexpected error during AI cross-examination: {e}"

    layer7_score = _compute_layer7_score(additional_findings_out, per_finding_verification, regions_out)
    score_calc = _compute_combined_score_with_ai(response, regions_out, per_finding_verification, layer7_score)

    # overall_assessment is NEVER used to auto-resolve/flip the verdict —
    # an explicit disagreement is only surfaced as a flag for a human to
    # look at. "inconclusive" is deliberately NOT treated as a disagreement
    # (it's a softer "can't confirm from visuals alone", not a contradiction).
    ai_disagreement_flag = bool(overall_assessment and overall_assessment["agrees_with_engine_verdict"] is False)
    ai_disagreement_message = (
        "⚠️ AI Review disagrees with the deterministic verdict — human review strongly recommended."
        if ai_disagreement_flag else None
    )

    result = {
        "available": True,
        "explanation": explanation,
        "explanation_prompt": explanation_prompt,
        "explanation_error": explanation_error,
        "regions": regions_out,
        "regions_error": regions_error,
        "per_finding_verification": per_finding_verification,
        "additional_findings": additional_findings_out,
        "overall_assessment": overall_assessment,
        "ai_disagreement_flag": ai_disagreement_flag,
        "ai_disagreement_message": ai_disagreement_message,
        "job_c_error": job_c_error,
        "layer7_score": layer7_score,
        "layer7_weight": score_calc["layer7_weight"],
        "job_b_template_downweight_points": score_calc["job_b_template_downweight_points"],
        "job_c_contradiction_downweight_points": score_calc["job_c_contradiction_downweight_points"],
        "downweight_applied": score_calc["downweight_applied"],
        "combined_score": response.combined_score,
        "combined_score_with_ai": score_calc["combined_score_with_ai"],
        "from_cache": False,
    }

    cached["ai_review"] = result
    return result
