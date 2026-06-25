"""
Document Forensics Engine — FastAPI Backend
Run: uvicorn main:app --reload --port 8000
Test: http://localhost:8000/docs
"""

import io
import json
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from metadata_extractor import MetadataExtractor, PRODUCER_DB, _DB_PATH as _PRODUCER_DB_PATH
from content_analyzer import ContentAnalyzer
from ocr_analyzer import OCRAnalyzer
from numeric_analyzer import NumericAnalyzer
from ela_analyzer import ELAAnalyzer
from pymupdf_analyzer import PyMuPDFAnalyzer
from xref_analyzer import XrefAnalyzer
from verdict_engine import combine
from models import (
    ForensicResponse, HealthResponse,
    LayerScores, SuspiciousLine, NumericAnomaly, ConfidenceDetail,
    FullMetadata, FontDetail, PageDetail,
    FusedFindingModel, FusionStats,
)
from signal_fusion import SignalFusion, FusedFinding

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

    if layer_scores["ocr"] >= 20:
        active_layers.append("OCR")
        key_signals.append(f"OCR anomaly score {layer_scores['ocr']}/100")

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


# ── File converter ─────────────────────────────────────────────────────────────

def convert_to_pdf(file_path: str, original_filename: str) -> str:
    """
    Convert image or Word document to PDF for analysis.
    Returns path to PDF file (may be same as input if already PDF).
    """
    ext = Path(original_filename).suffix.lower()

    if ext == ".pdf":
        return file_path

    if ext in (".jpg", ".jpeg", ".png"):
        import fitz
        img_doc  = fitz.open()
        img_page = img_doc.new_page()
        img_page.insert_image(img_page.rect, filename=file_path)
        tmp_fd, converted = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        img_doc.save(converted)
        img_doc.close()
        os.unlink(file_path)
        return converted

    if ext in (".docx", ".doc"):
        import io
        import shutil
        import subprocess
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from docx import Document as DocxDocument
        from PIL import Image as PILImage

        # Try LibreOffice first
        libreoffice_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            "/usr/bin/soffice",
            "/usr/local/bin/soffice",
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "soffice",
        ]
        soffice = None
        for path in libreoffice_paths:
            if shutil.which(path) or os.path.exists(path):
                soffice = path
                break

        if soffice:
            out_dir = tempfile.mkdtemp()
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf",
                 "--outdir", out_dir, file_path],
                timeout=30, capture_output=True
            )
            converted_files = [f for f in os.listdir(out_dir) if f.endswith(".pdf")]
            if converted_files:
                converted = os.path.join(out_dir, converted_files[0])
                os.unlink(file_path)
                return converted

        # Fallback: python-docx + reportlab
        doc = DocxDocument(file_path)
        tmp_fd, converted = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        c = canvas.Canvas(converted, pagesize=A4)
        w, h = A4

        all_images = []
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    img_data = rel.target_part.blob
                    pil_img  = PILImage.open(io.BytesIO(img_data))
                    if pil_img.mode not in ("RGB", "L"):
                        pil_img = pil_img.convert("RGB")
                    all_images.append(pil_img)
                except Exception:
                    continue

        if all_images:
            for pil_img in all_images:
                iw, ih = pil_img.size
                scale  = min((w - 40) / iw, (h - 40) / ih)
                buf    = io.BytesIO()
                pil_img.save(buf, format="PNG", optimize=False)
                buf.seek(0)
                c.drawImage(ImageReader(buf),
                            (w - iw * scale) / 2, (h - ih * scale) / 2,
                            width=iw * scale, height=ih * scale)
                c.showPage()
        else:
            y = h - 50
            for para in doc.paragraphs:
                if para.text.strip():
                    c.setFont("Helvetica", 11)
                    c.drawString(20, y, para.text[:120])
                    y -= 20
                    if y < 50:
                        c.showPage()
                        y = h - 50

        c.save()
        os.unlink(file_path)
        return converted

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {ext}. Supported: PDF, JPG, PNG, DOCX"
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
        page_rotation=meta_report.page_rotation,
        raw=meta_report.raw_metadata,
        structure=meta_report.structure,
        suspicious_content=meta_report.suspicious_content,
        dimensions=meta_report.dimensions_full,
        dates=meta_report.dates_full,
        authenticity=meta_report.authenticity,
    )


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
        doc_tmp = fitz.open(pdf_path)
        total_pages = len(doc_tmp)
        has_images = any(len(p.get_images()) > 0 for p in doc_tmp)
        doc_tmp.close()

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
            raise HTTPException(
                status_code=500,
                detail="Core analysis layers failed. File may be corrupted."
            )

        # Combine verdict
        verdict_obj = combine(
            meta_report, content_report, ocr_report,
            numeric_report, ela_report, pymupdf_report, xref_report
        )

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
        from location_highlighter import LocationHighlighter

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
