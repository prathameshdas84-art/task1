"""
Document Forensics Engine — FastAPI Backend
Run: uvicorn main:app --reload --port 8000
Test: http://localhost:8000/docs
"""

import asyncio
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
from pydantic import BaseModel
from typing import Optional

from metadata_extractor import MetadataExtractor, PRODUCER_DB, _DB_PATH as _PRODUCER_DB_PATH
from content_analyzer import ContentAnalyzer
from ocr_analyzer import OCRAnalyzer
from numeric_analyzer import NumericAnalyzer
from ela_analyzer import ELAAnalyzer
from pymupdf_analyzer import PyMuPDFAnalyzer
from xref_analyzer import XrefAnalyzer
from verdict_engine import (
    combine, WEIGHTS, UNCERTAIN_BAND,
    CONFIDENCE_BASE, CONFIDENCE_DISTANCE_MULTIPLIER, CONFIDENCE_CAP,
)
from models import (
    ForensicResponse, HealthResponse,
    LayerScores, SuspiciousLine, NumericAnomaly, ConfidenceDetail,
    FullMetadata, FontDetail, PageDetail,
    FusedFindingModel, FusionStats,
)
from signal_fusion import SignalFusion, FusedFinding
from forensic_calculator import ForensicCalculator
from hidden_text_extractor import HiddenTextExtractor


class CalcRequest(BaseModel):
    col_a_index: int
    col_b_index: int
    operation: str = "+-"
    balance_col_index: int
    starting_balance: Optional[float] = None
    tolerance: float = 1.0
    page_filter: Optional[int] = None
    # User-placed "end of table" marker (1-indexed page + PDF-point Y
    # position) — rows at or after this point are excluded from the
    # calculation. Optional: when omitted, the full auto-detected table
    # region is used, unchanged from prior behavior.
    end_page: Optional[int] = None
    end_y: Optional[float] = None
    # User-placed "start of table" marker — mirrors end_page/end_y but as
    # the top boundary: rows strictly above this point are excluded, and
    # this row's own printed balance is trusted directly as the opening
    # balance (see ForensicCalculator.resolve_opening_balance).
    start_page: Optional[int] = None
    start_y: Optional[float] = None

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
# OWN dates against each other, content/OCR never compare against metadata.
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


# ── Forensic Calculator endpoints ─────────────────────────────────────────────

@app.get("/calculator/columns/{file_id}", tags=["Calculator"])
async def get_calculator_columns(file_id: str):
    """
    Return detected numeric columns for a previously analysed document.
    Each column includes its x-position, sample values, and a likely_type hint
    ('balance', 'transaction', or 'unknown').
    """
    if file_id not in _analysis_cache:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found. Please analyse the document first."
        )
    pdf_path = _analysis_cache[file_id]["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=410,
            detail="PDF no longer available — please re-upload and analyse."
        )
    try:
        columns = ForensicCalculator().extract_columns(pdf_path)
        return {"file_id": file_id, "columns": columns, "total_columns": len(columns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column extraction failed: {e}")


@app.post("/calculator/run/{file_id}", tags=["Calculator"])
async def run_calculator(file_id: str, request: CalcRequest):
    """
    Run arithmetic running-balance verification on a previously analysed document.
    Returns every row with its expected vs printed balance and a mismatch severity flag.
    """
    if file_id not in _analysis_cache:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found. Please analyse the document first."
        )
    if request.operation not in ("+", "-", "*", "/", "+-"):
        raise HTTPException(status_code=400, detail="Invalid operation")
    pdf_path = _analysis_cache[file_id]["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=410,
            detail="PDF no longer available — please re-upload and analyse."
        )
    try:
        result = ForensicCalculator().run_calculation(pdf_path, request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculator failed: {e}")


@app.post("/calculator/run-stream/{file_id}", tags=["Calculator"])
async def run_calculator_stream(file_id: str, request: CalcRequest):
    """
    Stream arithmetic running-balance verification row-by-row as SSE events.
    Event types: 'columns', 'row', 'done', 'error'.
    """
    if file_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if request.operation not in ("+", "-", "*", "/", "+-"):
        raise HTTPException(status_code=400, detail="Invalid operation")
    pdf_path = _analysis_cache[file_id]["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=410, detail="PDF no longer available.")

    async def generate():
        try:
            calc = ForensicCalculator()
            columns = calc.extract_columns(pdf_path)
            yield f"data: {json.dumps({'type': 'columns', 'data': columns})}\n\n"
            await asyncio.sleep(0)

            result = calc.run_calculation(pdf_path, request)

            if result.get("error"):
                yield f"data: {json.dumps({'type': 'error', 'data': result['error']})}\n\n"
                return

            rows = result.get("rows", [])
            total = len(rows)

            for row in rows:
                payload = {
                    "type":       "row",
                    "data":       row,
                    "current":    row["row_num"],
                    "total":      total,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.05)

            summary_payload = {
                "type":    "done",
                "data": {
                    "total_rows":                     result["total_rows"],
                    "mismatch_count":                 result["mismatch_count"],
                    "mismatch_rows":                  result["mismatch_rows"],
                    "opening_balance_method":         result["opening_balance_method"],
                    "opening_balance_confidence":     result["opening_balance_confidence"],
                    "opening_balance_anomaly":        result["opening_balance_anomaly"],
                    "opening_balance_anomaly_reason": result["opening_balance_anomaly_reason"],
                    "summary":                        result["summary"],
                },
            }
            yield f"data: {json.dumps(summary_payload)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


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
