"""
Image-document analysis routes — POST /analyze-image plus the evidence
image endpoints for that pipeline.

This is the SEPARATE upload path for direct JPG/PNG uploads (photographed
IDs, certificates, stamped/signed documents) targeted by
analyzers/image_document_analyzer.py. It never touches the PDF pipeline:

  * PDFs — including scanned/mixed PDFs with embedded raster pages — are
    rejected here with a pointer to POST /analyze. ela_analyzer.py already
    runs the image-based checks (noise consistency, digital erasure) for
    raster PDF pages inside that pipeline; routing them here as well would
    silently double-count the same smoothing signal across two detectors.
  * The existing /analyze route continues to accept JPG/PNG by converting
    to PDF, exactly as before — this endpoint is additive, not a
    replacement.

Output telemetry matches the PDF pipeline's: findings carry page/bbox and
flow through the SAME SignalFusion agreement logic (via fuse()'s additive
extra_findings input), and the verdict uses verdict_engine's constants
with the additive WEIGHTS["image_document"] entry.
"""

import io
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from analyzers.image_document_analyzer import (
    ImageDocumentAnalyzer, normalize_for_fusion,
)
from fusion.signal_fusion import SignalFusion
from fusion.verdict_engine import (
    WEIGHTS, THRESHOLDS, THRESHOLD, UNCERTAIN_BAND,
    CONFIDENCE_BASE, CONFIDENCE_DISTANCE_MULTIPLIER, CONFIDENCE_CAP,
)

router = APIRouter()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Separate cache from api/analysis_cache.py on purpose: that cache's
# consumers (/annotated-image, /hidden-text) all read entry["pdf_path"]
# and would crash on an image entry. Image analyses are only ever read
# back by the two endpoints in this module.
MAX_CACHED_IMAGE_ANALYSES = 50
_image_analysis_cache: OrderedDict = OrderedDict()

# Box colors per evidence check for the annotated overlay (BGR).
_CHECK_COLORS = {
    "check1_local_variance": (0, 0, 255),     # red — inpaint smoothing
    "check5_edge_sharpness": (0, 165, 255),   # orange — overlay text edges
    "check6_copy_move":      (255, 0, 255),   # magenta — clone consensus
    "check8_stamp_texture":  (255, 0, 0),     # blue — flat ink fill
    "check9_stamp_boundary": (0, 255, 255),   # yellow — cutout boundary
}


@router.post("/analyze-image", tags=["Image Forensics"])
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a DIRECT image upload (JPG/PNG) for signs of tampering:
    AI-inpainting removal, social-media-style text/sticker overlays, and
    pasted stamps/signatures.

    PDFs belong on POST /analyze — scanned/mixed PDFs keep their existing
    ELA-based image checks there (see module docstring for why).
    """
    start_time = time.time()

    ext = Path(file.filename).suffix.lower()
    if ext == ".pdf":
        raise HTTPException(
            status_code=400,
            detail="PDFs (including scanned PDFs) are analyzed by POST /analyze "
                   "— this endpoint is only for direct JPG/PNG uploads.",
        )
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. "
                   f"Supported here: {', '.join(sorted(IMAGE_EXTENSIONS))}",
        )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        os.close(tmp_fd)
        content = await file.read()
        file_size_kb = len(content) / 1024
        with open(tmp_path, "wb") as f:
            f.write(content)

        try:
            report = ImageDocumentAnalyzer().analyze(tmp_path)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Could not analyze the uploaded image: "
                       f"{type(e).__name__}: {e}",
            )

        # ── Verdict — same constants/formula as the PDF pipeline ──────
        weights = WEIGHTS["image_document"]
        combined = report.anomaly_score * weights["image_forensics"]
        effective_threshold = THRESHOLDS.get("image_document", THRESHOLD)
        verdict = "MODIFIED" if combined >= effective_threshold else "ORIGINAL"
        if abs(combined - effective_threshold) <= UNCERTAIN_BAND:
            verdict = "UNCERTAIN"
        distance = abs(combined - effective_threshold)
        confidence = min(
            CONFIDENCE_CAP,
            CONFIDENCE_BASE + int(distance * CONFIDENCE_DISTANCE_MULTIPLIER),
        )

        # ── Clean-verdict gate — same convention as /analyze: an ORIGINAL
        # verdict means the engine calls this image clean, so per-region
        # boxes are suppressed at the source (they'd draw false highlights).
        is_clean = verdict == "ORIGINAL"
        visible_anomalies = [] if is_clean else report.anomalies

        # ── Fusion — image checks enter as their own layers ────────────
        fused_findings, fusion_stats = SignalFusion().fuse(
            extra_findings=normalize_for_fusion(report) if not is_clean else [],
        )

        # ── Annotated overlay (per-check colored boxes) ────────────────
        annotated_png = None
        try:
            bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                for a in visible_anomalies:
                    x, y, w, h = a.bbox
                    color = _CHECK_COLORS.get(a.evidence_check, (0, 0, 255))
                    cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(bgr, a.evidence_check.replace("check", "C")[:20],
                                (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, color, 1, cv2.LINE_AA)
                ok, buf = cv2.imencode(".png", bgr)
                if ok:
                    annotated_png = buf.tobytes()
        except Exception:
            annotated_png = None

        analysis_id = str(uuid.uuid4())
        _image_analysis_cache[analysis_id] = {
            "heatmap_png": report.heatmap_png,
            "annotated_png": annotated_png,
            "verdict": verdict,
        }
        if len(_image_analysis_cache) > MAX_CACHED_IMAGE_ANALYSES:
            _image_analysis_cache.popitem(last=False)

        return {
            "verdict": verdict,
            "combined_score": round(combined, 1),
            "confidence": confidence,
            "effective_threshold": effective_threshold,
            "filename": file.filename,
            "file_size_kb": round(file_size_kb, 1),
            "document_class": "image_document",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "layers": {"image_forensics": report.anomaly_score},
            "signals": report.signals,
            "image_forensics": {
                "is_born_digital": report.is_born_digital,
                "jpeg_history_detected": report.jpeg_history_detected,
                "compression_history": report.compression_history,
                "stamp_detected": report.stamp_detected,
                "signature_detected": report.signature_detected,
                "anomalies": [
                    {
                        "type": a.type,
                        "page": a.page,
                        "bbox": [int(v) for v in a.bbox],   # [x, y, w, h]
                        "confidence": a.confidence,
                        "evidence_check": a.evidence_check,
                        "detail": a.detail,
                    }
                    for a in visible_anomalies
                ],
                "not_implemented": report.not_implemented,
                "metrics": report.metrics,
            },
            "fused_findings": [
                {
                    "page": f.page + 1,
                    "bbox": [float(v) for v in f.bbox],  # (x0,y0,x1,y1) px
                    "confirming_layers": f.confirming_layers,
                    "confidence": f.confidence,
                    "score": f.score,
                    "description": f.description,
                }
                for f in fused_findings
            ],
            "fusion_stats": fusion_stats,
            "analysis_id": analysis_id,
            "heatmap_url": f"/image-heatmap/{analysis_id}",
            "annotated_url": f"/image-annotated/{analysis_id}",
        }
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def _cached_png(analysis_id: str, key: str, label: str) -> StreamingResponse:
    if analysis_id not in _image_analysis_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Image analysis {analysis_id} not found. "
                   f"Results are cached for the session only.",
        )
    png = _image_analysis_cache[analysis_id].get(key)
    if not png:
        raise HTTPException(status_code=410, detail=f"{label} not available.")
    return StreamingResponse(
        io.BytesIO(png),
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{key}.png"',
            "X-Analysis-ID": analysis_id,
            "X-Verdict": _image_analysis_cache[analysis_id]["verdict"],
        },
    )


@router.get("/image-heatmap/{analysis_id}", tags=["Image Forensics"])
async def get_image_heatmap(analysis_id: str):
    """Check 10's near-white micro-contrast heatmap (COLORMAP_JET).
    Display-only evidence for a human reviewer — never a scoring input."""
    return _cached_png(analysis_id, "heatmap_png", "Heatmap")


@router.get("/image-annotated/{analysis_id}", tags=["Image Forensics"])
async def get_image_annotated(analysis_id: str):
    """The uploaded image with per-check colored anomaly boxes drawn on."""
    return _cached_png(analysis_id, "annotated_png", "Annotated image")
