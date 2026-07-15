"""GET /hidden-text/{file_id} — returns the recovered hidden/covered
text report for a cached analysis."""

import os

from fastapi import HTTPException

from api.analysis_cache import get_analysis
from utils.hidden_text_extractor import HiddenTextExtractor

from .base import router


@router.get("/hidden-text/{file_id}", tags=["Forensics"])
async def get_hidden_text(file_id: str):
    """
    Attempt to recover original text that was covered up by a later edit
    (white-out rectangles, layered text overlaps, or incremental-update
    revisions). Read-only — never modifies the analyzed PDF.
    """
    cached = get_analysis(file_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    pdf_path = cached["pdf_path"]

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=410,
            detail="PDF no longer available"
        )

    try:
        # Reuse the report already computed during /analyze when available
        # (identical, ungated output) so the recovery methods don't run twice;
        # fall back to computing on demand for any cache entry without it.
        report = cached.get("hidden_text_report")
        if report is None:
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
                    "plain_explanation": f.plain_explanation,
                    # missing = removed with nothing visible put in its place;
                    # replaced = different visible text put over the original.
                    "replacement_type": f.replacement_type,
                }
                for f in report.findings
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Hidden text extraction failed: {e}"
        )


