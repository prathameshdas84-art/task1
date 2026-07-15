"""GET /annotated-image/{analysis_id} — renders a cached analysis's
page with labeled finding boxes drawn on it."""

import io
import os

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from api.analysis_cache import get_analysis

from .base import router


@router.get("/annotated-image/{analysis_id}", tags=["Forensics"])
async def get_annotated_image(analysis_id: str, page: int = 1):
    """
    Get an annotated page image for a previous analysis.

    - **analysis_id**: ID returned from /analyze
    - **page**: Page number (1-indexed, default: 1)

    Always draws every individual per-layer marking. Cross-validated fusion is
    surfaced separately in the UI Overview tab and never replaces these boxes.

    Returns PNG image with red boxes (content font/spacing anomalies),
    yellow boxes (numeric outliers), purple boxes (ELA flat/pasted patches),
    cyan/gold boxes (overlay layers), green boxes (embedded-image findings),
    and dashed magenta boxes (hidden/stacked text) drawn on suspicious
    regions. Each box's label states the specific finding.
    """
    cached = get_analysis(analysis_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found — it was never run "
                   f"on this server or has been evicted."
        )

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
                numeric_anomalies=cached["numeric_anomalies"],
                ela_regions=cached["ela_regions"],
                overlay_regions=cached.get("overlay_regions", []),
                age_days=age_days,
                fused_findings=cached.get("fused_findings", []),
                text_stacking_findings=cached.get("text_stacking_findings", []),
                hidden_text_findings=cached.get("hidden_text_findings", []),
                # Scanned-pixel findings share the embedded-image dict shape
                # (0-indexed page, point-space bbox, "label") — drawn through
                # the same path; each box's label carries the distinction.
                embedded_image_findings=(cached.get("embedded_image_findings", [])
                                         + cached.get("scanned_pixel_findings", [])),
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

