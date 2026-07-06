"""
AI Review (Gemini) route — opt-in, supplementary, never part of /analyze.
Relocated verbatim out of main.py (Phase 2 folder reorganization) — no
logic changes; only the @app.post decorator became an APIRouter route and
imports were adjusted for the new package layout.

Every region Job B reviews is something the 6-layer engine ALREADY flagged
(fused findings, suspicious lines, numeric outliers, ELA regions, OCR word
anomalies, PyMuPDF overlays). Job C (below) is the one exception — it scans
full pages independently of what the 6 layers already found.
"""

import time

from fastapi import APIRouter, HTTPException

from models import ForensicResponse
from fusion.verdict_engine import WEIGHTS
from ai_review.gemini_advisor import GeminiAdvisor, GeminiNotConfigured, GeminiRequestError
from api.analysis_cache import _analysis_cache

router = APIRouter()

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


@router.post("/api/analysis/{analysis_id}/ai-review", tags=["AI Review"])
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
