"""
AI Review route — opt-in, supplementary, never part of /analyze. Backed by
one of two interchangeable providers, selected via the AI_REVIEW_PROVIDER
env var ("gemini" | "nvidia", default "gemini" — unset/existing setups get
byte-identical behavior to before NVIDIA support existed). Both providers
implement the exact same review_and_explain/independent_scan interface
(see ai_review/gemini_advisor.py's module docstring), so nothing below this
line needs to branch on which one is active except the one selection point.

The review is AT MOST two provider calls per click (was always three):

  1. Job C (independent full-page scan) — CONDITIONAL. Runs ONLY when
     _compute_needs_independent_scan() says the engine's own findings lack
     a confident location (see that function) — a plain rule check on the
     deterministic results already in hand, never an AI call. When the
     engine already has well-localized findings, this call — the expensive
     one, full-page renders + the whole analysis JSON — is skipped
     entirely. It runs FIRST (when it runs) so its results feed the one
     explanation below.
  2. Merged region-review + explanation — ALWAYS runs; former Job A and
     Job B combined into ONE request. Labels the engine's already-flagged
     region crops template-element/possible-edit/uncertain AND writes the
     plain-English explanation in the same reply, synthesizing engine
     findings + its own region verdicts + Job C's results when the scan
     ran — one narrative, no second round-trip.

So: confident-location path = 1 call, needs-scan path = 2 calls, vs the
old always-3. Which path was taken is logged and returned (scan_mode /
needs_independent_scan / ai_calls_made) so it's auditable.

Every region the merged call reviews is something the engine ALREADY
flagged (fused findings, suspicious lines, numeric outliers, ELA
regions, PyMuPDF overlays). Job C is the one exception
— it scans full pages independently of what the 6 layers already found.

The response also carries merged_findings — copies of the main report's
findings lists with per-finding ai_status (confirmed/contradicted/
unreviewed) and any Job C discoveries appended in the same list tagged
ai_discovered — so the frontend re-renders the SAME report in place rather
than showing a second parallel panel. The deterministic combined_score/
verdict/response are never mutated; they stay cached (and returned) as the
audit baseline.
"""

import logging
import os
import time

from fastapi import APIRouter, HTTPException

from models import ForensicResponse
from fusion.verdict_engine import WEIGHTS, UNCERTAIN_BAND, THRESHOLD
from analyzers.ela_analyzer import Z_THRESHOLD as ELA_Z_THRESHOLD
from utils.ai_retry import AIProviderNotConfigured, AIProviderRequestError
from ai_review.gemini_advisor import GeminiAdvisor
from ai_review.nvidia_advisor import NvidiaAdvisor
from api.analysis_cache import _analysis_cache

logger = logging.getLogger("document_forensics")

router = APIRouter()

# "gemini" | "nvidia" — Gemini is the default so unset/existing .env files
# behave exactly as before NVIDIA support existed. Read once at import time
# (not per-request) so it's obvious which provider is active for the whole
# process's lifetime, matching how GEMINI_API_KEY/NVIDIA_API_KEY are also
# read once at their own module's import time.
AI_REVIEW_PROVIDER = os.environ.get("AI_REVIEW_PROVIDER", "gemini").strip().lower()


def _make_advisor():
    """Instantiates the configured provider's advisor. Raises
    AIProviderNotConfigured (shared, provider-agnostic) if that provider's
    API key is missing — same graceful-degradation path regardless of
    which provider was selected."""
    if AI_REVIEW_PROVIDER == "nvidia":
        return NvidiaAdvisor(), "NVIDIA NIM"
    return GeminiAdvisor(), "Gemini"

MAX_AI_REVIEW_REGIONS     = 8   # cap regions sent to the provider's vision endpoint per click — bounds latency/cost
AI_REVIEW_CROP_PADDING_PT = 6   # PDF points of padding added around each flagged bbox when cropping

MAX_AI_REVIEW_PAGES = 5           # cap pages sent to Job C's independent page scan per click
JOB_C_RENDER_DPI     = 150        # matches the DPI region crops / /annotated-image already use
AI_REVIEW_INTER_CALL_DELAY_SECONDS = 1.5  # pause between Job C and the merged call (the only case
                                           # with two sequential calls now), on top of the active
                                           # provider's own per-call 429 retry/backoff, to reduce
                                           # rate-limit collisions — applies to either provider

# ── needs_independent_scan rule check — a PLAIN rule check on the
# deterministic results already in hand (never an AI call): Job C only runs
# when the engine's own findings genuinely lack a confident location.
# ELA regions all sit barely over the detection threshold → the bboxes
# exist but are low-confidence localizations. A combined score hovering
# near the modification threshold without a single HIGH-confidence
# cross-validated finding → the layers collectively suggest "something is
# off" without one strong, well-localized finding.
ELA_LOW_CONFIDENCE_Z_MARGIN = 1.0                 # z within [Z_THRESHOLD, Z_THRESHOLD+1) = barely detected
NEAR_THRESHOLD_BAND         = UNCERTAIN_BAND * 2  # "close to the threshold" — wider than the UNCERTAIN band itself

# ── Layer 7 (AI Review) scoring — feeds a SEPARATE combined_score_with_ai;
# never mutates the deterministic combined_score/layers computed in
# /analyze. Provider-agnostic: the same scoring math applies whichever of
# Gemini/NVIDIA NIM is active, since both produce the same result shapes.
# Per-finding down-weights: each merged-call region reclassified as a
# template-element, OR each Job C per_finding_verification entry the model
# marks "contradicted", subtracts its own downweight constant from THAT
# finding's own source layer score (floored at 0) — a targeted correction
# for a specific false positive, not a blanket layer override.
# Layer 7's own 0-100 "AI anomaly score" (Job C's additional_findings,
# confidence-weighted; corroboration or contradiction of existing findings)
# is added on top scaled by LAYER7_WEIGHT — kept low/conservative since
# it's a supplementary, non-deterministic signal, not a 7th vote of equal
# weight to the deterministic layers.
JOB_B_TEMPLATE_DOWNWEIGHT_POINTS   = 8    # points subtracted per region-review template-element reclassification
JOB_C_CONTRADICTION_DOWNWEIGHT_POINTS = 8 # points subtracted per Job C "contradicted" verification
LAYER7_WEIGHT                    = 0.10  # Layer 7 score's contribution to combined_score_with_ai
JOB_C_CONFIDENCE_POINTS = {"low": 8, "medium": 18, "high": 32}  # per Job C additional_finding
JOB_B_CORROBORATION_BONUS   = 5   # region review confirms an existing finding as a real possible-edit
JOB_B_CONTRADICTION_PENALTY = 5   # region review reclassifies an existing finding as a template element
JOB_C_SUPPORTED_BONUS       = 5   # Job C verification supports an existing finding
JOB_C_CONTRADICTED_PENALTY  = 5   # Job C verification contradicts an existing finding

_KNOWN_LAYER_KEYS = {"metadata", "content", "numeric", "ela", "pymupdf", "xref"}


def _compute_needs_independent_scan(cached: dict, response: ForensicResponse) -> tuple:
    """
    The Part-1 rule check: True when ANY of the following holds — i.e. the
    deterministic layers left the report without a confident location and
    the expensive independent full-page scan (Job C) can genuinely help.
    Returns (needs_independent_scan, human-readable reasons) — the reasons
    list is returned in the response for auditability.

      1. verdict == "UNCERTAIN" — the engine itself couldn't call it.
      2. Content/numeric findings exist with NO bbox at all — the finding
         is real but literally cannot be pointed at on the page.
      3. ELA findings exist but ALL sit barely over the detection
         threshold (z < Z_THRESHOLD + ELA_LOW_CONFIDENCE_Z_MARGIN) — the
         layer localized something, but only weakly. One strong region
         anywhere in the layer means it DID localize confidently, so
         "all", not "any".
      4. combined_score sits within NEAR_THRESHOLD_BAND of the effective
         threshold (post-adjustment — response.combined_score already
         includes the timeline/contradiction adjustments /analyze applied)
         with no HIGH-confidence cross-validated finding — the layers
         collectively suggest "something is off" without a single strong,
         well-localized finding. (UNCERTAIN itself is rule 1; this catches
         the decided-but-barely band around it.)
      5. MODIFIED verdict but nothing produced a croppable bbox anywhere —
         the exact "engine can't pinpoint a location" case Job C is for.
    """
    reasons = []

    if response.verdict == "UNCERTAIN":
        reasons.append("verdict is UNCERTAIN — the engine itself could not call this document either way")

    unlocated = sum(1 for sl in cached.get("suspicious_lines", []) if not getattr(sl, "bbox", None))
    unlocated += sum(1 for na in cached.get("numeric_anomalies", []) if not getattr(na, "bbox", None))
    if unlocated:
        reasons.append(f"{unlocated} content/numeric finding(s) carry no bounding box at all")

    ela_regions = cached.get("ela_regions", [])
    low_conf_cutoff = ELA_Z_THRESHOLD + ELA_LOW_CONFIDENCE_Z_MARGIN
    if ela_regions and all(r.z_score < low_conf_cutoff for r in ela_regions):
        reasons.append(
            f"all {len(ela_regions)} ELA region(s) sit near the detection threshold "
            f"(z < {low_conf_cutoff:g}) — low-confidence localization"
        )

    effective_threshold = cached.get("effective_threshold", THRESHOLD)
    has_high_conf_fused = any(f.confidence == "HIGH" for f in cached.get("fused_findings", []))
    distance = abs(response.combined_score - effective_threshold)
    if response.verdict != "UNCERTAIN" and distance <= NEAR_THRESHOLD_BAND and not has_high_conf_fused:
        reasons.append(
            f"combined score {response.combined_score:g} is within ±{NEAR_THRESHOLD_BAND:g} of the "
            f"{effective_threshold:g} threshold with no HIGH-confidence cross-validated finding"
        )

    if response.verdict == "MODIFIED" and not _gather_flagged_regions(cached):
        reasons.append("MODIFIED verdict but no finding produced a croppable bounding box — nothing to localize")

    return bool(reasons), reasons


def _gather_flagged_regions(cached: dict, max_regions: int = MAX_AI_REVIEW_REGIONS) -> list:
    """
    Fused (cross-validated) findings are added first since they're the
    highest-confidence subset; the rest only fill out the remaining cap.
    De-duplicates by (page, rounded bbox) since a fused finding and its own
    source-layer finding would otherwise both add nearly the same region.

    Each region carries a "provenance" (response-list name, index) pointing
    at the finding it came from in the /analyze response, so the region
    verdict can be merged back onto that exact finding as its ai_status
    (see _build_merged_findings). ELA/overlay regions have no per-finding
    list in the response (they surface as signals/annotations only), so
    their provenance is None — their verdicts still feed the score
    adjustment, they just have no list row to annotate.
    """
    regions = []

    def add(page, bbox, layer, description, provenance=None):
        if bbox and len(bbox) == 4:
            regions.append({
                "page": page,
                "bbox": tuple(float(v) for v in bbox),
                "layer": layer,
                "description": (description or "")[:200],
                "provenance": provenance,
            })

    for i, f in enumerate(cached.get("fused_findings", [])):
        add(f.page, f.bbox, "fusion (" + "+".join(f.confirming_layers) + ")", f.description,
            provenance=("fused_findings", i))
    for i, sl in enumerate(cached.get("suspicious_lines", [])):
        add(sl.page, sl.bbox, "content", sl.text, provenance=("suspicious_lines", i))
    for i, na in enumerate(cached.get("numeric_anomalies", [])):
        add(na.page, na.bbox, "numeric", na.text, provenance=("numeric_anomalies", i))
    for er in cached.get("ela_regions", []):
        add(er.page, er.bbox, "ela", f"ELA anomaly (z-score {er.z_score:.1f})")
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
    rendered pages so the model can cross-examine each finding against
    what's actually visible — a superset of the merged call's narrower
    explainer summary, since Job C also needs ELA findings and metadata to
    verify against."""
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
    already reflected in the deterministic layers, since the model was given
    the full engine output and told to only report what it missed), plus
    whether the region review / Job C corroborated or contradicted the
    engine's own findings."""
    score = 0.0
    for f in additional_findings:
        score += JOB_C_CONFIDENCE_POINTS.get(f["confidence"], 8)
    for r in job_b_regions:
        if r["label"] == "possible-edit":
            score += JOB_B_CORROBORATION_BONUS
        elif r["label"] == "template-element":
            score -= JOB_B_CONTRADICTION_PENALTY
    for v in job_c_verifications:
        if v["verdict"] == "supported":
            score += JOB_C_SUPPORTED_BONUS
        elif v["verdict"] == "contradicted":
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
    region-review template-element reclassifications, and Job C
    "contradicted" verifications — both floor at 0 per layer, neither is a
    blanket layer override."""
    adjusted = dict(response.layers.dict())
    downweight_applied = []

    for r in job_b_regions:
        if r["label"] != "template-element":
            continue
        for layer_name in _extract_layer_names(r["source_layer"]):
            before = adjusted.get(layer_name, 0)
            adjusted[layer_name] = max(0, before - JOB_B_TEMPLATE_DOWNWEIGHT_POINTS)
            downweight_applied.append({
                "source": "region_review",
                "layer": layer_name,
                "finding_description": r["engine_description"],
                "points": before - adjusted[layer_name],
            })

    for v in job_c_verifications:
        if v["verdict"] != "contradicted":
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


def _compute_ai_adjusted_verdict(combined_score_with_ai: float, effective_threshold: float) -> str:
    """Same MODIFIED/ORIGINAL/UNCERTAIN threshold logic verdict_engine.combine()
    uses for the deterministic verdict, applied to combined_score_with_ai
    instead — so the post-AI-review verdict label the UI displays is
    computed the same way the rest of the app labels verdicts, rather than
    left to the model's own judgment of what counts as "modified"."""
    distance = abs(combined_score_with_ai - effective_threshold)
    if distance <= UNCERTAIN_BAND:
        return "UNCERTAIN"
    return "MODIFIED" if combined_score_with_ai >= effective_threshold else "ORIGINAL"


def _build_merged_findings(response: ForensicResponse, cached: dict,
                            regions_out: list, additional_findings_out: list) -> dict:
    """
    The Part-3 merged structure: copies of the main report's findings lists
    (same shapes /analyze returned them in) with an ai_status stamped on
    each item — "confirmed" (region review called it a possible edit),
    "contradicted" (region review called it a template element; ai_note
    carries the model's one-line reason), or "unreviewed" (this finding
    wasn't sent for review, or the model was uncertain/unavailable on it).
    Job C discoveries are appended to the SAME fused_findings list tagged
    ai_discovered — distinguishable, but living in the one list, not a
    separate section. The frontend swaps these lists in over the originals
    in place; the originals stay untouched in the cached response for audit.
    """
    status_by = {}
    for r in regions_out:
        prov = r.get("provenance")
        if not prov:
            continue
        if r["label"] == "possible-edit":
            status = "confirmed"
        elif r["label"] == "template-element":
            status = "contradicted"
        else:
            # "uncertain" — reviewed but inconclusive; grouped with
            # unreviewed for display since neither changes the finding.
            status = "unreviewed"
        status_by[tuple(prov)] = (status, r["reasoning"])

    def annotate(dicts, list_name):
        out = []
        for i, d in enumerate(dicts):
            status, note = status_by.get((list_name, i), ("unreviewed", None))
            d = dict(d)
            d["ai_status"] = status
            if note:
                d["ai_note"] = note
            out.append(d)
        return out

    merged = {
        "fused_findings": annotate([f.dict() for f in response.fused_findings], "fused_findings"),
        "suspicious_lines": annotate([s.dict() for s in response.suspicious_lines], "suspicious_lines"),
        "numeric_anomalies": annotate([n.dict() for n in response.numeric_anomalies], "numeric_anomalies"),
    }

    for f in additional_findings_out:
        merged["fused_findings"].append({
            "page": f["page"],
            "bbox": f["bbox"],
            "confirming_layers": ["ai_review"],
            "confidence": (f["confidence"] or "low").upper(),
            "score": None,
            "description": f["description"],
            "ai_discovered": True,
        })

    return merged


_UNAVAILABLE_PREFIX = "AI review unavailable for this item — "


def _extract_failure_reason(reasoning: str) -> str:
    """A region-verdict 'unavailable' reasoning is always
    '<_UNAVAILABLE_PREFIX><the actual error>' — strip the prefix so
    failure messages state the real reason (rate limit, timeout, etc.)
    once, instead of the per-item wrapper text."""
    if reasoning and reasoning.startswith(_UNAVAILABLE_PREFIX):
        return reasoning[len(_UNAVAILABLE_PREFIX):]
    return reasoning or "an unknown error"


def _scan_audit_fields(needs_scan: bool, scan_reasons: list, scan_mode: str, ai_calls_made: int) -> dict:
    """The Part-1 auditability fields, present on EVERY response shape
    (success, not-configured, hard failure) so a consumer can always see
    which path was chosen and how many provider calls it cost."""
    return {
        "needs_independent_scan": needs_scan,
        "needs_independent_scan_reasons": scan_reasons,
        "scan_mode": scan_mode,
        "ai_calls_made": ai_calls_made,
    }


def _build_hard_failure_result(provider_label: str, response: ForensicResponse, reason: str,
                                audit_fields: dict) -> dict:
    """The FAIL-FAST result: one of the (at most two) AI calls failed
    outright after its own retries — a strong signal any further call
    would fail identically, so nothing further is attempted. Deliberately
    NOT cached (see the caller) so clicking retry re-attempts the whole
    pipeline fresh via this same endpoint, rather than replaying a stale
    failure."""
    message = (
        f"⚠️ AI Review unavailable right now ({reason}). "
        f"Try again in a minute. The deterministic 6-layer analysis above "
        f"is complete and unaffected."
    )
    return {
        "available": True,
        "provider": provider_label,
        "hard_failure": True,
        "hard_failure_message": message,
        "explanation": None,
        "explanation_prompt": None,
        "explanation_error": None,
        "regions": [],
        "regions_error": None,
        "per_finding_verification": [],
        "additional_findings": [],
        "overall_assessment": None,
        "ai_disagreement_flag": False,
        "ai_disagreement_message": None,
        "job_c_error": None,
        "layer7_score": 0,
        "layer7_weight": LAYER7_WEIGHT,
        "job_b_template_downweight_points": JOB_B_TEMPLATE_DOWNWEIGHT_POINTS,
        "job_c_contradiction_downweight_points": JOB_C_CONTRADICTION_DOWNWEIGHT_POINTS,
        "downweight_applied": [],
        "combined_score": response.combined_score,
        "combined_score_with_ai": None,
        "ai_adjusted_verdict": None,
        "merged_findings": None,
        "from_cache": False,
        **audit_fields,
    }


@router.post("/api/analysis/{analysis_id}/ai-review", tags=["AI Review"])
async def ai_review(analysis_id: str):
    """
    Opt-in supplementary AI review — ONLY invoked when the user clicks the
    AI-verify button in the UI. Never runs during /analyze, never mutates
    the cached ForensicResponse/combined_score/verdict. Runs, IN ORDER:

      Job C (CONDITIONAL — only when _compute_needs_independent_scan() says
             the engine's findings lack a confident location): genuine
             cross-examination — the model gets BOTH the rendered page
             images AND the engine's own full analysis JSON in one call,
             independently verifies each engine finding against the actual
             document, surfaces anything the engine missed, and gives its
             own overall assessment (never auto-applied to the verdict —
             surfaced as a flagged disagreement for human review instead).
             Runs FIRST so its results feed the one explanation below.
      Merged region-review + explanation (ALWAYS; former Job A + Job B in
             ONE request): template-vs-possible-edit labels for regions
             the engine already flagged, plus the plain-English
             explanation synthesizing engine findings + those same region
             verdicts + Job C's results when the scan ran.
      [Layer 7 score + combined_score_with_ai + merged_findings computed
       from the above — combined_score itself is never touched.]

    So the confident-location path costs 1 provider call, the needs-scan
    path 2 — vs the old always-3. scan_mode/needs_independent_scan(_reasons)
    /ai_calls_made in the response record which path ran and why. Fails
    gracefully per-call (API key missing, network error, rate limit)
    without ever raising past this endpoint or affecting the cached verdict.

    Cached per analysis_id: a second call returns the exact same result
    instead of re-calling the provider, so combined_score_with_ai stays
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

    # The Part-1 rule check — plain logic on the deterministic results
    # already in hand; decided BEFORE any provider call is made.
    needs_scan, scan_reasons = _compute_needs_independent_scan(cached, response)
    scan_mode = "full-scan" if needs_scan else "region-review-only"
    ai_calls_made = 0
    logger.info(
        "ai_review: analysis %s -> %s%s",
        analysis_id, scan_mode,
        (" (" + "; ".join(scan_reasons) + ")") if scan_reasons
        else " (engine findings are confidently localized)",
    )

    try:
        advisor, provider_label = _make_advisor()
    except AIProviderNotConfigured as e:
        return {
            "available": False,
            "provider": "NVIDIA NIM" if AI_REVIEW_PROVIDER == "nvidia" else "Gemini",
            "hard_failure": False,
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
            "merged_findings": None,
            **_scan_audit_fields(needs_scan, scan_reasons, scan_mode, ai_calls_made),
        }

    # ── Job C — CONDITIONAL, and FIRST when it runs, so the one explanation
    # below can synthesize its results (one narrative, not a bolted-on
    # paragraph). Cross-examines the engine's OWN findings against the
    # actual rendered pages in ONE combined call (full engine JSON + page
    # images together) — deliberately one call, not per-page: verifying
    # claims like "this header repeats across pages" needs more than one
    # page in view — so it uses a longer timeout and the same retry/backoff
    # as every other call to manage that.
    per_finding_verification = []
    additional_findings_out = []
    overall_assessment = None
    job_c_error = None
    job_c_ran = False
    if needs_scan:
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
                ai_calls_made += 1
                cross_exam = advisor.independent_scan(page_images, job_c_summary)
                job_c_ran = True
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
        except AIProviderRequestError as e:
            # FAIL FAST: the first call failed outright after its own
            # retries — the merged call would almost certainly fail
            # identically, so don't attempt it. Not cached, so a retry
            # re-runs the whole pipeline fresh.
            return _build_hard_failure_result(
                provider_label, response, str(e),
                _scan_audit_fields(needs_scan, scan_reasons, scan_mode, ai_calls_made))
        except Exception as e:
            job_c_error = f"Unexpected error during AI cross-examination: {e}"

        if ai_calls_made:
            time.sleep(AI_REVIEW_INTER_CALL_DELAY_SECONDS)

    # ── Merged region-review + explanation — ALWAYS runs, exactly ONE call.
    # Crops every already-flagged region and sends them alongside the
    # narrow explainer summary (the fields the former Job A used, plus Job
    # C's results when the scan ran) — the model labels the regions AND
    # writes the explanation in one reply.
    flagged_regions, crop_bytes_list = [], []
    for region in _gather_flagged_regions(cached):
        crop_bytes = _crop_region_image(pdf_path, region["page"], region["bbox"])
        if not crop_bytes:
            continue
        flagged_regions.append(region)
        crop_bytes_list.append(crop_bytes)

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
        "independent_scan_ran": job_c_ran,
    }
    if job_c_ran:
        analysis_summary.update({
            "job_c_per_finding_verification": per_finding_verification,
            "job_c_additional_findings": additional_findings_out,
            "job_c_overall_assessment": overall_assessment,
        })

    regions_out = []
    regions_error = None
    explanation, explanation_prompt, explanation_error = None, None, None
    try:
        ai_calls_made += 1
        label_results, explanation, explanation_prompt = advisor.review_and_explain(
            crop_bytes_list, analysis_summary)

        unavailable = [r for r in label_results if r["label"] == "unavailable"]
        if (label_results and len(unavailable) == len(label_results)
                and not (explanation and explanation.get("detail"))):
            # Nothing usable in the reply at all (every region unavailable
            # AND no explanation text) — same fail-fast handling as a
            # transport failure.
            return _build_hard_failure_result(
                provider_label, response,
                _extract_failure_reason(unavailable[0]["reasoning"]),
                _scan_audit_fields(needs_scan, scan_reasons, scan_mode, ai_calls_made))

        for region, label_result in zip(flagged_regions, label_results):
            if label_result["label"] == "unavailable":
                continue
            regions_out.append({
                "page": region["page"] + 1,  # 1-indexed for display
                "bbox": list(region["bbox"]),
                "source_layer": region["layer"],
                "engine_description": region["description"],
                "label": label_result["label"],
                "reasoning": label_result["reasoning"],
                "provenance": list(region["provenance"]) if region["provenance"] else None,
            })
        if unavailable:
            regions_error = (
                f"{len(unavailable)} of {len(label_results)} region(s) could not be "
                f"reviewed ({_extract_failure_reason(unavailable[0]['reasoning'])}) — "
                f"try again to review the rest."
            )
    except AIProviderRequestError as e:
        # The merged call failed outright — even if Job C succeeded, there
        # is no region review and no explanation to show, so fail fast (and
        # don't cache) rather than persist a mostly-empty result a retry
        # could never repair.
        return _build_hard_failure_result(
            provider_label, response, str(e),
            _scan_audit_fields(needs_scan, scan_reasons, scan_mode, ai_calls_made))
    except Exception as e:
        regions_error = f"Unexpected error during region review/explanation: {e}"
        explanation_error = regions_error

    # Layer 7 score + the FINAL combined_score_with_ai + the merged findings
    # structure the frontend swaps into the main report in place.
    layer7_score = _compute_layer7_score(additional_findings_out, per_finding_verification, regions_out)
    score_calc = _compute_combined_score_with_ai(response, regions_out, per_finding_verification, layer7_score)
    effective_threshold = cached.get("effective_threshold", THRESHOLD)
    ai_adjusted_verdict = _compute_ai_adjusted_verdict(score_calc["combined_score_with_ai"], effective_threshold)
    merged_findings = _build_merged_findings(response, cached, regions_out, additional_findings_out)

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
        "provider": provider_label,
        "hard_failure": False,
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
        # combined_score_with_ai is "the" displayed score post-click; the
        # deterministic combined_score stays alongside it as the secondary
        # audit field (never deleted, just not the primary display).
        "combined_score": response.combined_score,
        "combined_score_with_ai": score_calc["combined_score_with_ai"],
        "ai_adjusted_verdict": ai_adjusted_verdict,
        "merged_findings": merged_findings,
        "from_cache": False,
        **_scan_audit_fields(needs_scan, scan_reasons, scan_mode, ai_calls_made),
    }

    # Audit trail: the deterministic response object stays untouched in
    # cached["response"]; the AI-merged result is stored alongside it here.
    cached["ai_review"] = result
    return result
