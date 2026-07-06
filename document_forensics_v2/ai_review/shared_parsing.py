"""
Shared response-parsing/validation for AI Review providers (Gemini, NVIDIA
NIM, and any future provider). Both providers are instructed (via their own,
provider-specific prompts) to produce the exact same JSON contracts for Job
A/B/C — this module is the ONE place that validates/coerces that JSON into
the internal result structures the rest of the app (scoring, caching,
frontend) consumes, so a NVIDIA response and a Gemini response for the same
document produce byte-for-byte identically SHAPED results (values will
differ, since they're different models, but never the schema).

Each provider's own advisor module is responsible for:
  1. Building its own prompt text (phrasing may need to differ per model).
  2. Talking to its own transport (Gemini's native REST vs NVIDIA's
     OpenAI-compatible chat completions) and unwrapping ITS OWN response
     envelope down to the raw text the model produced.
  3. Calling the functions here to validate/coerce that raw text into the
     shared result shape.

Raises AIProviderRequestError (imported from utils.ai_retry, not redefined
here) for the ONE case with no meaningful partial result: Job C missing an
overall_assessment. Every other malformed/missing field degrades gracefully
(documented per-function) rather than losing the whole result.
"""

import json
import logging

from utils.ai_retry import AIProviderRequestError

logger = logging.getLogger("document_forensics")

REGION_LABELS = ("template-element", "possible-edit", "uncertain")
EDIT_CONFIDENCE_LABELS = ("low", "medium", "high")
CROSS_EXAM_VERDICTS = ("supported", "contradicted", "unverifiable")


def strip_json_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t.lstrip("`")
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def coerce_to_text(value) -> str:
    """Some models return a string field as a JSON array of paragraph/
    sentence strings instead of one string, despite the prompt asking for
    a single string — join rather than crash."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n\n".join(str(v).strip() for v in value if v).strip()
    return str(value).strip()


def parse_explanation(raw: str, provider_label: str = "AI provider") -> dict:
    """Job A. Best-effort structured parse — never raises. A malformed/
    non-JSON reply degrades to putting everything in 'detail' with no
    separated lead sentence, rather than losing the explanation outright
    (unlike Job C, where a missing overall_assessment IS treated as a hard
    failure — here there's always a reasonable degraded fallback)."""
    try:
        data = json.loads(strip_json_fences(raw))
        if isinstance(data, dict) and "detail" in data:
            return {
                "lead_sentence": coerce_to_text(data.get("lead_sentence")) or None,
                "detail": coerce_to_text(data.get("detail")),
            }
    except (ValueError, TypeError):
        logger.warning("%s: could not parse Job A JSON reply, falling back to raw text:\n%s", provider_label, raw)
    return {"lead_sentence": None, "detail": (raw or "").strip()}


def parse_region_batch(raw: str, expected_n: int, provider_label: str = "AI provider") -> list:
    """Job B (batched). Returns a list of length expected_n, in order, of
    {"label": one of REGION_LABELS or "unavailable", "reasoning": str}.
    Never raises — malformed/missing entries fill in as 'unavailable'
    rather than losing the whole batch."""
    parsed = {}
    try:
        data = json.loads(strip_json_fences(raw))
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                if isinstance(idx, int) and 1 <= idx <= expected_n:
                    parsed[idx] = item
    except (ValueError, TypeError):
        logger.warning("%s: could not parse Job B batch JSON reply:\n%s", provider_label, raw)

    out = []
    for i in range(1, expected_n + 1):
        item = parsed.get(i)
        if not item:
            out.append({
                "label": "unavailable",
                "reasoning": "AI review unavailable for this item — no response returned.",
            })
            continue
        label = item.get("label", "uncertain")
        if label not in REGION_LABELS:
            label = "uncertain"
        reasoning = (item.get("reasoning") or "").strip()[:280] or "No reasoning returned."
        out.append({"label": label, "reasoning": reasoning})
    return out


def parse_cross_examination(raw: str, provider_label: str = "AI provider") -> dict:
    """Job C. Individual malformed per_finding_verification/
    additional_findings entries are dropped (best-effort — one bad item
    shouldn't lose the rest), but a response missing overall_assessment
    entirely, or that isn't valid JSON, raises AIProviderRequestError —
    there's no meaningful partial result for "no verdict at all"."""
    try:
        data = json.loads(strip_json_fences(raw))
    except (ValueError, TypeError) as e:
        logger.warning("%s: could not parse Job C JSON reply:\n%s", provider_label, raw)
        raise AIProviderRequestError(f"Could not parse cross-examination response: {e}")
    if not isinstance(data, dict) or "overall_assessment" not in data:
        raise AIProviderRequestError("Cross-examination response is missing overall_assessment.")

    verifications = []
    for item in data.get("per_finding_verification", []) or []:
        if not isinstance(item, dict):
            continue
        verdict = item.get("verdict", "unverifiable")
        if verdict not in CROSS_EXAM_VERDICTS:
            verdict = "unverifiable"
        verifications.append({
            "engine_finding": (item.get("engine_finding") or "").strip()[:400],
            "layer": (item.get("layer") or "unknown").strip().lower(),
            "verdict": verdict,
            "reasoning": (item.get("reasoning") or "").strip()[:400],
        })

    additional = []
    for item in data.get("additional_findings", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        bbox_px = item.get("bbox_px")
        try:
            bbox_px = [float(v) for v in bbox_px] if bbox_px else None
            if bbox_px is not None and len(bbox_px) != 4:
                bbox_px = None
        except (TypeError, ValueError):
            bbox_px = None
        confidence = item.get("confidence", "low")
        if confidence not in EDIT_CONFIDENCE_LABELS:
            confidence = "low"
        additional.append({
            "page": page,
            "bbox_px": bbox_px,
            "description": (item.get("description") or "").strip()[:300],
            "confidence": confidence,
        })

    overall_raw = data.get("overall_assessment") or {}
    agrees = overall_raw.get("agrees_with_engine_verdict", "inconclusive")
    if agrees not in (True, False, "inconclusive"):
        agrees = "inconclusive"
    overall = {
        "agrees_with_engine_verdict": agrees,
        "reasoning": (overall_raw.get("reasoning") or "").strip()[:500],
    }

    return {
        "per_finding_verification": verifications,
        "additional_findings": additional,
        "overall_assessment": overall,
    }
