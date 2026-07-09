"""
Shared response-parsing/validation for AI Review providers (Gemini, NVIDIA
NIM, and any future provider). Both providers are instructed (via their own,
provider-specific prompts) to produce the exact same JSON contracts for the
merged region-review+explanation call (former Job A + Job B, now ONE
request) and the independent scan (Job C) — this module is the ONE place
that validates/coerces that JSON into
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


def _coerce_region_items(data, expected_n: int) -> list:
    """Coerces the model's region-verdict JSON array into a list of length
    expected_n, in order, of {"label": one of REGION_LABELS or
    "unavailable", "reasoning": str}. Malformed/missing entries fill in as
    'unavailable' rather than losing the whole batch."""
    parsed = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if isinstance(idx, int) and 1 <= idx <= expected_n:
                parsed[idx] = item

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


def _lenient_json_load(text: str):
    """json.loads with one extra chance: a reply truncated just before its
    final closing brace(s)/quote (seen in practice from NVIDIA NIM's text
    model) gets the missing closers appended and re-tried, instead of
    dumping an otherwise-perfect reply into the raw-text fallback."""
    t = strip_json_fences(text)
    try:
        return json.loads(t)
    except (ValueError, TypeError):
        pass
    for suffix in ("}", "}}", "}}}", '"}', '"}}', '"}}}'):
        try:
            return json.loads(t + suffix)
        except (ValueError, TypeError):
            continue
    return None


def parse_review_and_explanation(raw: str, expected_n: int,
                                  provider_label: str = "AI provider") -> tuple:
    """The MERGED region-review + explanation call (former Job A + Job B,
    now one request — see api/ai_review_routes.py). Returns
    (regions: list, explanation: dict):
      regions — length expected_n, in order, each {"label": one of
                REGION_LABELS or "unavailable", "reasoning": str}
      explanation — {"lead_sentence": str|None, "detail": str}

    Never raises. A reply that isn't a JSON object at all degrades to every
    region 'unavailable' plus the raw text as the explanation detail — the
    same per-half fallbacks the two separate calls used to have."""
    data = _lenient_json_load(raw)
    if not isinstance(data, dict):
        logger.warning("%s: could not parse merged review/explanation JSON reply, "
                       "falling back to raw text:\n%s", provider_label, raw)
        regions = [{
            "label": "unavailable",
            "reasoning": "AI review unavailable for this item — the model's reply could not be parsed.",
        } for _ in range(expected_n)]
        return regions, {"lead_sentence": None, "detail": (raw or "").strip()}

    regions = _coerce_region_items(data.get("regions"), expected_n)
    expl_raw = data.get("explanation")
    if not isinstance(expl_raw, dict):
        expl_raw = {}
    # Some models nest the whole explanation object INSIDE lead_sentence
    # ({"explanation": {"lead_sentence": {"lead_sentence": ..., "detail":
    # ...}}}) — unwrap rather than stringify a dict into the lead.
    if isinstance(expl_raw.get("lead_sentence"), dict):
        expl_raw = expl_raw["lead_sentence"]
    explanation = {
        "lead_sentence": coerce_to_text(expl_raw.get("lead_sentence")) or None,
        "detail": coerce_to_text(expl_raw.get("detail")),
    }
    return regions, explanation


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
