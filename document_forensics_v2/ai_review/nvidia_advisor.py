"""
NVIDIA NIM Advisor — opt-in, supplementary AI review. The ALTERNATE AI
Review provider to Gemini (ai_review/gemini_advisor.py), selected via
AI_REVIEW_PROVIDER=nvidia (Gemini stays the default when AI_REVIEW_PROVIDER
is unset — this file changes nothing about existing Gemini-only setups).

Implements the SAME two-method shared interface as GeminiAdvisor so
api/ai_review_routes.py never needs to know which provider answered:
  review_and_explain(images, analysis_summary)
      -> (regions: list, {"lead_sentence": str, "detail": str}, prompt)
  independent_scan(page_images, analysis_summary) -> dict  # Job C

Uses NVIDIA NIM's OpenAI-compatible endpoint (base_url
https://integrate.api.nvidia.com/v1, POST /chat/completions) via plain
`requests` — same as gemini_advisor.py, no `openai` SDK dependency needed
for a single POST. Two models, picked from NVIDIA's catalog at
build.nvidia.com/models (current as of implementation; both overridable
via env vars in case the catalog changes before you read this):
  - NVIDIA_MODEL_TEXT (text-only reasoning; used by the merged call only
    when there are NO region crops to attach): defaults to
    "nvidia/nemotron-3-super-120b-a12b" — a Nemotron reasoning model, run
    with enable_thinking=False so the response is a clean JSON object
    rather than chain-of-thought preamble plus JSON.
  - NVIDIA_MODEL_VISION (image understanding — the merged region-review +
    explanation call with crops attached, and Job C): defaults to
    "nvidia/nemotron-nano-12b-v2-vl" — Nemotron Nano 12B v2 VL, explicitly
    documented in NVIDIA's catalog for multi-image reasoning and document
    intelligence (invoices, receipts, forms, visual Q&A), not a
    video-focused model.

Retry/backoff/timeout handling, JSON-contract parsing, AND the merged
call's prompt text are SHARED with gemini_advisor.py via utils/ai_retry.py,
ai_review/shared_parsing.py, and ai_review/shared_prompts.py respectively —
only the request/response transport shape below (OpenAI-compatible chat
completions) is NVIDIA-specific. NvidiaNotConfigured/NvidiaRequestError are aliases of the
shared, provider-agnostic exception classes (not NVIDIA-specific
subclasses), so api/ai_review_routes.py catches exactly one exception type
regardless of which provider is active.
"""

import base64
import json
import logging
import os

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from utils.ai_retry import (
    AIProviderNotConfigured as NvidiaNotConfigured,
    AIProviderRequestError as NvidiaRequestError,
    post_with_retry,
)
from ai_review.shared_parsing import (
    parse_review_and_explanation,
    parse_cross_examination,
)
from ai_review.shared_prompts import build_review_and_explain_prompt as _build_review_and_explain_prompt

logger = logging.getLogger("document_forensics")

NVIDIA_API_KEY      = os.environ.get("NVIDIA_API_KEY", "").strip()
NVIDIA_MODEL_TEXT   = os.environ.get("NVIDIA_MODEL_TEXT", "nvidia/nemotron-3-super-120b-a12b").strip()
NVIDIA_MODEL_VISION = os.environ.get("NVIDIA_MODEL_VISION", "nvidia/nemotron-nano-12b-v2-vl").strip()
NVIDIA_API_BASE     = "https://integrate.api.nvidia.com/v1"

REQUEST_TIMEOUT_SECONDS        = 30
MERGED_REQUEST_TIMEOUT_SECONDS = 60
JOB_C_REQUEST_TIMEOUT_SECONDS  = 90

# Explicit per-call output budgets — the OpenAI chat completions spec
# doesn't guarantee a generous default if max_tokens is omitted, unlike
# Gemini's REST API. The merged region-review + explanation call has to
# fit region verdicts AND a multi-paragraph explanation in one reply
# (roughly the former Job A + Job B budgets combined). Job C gets the
# most since cross-examining every distinct engine finding across up to
# 5 pages can produce a large JSON array.
MERGED_MAX_TOKENS = 5120
JOB_C_MAX_TOKENS  = 8192


class NvidiaAdvisor:

    def __init__(self):
        if not _REQUESTS_AVAILABLE:
            raise NvidiaNotConfigured(
                "The 'requests' package is not installed — the AI Review "
                "feature is unavailable in this environment."
            )
        if not NVIDIA_API_KEY:
            raise NvidiaNotConfigured(
                "NVIDIA_API_KEY is not set — the AI Review feature is "
                "unavailable until this environment variable is configured."
            )

    # ── Merged region-review + explanation — former Job A + Job B in ONE
    # call, same contract as Gemini's review_and_explain ────────────────────

    def review_and_explain(self, images: list, analysis_summary: dict) -> tuple:
        """Same contract as GeminiAdvisor.review_and_explain(): images is a
        list of PNG crop bytes, each ONE already-flagged region (may be
        empty — explanation-only); analysis_summary carries the
        deterministic findings plus Job C's results when the caller ran
        the independent scan first. Returns (regions, explanation,
        prompt_sent). Raises NvidiaRequestError only if the call fails
        outright — a malformed reply degrades per-half instead.

        Uses the vision model when crops are attached; falls back to the
        text model for an explanation-only call (no images to reason
        over, and the text model is the stronger pure-reasoner)."""
        n = len(images)
        prompt = _build_review_and_explain_prompt(n, analysis_summary)
        logger.info("nvidia_advisor: merged review+explanation prompt sent:\n%s", prompt)

        if n:
            content = [{"type": "text", "text": prompt}]
            for i, img in enumerate(images, start=1):
                content.append({"type": "text", "text": f"REGION {i}:"})
                content.append(self._image_content(img))
            raw = self._chat_vision(content, max_tokens=MERGED_MAX_TOKENS,
                                    timeout=MERGED_REQUEST_TIMEOUT_SECONDS)
        else:
            raw = self._chat_text(prompt, max_tokens=MERGED_MAX_TOKENS,
                                  timeout=MERGED_REQUEST_TIMEOUT_SECONDS)

        regions, explanation = parse_review_and_explanation(raw, n, provider_label="NVIDIA NIM")
        return regions, explanation, prompt

    # ── Job C — genuine cross-examination, same contract as Gemini's
    # independent_scan ──────────────────────────────────────────────────────

    def independent_scan(self, page_images: list, analysis_summary: dict) -> dict:
        """Same contract as GeminiAdvisor.independent_scan(): page_images
        is a list of (page_number_1_indexed, png_bytes) FULL page renders;
        analysis_summary is the engine's own analysis JSON. Returns
        {"per_finding_verification": [...], "additional_findings": [...],
        "overall_assessment": {...}}. Raises NvidiaRequestError if the call
        fails outright or the reply can't be parsed into a usable result."""
        prompt = (
            "You are cross-examining an ALREADY-COMPUTED forensic analysis "
            "against the ACTUAL document. You are given two things: (1) "
            "full page image(s) of the real document below, and (2) the "
            "deterministic engine's own analysis JSON (verdict, "
            "combined_score, per-layer scores, signals, suspicious lines, "
            "numeric anomalies, ELA findings, cross-layer fused findings, "
            "and metadata).\n\n"
            "This is genuine cross-examination, NOT descriptive captioning "
            "and NOT passive agreement. Do NOT simply restate or agree "
            "with the engine's findings by default. For EVERY distinct "
            "finding you can identify in the analysis JSON below (each "
            "entry in 'signals', each 'suspicious_lines' entry, each "
            "'numeric_anomalies' entry, each 'ela_findings' entry, each "
            "'fused_findings' entry), independently look at the actual "
            "page image(s) and decide whether the visual/textual evidence "
            "in the REAL document supports or contradicts that specific "
            "claim. Cite the SPECIFIC evidence you see — e.g. if a font-"
            "size anomaly is claimed on one line, check whether that same "
            "size/style appears elsewhere in the document (a repeating "
            "header, a template field), which would CONTRADICT the "
            "'anomaly' framing, or whether it's genuinely inconsistent "
            "with its surroundings, which would SUPPORT it. Say plainly "
            "when you disagree with or cannot confirm something the "
            "engine claimed.\n\n"
            "Independently of the engine's findings, ALSO look for any "
            "regions that look edited/inconsistent that the engine did "
            "NOT flag — use the engine's own layer definitions (content/"
            "font consistency, numeric outliers, OCR word anomalies, ELA "
            "visual artifacts, PyMuPDF overlays/ghost text) as your frame "
            "of reference for what kind of inconsistency would matter.\n\n"
            "Finally, give ONE overall assessment: does the actual "
            "document evidence support the engine's overall verdict, "
            "contradict it, or is it inconclusive from visual inspection "
            "alone? Your reasoning MUST reference BOTH what you saw in "
            "the document AND what the engine reported — not just one or "
            "the other.\n\n"
            "ENGINE ANALYSIS JSON:\n"
            f"{json.dumps(analysis_summary, indent=2, default=str)}\n\n"
            "Respond with ONLY a JSON object (no markdown fences, no "
            "prose) in EXACTLY this shape:\n"
            '{"per_finding_verification": [{"engine_finding": "<restate '
            'the specific claim>", "layer": "<metadata|content|ocr|'
            'numeric|ela|pymupdf|xref|fusion>", "verdict": '
            '"supported|contradicted|unverifiable", "reasoning": '
            '"<specific visual/textual evidence from the actual pages>"}, '
            '...],\n'
            '"additional_findings": [{"page": <int>, "bbox_px": '
            '[x0,y0,x1,y1] or null, "description": "...", "confidence": '
            '"low|medium|high", "not_flagged_by_engine": true}, ...],\n'
            '"overall_assessment": {"agrees_with_engine_verdict": true|'
            'false|"inconclusive", "reasoning": "<must reference both '
            'the document evidence and the engine report>"}}\n\n'
            "Cover every distinct engine finding you can identify — do "
            "not skip ones that are inconvenient or default to agreeing. "
            "Return an empty array for additional_findings if you find "
            "nothing further."
        )
        content = [{"type": "text", "text": prompt}]
        for page_num, img in page_images:
            content.append({"type": "text", "text": f"PAGE {page_num}:"})
            content.append(self._image_content(img))

        raw = self._chat_vision(content, max_tokens=JOB_C_MAX_TOKENS, timeout=JOB_C_REQUEST_TIMEOUT_SECONDS)
        return parse_cross_examination(raw, provider_label="NVIDIA NIM")

    # ── shared transport (OpenAI-compatible chat completions) ──────────────

    @staticmethod
    def _image_content(image_bytes: bytes, mime_type: str = "image/png") -> dict:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}

    def _chat_text(self, prompt: str, max_tokens: int, timeout: float) -> str:
        # enable_thinking=False so the reasoning model returns a clean JSON
        # object instead of chain-of-thought preamble + JSON — the merged
        # call needs a directly-parseable reply, not a reasoning trace.
        return self._chat(
            NVIDIA_MODEL_TEXT,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=timeout,
            chat_template_kwargs={"enable_thinking": False},
        )

    def _chat_vision(self, content: list, max_tokens: int, timeout: float) -> str:
        return self._chat(
            NVIDIA_MODEL_VISION,
            [{"role": "user", "content": content}],
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def _chat(self, model: str, messages: list, max_tokens: int, timeout: float,
              chat_template_kwargs: dict = None) -> str:
        """Core call shared by every job. Retry/backoff/timeout handling is
        delegated to utils.ai_retry.post_with_retry (shared with
        gemini_advisor.py) — only the OpenAI-compatible chat-completions
        request/response shape below is NVIDIA-specific."""
        url = f"{NVIDIA_API_BASE}/chat/completions"
        json_body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,  # low temperature — this is a structured-JSON
                                 # extraction/verification task, not creative writing
        }
        if chat_template_kwargs:
            json_body["chat_template_kwargs"] = chat_template_kwargs

        resp = post_with_retry(
            url,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json_body=json_body,
            timeout=timeout,
            provider_label="NVIDIA NIM API",
        )

        try:
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise NvidiaRequestError("NVIDIA NIM API returned no choices")
            text = (choices[0].get("message", {}).get("content") or "").strip()
            if not text:
                raise NvidiaRequestError("NVIDIA NIM API returned an empty response")
            return text
        except (KeyError, ValueError, IndexError) as e:
            raise NvidiaRequestError(f"Could not parse NVIDIA NIM API response: {e}")
