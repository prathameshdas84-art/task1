"""
NVIDIA NIM Advisor — opt-in, supplementary AI review. The ALTERNATE AI
Review provider to Gemini (ai_review/gemini_advisor.py), selected via
AI_REVIEW_PROVIDER=nvidia (Gemini stays the default when AI_REVIEW_PROVIDER
is unset — this file changes nothing about existing Gemini-only setups).

Implements the SAME three-method shared interface as GeminiAdvisor so
api/ai_review_routes.py never needs to know which provider answered:
  explain(analysis_summary) -> ({"lead_sentence": str, "detail": str}, prompt)
  review_regions(images: list) -> list                    # Job B, batched
  independent_scan(page_images, analysis_summary) -> dict  # Job C

Uses NVIDIA NIM's OpenAI-compatible endpoint (base_url
https://integrate.api.nvidia.com/v1, POST /chat/completions) via plain
`requests` — same as gemini_advisor.py, no `openai` SDK dependency needed
for a single POST. Two models, picked from NVIDIA's catalog at
build.nvidia.com/models (current as of implementation; both overridable
via env vars in case the catalog changes before you read this):
  - NVIDIA_MODEL_TEXT (Job A, text-only reasoning): defaults to
    "nvidia/nemotron-3-super-120b-a12b" — a Nemotron reasoning model, run
    with enable_thinking=False for Job A so the response is a clean JSON
    object rather than chain-of-thought preamble plus JSON.
  - NVIDIA_MODEL_VISION (Job B/C, image understanding): defaults to
    "nvidia/nemotron-nano-12b-v2-vl" — Nemotron Nano 12B v2 VL, explicitly
    documented in NVIDIA's catalog for multi-image reasoning and document
    intelligence (invoices, receipts, forms, visual Q&A), not a
    video-focused model.

Retry/backoff/timeout handling and Job A/B/C JSON-contract parsing are
SHARED with gemini_advisor.py via utils/ai_retry.py and
ai_review/shared_parsing.py respectively — only the request/response
transport shape below (OpenAI-compatible chat completions) is
NVIDIA-specific. NvidiaNotConfigured/NvidiaRequestError are aliases of the
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
    parse_explanation,
    parse_region_batch,
    parse_cross_examination,
)

logger = logging.getLogger("document_forensics")

NVIDIA_API_KEY      = os.environ.get("NVIDIA_API_KEY", "").strip()
NVIDIA_MODEL_TEXT   = os.environ.get("NVIDIA_MODEL_TEXT", "nvidia/nemotron-3-super-120b-a12b").strip()
NVIDIA_MODEL_VISION = os.environ.get("NVIDIA_MODEL_VISION", "nvidia/nemotron-nano-12b-v2-vl").strip()
NVIDIA_API_BASE     = "https://integrate.api.nvidia.com/v1"

REQUEST_TIMEOUT_SECONDS       = 30
JOB_C_REQUEST_TIMEOUT_SECONDS = 90

# Explicit per-job output budgets — the OpenAI chat completions spec
# doesn't guarantee a generous default if max_tokens is omitted, unlike
# Gemini's REST API. Job C gets the most since cross-examining every
# distinct engine finding across up to 5 pages can produce a large JSON
# array.
JOB_A_MAX_TOKENS = 2048
JOB_B_MAX_TOKENS = 3072
JOB_C_MAX_TOKENS = 8192


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

    # ── Job A — plain-English explanation, run LAST (by the caller) so it
    # can synthesize Job B/C's own results — same contract as Gemini's ────

    def explain(self, analysis_summary: dict) -> tuple:
        """Same contract as GeminiAdvisor.explain(): analysis_summary may
        include job_b_region_verdicts/job_c_per_finding_verification/
        job_c_additional_findings/job_c_overall_assessment/
        combined_score_with_ai/ai_adjusted_verdict if this AI review's Job
        B/C already ran. Returns ({"lead_sentence": str, "detail": str},
        prompt_sent)."""
        prompt = (
            "You are explaining an ALREADY-COMPUTED forensic analysis to a "
            "non-technical reader. The JSON below includes the original "
            "deterministic findings AND — since this AI review's own "
            "region-verification and cross-examination jobs already ran — "
            "their results too: job_b_region_verdicts (per-region template-"
            "element/possible-edit calls), job_c_per_finding_verification "
            "(supported/contradicted/unverifiable per engine finding), "
            "job_c_additional_findings (things the engine missed), and "
            "job_c_overall_assessment. combined_score_with_ai and "
            "ai_adjusted_verdict already account for ALL of this.\n\n"
            "CRITICAL: your first sentence MUST plainly state whether this "
            "document is MODIFIED, ORIGINAL, or UNCERTAIN using EXACTLY the "
            "ai_adjusted_verdict value and the combined_score_with_ai "
            "number below — NOT the pre-AI-review combined_score. Do not "
            "compute your own verdict or use your own judgment about the "
            "threshold; state the verdict/score already given to you, "
            "as the very first thing you say.\n\n"
            "After that first sentence, explain the reasoning in plain "
            "English: translate z-scores, layer names, and jargon. If "
            "job_b_region_verdicts or job_c_per_finding_verification "
            "contain any 'template-element' or 'contradicted' entries, "
            "you MUST explicitly say so by name — e.g. 'our AI visual "
            "review found that several flagged header regions are "
            "actually standard template elements, not edits, which is "
            "why the adjusted score differs from the original score.' Do "
            "NOT silently restate the original findings list as if "
            "nothing challenged them. Do not re-litigate or second-guess "
            "the final numbers themselves — only explain how they were "
            "reached. Keep the detail section to 3-6 short paragraphs.\n\n"
            "EXISTING ANALYSIS + THIS AI REVIEW'S OWN RESULTS:\n"
            f"{json.dumps(analysis_summary, indent=2, default=str)}\n\n"
            "Respond with ONLY a JSON object (no markdown fences, no "
            "prose) in EXACTLY this shape:\n"
            '{"lead_sentence": "<the one required first sentence, stating '
            'the final verdict/score>", "detail": "<3-6 short paragraphs '
            'of supporting explanation; **bold** is fine for emphasis, no '
            'other markdown>"}'
        )
        logger.info("nvidia_advisor: Job A prompt sent:\n%s", prompt)

        raw = self._chat_text(prompt, max_tokens=JOB_A_MAX_TOKENS, timeout=REQUEST_TIMEOUT_SECONDS)
        return parse_explanation(raw, provider_label="NVIDIA NIM"), prompt

    # ── Job B (batched) — label MULTIPLE already-flagged region crops in
    # ONE call, same contract as Gemini's review_regions ────────────────────

    def review_regions(self, images: list) -> list:
        """Same contract as GeminiAdvisor.review_regions(): images is a
        list of PNG crop bytes, each ONE already-flagged region. Returns a
        list (same length/order) of {"label": ..., "reasoning": ...}. A
        batch-level failure marks EVERY region "unavailable" rather than
        raising past this call."""
        if not images:
            return []
        n = len(images)
        prompt = (
            f"You will be shown {n} cropped regions from a document, each "
            "already flagged by an automated forensic tool as unusual. Each "
            "crop is preceded by a line reading 'REGION <n>:'. For EACH "
            "region independently, decide: does it look like a repeating "
            "template element (a logo, letterhead, watermark, or standard "
            "printed header/footer) or does it look like inserted/edited "
            "content (retyped text, a pasted-in block, an obvious visual "
            "seam)? Judge each region using ONLY what is visible in that "
            "region's own crop — do not guess about the rest of the "
            "document, and do not let one region's verdict influence "
            "another's.\n\n"
            "Respond with ONLY a JSON array (no markdown fences, no prose), "
            "one object per region, in this exact shape:\n"
            '[{"index": 1, "label": "template-element|possible-edit|uncertain", '
            '"reasoning": "one sentence"}, ...]\n'
            f"Return exactly {n} objects, with \"index\" values 1 through {n}."
        )
        content = [{"type": "text", "text": prompt}]
        for i, img in enumerate(images, start=1):
            content.append({"type": "text", "text": f"REGION {i}:"})
            content.append(self._image_content(img))

        try:
            raw = self._chat_vision(content, max_tokens=JOB_B_MAX_TOKENS, timeout=REQUEST_TIMEOUT_SECONDS)
        except NvidiaRequestError as e:
            unavailable = {"label": "unavailable",
                           "reasoning": f"AI review unavailable for this item — {e}"}
            return [dict(unavailable) for _ in images]

        return parse_region_batch(raw, n, provider_label="NVIDIA NIM")

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
        # enable_thinking=False for Job A so the reasoning model returns a
        # clean JSON object instead of chain-of-thought preamble + JSON —
        # Job A needs a directly-parseable reply, not a reasoning trace.
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
