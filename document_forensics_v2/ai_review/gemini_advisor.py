"""
Gemini Advisor — opt-in, supplementary AI review. One of two interchangeable
AI Review providers (see ai_review/nvidia_advisor.py for the other) selected
via AI_REVIEW_PROVIDER — Gemini is the default when unset.

Every provider implements the SAME two-method shape so api/ai_review_
routes.py never needs to know which one is active:
  review_and_explain(images, analysis_summary)
      -> (regions: list, {"lead_sentence": str, "detail": str}, prompt)
  independent_scan(page_images, analysis_summary) -> dict  # Job C

Scope is two calls (formerly three — Job A and Job B are now ONE request):
  Merged region-review + explanation (review_and_explain) — in a single
          call: (1) for regions the deterministic engine ALREADY flagged,
          look at just that cropped image and say whether it reads as a
          template element (logo/letterhead/watermark) or a possible edit
          — never shown the whole page, never asked to hunt for new
          anomalies, all regions labeled in this one batched call; and
          (2) explain the EXISTING 6-layer verdict in plain English,
          synthesizing its OWN region verdicts from the same reply (and
          Job C's results, already present in analysis_summary when the
          caller ran the independent scan first). One request instead of
          two because the explanation needs exactly the information this
          call already has — no second round-trip required. Never asked
          to form its own opinion on whether the document is fake beyond
          the per-region verdicts and synthesizing what Job C found.
  Job C (independent_scan) — genuine cross-examination, NOT descriptive
          captioning and NOT passive agreement. The model is given BOTH the
          rendered page image(s) AND the engine's own full analysis JSON
          (verdict, layer scores, signals, suspicious lines, numeric
          anomalies, ELA findings, fused findings, metadata) in ONE request,
          and must: (1) independently verify each of the engine's own
          findings against what's actually visible in the document —
          "supported" / "contradicted" / "unverifiable", with specific
          reasoning, not a default rubber-stamp; (2) independently surface
          anything the engine missed; (3) give one overall independent
          assessment of the verdict. This has to be a single combined call
          (not per-page) since cross-page reasoning — e.g. noticing a
          "flagged" header repeats identically on every other page — needs
          more than one page in view at once; it uses a longer timeout
          (JOB_C_REQUEST_TIMEOUT_SECONDS) and the same retry/backoff as
          every other call to manage that.

This module only talks to the Gemini API and parses its responses. It has no
knowledge of PDFs, scoring, or the analysis cache — main.py is responsible
for building the analysis JSON, rendering page images, converting Job C's
pixel bounding boxes to PDF points, and for making sure this module is only
ever invoked from the opt-in /ai-review endpoint, never from the default
upload/analyze flow.

Job A/B/C are all read-only with respect to verdict_engine.combine() — none
of them write back into a MetadataReport/ContentReport/etc. This module
never computes or returns a combined score itself; main.py is responsible
for translating Job B/C output into the SEPARATE combined_score_with_ai
number, which never overwrites the deterministic combined_score.

Retry/backoff/timeout handling is SHARED with every other AI Review
provider via utils/ai_retry.py — only the request/response shape below
(Gemini's native REST format) is provider-specific. GeminiNotConfigured/
GeminiRequestError are aliases of the shared, provider-agnostic exception
classes (not Gemini-specific subclasses), so api/ai_review_routes.py can
catch exactly one exception type regardless of which provider is active.
"""

import json
import logging
import os

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from utils.ai_retry import (
    AIProviderNotConfigured as GeminiNotConfigured,
    AIProviderRequestError as GeminiRequestError,
    post_with_retry,
)
from ai_review.shared_parsing import (
    parse_review_and_explanation,
    parse_cross_examination,
)
from ai_review.shared_prompts import build_review_and_explain_prompt as _build_review_and_explain_prompt

logger = logging.getLogger("document_forensics")

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_API_BASE  = "https://generativelanguage.googleapis.com/v1beta/models"

# Default for small requests. The merged region-review + explanation call
# carries the engine's analysis JSON + up to MAX_AI_REVIEW_REGIONS crops
# and produces region verdicts AND a multi-paragraph explanation in one
# reply, so it gets a middle budget. Job C sends full rendered page
# image(s) PLUS the engine's whole analysis JSON in one request — the
# largest payload and hardest reasoning task (per-finding verification +
# independent scan + verdict), so it gets the longest timeout.
REQUEST_TIMEOUT_SECONDS        = 30
MERGED_REQUEST_TIMEOUT_SECONDS = 60
JOB_C_REQUEST_TIMEOUT_SECONDS  = 90


class GeminiAdvisor:

    def __init__(self):
        if not _REQUESTS_AVAILABLE:
            raise GeminiNotConfigured(
                "The 'requests' package is not installed — the AI Review "
                "feature is unavailable in this environment."
            )
        if not GEMINI_API_KEY:
            raise GeminiNotConfigured(
                "GEMINI_API_KEY is not set — the AI Review feature is "
                "unavailable until this environment variable is configured."
            )
        self.model = GEMINI_MODEL

    # ── Merged region-review + explanation — former Job A + Job B in ONE
    # call: the same request that labels the engine's flagged region crops
    # also writes the plain-English explanation, since it already has every
    # input the explanation needs (engine findings + its own region verdicts
    # + Job C's results when the caller ran the scan first) ─────────────────

    def review_and_explain(self, images: list, analysis_summary: dict) -> tuple:
        """
        images: list of PNG crop bytes, each ONE already-flagged region
        (may be empty — the call is then explanation-only).
        analysis_summary: the deterministic findings (verdict, layers,
        signals, fused_findings, suspicious_lines, numeric_anomalies,
        summary) PLUS, when the caller ran the independent scan (Job C)
        FIRST, that scan's results (independent_scan_ran=True and the
        job_c_* fields) — so the ONE explanation synthesizes engine
        findings + this reply's own region verdicts + any newly-found
        locations, instead of a separate paragraph bolted on later.

        Returns (regions, explanation, prompt_sent):
          regions — same length/order as `images`, each {"label": one of
                    REGION_LABELS or "unavailable", "reasoning": str}
          explanation — {"lead_sentence": str|None, "detail": str}; a
                    malformed reply degrades per-half (regions
                    "unavailable", raw text as detail) rather than losing
                    everything.
        Raises GeminiRequestError only if the call fails outright
        (network/timeout/rate-limit after retries) — the caller treats
        that as the hard-failure fail-fast case.
        """
        n = len(images)
        prompt = _build_review_and_explain_prompt(n, analysis_summary)
        logger.info("gemini_advisor: merged review+explanation prompt sent:\n%s", prompt)

        parts = [{"text": prompt}]
        for i, img in enumerate(images, start=1):
            parts.append({"text": f"REGION {i}:"})
            parts.append(self._image_part(img))

        raw = self._generate(parts, timeout=MERGED_REQUEST_TIMEOUT_SECONDS)
        regions, explanation = parse_review_and_explanation(raw, n, provider_label="Gemini")
        return regions, explanation, prompt

    # ── Job C — genuine cross-examination: the engine's OWN findings AND the
    # actual rendered pages, reasoned over together in ONE call. ────────────

    def independent_scan(self, page_images: list, analysis_summary: dict) -> dict:
        """
        page_images: list of (page_number_1_indexed, png_bytes) — FULL page
        renders (never crops, unlike Job B).
        analysis_summary: the engine's own analysis JSON (verdict,
        combined_score, layer scores, signals, suspicious_lines,
        numeric_anomalies, ela_findings, fused_findings, metadata) — built
        by the caller (main.py).

        Sends BOTH inputs together so Gemini can independently verify each
        engine finding against what's actually visible in the document
        (never just accept the engine's framing), surface anything the
        engine missed, and give one independent overall assessment. This is
        deliberately ONE combined call, not per-page — verifying a claim
        like "this repeats identically across pages" requires more than one
        page in view at the same time.

        Returns {"per_finding_verification": [...], "additional_findings":
        [...], "overall_assessment": {...}} — see _parse_cross_examination
        for the exact per-item shape. Raises GeminiRequestError if the call
        fails outright (network/timeout/rate-limit after retries) or the
        reply can't be parsed into a usable result — unlike Job B's batch
        call, there is no meaningful partial-success default for "no
        overall assessment", so this propagates rather than silently
        returning an empty shell.
        """
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
        parts = [{"text": prompt}]
        for page_num, img in page_images:
            parts.append({"text": f"PAGE {page_num}:"})
            parts.append(self._image_part(img))

        raw = self._generate(parts, timeout=JOB_C_REQUEST_TIMEOUT_SECONDS)
        return parse_cross_examination(raw, provider_label="Gemini")

    # ── shared helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _image_part(image_bytes: bytes, mime_type: str = "image/png") -> dict:
        import base64
        return {"inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }}

    # ── Gemini REST call (plain HTTP via `requests` — already an installed
    # transitive dependency in this project; avoids pulling in the full
    # google-generativeai SDK) ──────────────────────────────────────────────

    def _generate(self, parts: list, timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
        """Core call shared by both calls. Retry/backoff/timeout handling is
        delegated to utils.ai_retry.post_with_retry (shared with every other
        AI Review provider) — only the request/response shape below (native
        Gemini REST, not OpenAI-compatible) is Gemini-specific. `timeout` is
        per-call so Job C's much larger full-page-image payloads can use a
        longer budget than the merged call's smaller one."""
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent"
        resp = post_with_retry(
            url,
            params={"key": GEMINI_API_KEY},
            json_body={"contents": [{"parts": parts}]},
            timeout=timeout,
            provider_label="Gemini API",
        )

        try:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                block_reason = data.get("promptFeedback", {}).get("blockReason")
                raise GeminiRequestError(
                    f"Gemini API returned no candidates"
                    + (f" (blocked: {block_reason})" if block_reason else "")
                )
            text_parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in text_parts).strip()
            if not text:
                raise GeminiRequestError("Gemini API returned an empty response")
            return text
        except (KeyError, ValueError, IndexError) as e:
            raise GeminiRequestError(f"Could not parse Gemini API response: {e}")
