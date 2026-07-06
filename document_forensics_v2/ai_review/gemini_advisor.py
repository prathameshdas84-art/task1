"""
Gemini Advisor — opt-in, supplementary AI review. One of two interchangeable
AI Review providers (see ai_review/nvidia_advisor.py for the other) selected
via AI_REVIEW_PROVIDER — Gemini is the default when unset.

Every provider implements the SAME three-method shape so api/ai_review_
routes.py never needs to know which one is active:
  explain(analysis_summary) -> ({"lead_sentence": str, "detail": str}, prompt)
  review_regions(images: list) -> list                    # Job B, batched
  independent_scan(page_images, analysis_summary) -> dict  # Job C

Scope is three jobs:
  Job A (explain) — explain the EXISTING 6-layer verdict (and, since it now
          runs LAST, this SAME AI review's own Job B/C results) in plain
          English. Never asked to form its own opinion on whether the
          document is fake beyond synthesizing what B/C already found.
  Job B (review_regions) — for regions the deterministic engine ALREADY
          flagged, look at just that cropped image and say whether it reads
          as a template element (logo/letterhead/watermark) or a possible
          edit. Never shown the whole page, never asked to hunt for new
          anomalies. Regions are labeled in ONE batched call rather than one
          HTTP call per region, to bound total call count against rate
          limits — the original one-region-at-a-time label_region() is kept
          as a thin wrapper around the batch call.
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
    REGION_LABELS,
    parse_explanation,
    parse_region_batch,
    parse_cross_examination,
)

logger = logging.getLogger("document_forensics")

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_API_BASE  = "https://generativelanguage.googleapis.com/v1beta/models"

# Job A (text explainer) / Job B (small region crops) default. Job C sends
# full rendered page image(s) PLUS the engine's whole analysis JSON in one
# request — a much larger payload, and a harder reasoning task (per-finding
# verification + independent scan + verdict), so it gets a longer timeout.
REQUEST_TIMEOUT_SECONDS       = 30
JOB_C_REQUEST_TIMEOUT_SECONDS = 90


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

    # ── Job A — plain-English explanation, run LAST so it can synthesize
    # Job B/C's own results rather than confidently restating a pre-AI-review
    # narrative while they simultaneously contradict it ─────────────────────

    def explain(self, analysis_summary: dict) -> tuple[dict, str]:
        """
        analysis_summary is a narrow, pre-built dict — the deterministic
        findings (layers, signals, fused_findings, suspicious_lines,
        numeric_anomalies, summary) PLUS, since Job A now runs AFTER Job B
        and Job C, that SAME AI review's own results if present:
        job_b_region_verdicts, job_c_per_finding_verification,
        job_c_additional_findings, job_c_overall_assessment, the FINAL
        combined_score_with_ai, and ai_adjusted_verdict (the verdict label
        already computed FROM combined_score_with_ai, using the same
        threshold logic as the deterministic engine — Job A is told to
        state this given value, not compute its own).

        Returns ({"lead_sentence": str, "detail": str}, prompt_sent) — the
        prompt is returned so callers can log/display it for auditability.
        A malformed reply degrades to {"lead_sentence": None, "detail":
        <raw text>} rather than losing the explanation outright.
        """
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
        logger.info("gemini_advisor: Job A prompt sent:\n%s", prompt)

        raw = self._generate_text(prompt)
        return parse_explanation(raw, provider_label="Gemini"), prompt

    # ── Job B — label ONE already-flagged region crop ──────────────────────────

    def label_region(self, image_bytes: bytes, mime_type: str = "image/png") -> dict:
        """
        image_bytes is a crop of ONLY the already-flagged bounding box (never
        the whole page) — the caller (main.py) is responsible for cropping.
        Returns {"label": one of REGION_LABELS, "reasoning": str}.
        """
        prompt = (
            "This image is a small cropped region from a document page that "
            "an automated forensic tool already flagged as unusual. Does this "
            "cropped region look like a repeating template element (a logo, "
            "letterhead, watermark, or standard printed header/footer) or "
            "does it look like an inserted/edited piece of content (retyped "
            "text, a pasted-in block, an obvious visual seam)? Answer only "
            "from what is visible in this crop — do not guess about the rest "
            "of the document.\n\n"
            "Respond in EXACTLY this format, two lines:\n"
            "LABEL: <template-element|possible-edit|uncertain>\n"
            "REASON: <one sentence>"
        )
        raw = self._generate_text(prompt, image_bytes=image_bytes, mime_type=mime_type)
        return self._parse_region_reply(raw)

    @staticmethod
    def _parse_region_reply(raw: str) -> dict:
        label = "uncertain"
        reason = raw.strip()[:280] if raw else "No usable response from the model."
        for line in (raw or "").splitlines():
            line = line.strip()
            if line.upper().startswith("LABEL:"):
                candidate = line.split(":", 1)[1].strip().lower()
                if candidate in REGION_LABELS:
                    label = candidate
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip() or reason
        return {"label": label, "reasoning": reason}

    # ── Job B (batched) — label MULTIPLE already-flagged region crops in ONE
    # Gemini call instead of one HTTP call per region ───────────────────────

    def review_regions(self, images: list) -> list:
        """
        images: list of PNG crop bytes, each ONE already-flagged region.
        Returns a list (same length/order as `images`) of
        {"label": one of REGION_LABELS or "unavailable", "reasoning": str}.
        A batch-level failure (rate limit surviving retries, network error,
        etc.) marks EVERY region "unavailable" rather than raising past this
        call — the caller shouldn't have to handle a total-failure exception
        differently from a partial one.
        """
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
        parts = [{"text": prompt}]
        for i, img in enumerate(images, start=1):
            parts.append({"text": f"REGION {i}:"})
            parts.append(self._image_part(img))

        try:
            raw = self._generate(parts)
        except GeminiRequestError as e:
            unavailable = {"label": "unavailable",
                           "reasoning": f"AI review unavailable for this item — {e}"}
            return [dict(unavailable) for _ in images]

        return parse_region_batch(raw, n, provider_label="Gemini")

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

    def _generate_text(self, prompt: str, image_bytes: bytes = None,
                        mime_type: str = "image/png",
                        timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
        parts = [{"text": prompt}]
        if image_bytes is not None:
            parts.append(self._image_part(image_bytes, mime_type))
        return self._generate(parts, timeout=timeout)

    def _generate(self, parts: list, timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
        """Core call shared by every job. Retry/backoff/timeout handling is
        delegated to utils.ai_retry.post_with_retry (shared with every other
        AI Review provider) — only the request/response shape below (native
        Gemini REST, not OpenAI-compatible) is Gemini-specific. `timeout` is
        per-call so Job C's much larger full-page-image payloads can use a
        longer budget than Job A/B's smaller ones."""
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
