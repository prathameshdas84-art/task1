"""
Gemini Advisor — opt-in, supplementary AI review.

Scope is three jobs:
  Job A — explain the EXISTING 6-layer verdict in plain English. Never asked
          to form its own opinion on whether the document is fake.
  Job B — for regions the deterministic engine ALREADY flagged, look at just
          that cropped image and say whether it reads as a template element
          (logo/letterhead/watermark) or a possible edit. Never shown the
          whole page, never asked to hunt for new anomalies. Regions are
          labeled in ONE batched Gemini call (label_regions_batch) rather
          than one HTTP call per region, to bound total call count against
          rate limits — the original one-region-at-a-time label_region()
          is kept as a thin wrapper around the batch call.
  Job C — genuine cross-examination (cross_examine_findings), NOT
          descriptive captioning and NOT passive agreement. Gemini is given
          BOTH the rendered page image(s) AND the engine's own full
          analysis JSON (verdict, layer scores, signals, suspicious lines,
          numeric anomalies, ELA findings, fused findings, metadata) in ONE
          request, and must: (1) independently verify each of the engine's
          own findings against what's actually visible in the document —
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
"""

import json
import logging
import os
import time

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

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

# Retry/backoff for transient failures only — a 429 rate limit OR a request
# timeout/connection error. Other 4xx/5xx are not retried since waiting
# won't fix a bad key or a malformed request.
MAX_RETRY_ATTEMPTS         = 3   # total attempts, including the first
RETRY_BACKOFF_BASE_SECONDS = 2   # attempt N (1-indexed) waits BASE * 2**(N-1)s: 2s, 4s

REGION_LABELS = ("template-element", "possible-edit", "uncertain")
EDIT_CONFIDENCE_LABELS = ("low", "medium", "high")
CROSS_EXAM_VERDICTS = ("supported", "contradicted", "unverifiable")


class GeminiNotConfigured(Exception):
    """No GEMINI_API_KEY is set — the caller should treat this as an
    unavailable feature, not an error to surface as a crash."""


class GeminiRequestError(Exception):
    """The Gemini API call failed (network error, non-2xx response, or an
    unparseable reply). Carries a short human-readable reason."""


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

    # ── Job A — plain-English explanation of the EXISTING verdict ──────────────

    def explain_report(self, analysis_summary: dict) -> tuple[str, str]:
        """
        analysis_summary is a narrow, pre-built dict — just the fields named
        in scope (layers, signals, fused_findings, suspicious_lines,
        numeric_anomalies, summary) — not the full raw API response, so the
        prompt stays focused on explaining, not re-analyzing.

        Returns (explanation_text, prompt_sent) — the prompt is returned so
        callers can log/display it for auditability.
        """
        prompt = (
            "You are explaining an ALREADY-COMPUTED forensic analysis result "
            "to a non-technical reader. Do not contradict or second-guess the "
            "verdict below. Do not introduce new evidence or suspicions of "
            "your own. Explain the reasoning behind it in plain English: "
            "translate z-scores, layer names, and technical jargon into "
            "language a normal person can follow. Keep it to 3-6 short "
            "paragraphs.\n\n"
            "EXISTING ANALYSIS RESULT (verdict + supporting signals, computed "
            "by a separate deterministic statistical engine — you are only "
            "explaining it):\n"
            f"{json.dumps(analysis_summary, indent=2, default=str)}"
        )
        logger.info("gemini_advisor: Job A prompt sent:\n%s", prompt)

        text = self._generate_text(prompt)
        return text, prompt

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

    def label_regions_batch(self, images: list) -> list:
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

        parsed = self._parse_batch_json(raw, n)
        out = []
        for i in range(1, n + 1):
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

    # ── Job C — genuine cross-examination: the engine's OWN findings AND the
    # actual rendered pages, reasoned over together in ONE call. ────────────

    def cross_examine_findings(self, page_images: list, analysis_summary: dict) -> dict:
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
            'numeric|ela|pymupdf|xref|fusion>", "gemini_verdict": '
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
        return self._parse_cross_examination(raw)

    # ── shared helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _image_part(image_bytes: bytes, mime_type: str = "image/png") -> dict:
        import base64
        return {"inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }}

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t.lstrip("`")
            t = t.strip()
            if t.endswith("```"):
                t = t[:-3]
        return t.strip()

    def _parse_batch_json(self, raw: str, expected_n: int) -> dict:
        """Returns {index: {"label": ..., "reasoning": ...}}. Best-effort —
        never raises; malformed/missing entries are simply absent so the
        caller fills them in as 'unavailable'."""
        result = {}
        try:
            data = json.loads(self._strip_json_fences(raw))
        except (ValueError, TypeError):
            logger.warning("gemini_advisor: could not parse Job B batch JSON reply:\n%s", raw)
            return result
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                if isinstance(idx, int) and 1 <= idx <= expected_n:
                    result[idx] = item
        return result

    def _parse_cross_examination(self, raw: str) -> dict:
        """Parses Job C's structured reply. Individual malformed per_finding_
        verification/additional_findings entries are dropped (best-effort —
        one bad item shouldn't lose the rest), but a response missing
        overall_assessment entirely, or that isn't valid JSON, raises
        GeminiRequestError — there's no meaningful partial result for "no
        verdict at all"."""
        try:
            data = json.loads(self._strip_json_fences(raw))
        except (ValueError, TypeError) as e:
            logger.warning("gemini_advisor: could not parse Job C JSON reply:\n%s", raw)
            raise GeminiRequestError(f"Could not parse cross-examination response: {e}")
        if not isinstance(data, dict) or "overall_assessment" not in data:
            raise GeminiRequestError("Cross-examination response is missing overall_assessment.")

        verifications = []
        for item in data.get("per_finding_verification", []) or []:
            if not isinstance(item, dict):
                continue
            verdict = item.get("gemini_verdict", "unverifiable")
            if verdict not in CROSS_EXAM_VERDICTS:
                verdict = "unverifiable"
            verifications.append({
                "engine_finding": (item.get("engine_finding") or "").strip()[:400],
                "layer": (item.get("layer") or "unknown").strip().lower(),
                "gemini_verdict": verdict,
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
        """Core call shared by every job. Retries (MAX_RETRY_ATTEMPTS total,
        exponential backoff) on a 429 rate limit AND on a transient request
        timeout/connection error — a slow response shouldn't fail on the
        first hit any more than a rate limit should. Other error statuses
        (bad key, malformed request) are not retried since waiting won't fix
        them. `timeout` is per-call so Job C's much larger full-page-image
        payloads can use a longer budget than Job A/B's smaller ones."""
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent"
        retryable_error = None

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                resp = requests.post(
                    url,
                    params={"key": GEMINI_API_KEY},
                    json={"contents": [{"parts": parts}]},
                    timeout=timeout,
                )
            except requests.Timeout:
                retryable_error = GeminiRequestError(
                    f"Gemini API request timed out after {timeout}s."
                )
                if attempt < MAX_RETRY_ATTEMPTS:
                    wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.info(
                        "gemini_advisor: request timed out (attempt %d/%d), retrying in %ds",
                        attempt, MAX_RETRY_ATTEMPTS, wait,
                    )
                    time.sleep(wait)
                    continue
                raise retryable_error
            except requests.ConnectionError as e:
                retryable_error = GeminiRequestError(f"Network error calling Gemini API: {e}")
                if attempt < MAX_RETRY_ATTEMPTS:
                    wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.info(
                        "gemini_advisor: connection error (attempt %d/%d), retrying in %ds",
                        attempt, MAX_RETRY_ATTEMPTS, wait,
                    )
                    time.sleep(wait)
                    continue
                raise retryable_error
            except requests.RequestException as e:
                raise GeminiRequestError(f"Network error calling Gemini API: {e}")

            if resp.status_code == 429:
                retryable_error = GeminiRequestError(
                    "Gemini API rate limit reached — try again shortly."
                )
                if attempt < MAX_RETRY_ATTEMPTS:
                    wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.info(
                        "gemini_advisor: rate limited (attempt %d/%d), retrying in %ds",
                        attempt, MAX_RETRY_ATTEMPTS, wait,
                    )
                    time.sleep(wait)
                    continue
                raise retryable_error

            if resp.status_code == 401 or resp.status_code == 403:
                raise GeminiRequestError("Gemini API rejected the configured API key.")
            if resp.status_code >= 400:
                raise GeminiRequestError(f"Gemini API returned HTTP {resp.status_code}: {resp.text[:300]}")

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

        raise retryable_error
