"""
Shared HTTP retry/backoff for AI Review provider calls (Gemini, NVIDIA NIM,
and any future provider) — extracted so the retry POLICY (which failures
are retryable, how backoff grows, what counts as a hard failure) lives in
exactly one place instead of being duplicated per-provider. Each provider
module still owns its own request-body shape and response parsing (those
differ completely between Gemini's native format and NVIDIA NIM's OpenAI-
compatible format) — this module only owns the transport-level retry loop.

Both exception types are intentionally provider-agnostic (not
Gemini-specific or NVIDIA-specific subclasses): api/ai_review_routes.py
catches exactly these two types regardless of which provider is active,
so nothing downstream needs a per-provider except-branch.
"""

import logging
import time

import requests

logger = logging.getLogger("document_forensics")

# Retry/backoff for transient failures only — a 429 rate limit, a 502/503/
# 504 capacity/gateway blip (NVIDIA NIM returns 503 "Worker local total
# request limit reached" under load — retryable, unlike a bad key), OR a
# request timeout/connection error. Other 4xx/5xx are not retried since
# waiting won't fix a bad key or a malformed request.
MAX_RETRY_ATTEMPTS         = 3   # total attempts, including the first
RETRY_BACKOFF_BASE_SECONDS = 2   # attempt N (1-indexed) waits BASE * 2**(N-1)s: 2s, 4s


class AIProviderNotConfigured(Exception):
    """No API key is set (or the provider's SDK/dependency is unavailable)
    for the currently-selected AI Review provider — callers should treat
    this as an unavailable feature, not an error to surface as a crash."""


class AIProviderRequestError(Exception):
    """An AI Review provider call failed (network error, timeout, rate
    limit after retries, non-2xx response, or an unparseable reply).
    Carries a short human-readable reason."""


def post_with_retry(url: str, *, headers: dict = None, params: dict = None, json_body: dict,
                     timeout: float, provider_label: str = "AI provider") -> requests.Response:
    """
    POSTs json_body to url, retrying up to MAX_RETRY_ATTEMPTS total tries
    with exponential backoff on: a 429 rate-limit response, a request
    timeout, or a connection error. Any other non-2xx status or exception
    raises immediately — waiting won't fix a bad key or a malformed
    request. Returns the raw successful Response; callers parse the body
    themselves, since each provider's response shape differs entirely.

    provider_label is used only in error message text (e.g. "Gemini API",
    "NVIDIA NIM API") so error strings stay provider-specific even though
    the retry mechanics are shared. `params` exists for providers (Gemini)
    that authenticate via a query string instead of a header.
    """
    retryable_error = None

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
        except requests.Timeout:
            retryable_error = AIProviderRequestError(
                f"{provider_label} request timed out after {timeout}s."
            )
            if attempt < MAX_RETRY_ATTEMPTS:
                wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.info(
                    "%s: request timed out (attempt %d/%d), retrying in %ds",
                    provider_label, attempt, MAX_RETRY_ATTEMPTS, wait,
                )
                time.sleep(wait)
                continue
            raise retryable_error
        except requests.ConnectionError as e:
            retryable_error = AIProviderRequestError(f"Network error calling {provider_label}: {e}")
            if attempt < MAX_RETRY_ATTEMPTS:
                wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.info(
                    "%s: connection error (attempt %d/%d), retrying in %ds",
                    provider_label, attempt, MAX_RETRY_ATTEMPTS, wait,
                )
                time.sleep(wait)
                continue
            raise retryable_error
        except requests.RequestException as e:
            raise AIProviderRequestError(f"Network error calling {provider_label}: {e}")

        if resp.status_code in (429, 502, 503, 504):
            retryable_error = AIProviderRequestError(
                f"{provider_label} rate limit reached — try again shortly."
                if resp.status_code == 429 else
                f"{provider_label} temporarily unavailable (HTTP {resp.status_code}) — try again shortly."
            )
            if attempt < MAX_RETRY_ATTEMPTS:
                wait = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.info(
                    "%s: transient HTTP %d (attempt %d/%d), retrying in %ds",
                    provider_label, resp.status_code, attempt, MAX_RETRY_ATTEMPTS, wait,
                )
                time.sleep(wait)
                continue
            raise retryable_error

        if resp.status_code in (401, 403):
            raise AIProviderRequestError(f"{provider_label} rejected the configured API key.")
        if resp.status_code >= 400:
            raise AIProviderRequestError(
                f"{provider_label} returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        return resp

    raise retryable_error
