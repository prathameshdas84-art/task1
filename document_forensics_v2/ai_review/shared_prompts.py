"""
Shared prompt construction for the merged region-review + explanation call
(former Job A + Job B, now ONE request — see api/ai_review_routes.py).

Both providers (Gemini, NVIDIA NIM) send this exact same prompt text; only
the transport-level image attachment format differs, and that stays in each
advisor. Job C's prompt remains duplicated inline per-advisor (pre-existing
convention); this new, longer two-task prompt is shared from the start so
the two providers' contracts can't drift apart.
"""

import json


def build_review_and_explain_prompt(n_regions: int, analysis_summary: dict) -> str:
    """One prompt, up to two tasks: (1) label the engine's already-flagged
    region crops template-element/possible-edit/uncertain — judged ONLY
    from each crop, exactly like the former standalone Job B; (2) write the
    plain-English explanation, synthesizing the engine findings in
    analysis_summary, this same reply's own region verdicts, and — when
    analysis_summary carries independent_scan_ran=True — the already-run
    Job C scan's results, into ONE narrative.

    The explanation deliberately does NOT state a final adjusted score:
    the score adjustment is computed deterministically by the caller FROM
    this reply's region verdicts, so it cannot be known inside the reply
    that produces them. The model is told to state the engine's own
    verdict/score and describe the adjustment direction qualitatively."""
    region_task = ""
    if n_regions:
        region_task = (
            f"TASK 1 — FLAGGED-REGION REVIEW: You will be shown {n_regions} "
            "cropped regions from the document, each already flagged by the "
            "deterministic engine as unusual. Each crop is preceded by a "
            "line reading 'REGION <n>:'. For EACH region independently, "
            "decide: does it look like a repeating template element (a "
            "logo, letterhead, watermark, or standard printed header/"
            "footer) or does it look like inserted/edited content (retyped "
            "text, a pasted-in block, an obvious visual seam)? Judge each "
            "region using ONLY what is visible in that region's own crop — "
            "do not use the analysis JSON below to decide a region's "
            "label, do not guess about the rest of the document, and do "
            "not let one region's verdict influence another's.\n\n"
        )

    scan_note = ""
    if analysis_summary.get("independent_scan_ran"):
        scan_note = (
            "The JSON below ALSO includes the results of an independent "
            "full-page AI scan that already ran: "
            "job_c_per_finding_verification (supported/contradicted/"
            "unverifiable per engine finding), job_c_additional_findings "
            "(locations the engine missed), and job_c_overall_assessment. "
            "Weave these into the SAME explanation — one synthesized "
            "narrative covering engine findings, your region verdicts, and "
            "any newly-found locations — not a separate paragraph bolted "
            "on. "
        )

    explain_header = "TASK 2 — " if n_regions else "YOUR TASK — "
    own_verdicts = (
        " AND your own TASK 1 region verdicts from this same reply"
        if n_regions else ""
    )
    challenged = (
        "If any of your TASK 1 verdicts is 'template-element', or any "
        "independent-scan verification is 'contradicted', you MUST "
        "explicitly say so by name — e.g. 'our AI visual review found "
        "that several flagged header regions are actually standard "
        "template elements, not edits' — and note that the adjusted "
        "score shown alongside this explanation will be LOWER than the "
        "engine score because of it. Do NOT silently restate the "
        "original findings list as if nothing challenged them. "
    )

    return (
        "You are performing an AI review of an ALREADY-COMPUTED forensic "
        "document analysis. Complete "
        + ("BOTH tasks below in ONE reply.\n\n" if n_regions else "the task below.\n\n")
        + region_task
        + explain_header
        + "PLAIN-ENGLISH EXPLANATION: Using the deterministic engine's "
        "analysis JSON below"
        + own_verdicts
        + ", explain the analysis to a non-technical reader — translate "
        "z-scores, layer names, and jargon into plain English. "
        + scan_note
        + "CRITICAL: your first sentence MUST plainly state whether the "
        "engine found this document MODIFIED, ORIGINAL, or UNCERTAIN "
        "using EXACTLY the 'verdict' value and the 'combined_score' "
        "number in the JSON — do not compute your own verdict or apply "
        "your own threshold. "
        + challenged
        + "Do NOT state a precise adjusted score anywhere — the system "
        "recomputes the adjusted score from your region verdicts after "
        "this reply; describe the direction of any adjustment "
        "qualitatively instead. Keep the detail section to 3-6 short "
        "paragraphs.\n\n"
        "ENGINE ANALYSIS JSON:\n"
        f"{json.dumps(analysis_summary, indent=2, default=str)}\n\n"
        "Respond with ONLY a JSON object (no markdown fences, no prose) "
        "in EXACTLY this shape:\n"
        '{"regions": [{"index": 1, "label": '
        '"template-element|possible-edit|uncertain", "reasoning": "one '
        'sentence"}, ...],\n'
        '"explanation": {"lead_sentence": "<the one required first '
        "sentence, stating the engine's verdict and combined score>\", "
        '"detail": "<3-6 short paragraphs of supporting explanation; '
        '**bold** is fine for emphasis, no other markdown>"}}\n'
        + (
            f'Return exactly {n_regions} objects in "regions", with '
            f'"index" values 1 through {n_regions}.'
            if n_regions
            else 'Return "regions" as an empty array [].'
        )
    )
