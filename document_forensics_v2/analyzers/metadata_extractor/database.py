"""Producer/creator fingerprint database, scoring weights, and the
small parse helpers (PDF dates, source identification)."""

import json
import os
import re
from datetime import datetime
from typing import Optional

from .models import SourceInfo



# ── Source fingerprint database ────────────────────────────────────────────────
# Producer/creator keyword fingerprints live in producer_database.json, not
# hardcoded here — that file can be extended (new tools, new categories)
# without touching this module. Categories map to the is_online_tool /
# is_editor / is_generator / is_scanner flags below.

# producer_database.json lives in the project-root data/ folder — go up
# two levels (this file sits in analyzers/metadata_extractor/) to reach it.
_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "producer_database.json",
)

_ONLINE_EDITOR_CATEGORY  = "online_editor"
_DESKTOP_EDITOR_CATEGORY = "desktop_editor"
_GENERATOR_CATEGORIES    = ("pdf_library",)
_SCANNER_CATEGORY        = "scanner"


def _load_producer_db() -> list:
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["producers"]
    except Exception:
        return []


PRODUCER_DB = _load_producer_db()

# Tolerance windows for date-based anomaly checks. Both are heuristic and
# were chosen, not measured — XMP/DocInfo clock skew of under a minute is
# common from normal save round-trips; a sub-5-second creation-to-modification
# gap is what an automated pipeline (e.g. a converter) produces, a human
# editing session never finishes that fast.
XMP_MISMATCH_TOLERANCE_SECONDS       = 60
INSTANT_TIMESTAMP_TOLERANCE_SECONDS  = 5

# Anomaly score weights — each constant name matches the signal that adds it
# in _detect_anomalies(). Centralized here instead of as inline literals so
# the relative weighting is visible in one place and easy to retune.
SCORE_ONLINE_TOOL          = 40
SCORE_EDITOR_MEDIUM         = 20
SCORE_UNKNOWN_SOURCE        = 15
SCORE_METADATA_STRIPPED     = 25
SCORE_XMP_MISMATCH          = 30
SCORE_INSTANT_TIMESTAMP     = 15
SCORE_MODIFIED_LATER        = 5
SCORE_MULTIPLE_PRODUCERS    = 20
SCORE_XMP_PRODUCER_MISMATCH = 15
SCORE_POSSIBLE_IMG_CONVERT  = 15
SCORE_XMP_METADATA_DATE_MISMATCH = 20
SCORE_ROTATION_INCONSISTENCY     = 15

# suspicion -> anomaly score. The producer database now carries an explicit
# suspicion per entry (e.g. PDF24 is MEDIUM, not the same HIGH bucket as
# Smallpdf/iLovePDF), so _detect_anomalies() scores directly off this field
# instead of a separately-derived is_online_tool flag that would force every
# "online_editor"-category entry into the HIGH/40-point bucket regardless of
# what suspicion the database actually assigned it.
SUSPICION_SCORE = {
    "HIGH": SCORE_ONLINE_TOOL,
    "MEDIUM": SCORE_EDITOR_MEDIUM,
    "UNKNOWN": SCORE_UNKNOWN_SOURCE,
    "LOW": 0,
}


# ── Data structures ────────────────────────────────────────────────────────────



def _parse_pdf_date(date_str: str) -> Optional[datetime]:
    """Parse PDF date format: D:YYYYMMDDHHmmSSOHH'mm'"""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    # Remove D: prefix
    if date_str.startswith("D:"):
        date_str = date_str[2:]
    # Try parsing just the numeric part
    match = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", date_str)
    if match:
        try:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), int(match.group(5)), int(match.group(6))
            )
        except ValueError:
            return None
    # Try ISO format (XMP)
    try:
        return datetime.fromisoformat(date_str[:19])
    except Exception:
        return None


# ── Source identifier ──────────────────────────────────────────────────────────

def _identify_source(producer: str, creator: str) -> SourceInfo:
    """Match producer/creator against the producer_database.json fingerprints."""
    combined = f"{producer} {creator}".lower()

    identified_name = "Unknown"
    suspicion_level = "UNKNOWN"
    category = "unknown"

    for entry in PRODUCER_DB:
        if entry["pattern"] in combined:
            identified_name = entry["name"]
            suspicion_level = entry["suspicion"]
            category = entry["category"]
            break

    is_online = category == _ONLINE_EDITOR_CATEGORY
    is_editor = category == _DESKTOP_EDITOR_CATEGORY
    is_gen    = category in _GENERATOR_CATEGORIES
    is_scan   = category == _SCANNER_CATEGORY

    return SourceInfo(
        raw_producer=producer,
        raw_creator=creator,
        identified_name=identified_name,
        suspicion_level=suspicion_level,
        is_online_tool=is_online,
        is_editor=is_editor,
        is_generator=is_gen,
        is_scanner=is_scan,
    )


# ── Main Extractor ─────────────────────────────────────────────────────────────

