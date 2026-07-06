"""
System routes — /health, /producers. Relocated verbatim out of main.py
(Phase 2 folder reorganization) — no logic changes; only the @app.get
decorator became an APIRouter route and imports were adjusted for the new
package layout.
"""

import json

from fastapi import APIRouter

from analyzers.metadata_extractor import PRODUCER_DB, _DB_PATH as _PRODUCER_DB_PATH
from models import HealthResponse

router = APIRouter()

# Top-level metadata (version/description) from producer_database.json —
# metadata_extractor.PRODUCER_DB only exposes the flat "producers" list
# (that's the shape _identify_source() needs), so these are read separately
# for the /producers endpoint.
try:
    with open(_PRODUCER_DB_PATH, "r", encoding="utf-8") as _f:
        _producer_db_raw = json.load(_f)
    PRODUCER_DB_VERSION     = _producer_db_raw.get("version", "unknown")
    PRODUCER_DB_DESCRIPTION = _producer_db_raw.get("description", "")
except Exception:
    PRODUCER_DB_VERSION     = "unknown"
    PRODUCER_DB_DESCRIPTION = ""


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if the API is running."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        layers=["metadata", "content", "ocr", "numeric", "ela", "pymupdf"],
    )


@router.get("/producers", tags=["System"])
async def list_producers():
    """
    Return the full producer/creator fingerprint database (from
    producer_database.json) so callers can see what sources are recognized
    and at what suspicion level, without reading the JSON file directly.
    """
    return {
        "version": PRODUCER_DB_VERSION,
        "description": PRODUCER_DB_DESCRIPTION,
        "count": len(PRODUCER_DB),
        "producers": PRODUCER_DB,
    }
