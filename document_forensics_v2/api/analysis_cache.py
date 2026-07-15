"""
Analysis cache — stores the last MAX_CACHED_ANALYSES analysis results +
pdf paths so /annotated-image and /hidden-text can re-render a page
without re-uploading. OrderedDict evicts the oldest entry once the limit
is reached.

The in-memory dict is the fast path, but it does not survive a server
restart (including uvicorn --reload restarts triggered by any source-file
edit) — and the analyzed PDF itself DOES survive, since it lives in the
OS temp dir and is only deleted on eviction. So every entry is also
persisted to a disk spool (pickle, minus any non-picklable render cache),
and get_analysis() rehydrates from the spool on a memory miss. That turns
"server restarted → re-upload and re-analyze everything" into a
transparent reload.

Lives in its own module (rather than in main.py or api/analysis_routes)
so the analyze/render/hidden-text endpoint modules share one instance.
"""

import logging
import os
import pickle
import tempfile
from collections import OrderedDict

logger = logging.getLogger(__name__)

MAX_CACHED_ANALYSES = 100
_analysis_cache: OrderedDict = OrderedDict()

# ── Disk spool (restart survival) ───────────────────────────────────────────
# One pickle per analysis_id. analysis_id is a server-generated uuid4 (never
# user input), so the filename is safe and unpickling only ever reads files
# this process (or a previous run of it) wrote.
SPOOL_DIR = os.path.join(tempfile.gettempdir(), "forensics_analysis_cache")

# Keys that must not be persisted: highlighted_pages holds rendered PIL
# images (large, and regenerable from the rest of the entry on demand).
_TRANSIENT_KEYS = ("highlighted_pages",)


def _spool_path(analysis_id: str) -> str:
    return os.path.join(SPOOL_DIR, f"{analysis_id}.pkl")


def persist_analysis(analysis_id: str, entry: dict) -> None:
    """Write a cache entry to the disk spool (best-effort — a persistence
    failure must never fail the /analyze request that produced the entry)."""
    try:
        os.makedirs(SPOOL_DIR, exist_ok=True)
        slim = {k: v for k, v in entry.items() if k not in _TRANSIENT_KEYS}
        with open(_spool_path(analysis_id), "wb") as f:
            pickle.dump(slim, f)
        _prune_spool()
    except Exception as e:
        logger.warning("analysis_cache: could not persist %s to spool: %s",
                       analysis_id, e)


def _prune_spool() -> None:
    """Keep the spool bounded to MAX_CACHED_ANALYSES entries (oldest first),
    deleting each pruned entry's analyzed PDF alongside its pickle."""
    try:
        pkls = [os.path.join(SPOOL_DIR, n) for n in os.listdir(SPOOL_DIR)
                if n.endswith(".pkl")]
        if len(pkls) <= MAX_CACHED_ANALYSES:
            return
        pkls.sort(key=os.path.getmtime)
        for path in pkls[:len(pkls) - MAX_CACHED_ANALYSES]:
            try:
                with open(path, "rb") as f:
                    old = pickle.load(f)
                old_pdf = old.get("pdf_path")
                if old_pdf and os.path.exists(old_pdf):
                    os.unlink(old_pdf)
            except Exception:
                pass
            os.unlink(path)
    except Exception:
        pass


def get_analysis(analysis_id: str):
    """Fetch a cache entry: memory first, then the disk spool. A spool hit
    is rehydrated into memory (with normal eviction) so subsequent page
    requests take the fast path again. Returns None when the id is unknown
    or its analyzed PDF no longer exists."""
    entry = _analysis_cache.get(analysis_id)
    if entry is not None:
        return entry

    path = _spool_path(analysis_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            entry = pickle.load(f)
    except Exception as e:
        logger.warning("analysis_cache: spool entry %s unreadable: %s",
                       analysis_id, e)
        return None
    if not os.path.exists(entry.get("pdf_path", "")):
        # The analyzed PDF is gone (temp cleanup) — the entry can't serve
        # renders anymore, so drop it.
        try:
            os.unlink(path)
        except Exception:
            pass
        return None

    _analysis_cache[analysis_id] = entry
    while len(_analysis_cache) > MAX_CACHED_ANALYSES:
        _analysis_cache.popitem(last=False)
    logger.info("analysis_cache: rehydrated %s from disk spool", analysis_id)
    return entry
