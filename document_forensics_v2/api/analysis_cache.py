"""
In-memory analysis cache — stores the last MAX_CACHED_ANALYSES analysis
results + pdf paths so /annotated-image, /hidden-text, and /ai-review can
re-render/re-review a page without re-uploading. OrderedDict evicts the
oldest entry once the limit is reached.

Lives in its own module (rather than in main.py or api/analysis_routes.py)
because BOTH api/analysis_routes.py (which writes to it in /analyze and
reads it in /annotated-image, /hidden-text) and api/ai_review_routes.py
(which reads/writes it in /ai-review) need the exact same dict instance —
having one router import internals from the other would be backwards, so
this shared piece of state gets its own leaf module instead.

Relocated verbatim out of main.py (Phase 2 folder reorganization) — no
logic changes.
"""

from collections import OrderedDict

MAX_CACHED_ANALYSES = 100
_analysis_cache: OrderedDict = OrderedDict()
