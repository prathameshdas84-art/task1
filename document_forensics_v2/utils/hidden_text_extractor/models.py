"""Finding/report dataclasses for hidden-text recovery and
coordinate-collision text stacking."""

from dataclasses import dataclass


@dataclass
class HiddenTextFinding:
    page: int
    method: str         # how it was found
    original_text: str  # the hidden/original text
    covering_text: str  # what was placed on top
    bbox: tuple          # location on page
    confidence: str      # HIGH / MEDIUM / LOW
    description: str     # human readable explanation
    field_type: str = "unknown"   # auto-detected: name/amount/date/id_number/address/score/unknown
    plain_explanation: str = ""   # human readable explanation of HOW it was done
    # "replaced" — original was hidden AND different visible text was put in its
    # place (the classic, already-working case). "missing" — original was
    # hidden/removed with NOTHING visibly put in its place (covering_text is
    # empty/whitespace after normalization). Classified centrally in analyze();
    # defaults to "replaced" so any finding built without going through analyze()
    # keeps the historical behavior.
    replacement_type: str = "replaced"   # "replaced" | "missing"


@dataclass
class HiddenTextReport:
    findings: list        # list of HiddenTextFinding
    total_found: int
    recovery_summary: str
    signals: list          # for main report
    conclusion: str = ""    # plain-English summary of the tampering


@dataclass
class TextStackingFinding:
    """A location where 2+ DISTINCT text runs occupy the SAME coordinates.

    NOTE: `page` is 0-indexed (matching the internal fusion/analysis-route
    convention where response building adds +1), NOT 1-indexed like
    HiddenTextFinding — this structure's only consumer is signal_fusion via
    the extra_findings path, which works in 0-indexed page space."""
    page: int              # 0-indexed
    bbox: tuple            # union of the colliding runs, (x0, y0, x1, y1) PDF points
    texts: list            # the distinct colliding text values (raw, order preserved)
    overlap_fraction: float  # strongest pairwise overlap in the cluster (0.0-1.0)
    confidence: str        # always "HIGH" — a coordinate collision is a strong signal
    score: float           # 0.0-1.0 fusion score
    description: str        # human readable explanation

