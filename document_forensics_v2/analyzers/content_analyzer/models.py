"""Content-layer dataclasses: per-line profile, suspicious line, and
the layer report."""

from dataclasses import dataclass, field


@dataclass
class LineProfile:
    page: int
    line_num: int
    text: str
    font_name: str
    font_size: float
    char_spacing: float
    word_spacing: float
    line_height: float
    bbox: tuple            # (x0, y0, x1, y1) PDF points
    noise: float           # visual noise of line region
    sharpness: float       # visual sharpness of line region
    char_widths: list = field(default_factory=list)  # per-word avg char width samples, for CV check


@dataclass
class SuspiciousLine:
    page: int
    line_num: int
    text: str
    bbox: tuple
    anomalies: list[str]   # what specifically is wrong
    score: float           # 0.0 - 1.0


@dataclass
class ContentReport:
    total_lines: int
    suspicious_lines: list[SuspiciousLine]
    dominant_font: str
    dominant_font_ratio: float   # 0.0 - 1.0
    font_count: int              # how many unique fonts
    anomaly_score: int           # 0-100
    signals: list[str]           # human readable summary signals
    pdf_type: str                # native_text | scanned | mixed

    # Every line THIS analyzer already classifies as structural/repeated
    # page furniture via _is_structural_line() (headers, footers, labels,
    # lines repeated on another page) — even though such lines never
    # become a SuspiciousLine finding here. Exposed for cross-layer fusion
    # (signal_fusion.py's contradiction detection) so OTHER layers' findings
    # on the same kind of region can be recognized too, not just content's
    # own font-mismatch pathway. Purely additive: no existing anomaly-
    # detection call site's behavior changes.
    structural_line_locations: list = field(default_factory=list)


# ── Feature extraction ─────────────────────────────────────────────────────────
