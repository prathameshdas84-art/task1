"""Report dataclasses, CHECK_POINTS-weighted scoring, and the fusion
adapter for the image-document pipeline."""

from dataclasses import dataclass, field

from .constants import CHECK_POINTS, NOT_IMPLEMENTED


def score_anomalies(anomalies: list) -> tuple:
    """CHECK_POINTS-weighted scoring — per_hit × confidence per anomaly,
    capped per check. Shared by analyze() and the scanned-page routing
    (utils/scanned_page_forensics), which must re-score after filtering
    QR-zone hits out so its fold magnitude matches its surviving findings.
    Returns (raw_score_float, per_check_scores_dict)."""
    score = 0.0
    per_check_scores = {}
    for check, cfg in CHECK_POINTS.items():
        hits = [a for a in anomalies if a.evidence_check == check]
        s = min(cfg["cap"], sum(cfg["per_hit"] * a.confidence for a in hits))
        per_check_scores[check] = round(s, 1)
        score += s
    return score, per_check_scores


@dataclass
class ImageAnomaly:
    type: str                # e.g. "inpaint_smoothing", "sharp_overlay_edge"
    bbox: tuple              # (x, y, w, h) in image pixels
    confidence: float        # 0.0-1.0
    evidence_check: str      # which check produced it (see CHECK_POINTS keys)
    page: int = 1            # always 1 for a single image
    detail: str = ""


@dataclass
class ImageForensicsReport:
    is_born_digital: bool
    jpeg_history_detected: bool
    compression_history: str   # single_compression | double_compression_suspected | uncertain | not_applicable
    stamp_detected: bool
    signature_detected: bool
    anomalies: list = field(default_factory=list)          # list[ImageAnomaly]
    not_implemented: list = field(default_factory=lambda: list(NOT_IMPLEMENTED))
    metrics: dict = field(default_factory=dict)
    anomaly_score: int = 0
    signals: list = field(default_factory=list)
    heatmap_png: bytes = None   # Check 10 — display-only evidence, never scored


def normalize_for_fusion(report: ImageForensicsReport) -> list:
    """Convert this report's anomalies into signal_fusion's normalized
    finding shape (dicts with layer/page/bbox/score/text). Each CHECK is
    its own fusion 'layer', so two different checks co-locating on the
    same region cross-validate through the existing 2+-layer agreement
    logic with zero special-casing. bboxes convert (x,y,w,h) → (x0,y0,x1,y1);
    pages stay 0-indexed inside fusion like every PDF layer's findings."""
    findings = []
    for a in report.anomalies:
        x, y, w, h = a.bbox
        findings.append({
            "layer": f"image_{a.evidence_check}",
            "page": a.page - 1,
            "bbox": (float(x), float(y), float(x + w), float(y + h)),
            "line_num": None,
            "text": a.detail or a.type,
            "score": float(a.confidence),
            "raw": a,
        })
    return findings
