"""ELA region/report dataclasses."""

from dataclasses import dataclass, field

from .constants import RENDER_DPI


@dataclass
class ELARegion:
    page: int
    bbox: tuple        # (x0, y0, x1, y1) in PDF points — resolution-independent
    mean_error: float
    z_score: float
    render_dpi: float = RENDER_DPI  # DPI this region's block was measured at

    # Multi-scale / multi-signal confirmation metadata (high-resolution
    # analysis — see RENDER_SCALES). Populated only for ELA-derived regions;
    # noise-consistency regions set noise_anomaly directly without going
    # through scale confirmation.
    confirmed_scales: list = field(default_factory=list)  # e.g. ["low","medium","high"]
    sharpness_anomaly: bool = False
    noise_anomaly: bool = False
    erasure_anomaly: bool = False
    score_weight: float = 1.0  # scanned-doc header/footer zones count for less (see SCANNED_* constants)

    # Flat/pasted-patch detection (see FLAT_ZONE_* constants). flat_confidence
    # is the shared detector's 0-1 confidence; stamp_associated marks a patch
    # that geometrically contains/surrounds an isolated stamp-ink component.
    flat_zone_anomaly: bool = False
    stamp_associated: bool = False
    flat_confidence: float = 0.0
    detail: str = ""


@dataclass
class ELAReport:
    pdf_type: str
    anomaly_score: int
    regions: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    incremental_updates: dict = field(default_factory=dict)


