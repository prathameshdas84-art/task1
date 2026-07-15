from dataclasses import dataclass

@dataclass
class OverlayRegion:
    page: int
    bbox: tuple          # (x0, y0, x1, y1) in PDF points
    overlay_type: str    # "covering_rect" | "image_overlay" | "char_spacing" | "ghost_text" | "coordinate_overwrite"
    reason: str


@dataclass
class PyMuPDFReport:
    pages_analyzed: int
    overlay_regions: list[OverlayRegion]
    anomaly_score: int
    signals: list[str]
