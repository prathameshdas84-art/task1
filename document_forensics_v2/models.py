"""
Pydantic response models for the Document Forensics Engine API.
"""

from pydantic import BaseModel
from typing import Optional


class LayerScores(BaseModel):
    metadata: int
    content: int
    ocr: int
    numeric: int
    ela: int
    pymupdf: int = 0


class SuspiciousLine(BaseModel):
    page: int
    line_num: int
    text: str
    anomaly_score_pct: int        # 0-100
    reasons: list[str]


class NumericAnomaly(BaseModel):
    page: int
    line_num: int
    text: str
    value: float
    z_score: float
    reason: str


class ConfidenceDetail(BaseModel):
    score: int                    # 0-100
    label: str                    # "LOW" | "MEDIUM" | "HIGH" | "VERY HIGH"
    explanation: str              # human readable explanation of what drove the verdict


class FontDetail(BaseModel):
    name: str
    type: str
    encoding: str
    embedded: bool


class PageDetail(BaseModel):
    page_number: int
    width_pt: float
    height_pt: float
    rotation: int


class FullMetadata(BaseModel):
    # Basic
    producer: Optional[str] = None
    creator: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None

    # Dates
    created: Optional[str] = None
    modified: Optional[str] = None
    was_modified: bool = False
    modification_interval: Optional[str] = None

    # Edit age — how long ago the document was last modified (PDFs store only
    # the LAST mod date, so this is overall age, not a per-edit history)
    edit_age_days: Optional[int] = None
    edit_age_human: Optional[str] = None
    is_recent_edit: bool = False
    is_very_recent_edit: bool = False

    # Forensic flags
    xmp_mismatch: bool = False
    multiple_producers: bool = False
    source_risk: str = "UNKNOWN"
    source_name: str = "Unknown"

    # Structure
    pdf_version: Optional[str] = None
    total_pages: int = 1
    document_id: Optional[str] = None

    # Content flags
    has_javascript: bool = False
    has_open_action: bool = False
    has_embedded_files: bool = False
    has_images: bool = False

    # Security
    is_encrypted: bool = False
    permissions: dict = {}

    # Fonts
    font_count: int = 0
    fonts: list[FontDetail] = []

    # Pages
    page_details: list[PageDetail] = []

    # Raw XMP fields
    xmp_fields: dict = {}

    # Comprehensive forensic-report sections (commercial-tool parity)
    raw: dict = {}                  # all original /Info + XMP fields
    structure: dict = {}            # page-by-page content structure
    suspicious_content: dict = {}   # JavaScript / actions / embedded files
    dimensions: dict = {}           # page size + standard format
    dates: dict = {}                # enriched date analysis
    authenticity: dict = {}         # overall 0-100 authenticity score


class FusedFindingModel(BaseModel):
    page: int
    bbox: list[float]
    confirming_layers: list[str]
    confidence: str               # HIGH, MEDIUM, LOW
    score: int
    description: str


class FusionStats(BaseModel):
    total_findings_input: int = 0
    high_confidence_findings: int = 0
    single_layer_suppressed: int = 0
    fusion_groups: int = 0


class ForensicResponse(BaseModel):
    # Core verdict
    verdict: str                  # "MODIFIED" | "ORIGINAL" | "UNCERTAIN"
    combined_score: float         # 0-100
    confidence: ConfidenceDetail

    # Document info
    filename: str
    file_size_kb: float
    pdf_type: str
    document_source: str
    processing_time_seconds: float

    # Layer breakdown
    layers: LayerScores

    # Evidence
    signals: list[str]
    suspicious_lines: list[SuspiciousLine]
    numeric_anomalies: list[NumericAnomaly]

    # Cross-layer signal fusion — high-confidence findings confirmed by 2+
    # independent layers, plus suppression statistics
    fused_findings: list[FusedFindingModel] = []
    fusion_stats: Optional[FusionStats] = None

    # Summary
    summary: str

    # Session metadata
    analysis_id: Optional[str] = None
    total_pages: int = 1

    # Full extended metadata (Layer 1 deep dive)
    metadata: Optional[FullMetadata] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    layers: list[str]
