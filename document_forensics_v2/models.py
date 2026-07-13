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
    xref: int = 0


class SuspiciousLine(BaseModel):
    page: int
    line_num: int
    text: str
    anomaly_score_pct: int        # 0-100
    reasons: list[str]
    bbox: Optional[list[float]] = None


class NumericAnomaly(BaseModel):
    page: int
    line_num: int
    text: str
    value: float
    z_score: float
    reason: str
    bbox: Optional[list[float]] = None


class ConfidenceDetail(BaseModel):
    score: int                    # 0-100
    label: str                    # "LOW" | "MEDIUM" | "HIGH" | "VERY HIGH"
    explanation: str              # human readable explanation of what drove the verdict


class FontDetail(BaseModel):
    name: str
    type: str
    encoding: str
    embedded: bool
    tool_signature: Optional[dict] = None


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
    trapped: Optional[str] = None

    # Content flags
    has_javascript: bool = False
    js_context: str = "none"  # "none" | "names_tree" | "open_action" | "page_level"
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

    # ICC color profiles + page rotation consistency
    icc_profiles: list[str] = []
    has_icc_profiles: bool = False
    icc_profile_details: list = []  # parsed profile description/creator/manufacturer
    page_rotation: dict = {}

    # Comprehensive forensic-report sections (commercial-tool parity)
    raw: dict = {}                  # all original /Info + XMP fields
    structure: dict = {}            # page-by-page content structure
    suspicious_content: dict = {}   # JavaScript / actions / embedded files
    dimensions: dict = {}           # page size + standard format
    dates: dict = {}                # enriched date analysis
    authenticity: dict = {}         # overall 0-100 authenticity score

    # Phase 2 — completeness extensions
    xmp_mm: dict = {}               # xmpMM:DocumentID/InstanceID/History
    trailer_ids: dict = {}          # both trailer /ID entries + comparison
    object_level_dates: list = []   # /ModDate,/CreationDate found on non-Info objects
    revision_info: dict = {}        # %%EOF count / /Prev pointer (informational)


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


class TextStackingFindingModel(BaseModel):
    """A coordinate-collision text-stacking finding — 2+ different text values
    occupying the same location (new text placed over original without removing
    it). Surfaced so the UI can show both/all colliding values per location,
    not just an annotated box."""
    page: int                     # 1-indexed for display
    bbox: list[float]
    texts: list[str]              # the distinct colliding text values
    overlap_fraction: float       # 0.0-1.0 strongest pairwise overlap
    confidence: str               # HIGH
    description: str


class EmbeddedImageFindingModel(BaseModel):
    """A finding from running the standalone image pipeline's checks on an
    embedded raster image XObject extracted from the PDF (utils/
    embedded_image_forensics) — distinct from page-level ELA findings, which
    analyze whole-page renders. bbox is already mapped into the parent
    page's point space through the image's placement rect."""
    page: int                     # 1-indexed for display
    bbox: list[float]
    label: str                    # e.g. "Embedded Image: Cutout Edge"
    detail: str
    confidence: float             # 0.0-1.0
    evidence_check: str           # image-pipeline check id, e.g. "check9_stamp_boundary"


class ContradictedFindingModel(BaseModel):
    page: int
    bbox: list[float]
    layer: str                    # the layer whose finding is contradicted
    original_description: str     # preserved — the original finding, never deleted
    contradiction_rule: str       # "cross_page_repetition" | "numeric_vs_structural_context"
    contradicting_evidence: str
    weight_reduction_points: int


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

    # OCR word-level anomalies (font size / color / position vs document
    # baseline) and the document-wide stats they were compared against.
    ocr_word_anomalies: list = []
    ocr_stats: dict = {}

    # Incremental-update / old-object-recovery findings (Layer 5 / ELA) —
    # %%EOF/xref counts, /Prev trailer pointer, and any shadowed earlier
    # object versions recovered from the raw file bytes.
    incremental_updates: dict = {}

    # Cross-layer signal fusion — high-confidence findings confirmed by 2+
    # independent layers, plus suppression statistics
    fused_findings: list[FusedFindingModel] = []
    fusion_stats: Optional[FusionStats] = None

    # Contradiction-aware fusion (Phase 1, additive) — findings whose layer
    # score was reduced (never deleted) because independent structural
    # evidence from another layer undermined them.
    contradicted_findings: list[ContradictedFindingModel] = []

    # Coordinate-collision text stacking — locations where 2+ different text
    # values occupy the same coordinates (new text placed over original without
    # removing it). Empty for documents with no such collision.
    text_stacking_findings: list[TextStackingFindingModel] = []

    # Embedded-image forensics — the image pipeline's checks run on raster
    # image XObjects extracted from the PDF itself. Empty when no embedded
    # image qualifies or none shows anomalies. (Declared here because
    # response_model filtering drops undeclared keys from the payload.)
    embedded_image_findings: list[EmbeddedImageFindingModel] = []

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
