"""Metadata report dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SourceInfo:
    raw_producer: str
    raw_creator: str
    identified_name: str
    suspicion_level: str      # "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
    is_online_tool: bool
    is_editor: bool
    is_generator: bool        # auto-generated (system/script)
    is_scanner: bool


@dataclass
class MetadataReport:
    # Raw fields
    producer: str
    creator: str
    author: str
    creation_date: Optional[datetime]
    modification_date: Optional[datetime]
    title: str
    subject: str
    keywords: str

    # XMP fields
    xmp_create_date: Optional[datetime]
    xmp_modify_date: Optional[datetime]
    xmp_metadata_date: Optional[datetime]
    xmp_producer: str
    xmp_creator_tool: str

    # Derived
    source: SourceInfo
    time_delta_seconds: Optional[float]   # ModDate - CreationDate
    xmp_docinfo_mismatch: bool            # XMP vs DocInfo dates differ
    multiple_producers: bool              # creator != producer significantly
    metadata_stripped: bool              # key fields missing

    # Anomalies found
    anomalies: list[str] = field(default_factory=list)
    anomaly_score: int = 0               # 0-100

    # Extended structural/forensic fields
    pdf_version: Optional[str] = None
    fonts: list = field(default_factory=list)
    is_encrypted: bool = False
    permissions: dict = field(default_factory=dict)
    page_details: list = field(default_factory=list)
    has_embedded_files: bool = False
    has_javascript: bool = False
    js_context: str = "none"  # "none" | "names_tree" | "open_action" | "page_level"
    has_open_action: bool = False
    document_id: Optional[str] = None
    xmp_fields: dict = field(default_factory=dict)
    icc_profiles: list = field(default_factory=list)
    has_icc_profiles: bool = False
    page_rotation: dict = field(default_factory=dict)

    # Phase 2 — completeness extensions
    trapped: Optional[str] = None                                # /Trapped Info-dict entry
    xmp_mm: dict = field(default_factory=dict)                   # xmpMM:DocumentID/InstanceID/History
    trailer_ids: dict = field(default_factory=dict)               # both /ID entries + comparison
    object_level_dates: list = field(default_factory=list)        # /ModDate,/CreationDate on non-Info objects
    icc_profile_details: list = field(default_factory=list)       # parsed ICC profile description/creator
    revision_info: dict = field(default_factory=dict)             # %%EOF count / /Prev pointer (informational)

    # Overall modification age (PDFs store only the LAST mod date, not a
    # per-edit history). Populated by _compute_edit_age() — see that method.
    edit_age: dict = field(default_factory=dict)

    # Comprehensive forensic-report sections (commercial-tool parity).
    # Each is populated by its own _extract_*/_enhance_*/_compute_* helper.
    raw_metadata: dict = field(default_factory=dict)        # every /Info + XMP key
    structure: dict = field(default_factory=dict)            # page-by-page content
    suspicious_content: dict = field(default_factory=dict)   # JS / actions / files
    dimensions_full: dict = field(default_factory=dict)      # page size + format
    dates_full: dict = field(default_factory=dict)           # enriched date analysis
    authenticity: dict = field(default_factory=dict)         # overall 0-100 score


# ── Date parser ────────────────────────────────────────────────────────────────

