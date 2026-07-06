"""
Metadata Extractor — Document Forensics Engine
Extracts all metadata from any PDF and identifies its origin.
"""

import json
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import fitz      # PyMuPDF
import pikepdf
from pypdf import PdfReader


# ── Source fingerprint database ────────────────────────────────────────────────
# Producer/creator keyword fingerprints live in producer_database.json, not
# hardcoded here — that file can be extended (new tools, new categories)
# without touching this module. Categories map to the is_online_tool /
# is_editor / is_generator / is_scanner flags below.

# producer_database.json lives at the project root, not alongside this file
# (this module is inside analyzers/ as of the Phase 2 folder reorg) — go up
# one level from analyzers/ to reach it.
_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "producer_database.json")

_ONLINE_EDITOR_CATEGORY  = "online_editor"
_DESKTOP_EDITOR_CATEGORY = "desktop_editor"
_GENERATOR_CATEGORIES    = ("pdf_library",)
_SCANNER_CATEGORY        = "scanner"


def _load_producer_db() -> list:
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["producers"]
    except Exception:
        return []


PRODUCER_DB = _load_producer_db()

# Tolerance windows for date-based anomaly checks. Both are heuristic and
# were chosen, not measured — XMP/DocInfo clock skew of under a minute is
# common from normal save round-trips; a sub-5-second creation-to-modification
# gap is what an automated pipeline (e.g. a converter) produces, a human
# editing session never finishes that fast.
XMP_MISMATCH_TOLERANCE_SECONDS       = 60
INSTANT_TIMESTAMP_TOLERANCE_SECONDS  = 5

# Anomaly score weights — each constant name matches the signal that adds it
# in _detect_anomalies(). Centralized here instead of as inline literals so
# the relative weighting is visible in one place and easy to retune.
SCORE_ONLINE_TOOL          = 40
SCORE_EDITOR_MEDIUM         = 20
SCORE_UNKNOWN_SOURCE        = 15
SCORE_METADATA_STRIPPED     = 25
SCORE_XMP_MISMATCH          = 30
SCORE_INSTANT_TIMESTAMP     = 15
SCORE_MODIFIED_LATER        = 5
SCORE_MULTIPLE_PRODUCERS    = 20
SCORE_XMP_PRODUCER_MISMATCH = 15
SCORE_POSSIBLE_IMG_CONVERT  = 15
SCORE_XMP_METADATA_DATE_MISMATCH = 20
SCORE_ROTATION_INCONSISTENCY     = 15

# suspicion -> anomaly score. The producer database now carries an explicit
# suspicion per entry (e.g. PDF24 is MEDIUM, not the same HIGH bucket as
# Smallpdf/iLovePDF), so _detect_anomalies() scores directly off this field
# instead of a separately-derived is_online_tool flag that would force every
# "online_editor"-category entry into the HIGH/40-point bucket regardless of
# what suspicion the database actually assigned it.
SUSPICION_SCORE = {
    "HIGH": SCORE_ONLINE_TOOL,
    "MEDIUM": SCORE_EDITOR_MEDIUM,
    "UNKNOWN": SCORE_UNKNOWN_SOURCE,
    "LOW": 0,
}


# ── Data structures ────────────────────────────────────────────────────────────

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

def _parse_pdf_date(date_str: str) -> Optional[datetime]:
    """Parse PDF date format: D:YYYYMMDDHHmmSSOHH'mm'"""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    # Remove D: prefix
    if date_str.startswith("D:"):
        date_str = date_str[2:]
    # Try parsing just the numeric part
    match = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", date_str)
    if match:
        try:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), int(match.group(5)), int(match.group(6))
            )
        except ValueError:
            return None
    # Try ISO format (XMP)
    try:
        return datetime.fromisoformat(date_str[:19])
    except Exception:
        return None


# ── Source identifier ──────────────────────────────────────────────────────────

def _identify_source(producer: str, creator: str) -> SourceInfo:
    """Match producer/creator against the producer_database.json fingerprints."""
    combined = f"{producer} {creator}".lower()

    identified_name = "Unknown"
    suspicion_level = "UNKNOWN"
    category = "unknown"

    for entry in PRODUCER_DB:
        if entry["pattern"] in combined:
            identified_name = entry["name"]
            suspicion_level = entry["suspicion"]
            category = entry["category"]
            break

    is_online = category == _ONLINE_EDITOR_CATEGORY
    is_editor = category == _DESKTOP_EDITOR_CATEGORY
    is_gen    = category in _GENERATOR_CATEGORIES
    is_scan   = category == _SCANNER_CATEGORY

    return SourceInfo(
        raw_producer=producer,
        raw_creator=creator,
        identified_name=identified_name,
        suspicion_level=suspicion_level,
        is_online_tool=is_online,
        is_editor=is_editor,
        is_generator=is_gen,
        is_scanner=is_scan,
    )


# ── Main Extractor ─────────────────────────────────────────────────────────────

class MetadataExtractor:
    """
    Extracts all metadata from a PDF using three libraries:
    - pikepdf: DocInfo + XMP
    - PyMuPDF: cross-check
    - pypdf: trailer info
    """

    def extract(self, pdf_path: str) -> MetadataReport:
        # Extract from all sources
        docinfo   = self._extract_docinfo(pdf_path)
        xmp       = self._extract_xmp(pdf_path)
        fitz_meta = self._extract_fitz(pdf_path)

        # Merge — prefer pikepdf, fallback to fitz
        producer  = docinfo.get("producer") or fitz_meta.get("producer", "")
        creator   = docinfo.get("creator")  or fitz_meta.get("creator", "")
        author    = docinfo.get("author")   or fitz_meta.get("author", "")
        title     = docinfo.get("title")    or fitz_meta.get("title", "")
        subject   = docinfo.get("subject")  or fitz_meta.get("subject", "")
        keywords  = docinfo.get("keywords") or fitz_meta.get("keywords", "")

        creation_date     = _parse_pdf_date(docinfo.get("creation_date", ""))
        modification_date = _parse_pdf_date(docinfo.get("mod_date", ""))

        xmp_create   = _parse_pdf_date(xmp.get("xmp_create", ""))
        xmp_modify   = _parse_pdf_date(xmp.get("xmp_modify", ""))
        xmp_metadata = _parse_pdf_date(xmp.get("xmp_metadata", ""))
        xmp_producer = xmp.get("xmp_producer", "")
        xmp_creator  = xmp.get("xmp_creator", "")

        # Identify source
        source = _identify_source(producer, creator)

        # Time delta
        time_delta = None
        if creation_date and modification_date:
            time_delta = abs((modification_date - creation_date).total_seconds())

        # XMP vs DocInfo mismatch
        xmp_mismatch = False
        if xmp_modify and modification_date:
            diff = abs((xmp_modify - modification_date).total_seconds())
            if diff > XMP_MISMATCH_TOLERANCE_SECONDS:
                xmp_mismatch = True
        if xmp_create and creation_date:
            diff = abs((xmp_create - creation_date).total_seconds())
            if diff > XMP_MISMATCH_TOLERANCE_SECONDS:
                xmp_mismatch = True

        # Multiple producers
        multi_producer = False
        if producer and creator:
            p = producer.lower()
            c = creator.lower()
            # Different tool families = suspicious
            if p != c and not any(x in p for x in c.split()[:2]):
                multi_producer = True

        # Stripped metadata
        stripped = not producer and not creator and not creation_date

        # Extended structural fields
        pikepdf_extras = self._extract_pikepdf_extras(pdf_path)
        xmp_fields_full = self._extract_xmp_fields_full(pdf_path)

        # Build report
        report = MetadataReport(
            producer=producer,
            creator=creator,
            author=author,
            creation_date=creation_date,
            modification_date=modification_date,
            title=title,
            subject=subject,
            keywords=keywords,
            xmp_create_date=xmp_create,
            xmp_modify_date=xmp_modify,
            xmp_metadata_date=xmp_metadata,
            xmp_producer=xmp_producer,
            xmp_creator_tool=xmp_creator,
            source=source,
            time_delta_seconds=time_delta,
            xmp_docinfo_mismatch=xmp_mismatch,
            multiple_producers=multi_producer,
            metadata_stripped=stripped,
            pdf_version=self._extract_pdf_version(pdf_path),
            fonts=pikepdf_extras["fonts"],
            is_encrypted=pikepdf_extras["is_encrypted"],
            permissions=pikepdf_extras["permissions"],
            page_details=pikepdf_extras["page_details"],
            has_embedded_files=pikepdf_extras["has_embedded_files"],
            has_javascript=pikepdf_extras["has_javascript"],
            has_open_action=pikepdf_extras["has_open_action"],
            document_id=pikepdf_extras["document_id"],
            xmp_fields=xmp_fields_full,
            icc_profiles=pikepdf_extras["icc_profiles"],
            has_icc_profiles=pikepdf_extras["has_icc_profiles"],
        )

        # /Trapped is a Name (e.g. /True, /False, /Unknown) or absent entirely —
        # report the bare value, or None when genuinely not set (not a guess).
        trapped_raw = (docinfo.get("trapped") or "").lstrip("/")
        report.trapped = trapped_raw or None

        # Overall modification age (used for display + risk weighting)
        report.edit_age = self._compute_edit_age(modification_date)

        # Comprehensive forensic-report sections
        report.raw_metadata       = self._extract_raw_metadata(pdf_path)
        report.structure          = self._extract_structure(pdf_path)
        report.suspicious_content = self._extract_suspicious_content(pdf_path)
        report.js_context = report.suspicious_content.get("js_context", "none")
        # suspicious_content's check covers Names tree + OpenAction + page-level
        # JS actions; pikepdf_extras only checked the Names tree, so this is the
        # more complete signal — keep has_javascript consistent with js_context.
        report.has_javascript = report.suspicious_content.get("has_javascript", report.has_javascript)
        report.dimensions_full    = self._extract_dimensions(pdf_path)
        report.dates_full         = self._enhance_dates(creation_date, modification_date)
        report.page_rotation      = self._check_page_rotation_consistency(pdf_path)
        report.icc_profile_details = pikepdf_extras.get("icc_profile_details", [])

        # Phase 2 completeness extensions — XMP Media Management, trailer ID
        # pair, per-object dates, and revision/incremental-update facts.
        report.xmp_mm = {
            "document_id":          xmp_fields_full.get("xmpMM:DocumentID"),
            "instance_id":          xmp_fields_full.get("xmpMM:InstanceID"),
            "original_document_id": xmp_fields_full.get("xmpMM:OriginalDocumentID"),
            "rendition_class":      xmp_fields_full.get("xmpMM:RenditionClass"),
            "history":              self._extract_xmp_history(pdf_path),
        }
        report.trailer_ids        = self._extract_trailer_ids(pdf_path)
        report.object_level_dates = self._extract_object_level_dates(pdf_path)
        report.revision_info      = self._extract_revision_info(pdf_path)

        # Run anomaly detection
        self._detect_anomalies(report)

        # Authenticity score reads the anomaly flags, so compute it last
        report.authenticity = self._compute_authenticity_score(report)
        return report

    def _compute_edit_age(self, mod_date) -> dict:
        """
        Compute how long ago the document was last modified.

        PDFs only store the LAST modification date, not individual edit
        timestamps — so this is the age of the most recent edit, not a
        per-edit history. The risk_factor reflects that a very recently
        modified document is more suspicious than an old one.
        """
        if not mod_date:
            return {"days_ago": None, "human_readable": "Unknown",
                    "risk_factor": 1.0}

        now = datetime.now(timezone.utc)
        if mod_date.tzinfo is None:
            mod_date = mod_date.replace(tzinfo=timezone.utc)
        delta = now - mod_date

        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60

        if days == 0 and hours == 0:
            human = f"{minutes} minutes ago" if minutes > 0 else "Just now"
            risk = 1.5  # very recent = more suspicious
        elif days == 0:
            human = f"{hours} hours ago"
            risk = 1.3
        elif days < 7:
            human = f"{days} day{'s' if days > 1 else ''} ago"
            risk = 1.2
        elif days < 30:
            weeks = days // 7
            human = f"{weeks} week{'s' if weeks > 1 else ''} ago"
            risk = 1.0
        elif days < 365:
            months = days // 30
            human = f"{months} month{'s' if months > 1 else ''} ago"
            risk = 0.9
        else:
            years = days // 365
            human = f"{years} year{'s' if years > 1 else ''} ago"
            risk = 0.8

        return {
            "days_ago": days,
            "human_readable": human,
            "risk_factor": risk,
            "is_recent": days < 1,
            "is_very_recent": days == 0 and hours < 3,
        }

    # ── Comprehensive report sections ──────────────────────────────────────────

    def _extract_raw_metadata(self, pdf_path: str) -> dict:
        """Return every key found in the PDF's /Info dictionary AND XMP — no
        filtering. Mirrors the 'raw metadata' dump commercial tools show."""
        raw = {}
        try:
            with pikepdf.open(pdf_path) as pdf:
                if pdf.docinfo:
                    for key, value in pdf.docinfo.items():
                        clean_key = str(key).lstrip("/")
                        raw[clean_key] = str(value).strip()
                # Also try XMP fields
                try:
                    with pdf.open_metadata() as xmp:
                        for key in xmp:
                            raw[str(key).lower()] = str(xmp.get(key, "")).strip()
                except Exception:
                    pass
        except Exception:
            pass
        return raw

    def _extract_structure(self, pdf_path: str) -> dict:
        """Page-by-page structure: text presence, length, rotation, media box,
        plus document-level word-count estimate and content-type classification."""
        import fitz
        structure = {
            "total_pages": 0,
            "content_type": "unknown",
            "page_details": [],
            "has_text_content": False,
            "total_text_length": 0,
            "estimated_word_count": 0,
            "avg_text_per_page": 0,
        }
        try:
            doc = fitz.open(pdf_path)
            structure["total_pages"] = len(doc)
            total_chars = 0
            for i, page in enumerate(doc):
                text = page.get_text("text") or ""
                total_chars += len(text)
                structure["page_details"].append({
                    "page_number": i + 1,
                    "has_text": len(text.strip()) > 0,
                    "text_length": len(text),
                    "rotation": page.rotation,
                    "media_box": list(page.mediabox),
                })
            structure["total_text_length"] = total_chars
            structure["estimated_word_count"] = total_chars // 5
            structure["avg_text_per_page"] = (
                total_chars // len(doc) if len(doc) > 0 else 0
            )
            structure["has_text_content"] = total_chars > 50
            if total_chars > 200:
                structure["content_type"] = "text-based"
            elif total_chars > 0:
                structure["content_type"] = "mixed"
            else:
                structure["content_type"] = "image-based"
            doc.close()
        except Exception:
            pass
        return structure

    def _extract_suspicious_content(self, pdf_path: str) -> dict:
        """Structural scan (via the parsed pikepdf object tree) for active/
        embedded content (JavaScript, OpenAction, Launch actions, embedded
        files) — the malware-vector surface.

        This walks parsed PDF objects rather than searching raw file bytes.
        A raw byte search for tokens like b"/JS" matches ANY occurrence of
        that sequence anywhere in the file, including inside compressed/
        encoded image streams (scanned pages) — random binary bytes that
        happen to decode to "/JS" are a real, observed false-positive
        source (e.g. ASCII85-encoded JPEG data containing that byte
        sequence by pure coincidence). Resolving through pikepdf's object
        graph means only an actual /JS or /JavaScript dictionary key can
        ever match.
        """
        result = {
            "has_javascript": False,
            "has_open_actions": False,
            "has_launch_actions": False,
            "has_embedded_files": False,
            "findings": [],
            "risk_score": 0,
            "js_context": "none",  # "none" | "names_tree" | "open_action" | "page_level"
        }
        try:
            with pikepdf.open(pdf_path) as pdf:
                names = pdf.Root.get("/Names", {})
                has_names_js = "/JavaScript" in names

                open_action = pdf.Root.get("/OpenAction", None)
                open_action_is_js = False
                if open_action is not None:
                    try:
                        open_action_is_js = open_action.get("/S") == pikepdf.Name("/JavaScript")
                    except Exception:
                        pass

                has_page_js = False
                has_launch = False
                for page in pdf.pages:
                    for annot in page.get("/Annots", []):
                        try:
                            action = annot.get("/A", None)
                            if action is not None:
                                action_type = action.get("/S", None)
                                if action_type == pikepdf.Name("/JavaScript"):
                                    has_page_js = True
                                elif action_type == pikepdf.Name("/Launch"):
                                    has_launch = True
                            additional = annot.get("/AA", None)
                            if additional is not None:
                                for key in additional.keys():
                                    act = additional.get(key)
                                    if act is not None and act.get("/S", None) == pikepdf.Name("/JavaScript"):
                                        has_page_js = True
                        except Exception:
                            continue

                if has_names_js:
                    result["has_javascript"] = True
                    result["js_context"] = "names_tree"
                    result["findings"].append(
                        "Document-level JavaScript in /Names tree — executes "
                        "automatically on open, highly suspicious in documents "
                        "that should not run code"
                    )
                    result["risk_score"] += 40
                elif open_action_is_js:
                    result["has_javascript"] = True
                    result["js_context"] = "open_action"
                    result["findings"].append(
                        "JavaScript executes on document open — "
                        "suspicious in static documents"
                    )
                    result["risk_score"] += 35
                elif has_page_js:
                    result["has_javascript"] = True
                    result["js_context"] = "page_level"
                    result["findings"].append(
                        "Page/form-level JavaScript action found — typically "
                        "form-field scripting, lower risk than document-level JS"
                    )
                    result["risk_score"] += 10

                if has_launch:
                    result["has_launch_actions"] = True
                    result["findings"].append(
                        "Launch action detected — attempts to run an external "
                        "program. Very suspicious."
                    )
                    result["risk_score"] += 50

                if open_action is not None and not open_action_is_js:
                    result["has_open_actions"] = True
                    result["findings"].append(
                        "OpenAction detected — PDF executes an action on open"
                    )
                    result["risk_score"] += 20

                if "/EmbeddedFiles" in names:
                    result["has_embedded_files"] = True
                    result["findings"].append(
                        "Embedded files detected — document contains file attachments"
                    )
                    result["risk_score"] += 15
        except Exception:
            pass
        return result

    def _extract_dimensions(self, pdf_path: str) -> dict:
        """First-page dimensions in pt and mm, orientation, and standard
        paper-format detection (A4/Letter/A5)."""
        import fitz
        result = {
            "width_pt": 0, "height_pt": 0,
            "width_mm": 0, "height_mm": 0,
            "orientation": "portrait",
            "format": "Unknown",
        }
        try:
            doc = fitz.open(pdf_path)
            if len(doc) > 0:
                page = doc[0]
                rect = page.rect
                result["width_pt"] = round(rect.width, 2)
                result["height_pt"] = round(rect.height, 2)
                # Convert pt to mm (1 pt = 0.3528 mm)
                result["width_mm"] = round(rect.width * 0.3528)
                result["height_mm"] = round(rect.height * 0.3528)
                result["orientation"] = (
                    "landscape" if rect.width > rect.height else "portrait"
                )
                # Detect standard formats
                w, h = result["width_mm"], result["height_mm"]
                if (abs(w - 210) < 3 and abs(h - 297) < 3) or \
                   (abs(w - 297) < 3 and abs(h - 210) < 3):
                    result["format"] = "A4"
                elif (abs(w - 216) < 3 and abs(h - 279) < 3) or \
                     (abs(w - 279) < 3 and abs(h - 216) < 3):
                    result["format"] = "Letter"
                elif (abs(w - 148) < 3 and abs(h - 210) < 3):
                    result["format"] = "A5"
            doc.close()
        except Exception:
            pass
        return result

    def _enhance_dates(self, creation_date, modification_date) -> dict:
        """Rich date analysis — ISO8601, human-readable, relative age, epoch,
        timezone — for both creation and modification, plus a quality block."""
        from datetime import datetime, timezone

        def format_date(d):
            if not d:
                return None
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - d
            days = delta.days

            if days == 0:
                hours = delta.seconds // 3600
                relative = f"{hours} hours ago" if hours > 0 else "Just now"
            elif days < 7:
                relative = f"{days} day{'s' if days > 1 else ''} ago"
            elif days < 30:
                relative = f"{days // 7} week{'s' if days // 7 > 1 else ''} ago"
            elif days < 365:
                relative = f"{days // 30} month{'s' if days // 30 > 1 else ''} ago"
            else:
                relative = f"{days // 365} year{'s' if days // 365 > 1 else ''} ago"

            return {
                "iso8601": d.isoformat(),
                "formatted": d.strftime("%Y-%m-%d %H:%M:%S"),
                "human": d.strftime("%B %d, %Y at %I:%M %p"),
                "relative": relative,
                "age_days": days,
                "timestamp": int(d.timestamp()),
                "timezone": str(d.tzinfo) if d.tzinfo else None,
            }

        result = {
            "created": format_date(creation_date),
            "modified": format_date(modification_date),
            "was_modified": False,
            "modification_seconds": 0,
            "quality": {"score": 100, "issues": [], "confidence": "high"},
        }

        if creation_date and modification_date:
            if creation_date.tzinfo is None:
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            if modification_date.tzinfo is None:
                modification_date = modification_date.replace(tzinfo=timezone.utc)
            delta_seconds = (modification_date - creation_date).total_seconds()
            result["modification_seconds"] = delta_seconds
            result["was_modified"] = delta_seconds > 0

        return result

    def _compute_authenticity_score(self, report, combined_score=None,
                                    verdict=None) -> dict:
        """Overall 0-100 authenticity score.

        Starts from the metadata-level anomaly flags, then — when available —
        folds in the cross-layer forensic result (combined_score + verdict)
        so the headline number reflects content/OCR/numeric/ELA/PyMuPDF
        findings, not just metadata. combined_score/verdict are None when this
        is computed during extract() (before the layers run); main.py
        recomputes with them populated once combine() has produced a verdict."""
        score = 100
        issues = []

        if report.xmp_docinfo_mismatch:
            score -= 25
            issues.append("XMP/DocInfo timestamp mismatch")

        if report.multiple_producers:
            score -= 15
            issues.append("Multiple producers detected")

        if report.metadata_stripped:
            score -= 30
            issues.append("Metadata stripped")

        if report.time_delta_seconds and report.time_delta_seconds > 0:
            if report.time_delta_seconds < 60:
                score -= 10
                issues.append("Very rapid modification")

        # Factor in overall verdict
        if combined_score is not None and combined_score > 0:
            # Reduce authenticity by combined score impact
            verdict_penalty = min(80, combined_score)
            score -= verdict_penalty
            if combined_score > 40:
                issues.append(f"Significant tamper signals detected (score: {combined_score:.1f})")

        if verdict == "MODIFIED":
            score = min(score, 30)  # cap at 30 if modified
            issues.append("Document classified as MODIFIED by forensic analysis")
        elif verdict == "UNCERTAIN":
            score = min(score, 60)
            issues.append("Document classification UNCERTAIN — review recommended")

        score = max(0, score)

        confidence = (
            "high" if score >= 80
            else "medium" if score >= 50
            else "low"
        )

        return {
            "score": score,
            "issues": issues,
            "assessment": (
                "high_confidence" if score >= 80
                else "medium_confidence" if score >= 50
                else "low_confidence_likely_tampered"
            ),
            "confidence": confidence,
        }

    def detect_image_conversion(self, pdf_path: str) -> bool:
        """
        Returns True if this PDF appears to be a digital document
        that was converted to image format (possible tamper hiding).
        Signals:
        - No embedded text (checked via pdfplumber)
        - Producer is unknown or image tool
        - Single full-page image object per page
        """
        try:
            import pdfplumber
            import fitz
            with pdfplumber.open(pdf_path) as pdf:
                has_text = any(
                    p.extract_text() and len(p.extract_text().strip()) > 20
                    for p in pdf.pages[:3]
                )
            if has_text:
                return False  # has text layer — not image-only

            # Check if pages contain single large image objects
            doc = fitz.open(pdf_path)
            image_only_pages = 0
            for page in doc:
                images = page.get_images()
                blocks = page.get_text("blocks")
                if len(images) >= 1 and len(blocks) == 0:
                    image_only_pages += 1
            doc.close()

            return image_only_pages > 0
        except Exception:
            return False

    # ── Extraction helpers ─────────────────────────────────────────────────────

    def _extract_docinfo(self, pdf_path: str) -> dict:
        result = {}
        try:
            with pikepdf.open(pdf_path) as pdf:
                info = pdf.docinfo
                result["producer"]      = str(info.get("/Producer", "")).strip()
                result["creator"]       = str(info.get("/Creator", "")).strip()
                result["author"]        = str(info.get("/Author", "")).strip()
                result["title"]         = str(info.get("/Title", "")).strip()
                result["subject"]       = str(info.get("/Subject", "")).strip()
                result["keywords"]      = str(info.get("/Keywords", "")).strip()
                result["creation_date"] = str(info.get("/CreationDate", "")).strip()
                result["mod_date"]      = str(info.get("/ModDate", "")).strip()
                result["trapped"]       = str(info.get("/Trapped", "")).strip()
        except Exception:
            pass
        return result

    def _extract_xmp(self, pdf_path: str) -> dict:
        result = {}
        try:
            with pikepdf.open(pdf_path) as pdf:
                with pdf.open_metadata() as meta:
                    result["xmp_create"]   = str(meta.get("xmp:CreateDate", ""))
                    result["xmp_modify"]   = str(meta.get("xmp:ModifyDate", ""))
                    result["xmp_metadata"] = str(meta.get("xmp:MetadataDate", ""))
                    result["xmp_producer"] = str(meta.get("pdf:Producer", ""))
                    result["xmp_creator"]  = str(meta.get("xmp:CreatorTool", ""))
        except Exception:
            pass
        return result

    def _extract_fitz(self, pdf_path: str) -> dict:
        result = {}
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata
            result["producer"]      = meta.get("producer", "")
            result["creator"]       = meta.get("creator", "")
            result["author"]        = meta.get("author", "")
            result["title"]         = meta.get("title", "")
            result["subject"]       = meta.get("subject", "")
            result["keywords"]      = meta.get("keywords", "")
            result["creation_date"] = meta.get("creationDate", "")
            result["mod_date"]      = meta.get("modDate", "")
            doc.close()
        except Exception:
            pass
        return result

    def _extract_pdf_version(self, pdf_path: str) -> Optional[str]:
        try:
            with open(pdf_path, "rb") as f:
                header = f.read(8).decode("latin-1", errors="replace")
            return header[5:8] if header.startswith("%PDF-") else None
        except Exception:
            return None

    def _extract_pikepdf_extras(self, pdf_path: str) -> dict:
        """
        Fonts, encryption/permissions, page dimensions, embedded files,
        JavaScript, and document ID — all read from one pikepdf handle
        instead of reopening the file for each.
        """
        result = {
            "fonts": [],
            "is_encrypted": False,
            "permissions": {},
            "page_details": [],
            "has_embedded_files": False,
            "has_javascript": False,
            "has_open_action": False,
            "document_id": None,
            "icc_profiles": [],
            "has_icc_profiles": False,
            "icc_profile_details": [],
        }
        try:
            with pikepdf.open(pdf_path) as pdf:
                result["is_encrypted"] = pdf.is_encrypted
                if pdf.is_encrypted:
                    # pikepdf exposes permissions via pdf.allow (a Permissions
                    # namedtuple), not as flat pdf.allow_* attributes.
                    perms = pdf.allow
                    result["permissions"] = {
                        "print": bool(perms.print_highres or perms.print_lowres),
                        "modify": bool(perms.modify_other),
                        "copy": bool(perms.extract),
                        "annotate": bool(perms.modify_annotation),
                    }

                names = pdf.Root.get("/Names", {})
                result["has_embedded_files"] = "/EmbeddedFiles" in names
                result["has_javascript"] = "/JavaScript" in names
                result["has_open_action"] = "/OpenAction" in pdf.Root

                try:
                    file_id = pdf.trailer.get("/ID", [])
                    if file_id:
                        result["document_id"] = str(file_id[0])
                except Exception:
                    pass

                fonts = []
                for page in pdf.pages:
                    resources = page.get("/Resources", {})
                    font_dict = resources.get("/Font", {}) if resources else {}
                    for font_name, font_ref in font_dict.items():
                        try:
                            font_obj = pdf.get_object(font_ref.objgen)
                            subtype = str(font_obj.get("/Subtype", "Unknown"))

                            # Composite (Type0/CID) fonts carry their
                            # /FontDescriptor on the descendant font, not on
                            # this dict directly — checking only font_obj
                            # itself under-reports "embedded" as False for
                            # every CID font even when it IS embedded.
                            font_descriptor = font_obj.get("/FontDescriptor", None)
                            if font_descriptor is None and subtype == "/Type0":
                                try:
                                    descendants = font_obj.get("/DescendantFonts", [])
                                    if descendants:
                                        desc_font = pdf.get_object(descendants[0].objgen)
                                        font_descriptor = desc_font.get("/FontDescriptor", None)
                                except Exception:
                                    font_descriptor = None

                            fonts.append({
                                "name": str(font_obj.get("/BaseFont", "Unknown")),
                                "type": subtype,
                                "encoding": str(font_obj.get("/Encoding", "Unknown")),
                                "embedded": font_descriptor is not None,
                                "tool_signature": self._extract_font_tool_signature(font_descriptor),
                            })
                        except Exception:
                            continue
                result["fonts"] = fonts

                page_details = []
                for i, page in enumerate(pdf.pages):
                    try:
                        media_box = [float(x) for x in page.MediaBox]
                        page_details.append({
                            "page_number": i + 1,
                            "width_pt": media_box[2],
                            "height_pt": media_box[3],
                            "rotation": int(page.get("/Rotate", 0)),
                        })
                    except Exception:
                        continue
                result["page_details"] = page_details

                # ICC color profiles — a document edited/re-exported with
                # different software typically embeds a different ICC
                # profile than the page(s) it was originally produced with,
                # so a mix of profiles across pages is a tamper signal.
                try:
                    icc_profiles = []
                    icc_profile_details = []
                    seen_streams = set()
                    for page in pdf.pages:
                        resources = page.get("/Resources", {})
                        color_spaces = resources.get("/ColorSpace", {}) if resources else {}
                        for name, cs in color_spaces.items():
                            try:
                                # pikepdf.Array (what cs actually is here for a
                                # real /ColorSpace array read from a file) does
                                # NOT subclass Python list, so an isinstance(cs,
                                # list) check silently never matched a real
                                # document — length/indexing still works on
                                # it directly, so drop the type gate.
                                if (len(cs) > 1 and str(cs[0]) == "/ICCBased"):
                                    icc_profiles.append(str(name))
                                    stream_obj = cs[1]
                                    stream_key = getattr(stream_obj, "objgen", None)
                                    if stream_key in seen_streams:
                                        continue
                                    seen_streams.add(stream_key)
                                    detail = {
                                        "resource_name": str(name),
                                        "n_components": int(stream_obj.get("/N", 0)) or None,
                                        "alternate": str(stream_obj.get("/Alternate", "")) or None,
                                    }
                                    try:
                                        profile_bytes = bytes(stream_obj.read_bytes())
                                        detail.update(self._parse_icc_profile(profile_bytes))
                                    except Exception:
                                        pass
                                    icc_profile_details.append(
                                        {k: v for k, v in detail.items() if v is not None}
                                    )
                            except Exception:
                                continue
                    result["icc_profiles"] = icc_profiles
                    result["has_icc_profiles"] = bool(icc_profiles)
                    result["icc_profile_details"] = icc_profile_details
                except Exception:
                    pass
        except Exception:
            pass
        return result

    def _extract_xmp_fields_full(self, pdf_path: str) -> dict:
        """
        Full XMP extraction via pikepdf's open_metadata() — every namespaced
        XMP key, plus Dublin Core fields and XMP timestamps pulled out
        explicitly so callers don't have to know the namespace syntax.

        Also flags xmp:MetadataDate != xmp:ModifyDate: MetadataDate tracks
        when the metadata packet itself was last touched, ModifyDate tracks
        when the document content was last touched. They diverge when
        something rewrote the metadata (e.g. stripped/re-injected by a
        tool) without going through the normal content-save path that
        would have updated both together — a sign of post-creation
        tampering with the metadata layer specifically.
        """
        result = {}
        try:
            with pikepdf.open(pdf_path) as pdf:
                with pdf.open_metadata() as xmp:
                    for key in xmp:
                        try:
                            result[str(key)] = str(xmp.get(key, ""))
                        except Exception:
                            pass

                    dc_fields = [
                        "dc:title", "dc:creator", "dc:description",
                        "dc:publisher", "dc:date", "dc:format",
                        "dc:identifier", "dc:source", "dc:language",
                    ]
                    for f in dc_fields:
                        try:
                            val = xmp.get(f)
                            if val:
                                result[f] = str(val)
                        except Exception:
                            pass

                    xmp_timestamps = [
                        "xmp:CreateDate", "xmp:ModifyDate",
                        "xmp:MetadataDate", "xmp:CreatorTool",
                    ]
                    for f in xmp_timestamps:
                        try:
                            val = xmp.get(f)
                            if val:
                                result[f] = str(val)
                        except Exception:
                            pass

                    # xmpMM: (Media Management) namespace — DocumentID/InstanceID
                    # persist across incremental saves while identifying the
                    # abstract "document" vs a specific saved rendition; a
                    # mismatched InstanceID across what claims to be the same
                    # DocumentID is a sign of a resave. History is extracted
                    # separately below since it's a structured array, not a
                    # simple string property.
                    xmpmm_fields = [
                        "xmpMM:DocumentID", "xmpMM:InstanceID",
                        "xmpMM:OriginalDocumentID", "xmpMM:RenditionClass",
                    ]
                    for f in xmpmm_fields:
                        try:
                            val = xmp.get(f)
                            if val:
                                result[f] = str(val)
                        except Exception:
                            pass

                    xmp_meta_date = result.get("xmp:MetadataDate")
                    xmp_mod_date  = result.get("xmp:ModifyDate")
                    if xmp_meta_date and xmp_mod_date and xmp_meta_date != xmp_mod_date:
                        result["_metadata_date_mismatch"] = True
        except Exception:
            pass
        return result

    # ── xmpMM:History (structured XMP array) ───────────────────────────────────

    _XMP_NS = {
        "rdf":   "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
        "stEvt": "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
    }

    def _extract_xmp_history(self, pdf_path: str) -> list:
        """
        xmpMM:History is a structured rdf:Seq of stEvt: event records (action,
        when, softwareAgent, instanceID, changed) — not a simple string
        property, so it can't be read the same way as xmp:CreateDate etc.
        Parsed directly from the raw /Metadata XML via ElementTree (stdlib)
        rather than through pikepdf's scalar-property XMP accessor, which
        isn't built for structured/array values.
        """
        history = []
        try:
            with pikepdf.open(pdf_path) as pdf:
                metadata_stream = pdf.Root.get("/Metadata", None)
                if metadata_stream is None:
                    return history
                xml_bytes = bytes(metadata_stream.read_bytes())
            root = ET.fromstring(xml_bytes)
            ns = self._XMP_NS
            for history_el in root.iter(f"{{{ns['xmpMM']}}}History"):
                seq = history_el.find(f"{{{ns['rdf']}}}Seq")
                if seq is None:
                    continue
                for li in seq.findall(f"{{{ns['rdf']}}}li"):
                    entry = {}
                    for child in li:
                        tag = child.tag.split("}")[-1]
                        if child.text and child.text.strip():
                            entry[tag] = child.text.strip()
                    for attr_key, attr_val in li.attrib.items():
                        tag = attr_key.split("}")[-1]
                        if tag == "parseType":
                            continue  # RDF serialization detail, not event data
                        if tag not in entry and attr_val:
                            entry[tag] = attr_val
                    if entry:
                        history.append(entry)
        except Exception:
            pass
        return history

    # ── Trailer /ID pair ────────────────────────────────────────────────────────

    @staticmethod
    def _id_to_hex(value) -> str:
        try:
            return bytes(value).hex()
        except Exception:
            return str(value)

    def _extract_trailer_ids(self, pdf_path: str) -> dict:
        """
        The trailer /ID entry is a pair [original, current]. Per spec the
        first element is meant to stay constant across every save of a
        document's lineage while the second is regenerated on each save —
        so id[0] == id[1] means this is the only save that has ever
        happened, and a mismatch confirms the file has been resaved at
        least once since creation (expected for a normal edit, informational
        either way — not itself evidence of tampering).
        """
        result = {"id_original": None, "id_current": None, "match": None}
        try:
            with pikepdf.open(pdf_path) as pdf:
                file_id = pdf.trailer.get("/ID", None)
                if file_id and len(file_id) >= 1:
                    result["id_original"] = self._id_to_hex(file_id[0])
                if file_id and len(file_id) >= 2:
                    result["id_current"] = self._id_to_hex(file_id[1])
                    result["match"] = result["id_original"] == result["id_current"]
        except Exception:
            pass
        return result

    # ── Object-level dates ──────────────────────────────────────────────────────

    def _extract_object_level_dates(self, pdf_path: str) -> list:
        """
        Some tools stamp /ModDate or /CreationDate on individual indirect
        objects (embedded file specs, form-field/annotation dicts, etc.), not
        just the top-level /Info dictionary. Walk every indirect object and
        surface any such dates found outside of /Info, which is already
        reported separately — a per-object date that disagrees with the
        document's overall metadata dates can pinpoint which part of the
        file was touched.
        """
        results = []
        try:
            with pikepdf.open(pdf_path) as pdf:
                info_ref = pdf.trailer.get("/Info", None)
                info_objgen = None
                if info_ref is not None:
                    try:
                        info_objgen = info_ref.objgen
                    except Exception:
                        pass

                for obj in pdf.objects:
                    try:
                        if not hasattr(obj, "get"):
                            continue
                        objgen = getattr(obj, "objgen", None)
                        if info_objgen is not None and objgen == info_objgen:
                            continue  # already reported as top-level Info dates
                        for key in ("/ModDate", "/CreationDate"):
                            if key in obj:
                                raw_value = str(obj[key])
                                parsed = _parse_pdf_date(raw_value)
                                results.append({
                                    "object_id": objgen[0] if objgen else None,
                                    "field": key.lstrip("/"),
                                    "raw_value": raw_value,
                                    "parsed_iso": parsed.isoformat() if parsed else None,
                                })
                    except Exception:
                        continue
        except Exception:
            pass
        return results

    # ── Revision / incremental-update info (informational, not scored) ────────

    def _extract_revision_info(self, pdf_path: str) -> dict:
        """
        How many generations of the file exist, purely as descriptive
        metadata. This mirrors the same %%EOF-count / /Prev-pointer signal
        the ELA layer already scores separately (ela_analyzer._detect_
        incremental_updates) — duplicated here, independently computed, so
        the metadata report can show it as a fact alongside the rest of the
        trailer/Info data without reaching into another layer's module.
        Purely informational: does not feed metadata_extractor's own
        anomaly_score.
        """
        result = {
            "eof_marker_count": 0,
            "incremental_update_count": 0,
            "has_incremental_updates": False,
            "has_prev_trailer_pointer": False,
            "prev_trailer_offset": None,
        }
        try:
            with open(pdf_path, "rb") as f:
                raw = f.read()
            eof_count = raw.count(b"%%EOF")
            result["eof_marker_count"] = eof_count
            result["incremental_update_count"] = max(0, eof_count - 1)
            result["has_incremental_updates"] = eof_count > 1
        except Exception:
            pass
        try:
            with pikepdf.open(pdf_path) as pdf:
                prev = pdf.trailer.get("/Prev")
                if prev is not None:
                    result["has_prev_trailer_pointer"] = True
                    result["prev_trailer_offset"] = int(prev)
                    result["has_incremental_updates"] = True
        except Exception:
            pass
        return result

    # ── ICC profile parsing (raw bytes -> description/creator, stdlib struct) ──

    @staticmethod
    def _parse_icc_text_tag(data: bytes, offset: int, size: int) -> Optional[str]:
        """Decode an ICC 'desc'/'cprt'/plain-'text' tag's human-readable string."""
        chunk = data[offset:offset + size]
        if len(chunk) < 8:
            return None
        tag_type = chunk[0:4]
        try:
            if tag_type == b"desc" and len(chunk) >= 12:
                # textDescriptionType (ICC v2): sig(4) + reserved(4) + ascii
                # count(4, BE uint32) + that many ASCII bytes (NUL-terminated)
                ascii_count = struct.unpack(">I", chunk[8:12])[0]
                ascii_bytes = chunk[12:12 + ascii_count]
                text = ascii_bytes.rstrip(b"\x00").decode("latin-1", errors="replace")
                return text.strip() or None
            if tag_type == b"mluc" and len(chunk) >= 16:
                # multiLocalizedUnicodeType (ICC v4): take the first record
                num_records = struct.unpack(">I", chunk[8:12])[0]
                if num_records > 0:
                    rec_off = 16
                    length, str_offset = struct.unpack(">II", chunk[rec_off + 4:rec_off + 12])
                    text_bytes = chunk[str_offset:str_offset + length]
                    text = text_bytes.decode("utf-16-be", errors="replace")
                    return text.strip("\x00").strip() or None
            if tag_type == b"text":
                text = chunk[8:].split(b"\x00")[0].decode("latin-1", errors="replace")
                return text.strip() or None
        except Exception:
            return None
        return None

    def _parse_icc_profile(self, data: bytes) -> dict:
        """
        Parse an embedded ICC profile's header + tag table (pure stdlib
        struct — no colormanagement library needed) to surface its actual
        description/creator, not just the fact that a profile exists. A
        document re-exported through different software typically embeds a
        different ICC profile than the one it was originally produced with.
        """
        result = {}
        try:
            if len(data) < 132:
                return result
            version_bytes = data[8:12]
            result["version"] = f"{version_bytes[0]}.{version_bytes[1] >> 4}.{version_bytes[1] & 0x0F}"
            result["device_class"]     = data[12:16].decode("ascii", errors="replace").strip() or None
            result["color_space"]      = data[16:20].decode("ascii", errors="replace").strip() or None
            result["connection_space"] = data[20:24].decode("ascii", errors="replace").strip() or None
            result["platform"]         = data[40:44].decode("ascii", errors="replace").strip("\x00 ") or None
            result["manufacturer"]     = data[48:52].decode("ascii", errors="replace").strip("\x00 ") or None
            result["model"]            = data[52:56].decode("ascii", errors="replace").strip("\x00 ") or None
            result["creator_signature"] = data[80:84].decode("ascii", errors="replace").strip("\x00 ") or None

            tag_count = struct.unpack(">I", data[128:132])[0]
            for i in range(tag_count):
                entry_offset = 132 + i * 12
                if entry_offset + 12 > len(data):
                    break
                sig = data[entry_offset:entry_offset + 4]
                off, size = struct.unpack(">II", data[entry_offset + 4:entry_offset + 12])
                if sig == b"desc":
                    desc = self._parse_icc_text_tag(data, off, size)
                    if desc:
                        result["description"] = desc
                elif sig == b"cprt":
                    cprt = self._parse_icc_text_tag(data, off, size)
                    if cprt:
                        result["copyright"] = cprt
        except Exception:
            pass
        return {k: v for k, v in result.items() if v}

    # ── Embedded font internal metadata (survives even when /Info is stripped) ─

    _TT_NAME_IDS = {
        1: "family", 2: "subfamily", 3: "unique_id", 4: "full_name",
        5: "version", 6: "postscript_name", 8: "manufacturer",
        9: "designer", 11: "vendor_url", 13: "license",
    }

    def _parse_sfnt_name_table(self, data: bytes, font_format: str) -> Optional[dict]:
        """Parse a TrueType/OpenType 'name' table — the font program's own
        embedded authoring-tool signature (family/version/manufacturer/
        designer strings), independent of the PDF's own /Info dictionary."""
        try:
            if len(data) < 12:
                return None
            num_tables = struct.unpack(">H", data[4:6])[0]
            name_table = None
            for i in range(num_tables):
                rec_off = 12 + i * 16
                if rec_off + 16 > len(data):
                    break
                tag = data[rec_off:rec_off + 4]
                offset, length = struct.unpack(">II", data[rec_off + 8:rec_off + 16])
                if tag == b"name":
                    name_table = data[offset:offset + length]
                    break
            if not name_table or len(name_table) < 6:
                return {"font_format": font_format, "name_table_found": False}

            count, string_offset = struct.unpack(">HH", name_table[2:6])
            fields = {}
            for i in range(count):
                rec_off = 6 + i * 12
                if rec_off + 12 > len(name_table):
                    break
                platform_id, _enc_id, _lang_id, name_id, length, offset = struct.unpack(
                    ">HHHHHH", name_table[rec_off:rec_off + 12]
                )
                if name_id not in self._TT_NAME_IDS:
                    continue
                str_start = string_offset + offset
                raw = name_table[str_start:str_start + length]
                try:
                    text = raw.decode("mac_roman" if platform_id == 1 else "utf-16-be",
                                       errors="replace").strip()
                except Exception:
                    continue
                key = self._TT_NAME_IDS[name_id]
                if text and key not in fields:
                    fields[key] = text
            return {"font_format": font_format, "name_table_found": bool(fields), **fields}
        except Exception:
            return None

    def _parse_type1_font_info(self, data: bytes) -> Optional[dict]:
        """Type1 fonts carry a cleartext PostScript /FontInfo dict (Notice,
        FullName, FamilyName, Weight, version) before the encrypted eexec
        section — plain regex extraction, no PostScript interpreter needed."""
        try:
            eexec_idx = data.find(b"eexec")
            cleartext = data[:eexec_idx] if eexec_idx != -1 else data[:4096]
            text = cleartext.decode("latin-1", errors="replace")
            patterns = {
                "full_name": r"/FullName\s*\(([^)]*)\)",
                "family":    r"/FamilyName\s*\(([^)]*)\)",
                "weight":    r"/Weight\s*\(([^)]*)\)",
                "version":   r"/version\s*\(([^)]*)\)",
                "notice":    r"/Notice\s*\(([^)]*)\)",
            }
            fields = {}
            for key, pat in patterns.items():
                m = re.search(pat, text)
                if m and m.group(1).strip():
                    fields[key] = m.group(1).strip()
            return {"font_format": "Type1", "name_table_found": bool(fields), **fields}
        except Exception:
            return None

    def _parse_cff_font_info(self, data: bytes) -> Optional[dict]:
        """Bare CFF (Type1C/CIDFontType0C) has no sfnt 'name' table — read
        just its Name INDEX (first structure in the format) for the
        font's internal name, per the CFF spec's fixed INDEX layout."""
        try:
            if len(data) < 4:
                return None
            hdr_size = data[2]
            pos = hdr_size
            count = struct.unpack(">H", data[pos:pos + 2])[0]
            pos += 2
            names = []
            if count > 0:
                off_size = data[pos]
                pos += 1
                offsets = []
                for _ in range(count + 1):
                    offsets.append(int.from_bytes(data[pos:pos + off_size], "big"))
                    pos += off_size
                data_start = pos - 1
                for i in range(count):
                    s = data_start + offsets[i]
                    e = data_start + offsets[i + 1]
                    names.append(data[s:e].decode("latin-1", errors="replace"))
            return {
                "font_format": "CFF (bare)",
                "name_table_found": bool(names),
                "family": names[0] if names else None,
            }
        except Exception:
            return None

    def _extract_font_tool_signature(self, font_descriptor) -> Optional[dict]:
        """
        Read the embedded font PROGRAM's own internal metadata (TrueType/OTF
        'name' table, Type1 FontInfo, or bare-CFF Name INDEX). This can
        reveal the actual authoring/subsetting tool even when the PDF's own
        /Info dictionary has been stripped — a font subsetter's signature
        (e.g. FontForge, a specific version string) doesn't get cleared by
        stripping document-level metadata.
        """
        if font_descriptor is None:
            return None
        try:
            if "/FontFile2" in font_descriptor:
                data = bytes(font_descriptor["/FontFile2"].read_bytes())
                return self._parse_sfnt_name_table(data, "TrueType")
            if "/FontFile3" in font_descriptor:
                ff3 = font_descriptor["/FontFile3"]
                subtype = str(ff3.get("/Subtype", ""))
                data = bytes(ff3.read_bytes())
                if "OpenType" in subtype:
                    return self._parse_sfnt_name_table(data, "OpenType (CFF)")
                return self._parse_cff_font_info(data)
            if "/FontFile" in font_descriptor:
                data = bytes(font_descriptor["/FontFile"].read_bytes())
                return self._parse_type1_font_info(data)
        except Exception:
            return None
        return None

    def _check_page_rotation_consistency(self, pdf_path: str) -> dict:
        """
        Pages from different source documents merged into one PDF often
        carry different /Rotate values (each source page kept its own
        viewer rotation) — a mix of rotations across pages is a sign of
        document recombination, not a single coherent scan/export.
        """
        result = {"consistent": True, "rotations": [], "anomaly": False}
        try:
            doc = fitz.open(pdf_path)
            rotations = [page.rotation for page in doc]
            doc.close()
            result["rotations"] = rotations
            unique_rotations = set(rotations)

            if len(unique_rotations) > 1:
                result["consistent"] = False
                result["anomaly"] = True
                result["anomaly_reason"] = (
                    f"Pages have inconsistent rotation: {sorted(unique_rotations)} "
                    f"— may indicate pages from different documents were merged"
                )
        except Exception:
            pass
        return result

    # ── Anomaly detection ──────────────────────────────────────────────────────

    def _detect_anomalies(self, report: MetadataReport):
        # Suppress flags for our own Word-to-PDF conversion output
        # ReportLab + anonymous creator = internal converter, not tampering
        if "reportlab" in report.producer.lower() and \
           report.creator.lower() in ("anonymous", "", "—"):
            report.anomalies = []
            report.anomaly_score = 0
            return  # nothing to flag — this is our own converter output

        score = 0
        anomalies = []

        # 1-3. Score directly off the database's suspicion field — each entry
        # in producer_database.json carries its own suspicion level (e.g.
        # PDF24 is MEDIUM, Smallpdf/iLovePDF are HIGH), so the score should
        # follow that per-entry value rather than a category shortcut that
        # would force every online-editor-category entry into the same
        # HIGH/40-point bucket regardless of what the database says.
        if report.source.suspicion_level == "HIGH":
            anomalies.append(
                f"High-suspicion source detected: {report.source.identified_name} — "
                f"document was processed by a tool with elevated tamper risk"
            )
            score += SUSPICION_SCORE["HIGH"]

        elif report.source.suspicion_level == "MEDIUM":
            anomalies.append(
                f"PDF editor detected: {report.source.identified_name} — "
                f"document may have been modified"
            )
            score += SUSPICION_SCORE["MEDIUM"]

        elif report.source.suspicion_level == "UNKNOWN":
            # An unidentified source is neutral information on its own — it
            # only means this layer can't fingerprint the tool, not that
            # something is wrong. State plainly which case this is (nothing
            # to fingerprint vs. a real-but-unrecognized value) instead of
            # implying suspicion; let signal fusion with other layers do the
            # actual suspicion-weighing.
            if not report.producer and not report.creator:
                anomalies.append(
                    "Producer/creator absent — cannot fingerprint source. "
                    "This is neutral information, not evidence of tampering "
                    "on its own."
                )
            else:
                anomalies.append(
                    f"Producer/creator ('{report.producer or report.creator}') does "
                    f"not match any tool in the recognized fingerprint database — "
                    f"the source application is unidentified. This is neutral "
                    f"information on its own, not evidence of tampering."
                )
            score += SUSPICION_SCORE["UNKNOWN"]

        # 4. Metadata stripped completely
        if report.metadata_stripped:
            anomalies.append(
                "Critical metadata is missing — producer, creator, and "
                "creation date are all absent (may have been deliberately stripped)"
            )
            score += SCORE_METADATA_STRIPPED

        # 5. XMP vs DocInfo mismatch
        if report.xmp_docinfo_mismatch:
            anomalies.append(
                "XMP metadata does not match DocInfo metadata — "
                "dates differ between layers, suggesting the document was "
                "re-processed after creation"
            )
            score += SCORE_XMP_MISMATCH

        # XMP MetadataDate vs ModifyDate mismatch (full XMP extraction)
        if report.xmp_fields.get("_metadata_date_mismatch"):
            anomalies.append(
                "XMP MetadataDate differs from XMP ModifyDate — "
                "document metadata was updated separately from content, "
                "suggesting post-creation modification"
            )
            score += SCORE_XMP_METADATA_DATE_MISMATCH

        # Page rotation inconsistency — signs of document recombination
        if report.page_rotation.get("anomaly"):
            anomalies.append(report.page_rotation["anomaly_reason"])
            score += SCORE_ROTATION_INCONSISTENCY

        # 6. Time delta anomaly
        if report.time_delta_seconds is not None:
            if report.time_delta_seconds < XMP_MISMATCH_TOLERANCE_SECONDS:
                anomalies.append(
                    f"Document created and modified within "
                    f"{report.time_delta_seconds:.0f} seconds — "
                    f"suggests automated processing or rapid re-save"
                )
                score += SCORE_INSTANT_TIMESTAMP
            elif report.time_delta_seconds > 0:
                mins = report.time_delta_seconds / 60
                anomalies.append(
                    f"Document was modified {mins:.0f} minutes after creation"
                )
                score += SCORE_MODIFIED_LATER

        # 7. Multiple producers
        if report.multiple_producers:
            anomalies.append(
                f"Conflicting generators: producer='{report.producer}' "
                f"creator='{report.creator}' — two different tools involved"
            )
            score += SCORE_MULTIPLE_PRODUCERS

        # 8. XMP producer differs from DocInfo producer
        if (report.xmp_producer and report.producer and
                report.xmp_producer.lower() != report.producer.lower()):
            anomalies.append(
                f"XMP producer '{report.xmp_producer}' differs from "
                f"DocInfo producer '{report.producer}'"
            )
            score += SCORE_XMP_PRODUCER_MISMATCH

        # 9. Possible digital-to-image conversion (tamper-hiding technique):
        # an unrecognized/stripped producer combined with an instant
        # creation-to-modification gap is what an automated image-export
        # pipeline produces (a human editing session never finishes in
        # under INSTANT_TIMESTAMP_TOLERANCE_SECONDS).
        if (report.metadata_stripped or
            report.source.suspicion_level == "UNKNOWN") and \
            report.time_delta_seconds is not None and \
            report.time_delta_seconds < INSTANT_TIMESTAMP_TOLERANCE_SECONDS:
            anomalies.append(
                "Document metadata suggests possible digital-to-image conversion — "
                "unknown producer with instant creation/modification timestamp "
                "is common when screenshots or image exports are used to hide edits"
            )
            score += SCORE_POSSIBLE_IMG_CONVERT

        # 10. Anonymous / cleared author+creator on a document that was
        # actually modified — a deliberately sanitized identity combined with
        # a real edit history is a classic "hide who edited this" pattern.
        ANONYMOUS_PATTERNS = [
            "(anonymous)", "(unspecified)", "anonymous",
            "unknown", "(unknown)", "user", "owner", ""
        ]

        author_lower = (report.author or "").lower().strip()
        creator_lower = (report.creator or "").lower().strip()

        # The "" entry in ANONYMOUS_PATTERNS would make `p in field` vacuously
        # true for ANY value (every string contains ""), which would fire the
        # check on documents carrying a real author name. The truly-empty case
        # is handled by the explicit `not field` clause, so the empty pattern
        # is skipped in the membership test (`p and ...`).
        author_cleared = (not author_lower) or any(
            p and p in author_lower for p in ANONYMOUS_PATTERNS)
        creator_cleared = (not creator_lower) or any(
            p and p in creator_lower for p in ANONYMOUS_PATTERNS)

        # Calculate if document was actually modified
        was_modified = (report.creation_date and report.modification_date and
                        report.creation_date != report.modification_date)

        if author_cleared and creator_cleared and was_modified:
            anomalies.append(
                "Author and creator metadata deliberately cleared while "
                "document shows modification history — pattern consistent "
                "with intentional metadata sanitization to hide edit origin"
            )
            score += 30

        elif author_cleared and was_modified:
            anomalies.append(
                "Author metadata cleared while document was modified — "
                "possible attempt to hide edit origin"
            )
            score += 15

        report.anomalies = anomalies
        report.anomaly_score = min(100, score)
