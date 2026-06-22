"""
Metadata Extractor — Document Forensics Engine
Extracts all metadata from any PDF and identifies its origin.
"""

import json
import os
import re
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

_DB_PATH = os.path.join(os.path.dirname(__file__), "producer_database.json")

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
    has_open_action: bool = False
    document_id: Optional[str] = None
    xmp_fields: dict = field(default_factory=dict)

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
        )

        # Overall modification age (used for display + risk weighting)
        report.edit_age = self._compute_edit_age(modification_date)

        # Comprehensive forensic-report sections
        report.raw_metadata       = self._extract_raw_metadata(pdf_path)
        report.structure          = self._extract_structure(pdf_path)
        report.suspicious_content = self._extract_suspicious_content(pdf_path)
        report.dimensions_full    = self._extract_dimensions(pdf_path)
        report.dates_full         = self._enhance_dates(creation_date, modification_date)

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
        """Byte-level scan for active/embedded content (JavaScript, OpenAction,
        Launch actions, embedded files) — the malware-vector surface."""
        result = {
            "has_javascript": False,
            "has_open_actions": False,
            "has_launch_actions": False,
            "has_embedded_files": False,
            "findings": [],
            "risk_score": 0,
        }
        try:
            with open(pdf_path, "rb") as f:
                raw_bytes = f.read()

            if b"/JS" in raw_bytes or b"/JavaScript" in raw_bytes:
                result["has_javascript"] = True
                result["findings"].append("JavaScript detected")
                result["risk_score"] += 40

            if b"/OpenAction" in raw_bytes:
                result["has_open_actions"] = True
                result["findings"].append("OpenAction detected — executes code on open")
                result["risk_score"] += 30

            if b"/Launch" in raw_bytes:
                result["has_launch_actions"] = True
                result["findings"].append("Launch action detected — runs external programs")
                result["risk_score"] += 50

            if b"/EmbeddedFile" in raw_bytes:
                result["has_embedded_files"] = True
                result["findings"].append("Embedded files detected")
                result["risk_score"] += 20
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
                            fonts.append({
                                "name": str(font_obj.get("/BaseFont", "Unknown")),
                                "type": str(font_obj.get("/Subtype", "Unknown")),
                                "encoding": str(font_obj.get("/Encoding", "Unknown")),
                                "embedded": "/FontDescriptor" in font_obj,
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
        except Exception:
            pass
        return result

    def _extract_xmp_fields_full(self, pdf_path: str) -> dict:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf_plumb:
                if pdf_plumb.metadata:
                    # Cast to str — pdfplumber metadata values aren't
                    # guaranteed JSON-serializable as-is.
                    return {str(k): str(v) for k, v in pdf_plumb.metadata.items()}
        except Exception:
            pass
        return {}

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
            anomalies.append(
                "Document source is unrecognized — producer/creator fields "
                "do not match any known tool"
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
