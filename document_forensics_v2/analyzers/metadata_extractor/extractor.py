"""MetadataExtractor core — the extract() orchestration, edit-age and
authenticity scoring, and anomaly detection. Extraction internals live
in the mixin modules."""

import re
from datetime import datetime, timezone

import fitz

from .database import (
    _identify_source, _parse_pdf_date,
    XMP_MISMATCH_TOLERANCE_SECONDS, INSTANT_TIMESTAMP_TOLERANCE_SECONDS,
    SCORE_ONLINE_TOOL, SCORE_EDITOR_MEDIUM, SCORE_UNKNOWN_SOURCE,
    SCORE_METADATA_STRIPPED, SCORE_XMP_MISMATCH, SCORE_INSTANT_TIMESTAMP,
    SCORE_MODIFIED_LATER, SCORE_MULTIPLE_PRODUCERS, SCORE_XMP_PRODUCER_MISMATCH,
    SCORE_POSSIBLE_IMG_CONVERT, SCORE_XMP_METADATA_DATE_MISMATCH,
    SCORE_ROTATION_INCONSISTENCY, SUSPICION_SCORE,
)
from .models import MetadataReport, SourceInfo
from .extraction_basic import BasicExtractionMixin
from .extraction_deep import DeepExtractionMixin
from .binary_parsing import BinaryParsingMixin


class MetadataExtractor(BasicExtractionMixin, DeepExtractionMixin,
                        BinaryParsingMixin):
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


    def _compute_authenticity_score(self, report, combined_score=None,
                                    verdict=None) -> dict:
        """Overall 0-100 authenticity score.

        Starts from the metadata-level anomaly flags, then — when available —
        folds in the cross-layer forensic result (combined_score + verdict)
        so the headline number reflects content/numeric/ELA/PyMuPDF
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
