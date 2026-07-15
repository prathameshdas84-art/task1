"""First-pass metadata extraction: raw /Info + XMP keys, document
structure, suspicious content, page dimensions, date formatting."""

from datetime import datetime, timezone

import fitz
import pikepdf


class BasicExtractionMixin:
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

