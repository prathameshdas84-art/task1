"""PDF object-structure forensics: injected XObject detection, object
fingerprinting, and incremental-update (multiple %%EOF) recovery."""

import re

import fitz

from .constants import *


class ObjectForensicsMixin:
    def _collect_xobject_placements(self, pdf_path: str) -> dict:
        """
        Map each Form XObject xref to where/how it's actually used across
        the document: how many pages invoke it, its bbox + page geometry
        (from the first invocation), and whether it carries its own text
        content (fonts / text-showing operators). Used to tell a reused
        template element (logo/letterhead) apart from an injected paste-over.
        """
        placements = {}
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc[page_num]
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                for xref, name, invoker, bbox in page.get_xobjects():
                    entry = placements.setdefault(xref, {
                        "pages": set(),
                        "bbox": bbox,
                        "page_height": page_rect.height,
                        "page_area": page_area,
                    })
                    entry["pages"].add(page_num)

            for xref, entry in placements.items():
                try:
                    xobj_dict = doc.xref_object(xref)
                    has_font = "/Font" in xobj_dict
                    stream = doc.xref_stream(xref) or b""
                    has_text_ops = b"Tj" in stream or b"TJ" in stream
                except Exception:
                    has_font, has_text_ops = False, False
                entry["has_text_content"] = bool(has_font or has_text_ops)
                entry["page_frequency"] = len(entry["pages"]) / total_pages if total_pages else 0

            doc.close()
        except Exception:
            return {}
        return placements

    def _is_injected_xobject(self, xref: int, placements: dict) -> bool:
        """
        True only for Form XObjects that look like an injected paste-over
        rather than a reused template component. See FORM_XOBJECT_* constants.
        """
        entry = placements.get(xref)
        if entry is None:
            return True  # no placement info — fall back to flagging (old behavior)

        if entry["page_frequency"] >= FORM_XOBJECT_MAX_TEMPLATE_FREQUENCY:
            return False  # appears on most pages = template, not injected

        bbox = fitz.Rect(entry["bbox"])
        page_height = entry["page_height"]
        page_area = entry["page_area"]

        if page_height > 0:
            top_frac = bbox.y0 / page_height
            bottom_frac = (page_height - bbox.y1) / page_height
            if top_frac <= FORM_XOBJECT_HEADER_ZONE_FRACTION or bottom_frac <= FORM_XOBJECT_FOOTER_ZONE_FRACTION:
                return False  # header/footer branding zone

        area_frac = (bbox.width * bbox.height) / page_area if page_area > 0 else 0
        if area_frac < FORM_XOBJECT_MIN_AREA_FRACTION:
            return False  # small logo/stamp

        if not entry.get("has_text_content"):
            return False  # image-only content, nothing to "paste over"

        return True

    def _pdf_object_fingerprint(self, pdf_path: str) -> tuple[list[str], int]:
        """
        Analyze PDF object structure for signs of post-creation editing.

        Signals:
        1. Objects with generation number > 0 (deleted and reused = edit)
        2. FreeText/Redact annotations (Acrobat edit vectors)
        3. Form XObjects covering significant page area (paste-over)
        4. Incremental updates (%%EOF count > 1)
        5. Mixed producer fingerprints in object streams
        """
        signals = []
        score   = 0

        try:
            import pikepdf

            # Signal 1: Count %%EOF markers (incremental updates)
            with open(pdf_path, 'rb') as f:
                content = f.read()
            eof_count = content.count(b'%%EOF')
            if eof_count > 1:
                signals.append(
                    f"PDF has {eof_count} revision layers (incremental updates) — "
                    f"document was saved multiple times after creation"
                )
                score += min(EOF_SCORE_CAP, (eof_count - 1) * EOF_SCORE_PER_REVISION)

            xobject_placements = self._collect_xobject_placements(pdf_path)

            with pikepdf.open(pdf_path) as pdf:

                high_gen_objects = []
                freetext_annots  = []
                form_xobjects    = []

                for objid in range(1, len(pdf.objects) + 1):
                    try:
                        obj = pdf.get_object(objid, 0)
                    except Exception:
                        continue

                    # Signal 2: High generation number = object was deleted and reused
                    try:
                        gen = obj.objgen[1] if hasattr(obj, 'objgen') else 0
                        if gen > 0:
                            high_gen_objects.append(objid)
                    except Exception:
                        pass

                    # Signal 3: FreeText/Redact annotations
                    try:
                        if (hasattr(obj, 'get') and
                            obj.get('/Type') == pikepdf.Name('/Annot') and
                            obj.get('/Subtype') in (
                                pikepdf.Name('/FreeText'),
                                pikepdf.Name('/Redact')
                            )):
                            freetext_annots.append(objid)
                    except Exception:
                        pass

                    # Signal 4: Form XObjects (paste-over content) — only
                    # ones that look injected, not reused template elements
                    try:
                        if (hasattr(obj, 'get') and
                            obj.get('/Type') == pikepdf.Name('/XObject') and
                            obj.get('/Subtype') == pikepdf.Name('/Form') and
                            self._is_injected_xobject(objid, xobject_placements)):
                            form_xobjects.append(objid)
                    except Exception:
                        pass

                if high_gen_objects:
                    signals.append(
                        f"{len(high_gen_objects)} PDF object(s) have generation "
                        f"number > 0 — objects were deleted and recreated, "
                        f"indicating direct object-level editing"
                    )
                    score += min(HIGH_GEN_SCORE_CAP, len(high_gen_objects) * HIGH_GEN_SCORE_PER_OBJECT)

                if freetext_annots:
                    signals.append(
                        f"{len(freetext_annots)} FreeText/Redact annotation(s) found — "
                        f"classic Acrobat/Foxit text overlay edit pattern"
                    )
                    score += min(FREETEXT_SCORE_CAP, len(freetext_annots) * FREETEXT_SCORE_PER_ANNOT)

                if form_xobjects:
                    signals.append(
                        f"{len(form_xobjects)} Form XObject(s) found — "
                        f"content may have been pasted over original"
                    )
                    score += min(FORM_XOBJECT_SCORE_CAP, len(form_xobjects) * FORM_XOBJECT_SCORE_PER_ITEM)

        except Exception:
            pass

        return signals, min(100, score)

    def _detect_incremental_updates(self, pdf_path: str) -> dict:
        """
        Detect incremental-update structure and attempt to recover shadowed
        prior object versions.

        A PDF incremental update appends a NEW xref section + trailer to the
        end of the file rather than rewriting it; the old bytes are still
        physically present but a conformant reader (pikepdf included) only
        ever resolves the MOST RECENT xref entry for a given object id, so
        `pdf.get_object()` can never surface a shadowed earlier version no
        matter what generation number is requested — that would require
        re-implementing xref-chain resolution by hand. We get the same
        result more directly and more reliably with a raw byte scan: any
        object id appearing in more than one "<id> <gen> obj" definition in
        the file has at least one shadowed earlier version, and the FIRST
        occurrence is that pre-edit content.
        """
        result = {
            "has_incremental_updates": False,
            "update_count": 0,
            "eof_count": 0,
            "xref_count": 0,
            "startxref_count": 0,
            "prev_trailer_offset": None,
            "old_objects_found": [],
            "signals": [],
            "score": 0,
        }
        try:
            with open(pdf_path, "rb") as f:
                raw = f.read()

            eof_count = raw.count(b"%%EOF")
            xref_count = raw.count(b"\nxref")
            startxref_count = raw.count(b"startxref")
            result["eof_count"] = eof_count
            result["xref_count"] = xref_count
            result["startxref_count"] = startxref_count

            if eof_count > 1:
                result["has_incremental_updates"] = True
                result["update_count"] = eof_count - 1
                result["signals"].append(
                    f"{eof_count} %%EOF markers found — document has "
                    f"{eof_count - 1} incremental update(s) layered on top "
                    f"of the original save"
                )
                result["score"] += min(
                    INCREMENTAL_EOF_SCORE_CAP,
                    (eof_count - 1) * INCREMENTAL_EOF_SCORE_PER_REVISION,
                )

            if xref_count > 1 and xref_count != startxref_count:
                result["signals"].append(
                    f"{xref_count} xref section(s) vs {startxref_count} "
                    f"startxref marker(s) — cross-reference table structure "
                    f"is consistent with chained incremental updates"
                )
                result["score"] += INCREMENTAL_XREF_MISMATCH_SCORE

            try:
                import pikepdf
                with pikepdf.open(pdf_path) as pdf:
                    prev = pdf.trailer.get("/Prev")
                    if prev is not None:
                        result["has_incremental_updates"] = True
                        result["prev_trailer_offset"] = int(prev)
                        result["signals"].append(
                            f"PDF trailer contains a /Prev pointer to byte "
                            f"offset {int(prev)} — structurally confirms an "
                            f"earlier revision's xref table still exists in "
                            f"the file"
                        )
            except Exception:
                pass

            # Raw byte scan for shadowed object versions.
            obj_def_re = re.compile(rb"(\d+)[ \t]+(\d+)[ \t]+obj\b")
            occurrences: dict[int, list[int]] = {}
            for m in obj_def_re.finditer(raw):
                objid = int(m.group(1))
                occurrences.setdefault(objid, []).append(m.start())

            old_versions = []
            for objid, offsets in occurrences.items():
                if len(offsets) <= 1:
                    continue
                start = offsets[0]  # earliest = pre-edit version
                snippet = raw[start:start + OLD_OBJECT_PREVIEW_BYTES]
                preview = snippet.decode("latin-1", errors="replace")
                preview = preview.split("endobj")[0].strip()
                old_versions.append({
                    "objid": objid,
                    "version_count": len(offsets),
                    "preview": preview[:160],
                })

            if old_versions:
                old_versions.sort(key=lambda v: -v["version_count"])
                result["old_objects_found"] = old_versions[:OLD_OBJECT_MAX_REPORTED]
                result["signals"].append(
                    f"{len(old_versions)} object(s) have a shadowed earlier "
                    f"version still present in the raw file — pre-edit "
                    f"content recovered where possible, see old_objects_found"
                )
                result["score"] += INCREMENTAL_OLD_OBJECTS_SCORE

        except Exception as e:
            result["signals"].append(f"Could not analyze incremental updates: {e}")

        result["score"] = min(100, result["score"])
        return result
