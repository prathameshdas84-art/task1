"""Deep extraction: docinfo/XMP/fitz cross-reads, pikepdf extras, XMP
edit history, trailer IDs, object-level dates, revision info, and
page-rotation consistency."""

import re
import xml.etree.ElementTree as ET
from typing import Optional

import fitz
import pikepdf

from .database import _parse_pdf_date


class DeepExtractionMixin:
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

