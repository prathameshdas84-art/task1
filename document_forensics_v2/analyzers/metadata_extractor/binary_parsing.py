"""Binary blob parsers: ICC profile text tags, sfnt/Type1/CFF font
name tables, and font-tool signatures."""

import re
import struct
from typing import Optional


class BinaryParsingMixin:
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

