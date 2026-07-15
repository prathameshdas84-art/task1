"""Digital-signature validation and shadow-attack detection (post-
signature visual changes)."""

import re
from datetime import datetime

from .constants import *


class SignatureChecksMixin:

    @staticmethod
    def _parse_pdf_date(date_obj) -> "datetime | None":
        """Parse a PDF date value ('D:YYYYMMDDHHmmSS...') to a datetime."""
        if date_obj is None:
            return None
        date_str = str(date_obj).strip()
        if date_str.startswith("D:"):
            date_str = date_str[2:]
        match = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", date_str)
        if not match:
            return None
        try:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), int(match.group(5)), int(match.group(6))
            )
        except ValueError:
            return None

    @staticmethod
    def _find_signature_objects(pdf) -> list:
        """Return all /Type /Sig objects (signature dictionaries) in the PDF."""
        import pikepdf
        sigs = []
        for objid in range(1, len(pdf.objects) + 1):
            try:
                obj = pdf.get_object(objid, 0)
            except Exception:
                continue
            try:
                if hasattr(obj, 'get') and obj.get('/Type') == pikepdf.Name('/Sig'):
                    sigs.append(obj)
            except Exception:
                continue
        return sigs

    def _detect_shadow_attack(self, pdf_path: str) -> tuple[list[str], int]:
        """
        Shadow attacks append new content AFTER a digital signature using
        PDF incremental updates. The signature still cryptographically
        validates — it only ever covered the bytes that existed at signing
        time — but the visible content has changed since then.

        Detection logic:
        1. Count %%EOF markers — more than one means incremental updates exist.
        2. Find all /Sig signature dictionary objects.
        3. For each signature's /ByteRange [0, offset1, offset2, end],
           check whether it covers the entire file.
        4. Check whether any bytes exist after the signed range (content
           added after the last signature).
        """
        signals = []
        score   = 0

        try:
            import pikepdf

            with open(pdf_path, 'rb') as f:
                content = f.read()
            file_size = len(content)
            eof_count = content.count(b'%%EOF')

            with pikepdf.open(pdf_path) as pdf:
                sigs = self._find_signature_objects(pdf)

                if eof_count > 1 and sigs:
                    signals.append(
                        f"PDF has {eof_count} revision layers AND a digital "
                        f"signature is present — incremental updates after "
                        f"signing are the mechanism shadow attacks use"
                    )
                    score += SHADOW_EOF_SIG_SCORE

                for sig in sigs:
                    try:
                        byte_range = sig.get('/ByteRange')
                        if byte_range is None or len(byte_range) < 4:
                            continue
                        offset1 = int(byte_range[1])
                        offset2 = int(byte_range[2])
                        end     = int(byte_range[3])

                        # Does the signed range cover the whole file?
                        if offset1 + (end - offset2) != file_size:
                            signals.append(
                                f"Signature ByteRange does not cover the entire "
                                f"file ({offset1 + (end - offset2)} bytes signed "
                                f"vs {file_size} byte file) — gap exists between "
                                f"signed content and end of file"
                            )
                            score += SHADOW_BYTERANGE_GAP_SCORE

                        # Were bytes added after the signed range?
                        signed_end = offset2 + end
                        if file_size - signed_end > 0:
                            signals.append(
                                f"{file_size - signed_end} byte(s) exist after "
                                f"the signed range — objects were added to the "
                                f"file after this signature was applied"
                            )
                            score += SHADOW_OBJECTS_AFTER_SIG_SCORE
                    except Exception:
                        continue

        except Exception:
            pass

        return signals, min(100, score)

    def _validate_digital_signature(self, pdf_path: str) -> tuple[list[str], int]:
        """
        Validate the structural integrity of any digital signature present
        (this is a forensic ByteRange/date check, not a cryptographic
        signature verification — it checks for shadow-attack patterns and
        post-signing modification, not whether the signature itself is
        cryptographically authentic).
        """
        signals = []
        score   = 0

        try:
            import pikepdf

            with open(pdf_path, 'rb') as f:
                file_size = len(f.read())

            with pikepdf.open(pdf_path) as pdf:
                sigs = self._find_signature_objects(pdf)

                if not sigs:
                    return ["No digital signature present"], 0

                mod_date = self._parse_pdf_date(pdf.docinfo.get('/ModDate'))

                for sig in sigs:
                    sub_filter   = str(sig.get('/SubFilter', '')) or None
                    reason       = str(sig.get('/Reason', '')) or None
                    contact_info = str(sig.get('/ContactInfo', '')) or None
                    name         = str(sig.get('/Name', '')) or None
                    sign_date    = self._parse_pdf_date(sig.get('/M'))

                    detail_bits = [b for b in (
                        f"algorithm={sub_filter}" if sub_filter else None,
                        f"signer={name}" if name else None,
                        f"reason={reason}" if reason else None,
                        f"contact={contact_info}" if contact_info else None,
                    ) if b]
                    detail = f" ({', '.join(detail_bits)})" if detail_bits else ""

                    byte_range = sig.get('/ByteRange')
                    covers_full_file = False
                    if byte_range is not None and len(byte_range) >= 4:
                        offset1 = int(byte_range[1])
                        offset2 = int(byte_range[2])
                        end     = int(byte_range[3])
                        covers_full_file = (offset1 + (end - offset2) == file_size)

                    if not covers_full_file:
                        signals.append(
                            f"Digital signature does not cover entire file — "
                            f"shadow attack pattern{detail}"
                        )
                        score = max(score, SIG_BYTERANGE_GAP_SCORE)
                        continue

                    if sign_date and mod_date and mod_date > sign_date:
                        signals.append(
                            f"Document was modified after digital signature "
                            f"was applied (signed {sign_date}, modified {mod_date}){detail}"
                        )
                        score = max(score, SIG_MODIFIED_AFTER_SIGNING_SCORE)
                        continue

                    signals.append(f"Digital signature valid{detail}")

        except Exception:
            pass

        return signals, min(100, score)

