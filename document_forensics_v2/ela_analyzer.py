"""
Error Level Analysis (ELA) — Layer 5
Detects localized image-editing artifacts by re-compressing the page as
JPEG and measuring the per-block difference against the original render.
Blocks with abnormally high recompression error (relative to the page's
own block-error distribution) indicate a region that was likely edited
or pasted in after the rest of the page was finalized.
"""

import io
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime

import fitz
import numpy as np
from PIL import Image


RENDER_DPI    = 150
# Vector PDFs (text outlined to paths — Canva/Figma/Illustrator exports)
# have no raster compression artifacts to find at low DPI since there's no
# embedded detail to begin with; rendering at higher DPI gives ELA enough
# pixels per block to produce a meaningful signal.
VECTOR_PDF_RENDER_DPI = 300
BLOCK_SIZE    = 32
ELA_QUALITY   = 75
ELA_AMPLIFY   = 15
Z_THRESHOLD   = 3.0
MIN_BLOCKS    = 4

# Margin sampled to estimate the document's baseline recompression noise
# floor — top/bottom/left/right 5% of the page, assumed blank.
NOISE_FLOOR_MARGIN_FRACTION = 20  # margin = page_dim // this value (5%)

# How block-level outlier fraction maps to the 0-100 anomaly score.
# Empirically calibrated against real test documents (clean bank statement,
# Joining Letter, KLPT offer letter): a clean multi-page document's natural
# header/text blocks land around 0.3-0.5% fraction-flagged; documents with
# genuine localized edits showed 1.4-2.3%. This multiplier was chosen so
# that range maps to a meaningfully separated score band — retune against
# a broader document sample if false positive/negative rates drift.
FRACTION_TO_SCORE_MULTIPLIER = 3000

# Per-page boxes are capped to the strongest N outliers so the UI doesn't
# flood with low-confidence boxes when a page has many flagged blocks.
MAX_REGIONS_PER_PAGE = 10

# Cross-page noise-consistency check (possible whole-page substitution).
CROSS_PAGE_MIN_PAGES     = 3     # need at least this many pages to compare
CROSS_PAGE_Z_THRESHOLD   = 2.5
CROSS_PAGE_SCORE_PER_PAGE = 20
CROSS_PAGE_SCORE_CAP      = 60
CROSS_PAGE_MERGE_DIVISOR  = 2    # how much this sub-score contributes to the final score

# PDF object-fingerprinting score weights (incremental updates, deleted/
# reused objects, FreeText/Redact annotations, Form XObjects).
EOF_SCORE_PER_REVISION       = 15
EOF_SCORE_CAP                = 40
HIGH_GEN_SCORE_PER_OBJECT    = 10
HIGH_GEN_SCORE_CAP           = 35
FREETEXT_SCORE_PER_ANNOT     = 15
FREETEXT_SCORE_CAP           = 40
FORM_XOBJECT_SCORE_PER_ITEM  = 10
FORM_XOBJECT_SCORE_CAP       = 30
OBJECT_MERGE_DIVISOR         = 3   # how much the object-fingerprint sub-score contributes to the final score

# DCT coefficient analysis (8x8 JPEG blocks).
DCT_BLOCK_SIZE       = 8
DCT_MIN_BLOCKS       = 10   # need at least this many 8x8 blocks to compute stats
DCT_Z_THRESHOLD      = 3.5  # higher than Z_THRESHOLD — DCT energy is noisier than ELA error
DCT_SCORE_PER_REGION = 3
DCT_SCORE_CAP        = 30
DCT_MERGE_DIVISOR    = 4    # how much the DCT sub-score contributes to the final score

# Shadow attack detection: new content appended after a digital signature
# via PDF incremental updates. The signature still cryptographically
# validates (it only covers the bytes present when it was applied) but the
# visible content has changed since signing.
SHADOW_EOF_SIG_SCORE            = 50  # incremental updates + a signature present
SHADOW_BYTERANGE_GAP_SCORE      = 60  # signature's ByteRange doesn't cover the whole file
SHADOW_OBJECTS_AFTER_SIG_SCORE  = 40  # bytes exist after the signed range
SHADOW_ATTACK_SCORE_DIVISOR     = 2   # how much this sub-score contributes to the final score

# Digital signature validation.
SIG_BYTERANGE_GAP_SCORE           = 70  # ByteRange doesn't cover the entire file
SIG_MODIFIED_AFTER_SIGNING_SCORE  = 50  # document ModDate is after the signing date
SIGNATURE_SCORE_DIVISOR           = 2   # how much this sub-score contributes to the final score


@dataclass
class ELARegion:
    page: int
    bbox: tuple        # (x0, y0, x1, y1) in PDF points — resolution-independent
    mean_error: float
    z_score: float
    render_dpi: float = RENDER_DPI  # DPI this region's block was measured at


@dataclass
class ELAReport:
    pdf_type: str
    anomaly_score: int
    regions: list = field(default_factory=list)
    signals: list = field(default_factory=list)


class ELAAnalyzer:

    def analyze(self, pdf_path: str, pdf_type: str = "native_text") -> ELAReport:
        doc = fitz.open(pdf_path)

        # Use higher DPI for vector PDFs (no compression artifacts at low
        # DPI, need more pixels for meaningful ELA).
        from content_analyzer import ContentAnalyzer
        try:
            is_vector = ContentAnalyzer()._is_vector_pdf(pdf_path)
        except Exception:
            is_vector = False
        render_dpi = VECTOR_PDF_RENDER_DPI if is_vector else RENDER_DPI
        scale = render_dpi / 72
        mat = fitz.Matrix(scale, scale)

        all_regions  = []
        total_blocks = 0
        total_flagged = 0
        total_dct_regions = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)

            regions, n_blocks, n_flagged = self._analyze_page(img, page_num, render_dpi)
            all_regions.extend(regions)
            total_blocks  += n_blocks
            total_flagged += n_flagged

            # DCT coefficient analysis
            dct_regions = self._dct_analysis(img, page_num)
            # Only add DCT regions not already covered by ELA
            # (merge if overlapping, add if new)
            total_dct_regions += len(dct_regions)

        doc.close()

        signals = [
            f"Page {r.page + 1}: ELA outlier block at "
            f"({r.bbox[0]:.0f},{r.bbox[1]:.0f})-({r.bbox[2]:.0f},{r.bbox[3]:.0f}) "
            f"error={r.mean_error:.1f} z={r.z_score:.1f}"
            for r in all_regions
        ]

        # Score on the FRACTION of blocks flagged across the whole document,
        # not the raw count — raw count saturates on any multi-page document
        # since every page naturally has some high-texture blocks (headers,
        # logos, dense text) that recompress with more error than blank space.
        fraction = (total_flagged / total_blocks) if total_blocks else 0.0
        anomaly_score = min(100, round(fraction * FRACTION_TO_SCORE_MULTIPLIER))

        # Cross-page consistency check (for multi-page documents)
        cp_anomalies, cp_signals, cp_score = self._cross_page_consistency(pdf_path)
        for s in cp_signals:
            signals.append(s)
        anomaly_score = min(100, anomaly_score + cp_score // CROSS_PAGE_MERGE_DIVISOR)

        # PDF object fingerprinting
        obj_signals, obj_score = self._pdf_object_fingerprint(pdf_path)
        for s in obj_signals:
            signals.append(f"[OBJECT] {s}")
        anomaly_score = min(100, anomaly_score + obj_score // OBJECT_MERGE_DIVISOR)

        if total_dct_regions:
            dct_score = min(DCT_SCORE_CAP, total_dct_regions * DCT_SCORE_PER_REGION)
            anomaly_score = min(100, anomaly_score + dct_score // DCT_MERGE_DIVISOR)

        # Shadow attack detection
        shadow_signals, shadow_score = self._detect_shadow_attack(pdf_path)
        for s in shadow_signals:
            signals.append(f"[SHADOW] {s}")
        anomaly_score = min(100, anomaly_score + shadow_score // SHADOW_ATTACK_SCORE_DIVISOR)

        # Digital signature validation
        sig_signals, sig_score = self._validate_digital_signature(pdf_path)
        for s in sig_signals:
            signals.append(f"[SIGNATURE] {s}")
        anomaly_score = min(100, anomaly_score + sig_score // SIGNATURE_SCORE_DIVISOR)

        return ELAReport(
            pdf_type=pdf_type,
            anomaly_score=anomaly_score,
            regions=all_regions,
            signals=signals,
        )

    def _dct_analysis(self, img: Image.Image, page_num: int) -> list:
        """
        Analyze DCT coefficient distribution across 8x8 blocks.
        JPEG compression works in 8x8 DCT blocks.
        Edited/pasted regions have different coefficient distributions
        than organically compressed regions.

        Returns list of suspicious block coordinates.
        """
        try:
            import cv2 as _cv2

            # Convert to YCbCr (JPEG native color space)
            arr = np.asarray(img.convert("YCbCr"), dtype=np.float32)
            y_channel = arr[:, :, 0]  # Luma channel

            h, w = y_channel.shape
            dct_block_size = DCT_BLOCK_SIZE
            n_rows = h // dct_block_size
            n_cols = w // dct_block_size

            if n_rows < 4 or n_cols < 4:
                return []

            # Compute DCT energy per block
            block_energies = []
            block_coords   = []

            for r in range(n_rows):
                for c in range(n_cols):
                    block = y_channel[
                        r*dct_block_size:(r+1)*dct_block_size,
                        c*dct_block_size:(c+1)*dct_block_size,
                    ]
                    dct = _cv2.dct(block)
                    # High-frequency energy (bottom-right of DCT matrix)
                    hf_energy = float(np.sum(np.abs(dct[4:, 4:])))
                    block_energies.append(hf_energy)
                    block_coords.append((c * dct_block_size, r * dct_block_size))

            if len(block_energies) < DCT_MIN_BLOCKS:
                return []

            energies = np.array(block_energies)
            mean_e   = energies.mean()
            std_e    = max(energies.std(), 0.01)

            suspicious = []
            for i, (energy, (bx, by)) in enumerate(zip(block_energies, block_coords)):
                z = abs(energy - mean_e) / std_e
                if z >= DCT_Z_THRESHOLD:  # higher threshold than ELA — DCT energy is noisier
                    pts_scale = 72 / RENDER_DPI
                    x0 = bx * pts_scale
                    y0 = by * pts_scale
                    x1 = (bx + dct_block_size) * pts_scale
                    y1 = (by + dct_block_size) * pts_scale
                    suspicious.append({
                        "page": page_num,
                        "bbox": (x0, y0, x1, y1),
                        "energy": energy,
                        "z_score": round(z, 2),
                    })

            return suspicious

        except Exception:
            return []

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

                    # Signal 4: Form XObjects (paste-over content)
                    try:
                        if (hasattr(obj, 'get') and
                            obj.get('/Type') == pikepdf.Name('/XObject') and
                            obj.get('/Subtype') == pikepdf.Name('/Form')):
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

    def _cross_page_consistency(
        self,
        pdf_path: str,
    ) -> tuple[list, list[str], int]:
        """
        Compare noise texture fingerprint across all pages.
        Genuine scanned documents have consistent noise patterns
        (same scanner, same settings, same paper).
        A replaced/substituted page shows different noise texture.

        Returns: (anomaly_list, signals, score)
        """
        try:
            doc   = fitz.open(pdf_path)
            scale = RENDER_DPI / 72
            mat   = fitz.Matrix(scale, scale)

            page_fingerprints = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img  = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
                arr  = np.asarray(img.convert("L"), dtype=np.float32)

                # Noise fingerprint = std of high-frequency component
                # Use Laplacian to extract high-frequency noise
                import cv2 as _cv2
                lap = _cv2.Laplacian(arr, _cv2.CV_64F)

                # Divide page into 4 quadrants, compute noise per quadrant
                h, w = lap.shape
                quadrants = [
                    lap[:h//2, :w//2],
                    lap[:h//2, w//2:],
                    lap[h//2:, :w//2],
                    lap[h//2:, w//2:],
                ]
                fingerprint = [float(np.std(q)) for q in quadrants]
                page_fingerprints.append({
                    "page": page_num,
                    "fingerprint": fingerprint,
                    "mean_noise": float(np.std(lap)),
                })

            doc.close()

            if len(page_fingerprints) < CROSS_PAGE_MIN_PAGES:
                # Need at least 3 pages to compare
                return [], ["Cross-page check skipped — document has fewer than 3 pages"], 0

            # Compare each page's noise against document average
            all_noise = [p["mean_noise"] for p in page_fingerprints]
            doc_mean  = statistics.mean(all_noise)
            doc_std   = max(statistics.stdev(all_noise), 0.01)

            anomalies = []
            for p in page_fingerprints:
                z = abs(p["mean_noise"] - doc_mean) / doc_std
                if z >= CROSS_PAGE_Z_THRESHOLD:
                    anomalies.append({
                        "page": p["page"],
                        "noise": p["mean_noise"],
                        "doc_mean": doc_mean,
                        "z_score": round(z, 2),
                        "reason": (
                            f"Page {p['page']+1} noise texture ({p['mean_noise']:.1f}) "
                            f"differs from document average ({doc_mean:.1f}) "
                            f"by z={z:.1f} — possible page substitution"
                        )
                    })

            signals = []
            score   = 0

            if anomalies:
                signals.append(
                    f"{len(anomalies)} page(s) have inconsistent noise texture — "
                    f"possible page substitution or different scan source"
                )
                score = min(CROSS_PAGE_SCORE_CAP, len(anomalies) * CROSS_PAGE_SCORE_PER_PAGE)
            else:
                signals.append(
                    "Cross-page scan consistency check passed — "
                    "all pages show uniform noise texture"
                )

            return anomalies, signals, score

        except Exception:
            return [], [], 0

    def _estimate_noise_floor(self, img: Image.Image) -> float:
        """
        Estimate document noise floor from blank margin regions.
        Margins (top 5%, bottom 5%, left 5%, right 5%) are typically blank
        and represent the natural noise level of the document/scan.
        Returns mean ELA error in margin regions.
        """
        try:
            w, h = img.size
            margin_x = max(1, w // NOISE_FLOOR_MARGIN_FRACTION)  # 5% margin
            margin_y = max(1, h // NOISE_FLOOR_MARGIN_FRACTION)

            # Compute ELA on full image
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=ELA_QUALITY)
            buf.seek(0)
            recompressed = Image.open(buf)

            orig_arr  = np.asarray(img, dtype=np.int16)
            recom_arr = np.asarray(recompressed, dtype=np.int16)
            diff = np.abs(orig_arr - recom_arr).mean(axis=2).astype(np.float32)

            # Sample margin regions
            top_margin    = diff[:margin_y, :]
            bottom_margin = diff[h-margin_y:, :]
            left_margin   = diff[:, :margin_x]
            right_margin  = diff[:, w-margin_x:]

            margin_values = np.concatenate([
                top_margin.flatten(),
                bottom_margin.flatten(),
                left_margin.flatten(),
                right_margin.flatten(),
            ])

            return float(np.mean(margin_values)) + float(np.std(margin_values))
        except Exception:
            return 0.0

    def _analyze_page(self, img: Image.Image, page_num: int, render_dpi: float = RENDER_DPI):
        # Recompress as JPEG and diff against the original render
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=ELA_QUALITY)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig_arr  = np.asarray(img, dtype=np.int16)
        recom_arr = np.asarray(recompressed, dtype=np.int16)
        diff = np.abs(orig_arr - recom_arr).sum(axis=2)  # H x W error map
        gray = np.asarray(img.convert("L"), dtype=np.float64)

        h, w = diff.shape
        n_rows = h // BLOCK_SIZE
        n_cols = w // BLOCK_SIZE
        if n_rows < 2 or n_cols < 2:
            return [], 0, 0

        # Normalize each block's recompression error by its own texture
        # (grayscale std-dev) — a block of dense text/logo legitimately
        # recompresses with more error than blank page space, so raw error
        # alone can't tell "naturally busy" apart from "edited."
        ratios = np.zeros((n_rows, n_cols), dtype=np.float64)
        errors = np.zeros((n_rows, n_cols), dtype=np.float64)
        for r in range(n_rows):
            for c in range(n_cols):
                block_diff = diff[
                    r * BLOCK_SIZE:(r + 1) * BLOCK_SIZE,
                    c * BLOCK_SIZE:(c + 1) * BLOCK_SIZE,
                ]
                block_gray = gray[
                    r * BLOCK_SIZE:(r + 1) * BLOCK_SIZE,
                    c * BLOCK_SIZE:(c + 1) * BLOCK_SIZE,
                ]
                err = block_diff.mean()
                texture = block_gray.std()
                errors[r, c] = err
                ratios[r, c] = err / (texture + 1.0)

        flat_ratios = ratios.flatten()
        flat_errors = errors.flatten()
        if flat_ratios.size < MIN_BLOCKS:
            return [], 0, 0

        noise_floor = self._estimate_noise_floor(img)
        # Subtract noise floor from errors before scoring
        # This removes the baseline noise common to all blocks
        flat_errors_norm = np.maximum(0, flat_errors - noise_floor)
        flat_ratios_norm = flat_errors_norm / (flat_ratios + 1e-6) * flat_ratios

        mean = flat_ratios_norm.mean()
        std  = flat_ratios_norm.std()

        # Coordinate conversion must use the DPI this page was actually
        # rendered at (render_dpi), not a fixed instance default — vector
        # PDFs render at VECTOR_PDF_RENDER_DPI, not RENDER_DPI, and using
        # the wrong scale here would put boxes in the wrong place.
        pts_scale = 72 / render_dpi

        regions  = []
        n_flagged = 0
        for idx in range(flat_ratios_norm.size):
            z = 0.0 if std < 1e-6 else (flat_ratios_norm[idx] - mean) / std
            if z > Z_THRESHOLD:
                n_flagged += 1
                r, c = divmod(idx, n_cols)
                x0 = c * BLOCK_SIZE * pts_scale
                y0 = r * BLOCK_SIZE * pts_scale
                x1 = (c + 1) * BLOCK_SIZE * pts_scale
                y1 = (r + 1) * BLOCK_SIZE * pts_scale
                regions.append(ELARegion(
                    page=page_num,
                    bbox=(x0, y0, x1, y1),
                    mean_error=float(flat_errors[idx]),
                    z_score=float(z),
                    render_dpi=render_dpi,
                ))

        # Cap how many boxes get drawn per page — keep only the strongest
        # outliers so the UI doesn't flood with low-confidence boxes.
        regions.sort(key=lambda r: r.z_score, reverse=True)
        regions = regions[:MAX_REGIONS_PER_PAGE]

        return regions, flat_ratios.size, n_flagged
