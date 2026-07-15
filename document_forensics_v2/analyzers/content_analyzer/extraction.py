"""Text extraction: PDF-type detection and the three-tier word
extraction ladder (pdfplumber -> PyMuPDF -> raw pikepdf content
streams), page rendering, and line grouping."""

import statistics
from collections import Counter

import fitz
import numpy as np
import pdfplumber

from .constants import *
from .models import LineProfile


class TextExtractionMixin:
    def _detect_pdf_type(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total = len(pdf.pages)
                with_text = sum(
                    1 for p in pdf.pages
                    if p.extract_text() and len(p.extract_text().strip()) > NATIVE_TEXT_MIN_CHARS
                )
            ratio = with_text / total if total > 0 else 0
            if ratio >= NATIVE_TEXT_RATIO_THRESHOLD:
                pdf_type = "native_text"
            elif ratio >= MIXED_TEXT_RATIO_THRESHOLD:
                pdf_type = "mixed"
            else:
                pdf_type = "scanned"
        except Exception:
            pdf_type = "native_text"

        try:
            import pikepdf
            with pikepdf.open(pdf_path) as pdf:
                info = pdf.docinfo
                producer = str(info.get("/Producer","")).lower()
                creator = str(info.get("/Creator","")).lower()
                if any(kw in producer or kw in creator for kw in SCANNER_KEYWORDS):
                    return "scanned_native"
        except Exception:
            pass

        # Override: vector PDFs (text outlined to paths, e.g. Canva/Figma/
        # Illustrator exports) have no usable text layer despite rendering
        # visible content — treat as "scanned" so the pixel-based layers
        # (ELA and its raster checks) weight them appropriately.
        if self._is_vector_pdf(pdf_path):
            return "scanned"

        return pdf_type

    def _is_vector_pdf(self, pdf_path: str) -> bool:
        """
        Returns True if this PDF has no usable text layer — text is stored
        as vector paths, not text objects. Happens with Canva, Figma,
        Illustrator, and any tool that exports PDF with outlined/converted
        text. Not tool-specific: detects the underlying condition (no text
        objects despite visible page content) rather than matching producer/
        creator strings, so it generalizes to any vector-export pipeline.

        Detection: pdfplumber finds (near) no words across all pages AND the
        page still renders visible drawings — distinguishing "no text
        because vector-only" from "no text because blank page".
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_words = sum(
                    len(p.extract_words() or [])
                    for p in pdf.pages
                )
            if total_words > 10:
                return False  # has real text layer

            # Check if page has visual content despite no text
            doc = fitz.open(pdf_path)
            has_visual = False
            for page in doc:
                # If page has paths/drawings but no text = vector PDF
                paths = page.get_drawings()
                text = page.get_text("text").strip()
                if len(paths) > 10 and len(text) < 20:
                    has_visual = True
                    break
            doc.close()
            return has_visual
        except Exception:
            return False

    # ── Line extraction ────────────────────────────────────────────────────────

    def _extract_lines(self, pdf_path: str) -> list[LineProfile]:
        """
        Extract per-line features using pdfplumber + PyMuPDF for visuals.

        Word extraction falls back through three methods per page, for
        corporate PDFs with custom/non-standard font encodings that defeat
        the primary extractor:
          1. pdfplumber extract_words() — handles the vast majority of PDFs.
          2. PyMuPDF (fitz) text extraction — has its own independent
             encoding/CMap handling and recovers text pdfplumber misses.
          3. pikepdf raw content-stream decode — last resort, recovers
             literal Tj/TJ string operands directly with no font/encoding
             interpretation at all (see _extract_words_pikepdf_fallback).
          4. If all three find nothing on a page, that page contributes no
             lines — pixel-level layers (ELA noise/erasure/flat-zone and the
             embedded-image checks) carry the evidence for such pages, since
             they work from rendered/embedded pixels, not the text layer.
        """
        page_images = self._render_pages(pdf_path)
        all_lines   = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    words = page.extract_words(
                        extra_attrs=["fontname", "size"],
                        keep_blank_chars=False
                    )
                except Exception:
                    words = []

                if not words:
                    words = self._extract_words_fitz_fallback(pdf_path, page_num)
                if not words:
                    words = self._extract_words_pikepdf_fallback(pdf_path, page_num)
                if not words:
                    continue

                grouped = self._group_into_lines(words)
                img     = page_images.get(page_num)

                for line_num, line_words in enumerate(grouped):
                    if not line_words:
                        continue

                    text = " ".join(w["text"] for w in line_words).strip()
                    if not text or len(text) < 2:
                        continue

                    fonts = [w.get("fontname", "Unknown") for w in line_words]
                    sizes = [float(w.get("size", 11)) for w in line_words]
                    font_name = Counter(fonts).most_common(1)[0][0]
                    font_size = statistics.median(sizes)

                    x0 = min(w["x0"]     for w in line_words)
                    y0 = min(w["top"]    for w in line_words)
                    x1 = max(w["x1"]     for w in line_words)
                    y1 = max(w["bottom"] for w in line_words)

                    noise, sharpness = self._visual_features(
                        img, (x0, y0, x1, y1),
                        scale=self.RENDER_DPI / 72
                    )

                    all_lines.append(LineProfile(
                        page=page_num,
                        line_num=line_num,
                        text=text,
                        font_name=font_name,
                        font_size=font_size,
                        char_spacing=self._char_spacing(line_words),
                        word_spacing=self._word_spacing(line_words),
                        line_height=y1 - y0,
                        bbox=(x0, y0, x1, y1),
                        noise=noise,
                        sharpness=sharpness,
                        char_widths=self._char_widths(line_words),
                    ))

        return all_lines

    def _extract_words_fitz_fallback(self, pdf_path: str, page_num: int) -> list[dict]:
        """
        Fallback step 2: PyMuPDF text extraction, converted to the same
        word dict shape pdfplumber's extract_words() produces. PyMuPDF has
        its own font/encoding handling independent of pdfplumber's, so it
        can recover text from PDFs whose custom encoding defeats pdfplumber
        (no per-character font/size info is available this way, so
        fontname/size are filled with placeholders — downstream font-based
        checks simply won't fire on these lines, which is acceptable since
        the alternative is no text at all).
        """
        words = []
        try:
            doc = fitz.open(pdf_path)
            if page_num < len(doc):
                for x0, y0, x1, y1, word_text, *_ in doc[page_num].get_text("words"):
                    if word_text.strip():
                        words.append({
                            "text": word_text,
                            "x0": x0, "top": y0, "x1": x1, "bottom": y1,
                            "fontname": "Unknown",
                            "size": 11.0,
                        })
            doc.close()
        except Exception:
            pass
        return words

    def _extract_words_pikepdf_fallback(self, pdf_path: str, page_num: int) -> list[dict]:
        """
        Fallback step 3 (last resort): decode the page's raw content stream
        with pikepdf and recover literal strings straight from Tj/TJ
        text-showing operators. No font/encoding interpretation and no real
        text-positioning state machine is applied (that would require
        reimplementing the PDF rendering model), so recovered strings are
        placed on synthetic sequential lines spanning the page width rather
        than at their true coordinates. This keeps at least the document's
        text content available to the content/numeric layers when both
        pdfplumber and PyMuPDF fail to decode the encoding at all.
        """
        words = []
        try:
            import pikepdf
            with pikepdf.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return []
                page = pdf.pages[page_num]
                page_width = float(page.MediaBox[2] - page.MediaBox[0]) if "/MediaBox" in page else 612.0

                row = 0
                for operands, operator in pikepdf.parse_content_stream(page):
                    op = str(operator)
                    strings = []
                    if op == "Tj" and operands:
                        strings.append(operands[0])
                    elif op == "TJ" and operands:
                        for item in operands[0]:
                            if not isinstance(item, (int, float)):
                                strings.append(item)

                    for s in strings:
                        try:
                            text = bytes(s).decode("latin-1", errors="replace").strip()
                        except Exception:
                            continue
                        if not text:
                            continue
                        y = 20.0 * row
                        words.append({
                            "text": text,
                            "x0": 20.0, "top": y, "x1": min(page_width - 20.0, 20.0 + len(text) * 6.0), "bottom": y + 14.0,
                            "fontname": "Unknown",
                            "size": 11.0,
                        })
                        row += 1
        except Exception:
            pass
        return words

    def _render_pages(self, pdf_path: str) -> dict:
        images = {}
        try:
            doc   = fitz.open(pdf_path)
            scale = self.RENDER_DPI / 72
            mat   = fitz.Matrix(scale, scale)
            for i, page in enumerate(doc):
                try:
                    pix       = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    images[i] = np.frombuffer(
                        pix.samples, dtype=np.uint8
                    ).reshape(pix.h, pix.w, 3)
                except Exception:
                    continue
            doc.close()
        except Exception:
            pass
        return images

    def _group_into_lines(self, words: list) -> list[list]:
        if not words:
            return []
        sorted_words = sorted(words, key=lambda w: (round(w["top"] / 4) * 4, w["x0"]))
        lines, current, current_y = [], [sorted_words[0]], sorted_words[0]["top"]
        for w in sorted_words[1:]:
            if abs(w["top"] - current_y) <= 5:
                current.append(w)
            else:
                lines.append(sorted(current, key=lambda x: x["x0"]))
                current, current_y = [w], w["top"]
        if current:
            lines.append(sorted(current, key=lambda x: x["x0"]))
        return lines

