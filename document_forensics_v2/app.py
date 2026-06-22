import os
import tempfile
import fitz
import streamlit as st
from metadata_extractor import MetadataExtractor
from content_analyzer import ContentAnalyzer
from ocr_analyzer import OCRAnalyzer
from numeric_analyzer import NumericAnalyzer
from ela_analyzer import ELAAnalyzer
from pymupdf_analyzer import PyMuPDFAnalyzer
from verdict_engine import combine
from location_highlighter import LocationHighlighter, RENDER_DPI

# Common LibreOffice install locations across platforms — "soffice" alone
# (last entry) covers any install already on PATH, including most Linux
# package-manager installs and Homebrew on Mac.
LIBREOFFICE_PATHS = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "soffice",
]
LIBREOFFICE_CONVERT_TIMEOUT_SECONDS = 30

# Fallback PDF-rendering settings used when converting a Word doc that has
# no LibreOffice available (python-docx + reportlab path).
FALLBACK_FONT_NAME       = "Helvetica"  # a standard PDF font, always available without embedding
FALLBACK_FONT_SIZE       = 11
FALLBACK_PAGE_MARGIN     = 20
FALLBACK_TEXT_LINE_HEIGHT = 20
FALLBACK_TEXT_TOP_MARGIN  = 50

st.set_page_config(
    page_title="Document Forensics",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
.stApp { background: #0a0a0a; color: #e0e0e0; }
.verdict-modified {
    font-size: 2.5rem; font-weight: 900; color: #FF4444;
    text-align: center; padding: 1.5rem;
    border: 2px solid #FF4444; border-radius: 8px;
    background: #FF444412; margin: 1rem 0;
}
.verdict-original {
    font-size: 2.5rem; font-weight: 900; color: #00CC66;
    text-align: center; padding: 1.5rem;
    border: 2px solid #00CC66; border-radius: 8px;
    background: #00CC6612; margin: 1rem 0;
}
.score-row {
    display: flex; justify-content: space-between;
    background: #111; padding: 10px 16px;
    border-radius: 6px; margin: 4px 0; font-size: 0.88rem;
}
.signal-item {
    background: #141414; border-left: 3px solid #FF8C00;
    padding: 9px 14px; margin: 5px 0;
    border-radius: 4px; font-size: 0.85rem; line-height: 1.6;
}
.signal-clean { border-left-color: #00CC66; }
.signal-meta  { border-left-color: #FF4444; }
.signal-ocr   { border-left-color: #8888FF; }
.line-item {
    background: #111; border-left: 3px solid #FF4444;
    padding: 10px 14px; margin: 6px 0;
    border-radius: 4px; font-size: 0.82rem;
    font-family: monospace; line-height: 1.6;
}
.ocr-item {
    background: #111; border-left: 3px solid #8888FF;
    padding: 10px 14px; margin: 6px 0;
    border-radius: 4px; font-size: 0.82rem;
    font-family: monospace; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔬 Document Forensics Engine")
st.markdown("##### Upload any PDF — was it modified?")
st.divider()

uploaded = st.file_uploader(
    "Upload Document (PDF, Word, Image)",
    type=["pdf", "jpg", "jpeg", "png", "doc", "docx"]
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    # Convert image to PDF if needed
    file_ext = uploaded.name.lower().split(".")[-1]
    if file_ext in ("jpg", "jpeg", "png"):
        import fitz
        img_doc = fitz.open()
        img_page = img_doc.new_page()
        img_page.insert_image(img_page.rect, filename=pdf_path)
        converted_path = pdf_path.replace(".pdf", "_converted.pdf")
        img_doc.save(converted_path)
        img_doc.close()
        os.unlink(pdf_path)
        pdf_path = converted_path
        st.info("Image converted to PDF for analysis.")

    # Add Word conversion
    elif file_ext in ("doc", "docx"):
        try:
            import subprocess
            import shutil

            # Try using LibreOffice to convert docx to PDF
            soffice = None
            for path in LIBREOFFICE_PATHS:
                if shutil.which(path) or os.path.exists(path):
                    soffice = path
                    break

            if soffice:
                import tempfile
                out_dir = tempfile.mkdtemp()
                subprocess.run(
                    [soffice, "--headless", "--convert-to", "pdf",
                     "--outdir", out_dir, pdf_path],
                    timeout=LIBREOFFICE_CONVERT_TIMEOUT_SECONDS,
                    capture_output=True
                )
                # Find converted file
                converted = [f for f in os.listdir(out_dir) if f.endswith(".pdf")]
                if converted:
                    new_path = os.path.join(out_dir, converted[0])
                    os.unlink(pdf_path)
                    pdf_path = new_path
                    st.info(f"Word document converted to PDF for analysis.")
                else:
                    st.warning("Word conversion produced no output — analyzing as-is.")
            else:
                # LibreOffice not found — try python-docx + reportlab fallback
                try:
                    import tempfile
                    from docx import Document as DocxDocument
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.pagesizes import A4
                    import io

                    doc = DocxDocument(pdf_path)

                    tmp_fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf")
                    os.close(tmp_fd)

                    c = canvas.Canvas(tmp_pdf, pagesize=A4)
                    w, h = A4

                    # Extract inline images from the docx, preserving document order
                    from PIL import Image as PILImage
                    from reportlab.lib.utils import ImageReader

                    all_images = []
                    for rel in doc.part.rels.values():
                        if "image" in rel.reltype:
                            try:
                                img_data = rel.target_part.blob
                                pil_img = PILImage.open(io.BytesIO(img_data))
                                if pil_img.mode not in ("RGB", "L"):
                                    pil_img = pil_img.convert("RGB")
                                all_images.append(pil_img)
                            except Exception:
                                continue

                    images_added = 0
                    margin = FALLBACK_PAGE_MARGIN
                    for pil_img in all_images:
                        try:
                            img_w, img_h = pil_img.size
                            scale = min((w - 2 * margin) / img_w, (h - 2 * margin) / img_h)
                            draw_w = img_w * scale
                            draw_h = img_h * scale
                            x_pos = (w - draw_w) / 2
                            y_pos = (h - draw_h) / 2

                            img_tmp = io.BytesIO()
                            pil_img.save(img_tmp, format="PNG", optimize=False)
                            img_tmp.seek(0)
                            c.drawImage(
                                ImageReader(img_tmp), x_pos, y_pos,
                                width=draw_w, height=draw_h
                            )
                            c.showPage()
                            images_added += 1
                        except Exception:
                            continue

                    # If no images, add text content
                    if images_added == 0:
                        y = h - FALLBACK_TEXT_TOP_MARGIN
                        for para in doc.paragraphs:
                            if para.text.strip():
                                c.setFont(FALLBACK_FONT_NAME, FALLBACK_FONT_SIZE)
                                c.drawString(FALLBACK_PAGE_MARGIN, y, para.text[:120])
                                y -= FALLBACK_TEXT_LINE_HEIGHT
                                if y < FALLBACK_TEXT_TOP_MARGIN:
                                    c.showPage()
                                    y = h - FALLBACK_TEXT_TOP_MARGIN

                    c.save()
                    os.unlink(pdf_path)
                    pdf_path = tmp_pdf
                    st.info(f"Word document converted — {images_added} image(s) extracted.")
                    if images_added > 0:
                        st.warning(
                            "⚠️ This Word document contains image(s) only — "
                            "no text layer available. "
                            "Analysis will be limited to metadata and visual checks. "
                            "For accurate results, upload the original PDF directly."
                        )
                except ImportError:
                    st.error(
                        "Cannot convert Word documents — LibreOffice not installed. "
                        "Please install LibreOffice or convert to PDF manually."
                    )
                    st.stop()
        except Exception as e:
            st.error(f"Word conversion failed: {e}")
            st.stop()

    # ── Run all 6 layers ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        with st.spinner("Layer 1: Metadata..."):
            try:
                meta_report = MetadataExtractor().extract(pdf_path)
            except Exception as e:
                st.error(f"Metadata: {e}")
                meta_report = None

    with col2:
        with st.spinner("Layer 2: Content..."):
            try:
                content_report = ContentAnalyzer().analyze(pdf_path)
            except Exception as e:
                st.error(f"Content: {e}")
                content_report = None

    with col3:
        with st.spinner("Layer 3: OCR..."):
            try:
                ocr_report = OCRAnalyzer().analyze(pdf_path)
            except Exception as e:
                st.error(f"OCR: {e}")
                ocr_report = None

    with col4:
        with st.spinner("Layer 4: Numbers..."):
            try:
                numeric_report = NumericAnalyzer().analyze(pdf_path)
            except Exception as e:
                st.error(f"Numeric: {e}")
                numeric_report = None

    with col5:
        with st.spinner("Layer 5: ELA..."):
            try:
                ela_report = ELAAnalyzer().analyze(pdf_path, content_report.pdf_type if content_report else "native_text")
            except Exception as e:
                st.error(f"ELA: {e}")
                ela_report = None

    with col6:
        with st.spinner("Layer 6: PyMuPDF..."):
            try:
                pymupdf_report = PyMuPDFAnalyzer().analyze(pdf_path)
            except Exception as e:
                st.error(f"PyMuPDF: {e}")
                pymupdf_report = None

    if not meta_report or not content_report or not ocr_report:
        try: os.unlink(pdf_path)
        except: pass
        st.stop()

    is_image_converted = meta_report and \
        MetadataExtractor().detect_image_conversion(pdf_path)

    if is_image_converted:
        st.warning(
            "⚠️ This document appears to be a digital file converted to image format. "
            "This technique is sometimes used to hide text-level edits. "
            "OCR analysis has been applied but accuracy may be limited."
        )

    # ── Combine verdict ───────────────────────────────────────────────────────
    verdict = combine(meta_report, content_report, ocr_report, numeric_report, ela_report, pymupdf_report)

    # ── Verdict banner ────────────────────────────────────────────────────────
    # Document name display
    st.markdown(
        f'<div style="background:#111; padding:8px 16px; border-radius:6px; '
        f'margin-bottom:1rem; font-size:0.88rem; color:#888;">'
        f'📄 <strong style="color:#e0e0e0">{uploaded.name}</strong> &nbsp;|&nbsp; '
        f'{os.path.getsize(pdf_path)/1024:.1f} KB &nbsp;|&nbsp; '
        f'PDF Type: {content_report.pdf_type.replace("_"," ").title()}'
        f'</div>',
        unsafe_allow_html=True
    )

    if verdict.verdict == "MODIFIED":
        st.markdown(
            '<div class="verdict-modified">⚠️ MODIFIED</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="verdict-original">✅ NOT MODIFIED</div>',
            unsafe_allow_html=True
        )

    # ── Score breakdown ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="score-row">
        <span>Confidence</span><strong>{verdict.confidence}%</strong>
    </div>
    <div class="score-row">
        <span>Combined Score</span><strong>{verdict.combined_score}/100</strong>
    </div>
    <div class="score-row">
        <span>Layer 1 — Metadata</span><strong>{verdict.metadata_score}/100</strong>
    </div>
    <div class="score-row">
        <span>Layer 2 — Content</span><strong>{verdict.content_score}/100</strong>
    </div>
    <div class="score-row">
        <span>Layer 3 — OCR</span><strong>{verdict.ocr_score}/100</strong>
    </div>
    <div class="score-row">
        <span>Layer 4 — Numeric</span><strong>{verdict.numeric_score}/100</strong>
    </div>
    <div class="score-row">
        <span>Layer 5 — ELA</span><strong>{verdict.ela_score}/100</strong>
    </div>
    <div class="score-row">
        <span>Layer 6 — PyMuPDF</span><strong>{verdict.pymupdf_score}/100</strong>
    </div>
    <div class="score-row">
        <span>PDF Type</span>
        <strong>{verdict.pdf_type.replace("_"," ").title()}</strong>
    </div>
    <div class="score-row">
        <span>Document Source</span>
        <strong>{meta_report.source.identified_name}</strong>
    </div>
    <div class="score-row">
        <span>OCR Avg Confidence</span>
        <strong>{ocr_report.avg_confidence:.0f}%</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── All signals ───────────────────────────────────────────────────────────
    st.markdown("#### Evidence")
    for sig in verdict.all_signals:
        is_clean = any(x in sig.lower() for x in [
            "consistent", "no anomaly", "passed", "no confidence"
        ])
        is_meta = "[METADATA]" in sig
        is_ocr  = "[OCR]" in sig
        if is_clean:
            css = "signal-clean"
            icon = "✓"
        elif is_meta:
            css = "signal-meta"
            icon = "⚡"
        elif is_ocr:
            css = "signal-ocr"
            icon = "🔍"
        else:
            css = "signal-item"
            icon = "⚡"
        st.markdown(
            f'<div class="{css}">{icon} {sig}</div>',
            unsafe_allow_html=True
        )

    # ── Location Detection (Phase 2) ─────────────────────────────────────────
    if verdict.verdict == "MODIFIED" and (
        content_report.suspicious_lines or ocr_report.suspicious_regions or
        (numeric_report and numeric_report.anomalies) or
        (ela_report and ela_report.regions) or
        (pymupdf_report and pymupdf_report.overlay_regions)
    ):
        st.markdown("#### 📍 Modified Location Detection")
        st.markdown(
            '<div style="background:#111; padding:10px 16px; border-radius:6px; '
            'margin-bottom:1rem; font-size:0.85rem; color:#888;">'
            '🔴 Red = font/spacing &nbsp;|&nbsp; '
            '🟠 Orange = OCR &nbsp;|&nbsp; '
            '🟡 Yellow = numeric &nbsp;|&nbsp; '
            '🟣 Purple = ELA (z≥4.0 only) &nbsp;|&nbsp; '
            '🩵 Cyan = white-rect overlay &nbsp;|&nbsp; '
            '🟪 Magenta = image overlay</div>',
            unsafe_allow_html=True
        )

        try:
            highlighter = LocationHighlighter(pdf_path)
            highlighted_pages = highlighter.highlight_pages(
                suspicious_lines=content_report.suspicious_lines,
                ocr_regions=ocr_report.suspicious_regions,
                numeric_anomalies=numeric_report.anomalies if numeric_report else [],
                ela_regions=ela_report.regions if ela_report else [],
                overlay_regions=pymupdf_report.overlay_regions if pymupdf_report else [],
            )

            if highlighted_pages:
                # Get total pages
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                doc.close()

                # Page selector for ALL pages
                selected_page = st.selectbox(
                    f"Select page (1-{total_pages}):",
                    list(range(total_pages)),
                    format_func=lambda x: f"Page {x+1}" + (" 🔴" if x in highlighted_pages else "")
                )

                # Show page — with boxes if flagged, clean if not
                if selected_page in highlighted_pages:
                    page_img = highlighted_pages[selected_page]
                    caption = f"Page {selected_page+1} — suspicious regions highlighted"
                else:
                    # Render clean page
                    doc2 = fitz.open(pdf_path)
                    page = doc2[selected_page]
                    scale = RENDER_DPI / 72
                    mat = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    from PIL import Image
                    page_img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
                    doc2.close()
                    caption = f"Page {selected_page+1} — no anomalies detected"

                st.image(page_img, use_column_width=True, caption=caption)

                # Show line details below image
                page_lines = [
                    sl for sl in content_report.suspicious_lines
                    if sl.page == selected_page
                ]
                if page_lines:
                    st.markdown("**Flagged lines on this page:**")
                    for sl in page_lines:
                        pct = int(sl.score * 100)
                        st.markdown(
                            f'<div class="line-item">'
                            f'<strong>Line {sl.line_num+1}</strong> '
                            f'<span style="color:#FF8C00">({pct}% anomaly)</span><br>'
                            f'<span style="color:#ccc">"{sl.text}"</span><br>'
                            f'<span style="color:#666">'
                            f'{"  |  ".join(sl.anomalies[:2])}'
                            f'</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.info("No specific regions could be highlighted on this document.")

        except Exception as e:
            st.warning(f"Location highlighting unavailable: {e}")

    # ── Content suspicious lines ──────────────────────────────────────────────
    if content_report.suspicious_lines:
        st.markdown(
            f"#### Suspicious Lines — Content "
            f"({len(content_report.suspicious_lines)} flagged)"
        )
        for sl in content_report.suspicious_lines:
            pct     = int(sl.score * 100)
            reasons = " | ".join(sl.anomalies[:2])
            st.markdown(
                f'<div class="line-item">'
                f'<strong>Page {sl.page+1} · Line {sl.line_num+1}</strong> '
                f'<span style="color:#FF8C00">({pct}% anomaly)</span><br>'
                f'<span style="color:#ccc">"{sl.text}"</span><br>'
                f'<span style="color:#666">{reasons}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── OCR suspicious regions ────────────────────────────────────────────────
    if ocr_report.suspicious_regions:
        st.markdown(
            f"#### Suspicious Regions — OCR "
            f"({len(ocr_report.suspicious_regions)} flagged)"
        )
        for r in ocr_report.suspicious_regions:
            st.markdown(
                f'<div class="ocr-item">'
                f'<strong>Page {r.page+1}</strong> '
                f'<span style="color:#8888FF">'
                f'(confidence {r.confidence:.0f}% vs page avg {r.page_avg_confidence:.0f}%)'
                f'</span><br>'
                f'<span style="color:#ccc">"{r.text}"</span><br>'
                f'<span style="color:#666">{r.reason}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Full metadata ─────────────────────────────────────────────────────────
    with st.expander("📋 Full Metadata"):
        def fmt(d):
            return d.strftime("%Y-%m-%d %H:%M:%S") if d else "—"
        st.json({
            "Producer":            meta_report.producer or "—",
            "Creator":             meta_report.creator  or "—",
            "Author":              meta_report.author   or "—",
            "Created":             fmt(meta_report.creation_date),
            "Modified":            fmt(meta_report.modification_date),
            "XMP Mismatch":        meta_report.xmp_docinfo_mismatch,
            "Multiple Producers":  meta_report.multiple_producers,
            "Metadata Stripped":   meta_report.metadata_stripped,
            "Source Risk":         meta_report.source.suspicion_level,
            "OCR Pages Analyzed":  ocr_report.pages_analyzed,
            "OCR Avg Confidence":  f"{ocr_report.avg_confidence:.1f}%",
            "OCR/Embedded Mismatch": ocr_report.ocr_vs_embedded_mismatch,
            "Mismatch Ratio":      f"{ocr_report.mismatch_ratio:.0%}",
        })

    try:
        os.unlink(pdf_path)
    except PermissionError:
        pass
