"""
File conversion — turns an uploaded image or Word document into a PDF so
the rest of the pipeline only ever has to deal with one format. Relocated
verbatim out of main.py (Phase 2 folder reorganization) — no logic changes.
"""

import os
import tempfile
from pathlib import Path

from fastapi import HTTPException


def convert_to_pdf(file_path: str, original_filename: str) -> str:
    """
    Convert image or Word document to PDF for analysis.
    Returns path to PDF file (may be same as input if already PDF).
    """
    ext = Path(original_filename).suffix.lower()

    if ext == ".pdf":
        return file_path

    if ext in (".jpg", ".jpeg", ".png"):
        import fitz
        img_doc  = fitz.open()
        img_page = img_doc.new_page()
        img_page.insert_image(img_page.rect, filename=file_path)
        tmp_fd, converted = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        img_doc.save(converted)
        img_doc.close()
        os.unlink(file_path)
        return converted

    if ext in (".docx", ".doc"):
        import io
        import shutil
        import subprocess
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from docx import Document as DocxDocument
        from PIL import Image as PILImage

        # Try LibreOffice first
        libreoffice_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            "/usr/bin/soffice",
            "/usr/local/bin/soffice",
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "soffice",
        ]
        soffice = None
        for path in libreoffice_paths:
            if shutil.which(path) or os.path.exists(path):
                soffice = path
                break

        if soffice:
            out_dir = tempfile.mkdtemp()
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf",
                 "--outdir", out_dir, file_path],
                timeout=30, capture_output=True
            )
            converted_files = [f for f in os.listdir(out_dir) if f.endswith(".pdf")]
            if converted_files:
                converted = os.path.join(out_dir, converted_files[0])
                os.unlink(file_path)
                return converted

        # Fallback: python-docx + reportlab
        doc = DocxDocument(file_path)
        tmp_fd, converted = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        c = canvas.Canvas(converted, pagesize=A4)
        w, h = A4

        all_images = []
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    img_data = rel.target_part.blob
                    pil_img  = PILImage.open(io.BytesIO(img_data))
                    if pil_img.mode not in ("RGB", "L"):
                        pil_img = pil_img.convert("RGB")
                    all_images.append(pil_img)
                except Exception:
                    continue

        if all_images:
            for pil_img in all_images:
                iw, ih = pil_img.size
                scale  = min((w - 40) / iw, (h - 40) / ih)
                buf    = io.BytesIO()
                pil_img.save(buf, format="PNG", optimize=False)
                buf.seek(0)
                c.drawImage(ImageReader(buf),
                            (w - iw * scale) / 2, (h - ih * scale) / 2,
                            width=iw * scale, height=ih * scale)
                c.showPage()
        else:
            y = h - 50
            for para in doc.paragraphs:
                if para.text.strip():
                    c.setFont("Helvetica", 11)
                    c.drawString(20, y, para.text[:120])
                    y -= 20
                    if y < 50:
                        c.showPage()
                        y = h - 50

        c.save()
        os.unlink(file_path)
        return converted

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {ext}. Supported: PDF, JPG, PNG, DOCX"
    )
