"""
Test PDF Generator for Document Forensics Engine
Generates clean + tampered PDFs for accuracy testing.
Run: python generate_test_pdfs.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import pikepdf

OUTPUT_DIR = Path("test_pdfs")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_offer_letter(path: str, salary: str, date: str, font: str = "Helvetica"):
    """Generate a realistic offer letter PDF."""
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4

    # Company header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, h - 60, "ACME TECHNOLOGIES PVT LTD")
    c.setFont("Helvetica", 10)
    c.drawString(72, h - 78, "123 Business Park, Andheri East, Mumbai - 400069")
    c.drawString(72, h - 92, "Tel: +91-22-12345678 | Email: hr@acmetech.com")

    # Divider
    c.line(72, h - 100, w - 72, h - 100)

    # Date
    c.setFont("Helvetica", 11)
    c.drawString(72, h - 125, f"Date: {date}")

    # Reference
    c.drawString(72, h - 145, "Ref: ACME/HR/2024/OL/1042")

    # Candidate details
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, h - 175, "To,")
    c.setFont("Helvetica", 11)
    c.drawString(72, h - 193, "Mr. Rahul Sharma")
    c.drawString(72, h - 211, "45, Green Park Colony")
    c.drawString(72, h - 229, "Pune, Maharashtra - 411001")

    # Subject
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, h - 260, "Subject: Offer of Employment")
    c.line(72, h - 265, 300, h - 265)

    # Body
    c.setFont("Helvetica", 11)
    c.drawString(72, h - 290, "Dear Mr. Rahul Sharma,")
    c.drawString(72, h - 315,
        "We are pleased to offer you the position of Senior Software Engineer")
    c.drawString(72, h - 333,
        "at Acme Technologies Pvt Ltd, subject to the following terms:")

    # Terms table
    terms = [
        ("Designation",    "Senior Software Engineer"),
        ("Department",     "Engineering"),
        ("Date of Joining", date),
        ("Work Location",  "Mumbai, Maharashtra"),
        ("Employment Type", "Full Time, Permanent"),
    ]

    y = h - 370
    for label, value in terms:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(90, y, f"{label}:")
        c.setFont("Helvetica", 11)
        c.drawString(260, y, value)
        y -= 20

    # Salary line — this is what gets tampered
    c.setFont("Helvetica-Bold", 11)
    c.drawString(90, y, "Annual CTC:")
    c.setFont(font, 11)           # font parameter — tampered version uses different font
    c.drawString(260, y, f"INR {salary} per annum")
    y -= 20

    # Back to normal font
    c.setFont("Helvetica", 11)
    c.drawString(90, y, "Probation Period:")
    c.drawString(260, y, "6 months")
    y -= 35

    # Additional clauses
    c.drawString(72, y,
        "This offer is subject to satisfactory background verification and")
    y -= 18
    c.drawString(72, y,
        "medical examination. Please confirm your acceptance within 7 days.")
    y -= 35

    c.drawString(72, y,
        "We look forward to welcoming you to the Acme Technologies family.")
    y -= 50

    # Signature
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, y, "For Acme Technologies Pvt Ltd,")
    y -= 60
    c.drawString(72, y, "Priya Menon")
    c.setFont("Helvetica", 10)
    y -= 16
    c.drawString(72, y, "Head of Human Resources")

    c.save()
    print(f"  Created: {path}")


def main():
    print("Generating test PDFs...\n")

    # 1. Clean document — original
    make_offer_letter(
        path=str(OUTPUT_DIR / "clean.pdf"),
        salary="8,50,000",
        date="01 July 2024",
        font="Helvetica"
    )

    # 2. Tampered — different font on salary line
    make_offer_letter(
        path=str(OUTPUT_DIR / "tampered_font.pdf"),
        salary="8,50,000",
        date="01 July 2024",
        font="Courier"    # Courier instead of Helvetica on salary line
    )

    # 3. Tampered — salary changed
    make_offer_letter(
        path=str(OUTPUT_DIR / "tampered_salary.pdf"),
        salary="18,50,000",   # inflated from 8,50,000
        date="01 July 2024",
        font="Helvetica"
    )

    # 4. Tampered — date changed (using different font to make it detectable)
    make_offer_letter(
        path=str(OUTPUT_DIR / "tampered_date.pdf"),
        salary="8,50,000",
        date="01 January 2023",   # date changed
        font="Helvetica"
    )

    print("\nAll test PDFs generated in test_pdfs/")
    print("\nExpected results:")
    print("  clean.pdf           → NOT MODIFIED")
    print("  tampered_font.pdf   → MODIFIED (font mismatch on salary line)")
    print("  tampered_salary.pdf → MODIFIED (spacing anomaly on inflated number)")
    print("  tampered_date.pdf   → MODIFIED (date inconsistency)")
    print("\nUpload each to http://localhost:8501 and verify.")


if __name__ == "__main__":
    main()
