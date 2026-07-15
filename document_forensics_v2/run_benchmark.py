"""
Document Forensics Engine - Accuracy Benchmarking Pipeline
This script dynamically generates 100 unique, realistic PDFs (50 clean, 50 tampered)
in memory, analyzes them using the engine's endpoint via TestClient, and reports
performance metrics (Accuracy, Precision, Recall, Confusion Matrix).

Run:
  ..\.venv\Scripts\python.exe run_benchmark.py
"""

import os
import sys
import random
from collections import Counter
from fastapi.testclient import TestClient
import fitz

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import app

# Candidate names to generate diverse PDFs
NAMES = [
    "Rahul Sharma", "Amit Patel", "Priya Singh", "Rohan Mehta", "Siddharth Rao",
    "Sneha Reddy", "Vikram Malhotra", "Ananya Iyer", "Karan Johar", "Deepika Padukone",
    "Aarav Mehta", "Ishita Gupta", "Kabir Kapoor", "Meera Nair", "Aditya Sen",
    "Neha Verma", "Arjun Reddy", "Riya Sen", "Devendra Fadnavis", "Pranab Mukherjee"
]

def generate_pdf_in_memory(is_clean=True, tamper_type=None) -> bytes:
    """
    Generates a realistic multi-line offer letter in-memory.
    If is_clean=True:
        - Uniform Helvetica font, matching dates, clean metadata.
    If is_clean=False:
        - Out-of-sequence metadata (modified 1 day later, conflicting creator/producer).
        - Applied white overlay rect and typed over in Courier font depending on tamper_type.
    """
    name = random.choice(NAMES)
    ref_id = random.randint(1000, 9999)
    base_salary = random.randint(6, 18) * 100000  # e.g., INR 6,00,000 to INR 18,00,000
    salary_str = f"{base_salary:,}"

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    # Set metadata
    if is_clean:
        doc.set_metadata({
            "creator": "anonymous",
            "producer": "ReportLab PDF Library - (opensource)",
            "creationDate": "D:20260715120000+05'30'",
            "modDate": "D:20260715120000+05'30'"
        })
    else:
        # Conflicting generators + 1-day modification gap
        doc.set_metadata({
            "creator": "Microsoft Office Word",
            "producer": "ReportLab PDF Library - (opensource)",
            "creationDate": "D:20260715120000+05'30'",
            "modDate": "D:20260716120000+05'30'"
        })

    # Base design elements and offer clauses to establish dominant font (Helvetica)
    lines = [
        ('ACME TECHNOLOGIES PVT LTD', 72, 80, 'hebo', 14),
        ('123 Business Park, Andheri East, Mumbai - 400069', 72, 100, 'helv', 10),
        ('Tel: +91-22-12345678 | Email: hr@acmetech.com', 72, 115, 'helv', 10),
        (f'Ref: ACME/HR/2026/OL/{ref_id}', 72, 150, 'helv', 11),
        ('To,', 72, 180, 'hebo', 11),
        (f'Mr. {name}', 72, 198, 'helv', 11),
        ('45, Green Park Colony', 72, 216, 'helv', 11),
        ('Pune, Maharashtra - 411001', 72, 234, 'helv', 11),
        ('Subject: Offer of Employment', 72, 260, 'hebo', 12),
        (f'Dear Mr. {name},', 72, 290, 'helv', 11),
        ('We are pleased to offer you the position of Senior Software Engineer', 72, 315, 'helv', 11),
        ('at Acme Technologies Pvt Ltd, subject to the following terms:', 72, 333, 'helv', 11),
        ('Designation: Senior Software Engineer', 90, 360, 'hebo', 11),
        ('Probation Period: 6 months', 90, 420, 'helv', 11),
        ('This offer is subject to satisfactory background verification.', 72, 450, 'helv', 11),
        ('We look forward to welcoming you to the Acme Technologies family.', 72, 470, 'helv', 11),
        ('For Acme Technologies Pvt Ltd,', 72, 510, 'hebo', 11),
        ('Priya Menon', 72, 560, 'hebo', 11),
        ('Head of Human Resources', 72, 575, 'helv', 10)
    ]

    # Standard clean fields
    date_text = 'Date of Birth: 15 July 2026'
    ctc_text = f'Annual CTC: INR {salary_str} per annum'

    lines.append((date_text, 72, 135, 'helv', 11))
    lines.append((ctc_text, 90, 400, 'hebo', 11))

    # Render base text
    for text, x, y, font, size in lines:
        page.insert_text(fitz.Point(x, y), text, fontname=font, fontsize=size, color=(0, 0, 0))

    if not is_clean:
        if not tamper_type:
            tamper_type = random.choice(["font", "salary", "date"])

        if tamper_type == "font":
            # Font swap: overwrite both Date and CTC in Courier to ensure clear font mismatch
            # Cover Date of Birth line
            page.draw_rect(fitz.Rect(70, 123, 250, 140), color=(1, 1, 1), fill=(1, 1, 1))
            page.insert_text(fitz.Point(72, 135), 'Date of Birth: 15 July 2026', fontname='cour', fontsize=11, color=(0.05, 0.05, 0.1))
            # Cover Annual CTC line
            page.draw_rect(fitz.Rect(85, 388, 350, 405), color=(1, 1, 1), fill=(1, 1, 1))
            page.insert_text(fitz.Point(90, 400), f'Annual CTC: INR {salary_str} per annum', fontname='cour', fontsize=11, color=(0.05, 0.05, 0.1))

        elif tamper_type == "salary":
            # Inflated salary: overwrite CTC with 10x value in Courier (triggers font and numeric outlier check)
            inflated_salary = base_salary * 10
            inf_salary_str = f"{inflated_salary:,}"
            page.draw_rect(fitz.Rect(85, 388, 350, 405), color=(1, 1, 1), fill=(1, 1, 1))
            page.insert_text(fitz.Point(90, 400), f'Annual CTC: INR {inf_salary_str} per annum', fontname='cour', fontsize=11, color=(0.05, 0.05, 0.1))
            # Cover Date to add stacking/overlay signals
            page.draw_rect(fitz.Rect(70, 123, 250, 140), color=(1, 1, 1), fill=(1, 1, 1))
            page.insert_text(fitz.Point(72, 135), 'Date of Birth: 15 July 2026', fontname='cour', fontsize=11, color=(0.05, 0.05, 0.1))

        elif tamper_type == "date":
            # Altered date: overwrite Date of Birth with backdated year (2018) in Courier (triggers timeline anomaly)
            page.draw_rect(fitz.Rect(70, 123, 250, 140), color=(1, 1, 1), fill=(1, 1, 1))
            page.insert_text(fitz.Point(72, 135), 'Date of Birth: 15 July 2018', fontname='cour', fontsize=11, color=(0.05, 0.05, 0.1))
            # Cover Annual CTC to add stacking/overlay signals
            page.draw_rect(fitz.Rect(85, 388, 350, 405), color=(1, 1, 1), fill=(1, 1, 1))
            page.insert_text(fitz.Point(90, 400), f'Annual CTC: INR {salary_str} per annum', fontname='cour', fontsize=11, color=(0.05, 0.05, 0.1))

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def main():
    print("=" * 60)
    print("      DOCUMENT FORENSICS ACCURACY BENCHMARK PIPELINE      ")
    print("=" * 60)

    client = TestClient(app)

    # Initialize statistics
    tp, tn, fp, fn = 0, 0, 0, 0
    false_positives = []
    false_negatives = []

    # Generate 50 Clean and 50 Tampered PDFs
    print("\n[+] Step 1: Generating synthetic test suite in-memory...")
    clean_pdfs = [(f"clean_{i+1}.pdf", generate_pdf_in_memory(is_clean=True)) for i in range(50)]
    
    # 50 Tampered PDFs with random or balanced tamper types
    tamper_types = ["font", "salary", "date"]
    tampered_pdfs = []
    for i in range(50):
        t_type = tamper_types[i % len(tamper_types)]
        tampered_pdfs.append((f"tampered_{i+1}_{t_type}.pdf", generate_pdf_in_memory(is_clean=False, tamper_type=t_type)))
        
    print(f"    - Generated 50 ORIGINAL (clean) documents")
    print(f"    - Generated 50 MODIFIED (tampered) documents (balanced types)")

    # Analyze Clean PDFs
    print("\n[+] Step 2: Running Clean PDFs through forensics engine...")
    for filename, pdf_bytes in clean_pdfs:
        res = client.post('/analyze', files={'file': (filename, pdf_bytes, 'application/pdf')})
        if res.status_code != 200:
            print(f"    [Error] Endpoint returned HTTP {res.status_code} for {filename}")
            continue
            
        verdict = res.json().get("verdict", "UNKNOWN")
        # Clean should be ORIGINAL. If MODIFIED or UNCERTAIN, it's a False Positive (FP)
        if verdict == "ORIGINAL":
            tn += 1
        else:
            fp += 1
            false_positives.append(filename)
            
        # Print progress dot
        sys.stdout.write(".")
        sys.stdout.flush()

    # Analyze Tampered PDFs
    print("\n\n[+] Step 3: Running Tampered PDFs through forensics engine...")
    for filename, pdf_bytes in tampered_pdfs:
        res = client.post('/analyze', files={'file': (filename, pdf_bytes, 'application/pdf')})
        if res.status_code != 200:
            print(f"    [Error] Endpoint returned HTTP {res.status_code} for {filename}")
            continue
            
        verdict = res.json().get("verdict", "UNKNOWN")
        # Tampered should be flagged as MODIFIED or UNCERTAIN. If ORIGINAL, it's a False Negative (FN)
        if verdict in ["MODIFIED", "UNCERTAIN"]:
            tp += 1
        else:
            fn += 1
            false_negatives.append(filename)

        # Print progress dot
        sys.stdout.write(".")
        sys.stdout.flush()

    print("\n\n" + "=" * 60)
    print("                     BENCHMARK RESULTS                      ")
    print("=" * 60)

    # Metrics calculations
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    print(f"Total Analyzed  : {total}")
    print(f"True Positives  : {tp:3d}  (Fake -> Flagged)")
    print(f"True Negatives  : {tn:3d}  (Clean -> Original)")
    print(f"False Positives : {fp:3d}  (Clean -> Flagged)")
    print(f"False Negatives : {fn:3d}  (Fake -> Original)")
    print("-" * 60)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Precision       : {precision:.2f}%")
    print(f"Recall (TPR)    : {recall:.2f}%")
    print("=" * 60)

    if false_positives:
        print("\n[!] False Positives (Clean documents flagged as Tampered/Uncertain):")
        for fn_name in false_positives:
            print(f"    - {fn_name}")
    else:
        print("\n[✓] Zero False Positives detected.")

    if false_negatives:
        print("\n[!] False Negatives (Tampered documents flagged as Original):")
        for fn_name in false_negatives:
            print(f"    - {fn_name}")
    else:
        print("[✓] Zero False Negatives detected.")
    print("=" * 60)

if __name__ == "__main__":
    main()
