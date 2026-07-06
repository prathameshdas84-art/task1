"""
Document Forensics Engine — FastAPI Backend
Run: start.bat (uses ..\.venv — do not run with a global/system Python;
     PyMuPDF is only correctly installed in .venv and fails at import
     time otherwise, before the app object is even created)
Test: http://localhost:8000/docs
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ai_review_routes import router as ai_review_router
from api.system_routes import router as system_router
from api.analysis_routes import router as analysis_router

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document Forensics Engine",
    description="""
Detect tampering and modifications in PDF documents.

## How it works
Upload any PDF, image, or Word document and receive a detailed forensic analysis:
- **Layer 1 — Metadata**: Who created/modified the document and when
- **Layer 2 — Content**: Font consistency, spacing anomalies, CIDFont edit detection
- **Layer 3 — OCR**: Embedded vs visible text comparison, confidence analysis
- **Layer 4 — Numeric**: Statistical outlier detection in number fields
- **Layer 5 — ELA**: Error Level Analysis, shadow attack detection, signature validation

## Supported formats
PDF, JPG, JPEG, PNG, DOCX, DOC

## Verdict
- **MODIFIED**: Evidence of tampering detected
- **ORIGINAL**: No significant tampering evidence found
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_review_router)
app.include_router(system_router)
app.include_router(analysis_router)
