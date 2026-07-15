"""
Metadata Extractor — Document Forensics Engine
Extracts all metadata from any PDF and identifies its origin.
"""

from .database import PRODUCER_DB, _DB_PATH, _identify_source, _parse_pdf_date
from .models import MetadataReport, SourceInfo
from .extractor import MetadataExtractor

__all__ = [
    "PRODUCER_DB", "MetadataReport", "SourceInfo", "MetadataExtractor",
]
