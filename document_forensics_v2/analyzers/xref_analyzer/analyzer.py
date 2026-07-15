import fitz  # PyMuPDF
import logging

# Try to import scipy for Spearman correlation
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .models import XrefAnomaly, XrefReport

logger = logging.getLogger("document_forensics")


class XrefAnalyzer:
    """Detects XREF stream ordering anomalies (Canva fingerprint)."""

    # Spearman correlation thresholds
    CORRELATION_STRONG_INVERSE_THRESHOLD = -0.4  # Strong Canva signature
    CORRELATION_WEAK_INVERSE_THRESHOLD = 0.0
    CORRELATION_RANDOM_THRESHOLD = 0.2

    # Score mapping
    XREF_STRONG_INVERSE_SCORE = 50
    XREF_WEAK_INVERSE_SCORE = 25
    XREF_RANDOM_SCORE = 10
    XREF_SCORE_CAP = 100

    # Producer/creator whitelist: known normal PDF creators
    PRODUCER_WHITELIST = [
        "reportlab",
        "itext",
        "libreoffice",
        "openoffice",
        "microsoft",
        "pdfkit",
        "ghostscript",
        "freepdf",
        "print to pdf",
        "macos",
        "preview",
        "cups",
    ]

    def analyze(self, pdf_path: str) -> XrefReport:
        """
        Analyze XREF stream ordering for signs of Canva export.
        
        Returns XrefReport with correlation scores and confidence.
        """
        if not HAS_SCIPY:
            return XrefReport(
                pages_analyzed=0,
                xref_anomalies=[],
                xref_score=0,
                signals=["scipy not installed — XREF analysis skipped (pip install scipy)"],
            )

        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return XrefReport(
                pages_analyzed=0,
                xref_anomalies=[],
                xref_score=0,
                signals=["Could not open PDF for XREF analysis"],
            )

        # Check if producer is in whitelist
        if self._should_skip(doc):
            doc.close()
            return XrefReport(
                pages_analyzed=len(doc),
                xref_anomalies=[],
                xref_score=0,
                signals=["Document producer is in whitelist (known normal PDF creator)"],
            )

        anomalies = []
        all_correlations = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text objects with stream order
                text_objects = self._extract_text_objects_with_order(page)
                
                if len(text_objects) < 3:
                    continue  # Need at least 3 objects to compute meaningful correlation
                
                # Compute Spearman correlation
                correlation, p_value = self._compute_spearman_correlation(text_objects)
                all_correlations.append(correlation)
                
                # Determine anomaly severity based on correlation
                if correlation < self.CORRELATION_STRONG_INVERSE_THRESHOLD:
                    severity = "high"
                    score_contribution = self.XREF_STRONG_INVERSE_SCORE
                elif correlation < self.CORRELATION_WEAK_INVERSE_THRESHOLD:
                    severity = "medium"
                    score_contribution = self.XREF_WEAK_INVERSE_SCORE
                elif correlation < self.CORRELATION_RANDOM_THRESHOLD:
                    severity = "low"
                    score_contribution = self.XREF_RANDOM_SCORE
                else:
                    continue  # Normal correlation, no anomaly
                
                anomalies.append(XrefAnomaly(
                    page=page_num,
                    anomaly_type="xref_inversion",
                    correlation=round(correlation, 3),
                    severity=severity,
                    confidence=min(95, abs(correlation) * 100),  # Higher |correlation| = higher confidence
                    reason=f"XREF stream order inversely correlated with visual order (correlation={correlation:.3f}) — possible Canva export fingerprint",
                ))
        except Exception as e:
            logger.warning(f"XREF analysis failed: {e}")
        finally:
            doc.close()

        # Compute overall score
        score = 0
        signals = []

        if all_correlations:
            avg_correlation = sum(all_correlations) / len(all_correlations)
            
            if avg_correlation < self.CORRELATION_STRONG_INVERSE_THRESHOLD:
                score = self.XREF_STRONG_INVERSE_SCORE
                signals.append(f"Strong XREF inversion detected (avg correlation={avg_correlation:.3f}) — strong Canva fingerprint")
            elif avg_correlation < self.CORRELATION_WEAK_INVERSE_THRESHOLD:
                score = self.XREF_WEAK_INVERSE_SCORE
                signals.append(f"Weak XREF inversion detected (avg correlation={avg_correlation:.3f}) — possible Canva export")
            elif avg_correlation < self.CORRELATION_RANDOM_THRESHOLD:
                score = self.XREF_RANDOM_SCORE
                signals.append(f"XREF ordering appears near-random (avg correlation={avg_correlation:.3f}) — possible PDF manipulation")

        return XrefReport(
            pages_analyzed=len(anomalies) if anomalies else 0,
            xref_anomalies=anomalies,
            xref_score=min(self.XREF_SCORE_CAP, score),
            signals=signals,
        )

    def _should_skip(self, doc: fitz.Document) -> bool:
        """Check if document producer is in whitelist."""
        try:
            metadata = doc.metadata
            if not metadata:
                return False
            
            producer = str(metadata.get("producer", "")).lower()
            creator = str(metadata.get("creator", "")).lower()
            
            for keyword in self.PRODUCER_WHITELIST:
                if keyword.lower() in producer or keyword.lower() in creator:
                    return True
            
            return False
        except Exception:
            return False

    def _extract_text_objects_with_order(self, page: fitz.Page) -> list[dict]:
        """
        Extract all text objects from page with their stream order.
        
        Returns list of dicts: {"text": str, "y_pos": float, "stream_order": int}
        
        PyMuPDF's get_texttrace() returns objects in their appearance order in
        the PDF's content stream (creation order), which is the "stream order".
        Visual order is determined by y-position (top-to-bottom).
        """
        text_objects = []
        
        try:
            # Get text trace (objects in stream order)
            trace = page.get_texttrace()
            
            if not trace:
                return []
            
            for stream_order, trace_item in enumerate(trace):
                # PyMuPDF's get_texttrace() returns dictionaries, but just in case, handle dict or other types
                if not isinstance(trace_item, dict):
                    continue
                # Retrieve the text from characters
                chars = trace_item.get("chars", [])
                if not chars:
                    # In some PyMuPDF versions it might be in different fields or we have to construct it
                    # Let's extract text by visual matching or check if we have text field
                    # If get_texttrace is dictionary, it might represent a run/span.
                    # We can use the characters in 'chars' to get text.
                    text = "".join(ch[1] if isinstance(ch, tuple) and len(ch) > 1 else str(ch) for ch in chars)
                else:
                    text = "".join(ch[1] if isinstance(ch, tuple) and len(ch) > 1 else str(ch) for ch in chars)
                
                if not text or len(text.strip()) < 1:
                    continue
                
                # Get bounding box for this text
                # PyMuPDF doesn't directly give y-position from trace,
                # so we extract from rawdict and match by text
                rawdict = page.get_text("rawdict")
                y_pos = self._find_y_position_for_text(text, rawdict, page)
                
                if y_pos is not None:
                    text_objects.append({
                        "text": text,
                        "y_pos": y_pos,
                        "stream_order": stream_order,
                    })
        except Exception as e:
            logger.debug(f"Failed to extract text trace: {e}")
            return []
        
        return text_objects

    def _find_y_position_for_text(self, target_text: str, rawdict: dict, page: fitz.Page) -> float:
        """Find y-position of target text in rawdict."""
        try:
            for block in rawdict.get("blocks", []):
                if block.get("type") != 0:  # text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = "".join(ch.get("c", "") for ch in span.get("chars", []))
                        if target_text.lower() in text.lower():
                            # Return top y-position of this span
                            bbox = span.get("bbox", (0, 0, 0, 0))
                            return bbox[1]  # y0
        except Exception:
            pass
        
        return None

    def _compute_spearman_correlation(self, text_objects: list[dict]) -> tuple[float, float]:
        """
        Compute Spearman rank correlation between stream order and visual order.
        
        Returns (correlation_coefficient, p_value)
        
        - correlation ≈ 1.0: Objects appear in order (normal)
        - correlation ≈ 0.0: Objects appear randomly
        - correlation < -0.4: Objects appear in reverse order (Canva signature)
        """
        if len(text_objects) < 3:
            return (0.0, 1.0)
        
        # Sort by y-position to get visual order
        sorted_by_y = sorted(text_objects, key=lambda x: x["y_pos"])
        visual_ranks = list(range(len(sorted_by_y)))
        
        # Get stream orders
        stream_orders = [text_objects.index(obj) for obj in text_objects]
        
        # Compute Spearman correlation
        try:
            correlation, p_value = stats.spearmanr(stream_orders, visual_ranks)
            return (float(correlation), float(p_value))
        except Exception:
            return (0.0, 1.0)
