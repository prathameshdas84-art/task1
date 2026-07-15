from dataclasses import dataclass, field

@dataclass
class XrefAnomaly:
    page: int
    anomaly_type: str  # "xref_inversion"
    correlation: float  # Spearman correlation coefficient
    severity: str  # "high", "medium", "low"
    confidence: float
    reason: str


@dataclass
class XrefReport:
    pages_analyzed: int
    xref_anomalies: list[XrefAnomaly] = field(default_factory=list)
    xref_score: int = 0  # 0-100
    signals: list[str] = field(default_factory=list)
