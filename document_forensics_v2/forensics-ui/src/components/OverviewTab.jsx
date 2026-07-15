// Overview tab: cross-validated fused findings, fusion stats, and the
// summary narrative.

import { LAYER_LABELS } from "../lib/constants";

export default function OverviewTab({ result }) {
  const fusedList = result?.fused_findings;
  return (
            <div className="tab-content">
              {fusedList && fusedList.length > 0 && (
                <div className="fusion-section">
                  <div className="fusion-title">
                    🎯 Cross-Validated Findings (High Confidence)
                    <span className="badge">{fusedList.length}</span>
                  </div>
                  <div className="fusion-subtitle">
                    These regions were flagged by MULTIPLE independent layers —
                    most likely real tampering, not false positives.
                  </div>
                  {fusedList.map((f, i) => (
                    <div key={i} className="fusion-card" style={{
                      borderLeft: `4px solid ${
                        f.confidence === "HIGH" ? "#ff4444" :
                        f.confidence === "MEDIUM" ? "#ff9800" : "#ffdd44"
                      }`,
                    }}>
                      <div className="fusion-header">
                        <span className="fusion-page">Page {f.page}</span>
                        <span className="fusion-conf" style={{color:
                          f.confidence === "HIGH" ? "#ff4444" :
                          f.confidence === "MEDIUM" ? "#ff9800" : "#ffdd44"
                        }}>
                          {f.confidence} CONFIDENCE
                        </span>
                        {f.score != null && <span className="fusion-score">{f.score}/100</span>}
                      </div>
                      <div className="fusion-layers">
                        Confirmed by:{" "}
                        {f.confirming_layers.map(l => (
                          <span key={l} className="layer-badge">{l}</span>
                        ))}
                      </div>
                      <div className="fusion-desc">
                        {f.description}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {result.fusion_stats && (
                <div className="fusion-stats">
                  📊 Signal Fusion:{" "}
                  {result.fusion_stats.high_confidence_findings} high-confidence findings,{" "}
                  {result.fusion_stats.single_layer_suppressed} single-layer signals suppressed
                  (likely false positives)
                </div>
              )}

              <div className="overview-grid">
                {Object.entries(result.layers).map(([key, score]) => {
                  const layerPrefix = {
                    metadata: "[METADATA]",
                    content:  "[CONTENT]",
                    numeric:  "[NUMERIC]",
                    ela:      "[ELA]",
                    pymupdf:  "[PYMUPDF]",
                  }[key];

                  const layerSignals = result.signals
                    .filter(s => s.includes(layerPrefix))
                    .map(s => s.replace(layerPrefix, "").trim())
                    .slice(0, 3); // show max 3 signals per card

                  return (
                    <div key={key} className="layer-card"
                      style={{ borderTop: `3px solid ${
                        score >= 50 ? "#ff4444" :
                        score >= 20 ? "#ffaa00" : "#00cc66"
                      }`}}>
                      <div className="layer-name">{LAYER_LABELS[key] || key}</div>
                      <div className="layer-score-bar">
                        <div
                          className="layer-score-fill"
                          style={{
                            width: `${score}%`,
                            background: score >= 50 ? "#ff4444"
                              : score >= 20 ? "#ffaa00" : "#00cc66",
                          }}
                        />
                      </div>
                      <div style={{display:"flex", justifyContent:"space-between",
                                   alignItems:"center", marginTop:4}}>
                        <div className="layer-score-num">{score}/100</div>
                        <div className="layer-status">
                          {score >= 20 ? "⚠️ Anomaly" :
                           score >= 10 ? "⚡ Minor"  : "✅ Clean"}
                        </div>
                      </div>

                      {/* What this layer checks */}
                      <div style={{
                        fontSize:"0.75rem", color:"#666",
                        marginTop:6, paddingTop:6,
                        borderTop:"1px solid #222"
                      }}>
                        {{
                          metadata: "Checks: producer, creator, dates, XMP",
                          content:  "Checks: fonts, spacing, CIDFont sessions",
                          numeric:  "Checks: statistical outliers in numbers",
                          ela:      "Checks: compression artifact consistency",
                          pymupdf:  "Checks: hidden overlays, char spacing",
                        }[key]}
                      </div>

                      {/* Why it scored what it did */}
                      {layerSignals.length > 0 && (
                        <div style={{marginTop:6}}>
                          {layerSignals.map((sig, i) => {
                            const isNegative = !sig.includes("passed") &&
                              !sig.includes("skipped") &&
                              !sig.includes("consistent") &&
                              !sig.includes("No significant");
                            return (
                              <div key={i} style={{
                                fontSize:"0.72rem",
                                color: isNegative ? "#ffaa44" : "#666",
                                marginTop:3,
                                paddingLeft:6,
                                borderLeft: isNegative
                                  ? "2px solid #ffaa44"
                                  : "2px solid #333",
                                lineHeight:1.3,
                              }}>
                                {sig.length > 80 ? sig.slice(0,80)+"..." : sig}
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {layerSignals.length === 0 && score === 0 && (
                        <div style={{
                          fontSize:"0.72rem", color:"#444",
                          marginTop:6, fontStyle:"italic"
                        }}>
                          No anomalies detected
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              <div className="summary-box">
                <div className="summary-title">📝 Summary</div>
                <div className="summary-text">{result.summary}</div>
              </div>
            </div>
  );
}
