// Layers tab: per-layer finding lists (content, numeric, text stacking).

export default function LayersTab({ result }) {
  const suspiciousList = result?.suspicious_lines;
  const numericList = result?.numeric_anomalies;
  const textStackingList = result?.text_stacking_findings ?? [];
  return (
            <div className="tab-content">
              {/* Suspicious lines */}
              {suspiciousList?.length > 0 && (
                <div className="section">
                  <div className="section-title">
                    🔴 Suspicious Lines — Content Layer
                    <span className="badge">{suspiciousList.length}</span>
                  </div>
                  {suspiciousList.map((sl, i) => (
                    <div key={i} className="finding-card red">
                      <div className="finding-header">
                        Page {sl.page} · Line {sl.line_num} ·{" "}
                        <span style={{ color: "#ff4444" }}>
                          {sl.anomaly_score_pct}% anomaly
                        </span>
                      </div>
                      <div className="finding-text">"{sl.text}"</div>
                      {sl.reasons.map((r, j) => (
                        <div key={j} className="finding-reason">→ {r}</div>
                      ))}
                    </div>
                  ))}
                </div>
              )}

              {/* Numeric anomalies */}
              {numericList?.length > 0 && (
                <div className="section">
                  <div className="section-title">
                    🟡 Numeric Anomalies — Numeric Layer
                    <span className="badge">{numericList.length}</span>
                  </div>
                  {numericList.map((na, i) => (
                    <div key={i} className="finding-card yellow">
                      <div className="finding-header">
                        Page {na.page} · Line {na.line_num} ·{" "}
                        <span style={{ color: "#ffdd00" }}>
                          z-score: {na.z_score}
                        </span>
                      </div>
                      <div className="finding-text">"{na.text}"</div>
                      <div className="finding-reason">
                        Value: {na.value.toLocaleString()} · {na.reason}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Text-stacking findings — 2+ different text values at the same
                  coordinates (new text placed over original without removing
                  it). Shows BOTH/all colliding values per location, not just
                  the annotated box. */}
              {textStackingList.length > 0 && (
                <div className="section">
                  <div className="section-title">
                    🟪 Hidden Text Found — Text Stacking
                    <span className="badge">{textStackingList.length}</span>
                  </div>
                  {textStackingList.map((ts, i) => (
                    <div key={i} className="finding-card"
                      style={{ borderLeft: "3px solid #ff00ff" }}>
                      <div className="finding-header">
                        Page {ts.page} ·{" "}
                        <span style={{ color: "#ff00ff" }}>
                          {ts.confidence} · {Math.round(ts.overlap_fraction * 100)}% overlap
                        </span>
                      </div>
                      <div className="finding-text">
                        {ts.texts.map((t, j) => (
                          <span key={j}>
                            {j > 0 && <span style={{ color: "#888" }}> vs </span>}
                            <span style={{ color: "#fff" }}>"{t}"</span>
                          </span>
                        ))}
                      </div>
                      <div className="finding-reason">
                        → Two or more different text runs occupy the same
                        location — new text placed over the original without
                        removing it.
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {suspiciousList?.length === 0 &&
               numericList?.length === 0 &&
               textStackingList.length === 0 && (
                <div className="empty-state">
                  ✅ No specific line-level anomalies detected
                </div>
              )}
            </div>
  );
}
