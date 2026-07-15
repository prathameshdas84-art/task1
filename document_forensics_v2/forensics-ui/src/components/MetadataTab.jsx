// Metadata tab: edit timeline, document structure, authenticity score,
// suspicious content, date analysis, and raw PDF metadata.

export default function MetadataTab({ result }) {
  return (
            <div className="tab-content">
              <div className="edit-timeline">
                <div className="timeline-title">📅 Document Timeline</div>
                <div className="timeline-row">
                  <span className="timeline-icon">📄</span>
                  <span className="timeline-label">Created:</span>
                  <span className="timeline-value">{result.metadata?.created}</span>
                </div>
                <div className="timeline-arrow">↓</div>
                <div className="timeline-row">
                  <span className="timeline-icon">✏️</span>
                  <span className="timeline-label">Modified:</span>
                  <span className="timeline-value" style={{
                    color: result.metadata?.is_very_recent_edit ? "#ff4444" : "#e0e0e0"
                  }}>
                    {result.metadata?.modified}
                    {result.metadata?.edit_age_human &&
                      <span style={{marginLeft:8, color:"#888", fontSize:"0.85em"}}>
                        ({result.metadata.edit_age_human})
                      </span>
                    }
                  </span>
                </div>
                <div className="timeline-interval">
                  Time between create & modify:{" "}
                  {result.metadata?.modification_interval || "0 seconds"}
                </div>
              </div>

              <div className="meta-grid">
                {[
                  ["Producer", result.metadata.producer],
                  ["Creator", result.metadata.creator],
                  ["Author", result.metadata.author],
                  ["Title", result.metadata.title],
                  ["Created", result.metadata.created],
                  ["Modified", result.metadata.modified],
                  ["Was Modified", result.metadata.was_modified ? "YES ⚠️" : "No"],
                  ["Modification Interval", result.metadata.modification_interval],
                  ["XMP Mismatch", result.metadata.xmp_mismatch ? "YES ⚠️" : "No"],
                  ["Multiple Producers", result.metadata.multiple_producers ? "YES ⚠️" : "No"],
                  ["Source Risk", result.metadata.source_risk],
                  ["PDF Version", result.metadata.pdf_version],
                  ["Total Pages", result.metadata.total_pages],
                  ["Encrypted", result.metadata.is_encrypted ? "YES" : "No"],
                  ["Has JavaScript", result.metadata.has_javascript ? "YES ⚠️" : "No"],
                  ["Has Embedded Files", result.metadata.has_embedded_files ? "YES ⚠️" : "No"],
                  ["Font Count", result.metadata.font_count],
                  ["Has Images", result.metadata.has_images ? "Yes" : "No"],
                  ["Has ICC Profile", result.metadata.has_icc_profiles ? "Yes" : "No"],
                  ["Page Rotation Consistent",
                    result.metadata.page_rotation?.consistent === false ? "NO ⚠️" : "Yes"],
                ].map(([label, value]) => value != null && (
                  <div key={label} className="meta-row">
                    <div className="meta-label">{label}</div>
                    <div className="meta-value"
                      style={{ color: String(value).includes("⚠️") ? "#ff8800" : "#e0e0e0" }}>
                      {String(value)}
                    </div>
                  </div>
                ))}
              </div>

              {/* ── Document Structure ── */}
              {result.metadata.structure && (
                <details className="meta-section" open>
                  <summary>📑 Document Structure</summary>
                  <div className="meta-grid">
                    {[
                      ["Content Type", result.metadata.structure.content_type],
                      ["Total Pages", result.metadata.structure.total_pages],
                      ["Estimated Word Count", result.metadata.structure.estimated_word_count],
                      ["Total Text Length", result.metadata.structure.total_text_length],
                      ["Avg Text / Page", result.metadata.structure.avg_text_per_page],
                      ["Has Text Content", result.metadata.structure.has_text_content ? "Yes" : "No"],
                    ].map(([label, value]) => value != null && (
                      <div key={label} className="meta-row">
                        <div className="meta-label">{label}</div>
                        <div className="meta-value">{String(value)}</div>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {/* ── Authenticity Score ── */}
              {result.metadata.authenticity && (
                <details className="meta-section" open>
                  <summary>🛡️ Authenticity Score</summary>
                  <div className="auth-score-box">
                    {(() => {
                      const a = result.metadata.authenticity;
                      const c = a.score >= 80 ? "#00cc66"
                              : a.score >= 50 ? "#ffaa00" : "#ff4444";
                      return (
                        <>
                          <div className="auth-score-num" style={{ color: c }}>
                            {a.score}<span className="auth-score-max">/100</span>
                          </div>
                          <div className="auth-score-meta">
                            <div className="auth-assessment" style={{ color: c }}>
                              {String(a.assessment || "").replace(/_/g, " ")}
                            </div>
                            <div className="auth-confidence">
                              Confidence: {a.confidence}
                            </div>
                            {a.issues?.length > 0 ? (
                              <ul className="auth-issues">
                                {a.issues.map((iss, i) => (
                                  <li key={i}>⚠️ {iss}</li>
                                ))}
                              </ul>
                            ) : (
                              <div className="auth-clean">✅ No authenticity issues detected</div>
                            )}
                          </div>
                        </>
                      );
                    })()}
                  </div>
                </details>
              )}

              {/* ── Suspicious Content ── */}
              {result.metadata.suspicious_content && (
                <details className="meta-section">
                  <summary>
                    ⚠️ Suspicious Content
                    {result.metadata.suspicious_content.risk_score > 0 && (
                      <span className="badge" style={{ background: "#ff4444" }}>
                        risk {result.metadata.suspicious_content.risk_score}
                      </span>
                    )}
                  </summary>
                  <div className="meta-grid">
                    {[
                      ["JavaScript", result.metadata.suspicious_content.has_javascript],
                      ["Open Actions", result.metadata.suspicious_content.has_open_actions],
                      ["Launch Actions", result.metadata.suspicious_content.has_launch_actions],
                      ["Embedded Files", result.metadata.suspicious_content.has_embedded_files],
                    ].map(([label, value]) => (
                      <div key={label} className="meta-row">
                        <div className="meta-label">{label}</div>
                        <div className="meta-value"
                          style={{ color: value ? "#ff8800" : "#e0e0e0" }}>
                          {value ? "YES ⚠️" : "No"}
                        </div>
                      </div>
                    ))}
                  </div>
                  {result.metadata.suspicious_content.findings?.length > 0 && (
                    <ul className="auth-issues">
                      {result.metadata.suspicious_content.findings.map((f, i) => (
                        <li key={i}>→ {f}</li>
                      ))}
                    </ul>
                  )}
                  {result.metadata.js_context && result.metadata.js_context !== "none" && (
                    <div style={{ marginTop: 8, fontSize: "0.85rem" }}>
                      <span style={{
                        color:
                          result.metadata.js_context === "names_tree" ? "#ff4444" :
                          result.metadata.js_context === "open_action" ? "#ff8800" :
                          "#ffdd00"
                      }}>
                        ⚡ JavaScript Context: {result.metadata.js_context}
                      </span>
                      <div style={{ color: "#888", fontSize: "0.8rem", marginTop: 2 }}>
                        {result.metadata.js_context === "names_tree" &&
                          "Document-level JS — runs automatically — HIGH risk"}
                        {result.metadata.js_context === "open_action" &&
                          "Executes on open — MEDIUM risk"}
                        {result.metadata.js_context === "page_level" &&
                          "Page/form level — LOW risk (likely form-field scripting)"}
                      </div>
                    </div>
                  )}
                </details>
              )}

              {/* ── Date Analysis ── */}
              {result.metadata.dates && (
                <details className="meta-section">
                  <summary>📅 Date Analysis</summary>
                  {["created", "modified"].map((k) => {
                    const d = result.metadata.dates[k];
                    if (!d) return (
                      <div key={k} className="meta-row">
                        <div className="meta-label">{k}</div>
                        <div className="meta-value">Unknown</div>
                      </div>
                    );
                    return (
                      <div key={k} className="date-block">
                        <div className="date-block-title">
                          {k === "created" ? "📄 Created" : "✏️ Modified"}
                        </div>
                        <div className="meta-grid">
                          {[
                            ["Human", d.human],
                            ["Relative", d.relative],
                            ["ISO 8601", d.iso8601],
                            ["Age (days)", d.age_days],
                            ["Timezone", d.timezone],
                          ].map(([label, value]) => value != null && (
                            <div key={label} className="meta-row">
                              <div className="meta-label">{label}</div>
                              <div className="meta-value">{String(value)}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                  <div className="meta-row">
                    <div className="meta-label">Was Modified</div>
                    <div className="meta-value" style={{
                      color: result.metadata.dates.was_modified ? "#ff8800" : "#e0e0e0"
                    }}>
                      {result.metadata.dates.was_modified ? "YES ⚠️" : "No"}
                    </div>
                  </div>
                </details>
              )}

              {/* ── Raw PDF Metadata ── */}
              {result.metadata.raw && Object.keys(result.metadata.raw).length > 0 && (
                <details className="meta-section">
                  <summary>
                    🗂️ Raw PDF Metadata
                    <span className="badge">{Object.keys(result.metadata.raw).length}</span>
                  </summary>
                  <div className="meta-grid">
                    {Object.entries(result.metadata.raw).map(([key, value]) => (
                      <div key={key} className="meta-row">
                        <div className="meta-label">{key}</div>
                        <div className="meta-value" style={{ wordBreak: "break-all" }}>
                          {String(value)}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {result.metadata.fonts?.length > 0 && (
                <div className="section">
                  <div className="section-title">
                    Fonts ({result.metadata.font_count})
                  </div>
                  {result.metadata.fonts.map((f, i) => (
                    <div key={i} className="font-row">
                      <span className="font-name">{f.name}</span>
                      <span className="font-type">{f.type}</span>
                      <span className={`font-embedded ${f.embedded ? "yes" : "no"}`}>
                        {f.embedded ? "Embedded ✅" : "Not Embedded ⚠️"}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
  );
}
