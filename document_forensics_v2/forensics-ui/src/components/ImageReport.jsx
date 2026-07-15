// Image-document pipeline report (POST /analyze-image) — a SEPARATE
// upload path from the PDF pipeline; renders its own response shape.

import { useState } from "react";

import { API, IMAGE_CHECK_STYLE, COMPRESSION_STYLE } from "../lib/constants";

function ImageEvidencePanel({ title, src, note }) {
  const [failed, setFailed] = useState(false);
  return (
    <div style={{ flex: "1 1 340px", minWidth: 280 }}>
      <div className="section-title" style={{ marginBottom: 8 }}>{title}</div>
      {failed ? (
        <div className="empty-state">Image not available for this analysis</div>
      ) : (
        <img
          src={src}
          alt={title}
          onError={() => setFailed(true)}
          style={{ width: "100%", borderRadius: 8, border: "1px solid #333" }}
        />
      )}
      {note && !failed && (
        <div style={{ fontSize: "0.72rem", color: "#8890a0", marginTop: 6 }}>{note}</div>
      )}
    </div>
  );
}

function ImageReport({ data }) {
  const forensics = data.image_forensics || {};
  const anomalies = forensics.anomalies || [];
  const notImplemented = forensics.not_implemented || [];
  const fused = data.fused_findings || [];
  const compression = COMPRESSION_STYLE[forensics.compression_history]
    || { text: forensics.compression_history, color: "#888", checked: true };

  const verdictColor = data.verdict === "MODIFIED" ? "#ff4444"
                     : data.verdict === "ORIGINAL" ? "#00cc66"
                     : "#ff9800";
  const verdictIcon = data.verdict === "MODIFIED" ? "⚠️"
                    : data.verdict === "ORIGINAL" ? "✅"
                    : "❓";

  const flagBadge = (on, onLabel, offLabel) => (
    <span style={{
      fontSize: "0.72rem", fontWeight: 700, padding: "3px 10px", borderRadius: 12,
      border: `1px solid ${on ? "#4488ff" : "#333"}`,
      color: on ? "#88bbff" : "#666", whiteSpace: "nowrap",
    }}>
      {on ? onLabel : offLabel}
    </span>
  );

  return (
    <div className="results">
      {/* Verdict banner — same visual language as the PDF report, with an
          explicit pipeline badge so the source is never ambiguous. */}
      <div className="verdict-banner" style={{ borderColor: verdictColor }}>
        <div className="verdict-main">
          <span className="verdict-icon">{verdictIcon}</span>
          <span className="verdict-text" style={{ color: verdictColor }}>{data.verdict}</span>
          <span style={{
            marginLeft: 10, fontSize: "0.7rem", fontWeight: 700,
            color: "#7dd3fc", border: "1px solid #0369a1",
            borderRadius: 12, padding: "2px 10px", whiteSpace: "nowrap",
          }}>
            🖼️ Image Forensics
          </span>
        </div>
        <div className="verdict-stats">
          <div className="stat-item">
            <div className="stat-label">Combined Score</div>
            <div className="stat-value">{Number(data.combined_score).toFixed(1)}/100</div>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <div className="stat-label">Certainty</div>
            <div className="stat-value">{data.confidence}%</div>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <div className="stat-label">Threshold</div>
            <div className="stat-value">{data.effective_threshold}</div>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <div className="stat-label">Class</div>
            <div className="stat-value">image document</div>
          </div>
        </div>
      </div>

      {/* Detection flags + compression-history info row */}
      <div className="section" style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
        {flagBadge(forensics.stamp_detected, "🖃 Stamp detected", "🖃 No stamp found")}
        {flagBadge(forensics.signature_detected, "✍️ Signature detected", "✍️ No signature found")}
        {flagBadge(forensics.is_born_digital, "💻 Born-digital image", "📷 Camera/scan image")}
        <span style={{
          fontSize: "0.75rem", padding: "3px 10px", borderRadius: 12,
          border: compression.checked ? `1px solid ${compression.color}` : "1px dashed #555",
          color: compression.color, fontStyle: compression.checked ? "normal" : "italic",
        }}>
          🗜️ {compression.text}
        </span>
        <span style={{ fontSize: "0.72rem", color: "#8890a0" }}>
          JPEG history: {forensics.jpeg_history_detected ? "detected" : "none found"}
        </span>
      </div>

      {/* Cross-validated fused findings, when 2+ checks agree on a region */}
      {fused.length > 0 && (
        <div className="fusion-section">
          <div className="fusion-title">
            🎯 Cross-Validated Findings (2+ checks agree)
            <span className="badge">{fused.length}</span>
          </div>
          {fused.map((f, i) => (
            <div key={i} className="finding-card red">
              <div className="finding-header">
                {f.confidence} · score {f.score} · {f.confirming_layers.join(" + ")}
              </div>
              <div className="finding-reason">{f.description}</div>
            </div>
          ))}
        </div>
      )}

      {/* Anomaly cards */}
      <div className="section">
        <div className="section-title">
          🔎 Detected Anomalies
          <span className="badge">{anomalies.length}</span>
        </div>
        {anomalies.length === 0 ? (
          <div className="empty-state">
            ✅ No tampering anomalies detected by the implemented checks
            (see the coverage note below for techniques this engine does not run)
          </div>
        ) : (
          anomalies.map((a, i) => {
            const style = IMAGE_CHECK_STYLE[a.evidence_check] || { color: "#ff4444", label: a.evidence_check };
            return (
              <div key={i} className="finding-card" style={{ borderLeft: `3px solid ${style.color}` }}>
                <div className="finding-header">
                  <span style={{ color: style.color }}>{style.label}</span>
                  {" · "}{a.type}{" · "}
                  confidence {(a.confidence * 100).toFixed(0)}%
                </div>
                <div className="finding-reason">
                  → {a.detail}
                </div>
                <div style={{ fontSize: "0.7rem", color: "#8890a0", marginTop: 4 }}>
                  {a.evidence_check} · region x={a.bbox[0]}, y={a.bbox[1]}, {a.bbox[2]}×{a.bbox[3]}px
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Evidence images */}
      <div className="section">
        <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
          <ImageEvidencePanel
            title="📍 Annotated Image"
            src={`${API}${data.annotated_url}`}
            note="Colored boxes mark each check's flagged regions (red: inpaint · orange: overlay edges · magenta: clone · blue: flat ink · yellow: cutout boundary)."
          />
          <ImageEvidencePanel
            title="🌡️ Near-White Heatmap"
            src={`${API}${data.heatmap_url}`}
            note="Display-only evidence for a human reviewer (Check 10) — never a scoring input."
          />
        </div>
      </div>

      {/* Engine signals */}
      {(data.signals || []).length > 0 && (
        <div className="section">
          <div className="section-title">⚡ Signals</div>
          {data.signals.map((s, i) => (
            <div key={i} className="finding-reason" style={{ padding: "3px 0" }}>{s}</div>
          ))}
        </div>
      )}

      {/* Coverage disclosure — collapsed by default, informational styling
          (NOT an error state). An empty anomalies list is not a clean bill
          of health on these techniques: the engine deliberately does not
          attempt them because they can't produce a reliable confidence
          number from a single image, and says why for each. */}
      {notImplemented.length > 0 && (
        <details className="section" style={{ border: "1px solid #333", borderRadius: 8, padding: "10px 14px" }}>
          <summary style={{ cursor: "pointer", fontWeight: 600, color: "#8890a0" }}>
            ℹ️ Coverage note — {notImplemented.length} advanced technique{notImplemented.length === 1 ? "" : "s"} intentionally
            skipped (expand for details)
          </summary>
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: "0.75rem", color: "#8890a0", marginBottom: 12 }}>
              This is informational, not an error. These techniques cannot
              produce a trustworthy confidence number from a single uploaded
              image (they need reference data this upload doesn't carry), so
              the engine reports them as not attempted rather than inventing
              a score. Every implemented check above ran normally.
            </div>
            {notImplemented.map((n, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, fontSize: "0.8rem", color: "#d0d0d0" }}>
                  {String(n.technique || "").replace(/_/g, " ")}
                </div>
                <div style={{ fontSize: "0.75rem", color: "#8890a0" }}>{n.reason}</div>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

export default ImageReport;
