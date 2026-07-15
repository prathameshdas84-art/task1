import { useState, useCallback, useMemo, useEffect } from "react";
import axios from "axios";
import "./App.css";

import { API } from "./lib/constants";
import ImageReport from "./components/ImageReport";
import HiddenTextPanel from "./components/HiddenTextPanel";
import OverviewTab from "./components/OverviewTab";
import LayersTab from "./components/LayersTab";
import SignalsTab from "./components/SignalsTab";
import MetadataTab from "./components/MetadataTab";
import { buildForensicReportPdf } from "./lib/pdfReport";

export default function App() {
  const [file, setFile]             = useState(null);
  const [dragging, setDragging]     = useState(false);
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState(null);
  const [error, setError]           = useState(null);
  const [activePage, setActivePage] = useState(1);
  const [activeTab, setActiveTab]   = useState("overview");
  const [zoom, setZoom]             = useState(100);
  const [reportGenerating, setReportGenerating] = useState(false);
  const [hiddenTextData, setHiddenTextData] = useState(null);
  // Location tab: annotated-image load failure + manual-retry counter.
  // A failed image load must never wipe the analysis result — the rest of
  // the report (overview, layers, JSON) is client-side state that is still
  // perfectly valid; only this one image fetch failed.
  const [imageError, setImageError] = useState(false);
  const [imageRetry, setImageRetry] = useState(0);

  // Image-document pipeline (POST /analyze-image) — a SEPARATE upload path,
  // not a replacement for /analyze. Its response shape differs from the PDF
  // pipeline's (confidence is a plain int, anomalies carry [x,y,w,h] boxes,
  // etc.), so the result lives in its own state var and renders through
  // ImageReport — never through the PDF report tree.
  const [uploadMode, setUploadMode]   = useState("pdf");   // "pdf" | "image"
  const [imageResult, setImageResult] = useState(null);

  // Mirrors the backend's own 400 rejections on /analyze-image so the user
  // reads the same message whether the check fires client- or server-side.
  const validateImageFile = (f) => {
    const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    if (ext === ".pdf") {
      return "PDFs (including scanned PDFs) are analyzed by the PDF Document tab — this path is only for direct JPG/PNG uploads.";
    }
    if (![".jpg", ".jpeg", ".png"].includes(ext)) {
      return `Unsupported file type '${ext}'. Supported here: .jpg, .jpeg, .png`;
    }
    return null;
  };

  const acceptFile = useCallback((f, mode) => {
    if (!f) return;
    if (mode === "image") {
      const problem = validateImageFile(f);
      if (problem) {
        setFile(null); setError(problem);
        return;
      }
    }
    setFile(f); setResult(null); setImageResult(null); setError(null);
    setHiddenTextData(null);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    acceptFile(e.dataTransfer.files[0], uploadMode);
  }, [acceptFile, uploadMode]);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setActivePage(1);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(`${API}/analyze`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(data);
      setActiveTab("overview");
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;
    const problem = validateImageFile(file);
    if (problem) { setError(problem); return; }
    setLoading(true);
    setImageResult(null);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(`${API}/analyze-image`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setImageResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Image analysis failed");
    } finally {
      setLoading(false);
    }
  };

  // analysis_id is a unique UUID per analysis and page is explicit, so the
  // URL itself is a stable cache key — Date.now() is not needed and would
  // cause the browser to re-fetch the image on every React render (zoom
  // change, tab switch, etc.), which also triggers stale-session 404s.
  const imageUrl = useMemo(
    () => result
      ? `${API}/annotated-image/${result.analysis_id}?page=${activePage}`
        + (imageRetry > 0 ? `&retry=${imageRetry}` : "")
      : null,
    [result, activePage, imageRetry]
  );

  // A new page/result/retry means a new fetch — clear any stale error.
  useEffect(() => { setImageError(false); }, [imageUrl]);

  // Hidden Text Recovery — fetched automatically once an analysis result
  // (with an analysis_id) exists, so /hidden-text/{analysis_id} is ready
  // by the time the panel below decides whether to render.
  useEffect(() => {
    if (!result || !result.analysis_id) return;

    fetch(`${API}/hidden-text/${result.analysis_id}`)
      .then(r => r.json())
      .then(data => setHiddenTextData(data))
      .catch(() => setHiddenTextData(null));
  }, [result]);


  const displayVerdict = result?.verdict;
  const displayScore   = result?.combined_score;
  const textStackingList = result?.text_stacking_findings ?? [];
  // Whether the annotated image draws a hidden-text (Missing/Replaced) box.
  // The backend suppresses those boxes on an ORIGINAL-verdict document (the
  // same is_clean gate every other box uses), so mirror that here to drive the
  // legend — the /hidden-text panel itself still lists findings regardless.
  const hiddenTextDrawn =
    (hiddenTextData?.total_found ?? 0) > 0 && result?.verdict !== "ORIGINAL";

  const verdictColor = displayVerdict === "MODIFIED" ? "#ff4444"
                     : displayVerdict === "ORIGINAL" ? "#00cc66"
                     : "#ff9800";  // orange for UNCERTAIN

  const verdictIcon = displayVerdict === "MODIFIED" ? "⚠️"
                    : displayVerdict === "ORIGINAL" ? "✅"
                    : "❓";  // question mark for UNCERTAIN

  const confidenceColor = {
    "VERY HIGH":    "#00cc66",
    HIGH:           "#88cc00",
    MEDIUM:         "#ffaa00",
    LOW:            "#ff4444",
    "REVIEW NEEDED": "#ff9800",
  }[result?.confidence?.label] || "#888";

  const handleCopy = () => {
    const text = JSON.stringify(result, null, 2);
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text)
        .then(() => alert("Copied to clipboard!"))
        .catch(() => fallbackCopy(text));
    } else {
      fallbackCopy(text);
    }
  };

  const fallbackCopy = (text) => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.left = "-999999px";
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand("copy");
      alert("Copied to clipboard!");
    } catch {
      alert("Copy failed — please select text manually");
    }
    document.body.removeChild(textarea);
  };

  const generateReport = async () => {
    if (!result || reportGenerating) return;
    setReportGenerating(true);

    // The Hidden Text panel's own fetch (triggered by the result-changed
    // effect) may still be in flight — or may never have been kicked off
    // for a stale/cached result — when the user hits Download. Report
    // generation can't rely on that passive state; it fetches directly so
    // the report always reflects real extraction results, never a
    // premature "nothing found" from data that just hadn't loaded yet.
    let hiddenTextForReport = hiddenTextData;
    if (result.analysis_id && hiddenTextForReport?.file_id !== result.analysis_id) {
      try {
        const { data } = await axios.get(`${API}/hidden-text/${result.analysis_id}`);
        hiddenTextForReport = data;
        setHiddenTextData(data);
      } catch {
        hiddenTextForReport = null;
      }
    }

    await buildForensicReportPdf(result, hiddenTextForReport);
  };

  return (
    <div className="root">
      {/* ── Header ── */}
      <div className="header">
        <div className="header-left">
          <span className="logo">🔬</span>
          <div>
            <div className="title">Document Forensics Engine</div>
            <div className="subtitle">Fully Deterministic Engine — No AI/ML</div>
          </div>
        </div>
        <div className="header-badge">v2.0</div>
      </div>

      {/* ── Upload mode toggle — routes to /analyze vs /analyze-image ── */}
      <div className="tab-bar" style={{ marginBottom: 12 }}>
        {[["pdf", "📄 PDF Document"], ["image", "🖼️ Image (ID Card / Stamp / Signature)"]].map(([mode, label]) => (
          <button
            key={mode}
            className={`tab-btn ${uploadMode === mode ? "active" : ""}`}
            onClick={() => {
              if (uploadMode === mode) return;
              setUploadMode(mode);
              // A file picked for one pipeline may be invalid for the other
              // (the image path takes only JPG/PNG) — clear it and any error
              // so the dropzone always reflects the active mode's rules.
              setFile(null); setError(null);
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Upload zone ── */}
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onClick={() => document.getElementById("fileInput").click()}
      >
        <input
          id="fileInput"
          type="file"
          accept={uploadMode === "image" ? ".jpg,.jpeg,.png" : ".pdf,.jpg,.jpeg,.png,.docx,.doc"}
          style={{ display: "none" }}
          onChange={(e) => {
            acceptFile(e.target.files[0], uploadMode);
            e.target.value = "";  // allow re-picking the same filename
          }}
        />
        {file ? (
          <div className="file-info">
            <span className="file-icon">{uploadMode === "image" ? "🖼️" : "📄"}</span>
            <div>
              <div className="file-name">{file.name}</div>
              <div className="file-size">{(file.size / 1024).toFixed(1)} KB</div>
            </div>
          </div>
        ) : uploadMode === "image" ? (
          <div className="drop-prompt">
            <div className="drop-icon">🖼️</div>
            <div className="drop-text">Drop a photographed ID, certificate, or stamped/signed document here</div>
            <div className="drop-sub">or click to browse · JPG, PNG only — PDFs go to the PDF Document tab</div>
          </div>
        ) : (
          <div className="drop-prompt">
            <div className="drop-icon">📁</div>
            <div className="drop-text">Drop PDF, Image, or Word document here</div>
            <div className="drop-sub">or click to browse · PDF, JPG, PNG, DOCX</div>
          </div>
        )}
      </div>

      {/* ── Analyze button ── */}
      <button
        className={`analyze-btn ${loading ? "loading" : ""}`}
        onClick={uploadMode === "image" ? analyzeImage : analyze}
        disabled={!file || loading}
      >
        {loading ? "🔍 Analyzing..."
          : uploadMode === "image" ? "🖼️ Analyze Image" : "🔬 Analyze Document"}
      </button>

      {/* ── Error ── */}
      {error && (
        <div className="error-box">
          ❌ {typeof error === "string" ? error : JSON.stringify(error)}
        </div>
      )}

      {/* ── Image-pipeline results — rendered ONLY through ImageReport; the
          response shape differs from the PDF pipeline's, so it must never
          fall into the PDF report tree below. ── */}
      {uploadMode === "image" && imageResult && <ImageReport data={imageResult} />}

      {/* ── Results (PDF pipeline) ── */}
      {uploadMode === "pdf" && result && (
        <div className="results">

          <div className="verdict-banner" style={{ borderColor: verdictColor }}>
            <div className="verdict-main">
              <span className="verdict-icon">{verdictIcon}</span>
              <span className="verdict-text" style={{ color: verdictColor }}>
                {displayVerdict}
              </span>
            </div>
            <div className="verdict-stats">
              <div className="stat-item">
                <div className="stat-label">Combined Score</div>
                <div className="stat-value">{displayScore.toFixed(1)}/100</div>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <div className="stat-label">Confidence</div>
                <div className="stat-value" style={{ color: confidenceColor }}>
                  {result.confidence.label}
                </div>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <div className="stat-label">Certainty</div>
                <div className="stat-value" style={{ color: confidenceColor }}>
                  {result.confidence.score}%
                </div>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <div className="stat-label">Doc Type</div>
                <div className="stat-value">
                  {result.pdf_type.replace(/_/g, " ")}
                </div>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <div className="stat-label">Source</div>
                <div className="stat-value">{result.document_source}</div>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <div className="stat-label">Last Modified</div>
                <div className="stat-value" style={{
                  color: result.metadata?.is_very_recent_edit ? "#ff4444"
                       : result.metadata?.is_recent_edit ? "#ff8800"
                       : "#e0e0e0"
                }}>
                  {result.metadata?.edit_age_human || "Unknown"}
                </div>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <div className="stat-label">Time</div>
                <div className="stat-value">{result.processing_time_seconds}s</div>
              </div>
            </div>
            <div className="confidence-explanation">
              💡 {result.confidence.explanation}
            </div>
          </div>

          {/* Hidden Text Recovery panel */}
          <HiddenTextPanel hiddenTextData={hiddenTextData} />
          {/* Tab bar */}
          <div className="tab-bar">
            {["overview", "layers", "location", "signals", "metadata", "json"].map(tab => (
              <button
                key={tab}
                className={`tab-btn ${activeTab === tab ? "active" : ""}`}
                onClick={() => setActiveTab(tab)}
              >
                {{ overview:"📊 Overview", layers:"🔬 Layers",
                   location:"📍 Location", signals:"⚡ Signals",
                   metadata:"📋 Metadata", json:"{ } JSON" }[tab]}
              </button>
            ))}
          </div>

          {result && (
            <div style={{
              display: "flex",
              justifyContent: "flex-end",
              marginBottom: "12px",
            }}>
              <button
                disabled={reportGenerating}
                onClick={async () => {
                  try {
                    await generateReport();
                  } catch (err) {
                    alert("Report generation failed: " + (err.message || err));
                  } finally {
                    setReportGenerating(false);
                  }
                }}
                style={{
                  padding: "10px 20px",
                  background: reportGenerating
                    ? "linear-gradient(135deg, #555, #777)"
                    : "linear-gradient(135deg, #1a4a8a, #2a6aaa)",
                  color: "#fff",
                  border: "none",
                  borderRadius: "8px",
                  cursor: reportGenerating ? "wait" : "pointer",
                  fontSize: "0.9rem",
                  fontWeight: "600",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  boxShadow: "0 2px 8px rgba(30,100,200,0.3)",
                }}
              >
                {reportGenerating ? "⏳ Generating Report..." : "📥 Download Full Report (PDF)"}
              </button>
            </div>
          )}

          {/* ── Tab: Overview ── */}
          {activeTab === "overview" && <OverviewTab result={result} />}
          {/* ── Tab: Layers ── */}
          {activeTab === "layers" && <LayersTab result={result} />}
          {/* ── Tab: Location ── */}
          {activeTab === "location" && (
            <div className="tab-content">
              {/* The annotated image always shows ALL individual layer markings
                  (red/yellow/orange/purple/cyan). Cross-validated fusion is a
                  separate highlighted section in the Overview tab. */}

              {/* Page selector */}
              <div className="page-selector">
                {Array.from({ length: result.total_pages }, (_, i) => i + 1).map(p => (
                  <button
                    key={p}
                    className={`page-btn ${activePage === p ? "active" : ""}`}
                    onClick={() => setActivePage(p)}
                  >
                    Page {p}
                  </button>
                ))}
              </div>

              {/* Legend — color tells you which LAYER flagged the region;
                  the label printed on each box tells you WHAT was found
                  (e.g. "Font Size Mismatch", "Balance Mismatch",
                  "Pasted Stamp: Flat Background"). */}
              <div className="legend">
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#ff4444" }} />
                  Red = Content layer (font / spacing)
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#ff00c8" }} />
                  Magenta = Overlay layer (pasted image)
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#ffdd00" }} />
                  Yellow = Numeric layer
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#00ccff" }} />
                  Cyan = Overlay layer (white-out cover-up)
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#ffc800" }} />
                  Gold = Overlay layer (ghost text)
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#b400ff" }} />
                  Purple = ELA layer (flat / pasted patch)
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#00be5a" }} />
                  Green = Embedded-image forensics
                </span>
                {(textStackingList.length > 0 || hiddenTextDrawn) && (
                  <span className="legend-item">
                    <span className="legend-dot" style={{
                      background: "#ff00ff",
                      border: "1px dashed #fff",
                    }} />
                    Magenta (dashed) = Hidden text — Missing / Replaced data
                  </span>
                )}
              </div>
              <div className="legend-section">
                <div className="legend-note">
                  Color = which detection layer flagged the region; the label
                  on each box states the specific finding.
                </div>
              </div>

              {/* Confidence note */}
              <div className="legend-section">
                <div className="legend-note">
                  ℹ️ Only strong signals are shown: cross-validated findings
                  (confirmed by 2+ independent layers), high-score anomalies,
                  or pages where 3+ layers fired simultaneously.
                  The "Modified: …" badge (top-right of page) shows document age.
                </div>
              </div>

              {/* Zoom controls */}
              <div className="zoom-controls">
                <button onClick={() => setZoom(z => Math.max(50, z - 25))}>
                  − Zoom Out
                </button>
                <span>{zoom}%</span>
                <button onClick={() => setZoom(z => Math.min(300, z + 25))}>
                  + Zoom In
                </button>
                <button onClick={() => setZoom(100)}>Reset</button>
              </div>

              {/* Annotated image — a load failure shows an inline retry
                  notice and leaves the rest of the report untouched. The
                  backend re-loads results from its disk spool after a
                  restart, so a retry usually succeeds; only a genuinely
                  evicted/unknown analysis needs a re-run. */}
              {imageError ? (
                <div style={{
                  margin: "16px 0", padding: "16px 20px",
                  border: "1px solid #fca5a5", borderRadius: 10,
                  background: "#fef2f2", color: "#991b1b", fontSize: 13,
                }}>
                  ⚠️ Couldn't load the annotated image for page {activePage}.
                  If the backend was just restarted, it may still be
                  reloading this analysis — try again. The rest of this
                  report is unaffected.
                  <button
                    onClick={() => setImageRetry(r => r + 1)}
                    style={{
                      marginLeft: 12, background: "#fff", color: "#991b1b",
                      border: "1px solid #fca5a5", borderRadius: 6,
                      padding: "4px 12px", fontSize: 12, fontWeight: 600,
                      cursor: "pointer",
                    }}
                  >
                    ↻ Retry
                  </button>
                </div>
              ) : (
                <div className="image-container" style={{ overflowX: "auto" }}>
                  <img
                    src={imageUrl}
                    alt={`Page ${activePage} annotated`}
                    className="annotated-image"
                    style={{ width: `${zoom}%`, maxWidth: "none" }}
                    onError={() => setImageError(true)}
                  />
                </div>
              )}
            </div>
          )}

          {/* ── Tab: Signals ── */}
          {activeTab === "signals" && <SignalsTab result={result} />}
          {/* ── Tab: Metadata ── */}
          {activeTab === "metadata" && result.metadata && <MetadataTab result={result} />}
          {/* ── Tab: JSON ── */}
          {activeTab === "json" && (
            <div className="tab-content">
              <div className="json-header">
                <span>Full API Response</span>
                <button className="copy-btn" onClick={handleCopy}>
                  📋 Copy JSON
                </button>
              </div>
              <pre className="json-box">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

