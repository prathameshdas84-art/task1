import { useState, useCallback, useMemo, useEffect } from "react";
import axios from "axios";
import jsPDF from "jspdf";
import { applyPlugin } from "jspdf-autotable";
import "./App.css";

// jspdf-autotable v5 dropped the auto-patched doc.autoTable() in favor of a
// standalone autoTable(doc, opts) export; applyPlugin() restores the
// prototype method (and doc.lastAutoTable) so the report-builder code below
// can call doc.autoTable(...) directly instead of threading every call
// through the functional API.
applyPlugin(jsPDF);

const API = "http://localhost:8000";

const LAYER_LABELS = {
  metadata: "Layer 1 — Metadata",
  content:  "Layer 2 — Content",
  ocr:      "Layer 3 — OCR",
  numeric:  "Layer 4 — Numeric",
  ela:      "Layer 5 — ELA",
  pymupdf:  "Layer 6 — PyMuPDF",
};

const SIGNAL_COLORS = {
  // [INCREMENTAL] is checked before [ELA] since it appears NESTED inside an
  // "[ELA]      [INCREMENTAL] ..." signal string — Object.keys(...).find()
  // below takes the first key whose substring matches, so the more specific
  // inner prefix has to come first or it'd always resolve to ELA's color.
  "[INCREMENTAL]": "#00e0d0",
  "[METADATA]": "#ff9944",
  "[CONTENT]":  "#ff4466",
  "[OCR]":      "#44aaff",
  "[NUMERIC]":  "#ffdd00",
  "[ELA]":      "#cc44ff",
  "[PYMUPDF]":  "#00ffcc",
};

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
  const [aiReview, setAiReview]             = useState(null);
  const [aiReviewLoading, setAiReviewLoading] = useState(false);
  const [aiReviewError, setAiReviewError]   = useState(null);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) {
      setFile(f); setResult(null); setError(null); setHiddenTextData(null);
      setAiReview(null); setAiReviewError(null);
    }
  }, []);

  const requestAiReview = async () => {
    if (!result?.analysis_id || aiReviewLoading) return;
    setAiReviewLoading(true);
    setAiReviewError(null);
    try {
      const { data } = await axios.post(`${API}/api/analysis/${result.analysis_id}/ai-review`);
      setAiReview(data);
    } catch (err) {
      setAiReviewError(err.response?.data?.detail || err.message || "AI review request failed");
    } finally {
      setAiReviewLoading(false);
    }
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setActivePage(1);
    setAiReview(null);
    setAiReviewError(null);
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

  // analysis_id is a unique UUID per analysis and page is explicit, so the
  // URL itself is a stable cache key — Date.now() is not needed and would
  // cause the browser to re-fetch the image on every React render (zoom
  // change, tab switch, etc.), which also triggers stale-session 404s.
  const imageUrl = useMemo(
    () => result
      ? `${API}/annotated-image/${result.analysis_id}?page=${activePage}`
      : null,
    [result?.analysis_id, activePage]
  );

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

  const fieldIcons = {
    name:       "👤",
    amount:     "💰",
    date:       "📅",
    id_number:  "🔢",
    address:    "📍",
    score:      "📊",
    unknown:    "📄",
  };

  const methodBadges = {
    white_rectangle_cover: {
      label: "White-out detected",
      color: "#dc2626",
      icon: "🟥",
    },
    text_overlap: {
      label: "Text layered over original",
      color: "#d97706",
      icon: "📑",
    },
    incremental_update: {
      label: "Previous version found",
      color: "#7c3aed",
      icon: "🕐",
    },
  };

  const verdictColor = result?.verdict === "MODIFIED" ? "#ff4444"
                     : result?.verdict === "ORIGINAL" ? "#00cc66"
                     : "#ff9800";  // orange for UNCERTAIN

  const verdictIcon = result?.verdict === "MODIFIED" ? "⚠️"
                    : result?.verdict === "ORIGINAL" ? "✅"
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

    const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
    const pageW = doc.internal.pageSize.getWidth();
    const pageH = doc.internal.pageSize.getHeight();
    const margin = 15;
    const contentW = pageW - margin * 2;
    let y = margin;

    // ── Helper functions ──────────────────────────────────────
    const addPage = () => {
      doc.addPage();
      y = margin;
      addHeader();
    };

    const checkY = (needed = 10) => {
      if (y + needed > pageH - margin) addPage();
    };

    const addHeader = () => {
      doc.setFillColor(15, 15, 15);
      doc.rect(0, 0, pageW, 12, "F");
      doc.setTextColor(200, 200, 200);
      doc.setFontSize(8);
      doc.setFont("helvetica", "normal");
      doc.text("Document Forensics Engine - Confidential Analysis Report", margin, 8);
      doc.text(`Page ${doc.internal.getCurrentPageInfo().pageNumber}`, pageW - margin, 8, { align: "right" });
      doc.setTextColor(0, 0, 0);
      y = Math.max(y, 18);
    };

    const addSectionTitle = (title, color = [30, 30, 30]) => {
      checkY(12);
      doc.setFillColor(...color);
      doc.rect(margin, y, contentW, 7, "F");
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(10);
      doc.setFont("helvetica", "bold");
      doc.text(title, margin + 3, y + 5);
      doc.setTextColor(0, 0, 0);
      y += 10;
    };

    const addText = (text, size = 9, bold = false, color = [0, 0, 0]) => {
      checkY(6);
      doc.setFontSize(size);
      doc.setFont("helvetica", bold ? "bold" : "normal");
      doc.setTextColor(...color);
      const lines = doc.splitTextToSize(String(text), contentW);
      lines.forEach(line => {
        checkY(5);
        doc.text(line, margin, y);
        y += 5;
      });
      doc.setTextColor(0, 0, 0);
    };

    // ══════════════════════════════════════════════════════════
    // PAGE 1 - COVER PAGE
    // ══════════════════════════════════════════════════════════

    // Background
    doc.setFillColor(10, 10, 30);
    doc.rect(0, 0, pageW, pageH, "F");

    // Title block
    doc.setFillColor(30, 30, 60);
    doc.rect(0, 60, pageW, 60, "F");

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.setFont("helvetica", "bold");
    doc.text("DOCUMENT FORENSICS", pageW / 2, 80, { align: "center" });
    doc.text("ANALYSIS REPORT", pageW / 2, 92, { align: "center" });

    // Verdict badge
    const verdictBg = result.verdict === "MODIFIED" ? [220, 50, 50]
                    : result.verdict === "ORIGINAL" ? [50, 180, 80]
                    : [220, 140, 30];
    doc.setFillColor(...verdictBg);
    doc.roundedRect(pageW / 2 - 35, 100, 70, 14, 3, 3, "F");
    doc.setFontSize(13);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(255, 255, 255);
    doc.text(result.verdict, pageW / 2, 109, { align: "center" });

    // Score
    doc.setFontSize(11);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(200, 200, 200);
    doc.text(`Combined Score: ${result.combined_score.toFixed(1)}/100`, pageW / 2, 122, { align: "center" });
    doc.text(`Confidence: ${result.confidence.label} (${result.confidence.score}%)`, pageW / 2, 130, { align: "center" });

    // File info
    doc.setFillColor(20, 20, 50);
    doc.rect(margin, 148, contentW, 40, "F");
    doc.setTextColor(180, 180, 255);
    doc.setFontSize(9);
    doc.text(`Filename: ${result.filename}`, margin + 5, 158);
    doc.text(`File Size: ${result.file_size_kb} KB`, margin + 5, 164);
    doc.text(`Document Type: ${result.pdf_type}`, margin + 5, 170);
    doc.text(`Source: ${result.document_source}`, margin + 5, 176);
    doc.text(`Processing Time: ${result.processing_time_seconds}s`, margin + 5, 182);

    // Report info
    doc.setTextColor(120, 120, 150);
    doc.setFontSize(8);
    const now = new Date();
    doc.text(`Report Generated: ${now.toLocaleString()}`, pageW / 2, 220, { align: "center" });
    doc.text("Document Forensics Engine v2.0 - 6-Layer Tamper Detection", pageW / 2, 227, { align: "center" });
    doc.text("Core engine: no AI/ML - pure statistical and structural analysis", pageW / 2, 234, { align: "center" });

    // ══════════════════════════════════════════════════════════
    // PAGE 2 - EXECUTIVE SUMMARY
    // ══════════════════════════════════════════════════════════
    addPage();

    addSectionTitle("1. EXECUTIVE SUMMARY", [40, 40, 100]);
    addText(result.summary, 10, false);
    y += 3;
    addText(result.confidence.explanation, 9, false, [60, 60, 60]);
    y += 5;

    // Layer scores table
    addSectionTitle("2. LAYER SCORES", [40, 40, 100]);

    const layerNames = {
      metadata: "Layer 1 - Metadata Analysis",
      content:  "Layer 2 - Content/Font Analysis",
      ocr:      "Layer 3 - OCR Analysis",
      numeric:  "Layer 4 - Numeric Outlier Detection",
      ela:      "Layer 5 - ELA Visual Analysis",
      pymupdf:  "Layer 6 - PyMuPDF Deep Analysis",
    };

    const layerRows = Object.entries(result.layers).map(([key, score]) => [
      layerNames[key] || key,
      `${score}/100`,
      score >= 50 ? "HIGH ANOMALY" : score >= 20 ? "ANOMALY" : score >= 10 ? "Minor" : "Clean",
    ]);

    doc.autoTable({
      startY: y,
      head: [["Detection Layer", "Score", "Status"]],
      body: layerRows,
      margin: { left: margin, right: margin },
      headStyles: { fillColor: [40, 40, 100], textColor: 255, fontStyle: "bold" },
      bodyStyles: { fontSize: 9 },
      alternateRowStyles: { fillColor: [245, 245, 255] },
      columnStyles: {
        0: { cellWidth: 100 },
        1: { cellWidth: 25, halign: "center" },
        2: { cellWidth: 45, halign: "center" },
      },
      didParseCell: (data) => {
        if (data.section === "body" && data.column.index === 2) {
          const val = data.cell.raw;
          if (val === "HIGH ANOMALY") data.cell.styles.textColor = [200, 30, 30];
          else if (val === "ANOMALY") data.cell.styles.textColor = [200, 100, 30];
          else if (val === "Clean") data.cell.styles.textColor = [30, 150, 60];
        }
      },
    });
    y = doc.lastAutoTable.finalY + 8;

    // ══════════════════════════════════════════════════════════
    // PAGE 3 - ENGINE SUMMARY (plain-English narrative — ALWAYS rendered
    // for every input regardless of verdict: states what the engine
    // concluded, then explicitly states whether hidden/original text was
    // recovered from beneath any edits, one way or the other. Not a raw
    // signal dump — this is the labeled, human-readable account.)
    // ══════════════════════════════════════════════════════════
    checkY(20);
    addSectionTitle("3. ENGINE SUMMARY & DETECTION FINDINGS", [40, 40, 100]);

    const fieldLabel = (ft) =>
      !ft || ft === "unknown" ? "Content" : ft.charAt(0).toUpperCase() + ft.slice(1);

    const methodPhrase = {
      white_rectangle_cover: "a white box placed over the original text with new text typed on top",
      text_overlap:          "new text layered directly over the original text",
      incremental_update:    "an edit made in a later saved revision of the file",
    };

    const verdictLead = {
      MODIFIED:  `This document was flagged as MODIFIED with a combined forensic score of ${result.combined_score.toFixed(1)}/100 (${result.confidence.label} confidence).`,
      ORIGINAL:  `This document was assessed as ORIGINAL with a combined forensic score of ${result.combined_score.toFixed(1)}/100 (${result.confidence.label} confidence).`,
      UNCERTAIN: `This document's evidence was inconclusive (UNCERTAIN) with a combined forensic score of ${result.combined_score.toFixed(1)}/100 — manual review is recommended.`,
    }[result.verdict] || `Combined forensic score: ${result.combined_score.toFixed(1)}/100.`;

    addText(`${verdictLead} ${result.summary}`, 9, false);
    y += 4;

    addText("Hidden Text Recovery", 9.5, true);
    if (hiddenTextForReport?.total_found > 0) {
      addText(
        "The underlying PDF text layers contain content that does not match what is visible when the " +
        "document is opened normally. The original text beneath each edit is still present in the file's " +
        "data and was recovered below:",
        9, false
      );
      y += 2;

      hiddenTextForReport.findings.forEach((f, i) => {
        checkY(18);
        addText(`${i + 1}. Page ${f.page} — ${fieldLabel(f.field_type)}`, 9.5, true, [127, 29, 29]);
        addText(`Original text hidden in the file: "${f.original_text}"`, 9, false, [21, 128, 61]);
        addText(`Text placed over it: "${f.covering_text}"`, 9, false, [185, 28, 28]);
        addText(
          `How it was done: ${f.plain_explanation || `This was done using ${methodPhrase[f.method] || f.method}.`}`,
          8.5, false, [80, 80, 80]
        );
        y += 3;
      });

      y += 1;
      addText("Conclusion", 9.5, true);
      addText(hiddenTextForReport.conclusion, 9, false);
      if (result.verdict === "MODIFIED") {
        y += 3;
        addText(
          "Next steps: treat this document with caution. If independent verification is required, request " +
          "confirmation directly from the original issuing organization referenced in the document, rather " +
          "than relying on the copy provided.",
          9, false, [60, 60, 60]
        );
      }
    } else {
      // Explicitly state the negative result too — every input gets a
      // definitive statement either way, never silence.
      addText(
        hiddenTextForReport?.summary
          || "No hidden or covered-up original text was found in the underlying PDF data for this document.",
        9, false, [60, 60, 60]
      );
    }
    y += 5;

    // ══════════════════════════════════════════════════════════
    // PAGE 4 - SPECIFIC FINDINGS
    // ══════════════════════════════════════════════════════════

    if (result.fused_findings?.length > 0) {
      checkY(20);
      addSectionTitle("4. CROSS-VALIDATED HIGH-CONFIDENCE FINDINGS", [150, 0, 50]);
      addText("These regions were confirmed by MULTIPLE independent layers:", 9, true);
      y += 3;

      result.fused_findings.forEach((f, i) => {
        checkY(20);
        doc.setFillColor(
          f.confidence === "HIGH" ? 255 : 240,
          f.confidence === "HIGH" ? 235 : 240,
          235
        );
        doc.rect(margin, y, contentW, 18, "F");
        doc.setDrawColor(
          f.confidence === "HIGH" ? 200 : 150,
          f.confidence === "HIGH" ? 50 : 100,
          50
        );
        doc.rect(margin, y, contentW, 18, "S");

        doc.setFontSize(9);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(150, 30, 30);
        doc.text(`Finding ${i + 1}: Page ${f.page} - ${f.confidence} CONFIDENCE (${f.score}/100)`, margin + 3, y + 5);
        doc.setFont("helvetica", "normal");
        doc.setTextColor(60, 60, 60);
        doc.text(`Layers: ${f.confirming_layers.join(", ")}`, margin + 3, y + 10);
        const descLines = doc.splitTextToSize(f.description, contentW - 6);
        doc.text(descLines[0] || "", margin + 3, y + 15);
        doc.setTextColor(0, 0, 0);
        y += 22;
      });
    }

    if (result.suspicious_lines?.length > 0) {
      checkY(20);
      addSectionTitle("5. SUSPICIOUS LINES - Content Layer", [180, 30, 30]);

      const slRows = result.suspicious_lines.map(sl => [
        `Page ${sl.page}, Line ${sl.line_num}`,
        sl.text.substring(0, 50) + (sl.text.length > 50 ? "..." : ""),
        `${sl.anomaly_score_pct}%`,
        sl.reasons.join("; "),
      ]);

      doc.autoTable({
        startY: y,
        head: [["Location", "Text", "Score", "Reasons"]],
        body: slRows,
        margin: { left: margin, right: margin },
        headStyles: { fillColor: [180, 30, 30], textColor: 255 },
        bodyStyles: { fontSize: 8 },
        alternateRowStyles: { fillColor: [255, 245, 245] },
        columnStyles: {
          0: { cellWidth: 28 },
          1: { cellWidth: 55 },
          2: { cellWidth: 15, halign: "center" },
          3: { cellWidth: 82 },
        },
      });
      y = doc.lastAutoTable.finalY + 8;
    }

    if (result.numeric_anomalies?.length > 0) {
      checkY(20);
      addSectionTitle("6. NUMERIC ANOMALIES - Statistical Layer", [160, 120, 0]);

      const naRows = result.numeric_anomalies.map(na => [
        `Page ${na.page}`,
        na.text.substring(0, 40) + (na.text.length > 40 ? "..." : ""),
        na.value.toLocaleString(),
        na.z_score.toFixed(1),
        na.reason.substring(0, 80),
      ]);

      doc.autoTable({
        startY: y,
        head: [["Page", "Line Text", "Value", "Z-Score", "Reason"]],
        body: naRows,
        margin: { left: margin, right: margin },
        headStyles: { fillColor: [160, 120, 0], textColor: 255 },
        bodyStyles: { fontSize: 8 },
        alternateRowStyles: { fillColor: [255, 252, 235] },
        columnStyles: {
          0: { cellWidth: 15 },
          1: { cellWidth: 45 },
          2: { cellWidth: 25, halign: "right" },
          3: { cellWidth: 18, halign: "center" },
          4: { cellWidth: 77 },
        },
      });
      y = doc.lastAutoTable.finalY + 8;
    }

    // ══════════════════════════════════════════════════════════
    // PAGE 5 - ANNOTATED IMAGES
    // (Full metadata / font inventory tables intentionally dropped from
    // this report — the report now shows only labeled findings, the
    // marked-up document pages, and the summary, not the raw technical
    // dump. That detail is still available in the app's Metadata tab.)
    // ══════════════════════════════════════════════════════════
    if (result.analysis_id && result.total_pages > 0) {
      addPage();
      addSectionTitle("7. ANNOTATED DOCUMENT PAGES", [80, 80, 80]);
      addText("Red/Orange/Yellow/Purple/Cyan/Gold boxes indicate detected anomaly locations.", 8, false, [80, 80, 80]);
      y += 3;

      // Fetch+decode every page image CONCURRENTLY first — sequentially
      // awaiting each fetch one at a time (the original approach) means a
      // 12-page document pays its full network round-trip latency 12 times
      // in a row, which is what made report generation look hung on larger
      // documents. The jsPDF drawing calls below still run in page order,
      // they just don't block on network I/O between pages anymore.
      const pageImages = await Promise.all(
        Array.from({ length: result.total_pages }, (_, i) => i + 1).map(async (pageNum) => {
          try {
            const imgUrl = `${API}/annotated-image/${result.analysis_id}?page=${pageNum}&t=${Date.now()}`;
            const response = await fetch(imgUrl);
            if (!response.ok) return { pageNum, base64: null };
            const blob = await response.blob();
            const reader = new FileReader();
            const base64 = await new Promise((resolve, reject) => {
              reader.onload = () => resolve(reader.result);
              reader.onerror = reject;
              reader.readAsDataURL(blob);
            });
            return { pageNum, base64 };
          } catch {
            return { pageNum, base64: null };
          }
        })
      );

      // Load each page image and embed in PDF
      for (const { pageNum, base64 } of pageImages) {
        try {
          if (!base64) {
            addText(`Page ${pageNum}: Could not load annotated image`, 8, false, [150, 50, 50]);
            y += 5;
            continue;
          }

          // Calculate image dimensions to fit page
          const imgMaxW = contentW;
          const imgMaxH = 160;

          checkY(imgMaxH + 15);

          addText(`Page ${pageNum} of ${result.total_pages}`, 9, true);
          y += 2;

          doc.addImage(base64, "PNG", margin, y, imgMaxW, imgMaxH, undefined, "FAST");
          y += imgMaxH + 8;

        } catch (err) {
          addText(`Page ${pageNum}: Could not load annotated image`, 8, false, [150, 50, 50]);
          y += 5;
        }
      }
    }

    // ══════════════════════════════════════════════════════════
    // FINAL PAGE - DISCLAIMER
    // ══════════════════════════════════════════════════════════
    addPage();
    addSectionTitle("DISCLAIMER & METHODOLOGY", [60, 60, 60]);

    const disclaimer = [
      "This report was generated by the Document Forensics Engine v2.0.",
      "",
      "METHODOLOGY: The engine uses 6 independent detection layers:",
      "  Layer 1 - Metadata Analysis: Checks producer, creator, timestamps, XMP consistency",
      "  Layer 2 - Content Analysis: Analyzes font consistency, CIDFont sessions, color per line",
      "  Layer 3 - OCR Analysis: Checks word-level size, color, baseline alignment",
      "  Layer 4 - Numeric Analysis: Statistical outlier detection using z-scores",
      "  Layer 5 - ELA Analysis: Error Level Analysis at multiple DPI scales",
      "  Layer 6 - PyMuPDF Analysis: Hidden overlays, ghost text, character spacing",
      "",
      "LIMITATIONS:",
      "  - This engine cannot guarantee 100% accuracy on all document types",
      "  - Print-scan-retype attacks may not be detectable",
      "  - Same-font, same-color edits may not produce statistical anomalies",
      "  - Results should be verified by a qualified document examiner",
      "  - This report is for investigative purposes only",
      "",
      "CORE 6-LAYER ENGINE: deterministic, no AI/ML. All 6 layers above use",
      "statistical and structural analysis only — no machine learning models",
      "or AI systems participate in computing the verdict or combined_score.",
      "",
      "OPTIONAL AI REVIEW (LAYER 7): If \"Ask AI\" was used for this document,",
      "Gemini contributes an ADDITIONAL, separate combined_score_with_ai —",
      "shown only when explicitly requested below, and it never overwrites",
      "or replaces the deterministic combined_score/verdict above.",
    ];

    disclaimer.forEach(line => {
      addText(line, 8.5, line.startsWith("Layer") || line.startsWith("LIMITATIONS") || line.startsWith("METHODOLOGY") || line.startsWith("NO AI"), [50, 50, 50]);
    });

    // ── Save ──────────────────────────────────────────────────
    const safeName = result.filename.replace(/[^a-zA-Z0-9._-]/g, "_");
    doc.save(`ForensicReport_${safeName}_${Date.now()}.pdf`);
  };

  return (
    <div className="root">
      {/* ── Header ── */}
      <div className="header">
        <div className="header-left">
          <span className="logo">🔬</span>
          <div>
            <div className="title">Document Forensics Engine</div>
            <div className="subtitle">Core 6-Layer Engine: Deterministic, No AI/ML · Optional AI Review (Layer 7): Gemini, Separate AI-Adjusted Score</div>
          </div>
        </div>
        <div className="header-badge">v2.0</div>
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
          accept=".pdf,.jpg,.jpeg,.png,.docx,.doc"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files[0];
            if (f) {
              setFile(f); setResult(null); setError(null); setHiddenTextData(null);
              setAiReview(null); setAiReviewError(null);
            }
          }}
        />
        {file ? (
          <div className="file-info">
            <span className="file-icon">📄</span>
            <div>
              <div className="file-name">{file.name}</div>
              <div className="file-size">{(file.size / 1024).toFixed(1)} KB</div>
            </div>
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
        onClick={analyze}
        disabled={!file || loading}
      >
        {loading ? "🔍 Analyzing..." : "🔬 Analyze Document"}
      </button>

      {/* ── Error ── */}
      {error && (
        <div className="error-box">
          ❌ {typeof error === "string" ? error : JSON.stringify(error)}
        </div>
      )}

      {/* ── Results ── */}
      {result && (
        <div className="results">

          {/* Verdict banner */}
          <div className="verdict-banner" style={{ borderColor: verdictColor }}>
            <div className="verdict-main">
              <span className="verdict-icon">{verdictIcon}</span>
              <span className="verdict-text" style={{ color: verdictColor }}>
                {result.verdict}
              </span>
            </div>
            <div className="verdict-stats">
              <div className="stat-item">
                <div className="stat-label">Combined Score</div>
                <div className="stat-value">{result.combined_score.toFixed(1)}/100</div>
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
          {hiddenTextData && hiddenTextData.total_found > 0 && (
            <div style={{
              marginTop: '24px',
              border: '2px solid #ef4444',
              borderRadius: '12px',
              overflow: 'hidden',
              boxShadow: '0 4px 12px rgba(239,68,68,0.15)',
              marginBottom: '16px',
            }}>

              {/* Header */}
              <div style={{
                backgroundColor: '#7f1d1d',
                color: 'white',
                padding: '16px 20px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}>
                <span style={{fontSize: '20px'}}>🔍</span>
                <div>
                  <div style={{fontWeight: 'bold', fontSize: '16px'}}>
                    HIDDEN TEXT RECOVERY
                  </div>
                  <div style={{fontSize: '12px', opacity: 0.85}}>
                    Original content found beneath the edits
                  </div>
                </div>
                <div style={{
                  marginLeft: 'auto',
                  backgroundColor: '#ef4444',
                  borderRadius: '20px',
                  padding: '4px 12px',
                  fontSize: '13px',
                  fontWeight: 'bold',
                }}>
                  {hiddenTextData.total_found} region
                  {hiddenTextData.total_found > 1 ? 's' : ''} found
                </div>
              </div>

              {/* Warning banner */}
              <div style={{
                backgroundColor: '#fef2f2',
                borderBottom: '1px solid #fecaca',
                padding: '12px 20px',
                fontSize: '13px',
                color: '#991b1b',
              }}>
                ⚠️ This document contains hidden layers.
                The following original content was found
                beneath the visible text:
              </div>

              {/* Findings */}
              <div style={{padding: '16px 20px'}}>
                {hiddenTextData.findings.map((finding, idx) => (
                  <div key={idx} style={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    marginBottom: '12px',
                    overflow: 'hidden',
                  }}>

                    {/* Finding header */}
                    <div style={{
                      backgroundColor: '#f8fafc',
                      padding: '10px 16px',
                      borderBottom: '1px solid #e5e7eb',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}>
                      <span>
                        {fieldIcons[finding.field_type] || '📄'}
                      </span>
                      <span style={{
                        fontWeight: 'bold',
                        fontSize: '14px',
                      }}>
                        Page {finding.page} — {
                          finding.field_type === 'unknown'
                            ? 'Content'
                            : finding.field_type.charAt(0).toUpperCase()
                              + finding.field_type.slice(1)
                        } Field
                      </span>
                      <span style={{
                        marginLeft: 'auto',
                        backgroundColor: methodBadges[finding.method]?.color || '#6b7280',
                        color: 'white',
                        fontSize: '11px',
                        padding: '2px 8px',
                        borderRadius: '4px',
                      }}>
                        {methodBadges[finding.method]?.icon} {' '}
                        {methodBadges[finding.method]?.label || finding.method}
                      </span>
                    </div>

                    {/* Original vs Replaced */}
                    <div style={{padding: '16px'}}>

                      {/* Side by side comparison */}
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '12px',
                        marginBottom: '12px',
                      }}>

                        {/* Original */}
                        <div style={{
                          backgroundColor: '#f0fdf4',
                          border: '1px solid #86efac',
                          borderRadius: '6px',
                          padding: '12px',
                        }}>
                          <div style={{
                            fontSize: '11px',
                            color: '#166534',
                            fontWeight: 'bold',
                            marginBottom: '6px',
                            textTransform: 'uppercase',
                            letterSpacing: '0.5px',
                          }}>
                            ✅ ORIGINAL (hidden in file)
                          </div>
                          <div style={{
                            fontSize: '15px',
                            fontWeight: 'bold',
                            color: '#15803d',
                            wordBreak: 'break-word',
                          }}>
                            {finding.original_text}
                          </div>
                        </div>

                        {/* Replaced with */}
                        <div style={{
                          backgroundColor: '#fef2f2',
                          border: '1px solid #fca5a5',
                          borderRadius: '6px',
                          padding: '12px',
                        }}>
                          <div style={{
                            fontSize: '11px',
                            color: '#991b1b',
                            fontWeight: 'bold',
                            marginBottom: '6px',
                            textTransform: 'uppercase',
                            letterSpacing: '0.5px',
                          }}>
                            ❌ REPLACED WITH (visible text)
                          </div>
                          <div style={{
                            fontSize: '15px',
                            fontWeight: 'bold',
                            color: '#dc2626',
                            wordBreak: 'break-word',
                          }}>
                            {finding.covering_text || 'Unknown'}
                          </div>
                        </div>
                      </div>

                      {/* How it was done */}
                      <div style={{
                        backgroundColor: '#f8fafc',
                        borderRadius: '6px',
                        padding: '10px 12px',
                        fontSize: '13px',
                        color: '#475569',
                        lineHeight: '1.5',
                      }}>
                        <span style={{
                          fontWeight: 'bold',
                          color: '#334155',
                        }}>
                          How it was done:{' '}
                        </span>
                        {finding.plain_explanation}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Conclusion */}
              {hiddenTextData.conclusion && (
                <div style={{
                  backgroundColor: '#1e293b',
                  color: 'white',
                  padding: '16px 20px',
                  fontSize: '13px',
                  lineHeight: '1.6',
                }}>
                  <div style={{
                    fontWeight: 'bold',
                    marginBottom: '6px',
                    fontSize: '14px',
                  }}>
                    CONCLUSION
                  </div>
                  {hiddenTextData.conclusion}
                </div>
              )}
            </div>
          )}

          {/* Show message if no hidden text found */}
          {hiddenTextData && hiddenTextData.total_found === 0 && (
            <div style={{
              marginTop: '16px',
              marginBottom: '16px',
              padding: '12px 16px',
              backgroundColor: '#f0fdf4',
              border: '1px solid #86efac',
              borderRadius: '8px',
              fontSize: '13px',
              color: '#166534',
            }}>
              ✅ No hidden text detected — visible content
              appears to be original
            </div>
          )}

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
          {activeTab === "overview" && (
            <div className="tab-content">
              {result.fused_findings && result.fused_findings.length > 0 && (
                <div className="fusion-section">
                  <div className="fusion-title">
                    🎯 Cross-Validated Findings (High Confidence)
                    <span className="badge">{result.fused_findings.length}</span>
                  </div>
                  <div className="fusion-subtitle">
                    These regions were flagged by MULTIPLE independent layers —
                    most likely real tampering, not false positives.
                  </div>
                  {result.fused_findings.map((f, i) => (
                    <div key={i} className="fusion-card" style={{
                      borderLeft: `4px solid ${
                        f.confidence === "HIGH" ? "#ff4444" :
                        f.confidence === "MEDIUM" ? "#ff9800" : "#ffdd44"
                      }`
                    }}>
                      <div className="fusion-header">
                        <span className="fusion-page">Page {f.page}</span>
                        <span className="fusion-conf" style={{color:
                          f.confidence === "HIGH" ? "#ff4444" :
                          f.confidence === "MEDIUM" ? "#ff9800" : "#ffdd44"
                        }}>
                          {f.confidence} CONFIDENCE
                        </span>
                        <span className="fusion-score">{f.score}/100</span>
                      </div>
                      <div className="fusion-layers">
                        Confirmed by: {f.confirming_layers.map(l => (
                          <span key={l} className="layer-badge">{l}</span>
                        ))}
                      </div>
                      <div className="fusion-desc">{f.description}</div>
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
                    ocr:      "[OCR]",
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
                          ocr:      "Checks: OCR confidence, text mismatch",
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
          )}

          {/* ── Tab: Layers ── */}
          {activeTab === "layers" && (
            <div className="tab-content">
              {/* Suspicious lines */}
              {result.suspicious_lines?.length > 0 && (
                <div className="section">
                  <div className="section-title">
                    🔴 Suspicious Lines — Content Layer
                    <span className="badge">{result.suspicious_lines.length}</span>
                  </div>
                  {result.suspicious_lines.map((sl, i) => (
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
              {result.numeric_anomalies?.length > 0 && (
                <div className="section">
                  <div className="section-title">
                    🟡 Numeric Anomalies — Numeric Layer
                    <span className="badge">{result.numeric_anomalies.length}</span>
                  </div>
                  {result.numeric_anomalies.map((na, i) => (
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

              {/* OCR word anomalies */}
              {result.ocr_word_anomalies?.length > 0 && (
                <div className="section">
                  <div className="section-title">
                    🔤 OCR Word Anomalies
                    <span className="badge">{result.ocr_word_anomalies.length}</span>
                  </div>
                  <div className="ocr-stats">
                    Avg font size: {result.ocr_stats?.avg_font_size}pt |{" "}
                    Avg brightness: {result.ocr_stats?.avg_color_brightness?.toFixed(0)} |{" "}
                    Words analyzed: {result.ocr_stats?.word_count}
                  </div>
                  {result.ocr_word_anomalies.map((wa, i) => (
                    <div key={i} className="finding-card"
                      style={{borderLeft: `3px solid ${
                        wa.anomaly_types.includes("color") ? "#ff00cc" : "#ff6400"
                      }`}}>
                      <div className="finding-header">
                        Page {wa.page} |
                        {wa.anomaly_types.includes("size") && " 📏 Size anomaly"}
                        {wa.anomaly_types.includes("color") && " 🎨 Color anomaly"}
                        {wa.anomaly_types.includes("confidence") && " ❓ Low confidence"}
                      </div>
                      <div className="finding-text">"{wa.word}"</div>
                      <div className="finding-reason">{wa.reason}</div>
                    </div>
                  ))}
                </div>
              )}

              {result.suspicious_lines?.length === 0 &&
               result.numeric_anomalies?.length === 0 &&
               result.ocr_word_anomalies?.length === 0 && (
                <div className="empty-state">
                  ✅ No specific line-level anomalies detected
                </div>
              )}
            </div>
          )}

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

              {/* Legend */}
              <div className="legend">
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#ff4444" }} />
                  Red = Font/spacing anomaly (Content layer)
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#ff00c8" }} />
                  Magenta = Color/ink anomaly (suspicious edit)
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#ffdd00" }} />
                  Yellow = Numeric value outlier
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#00ccff" }} />
                  Cyan = Hidden overlay detected
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: "#ffc800" }} />
                  Gold = Ghost text / overlapping layers
                </span>
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

              {/* Annotated image */}
              <div className="image-container" style={{ overflowX: "auto" }}>
                <img
                  src={imageUrl}
                  alt={`Page ${activePage} annotated`}
                  className="annotated-image"
                  style={{ width: `${zoom}%`, maxWidth: "none" }}
                  onError={() => {
                    setResult(null);
                    setError(
                      "Annotated image not found — the server may have restarted " +
                      "and lost this session's results. Please re-upload your PDF and analyze again."
                    );
                  }}
                />
              </div>
            </div>
          )}

          {/* ── Tab: Signals ── */}
          {activeTab === "signals" && (
            <div className="tab-content">
              <div className="section-title">
                ⚡ All Forensic Signals
                <span className="badge">{result.signals.length}</span>
              </div>
              {result.signals.map((sig, i) => {
                const prefix = Object.keys(SIGNAL_COLORS).find(p => sig.includes(p));
                const color = SIGNAL_COLORS[prefix] || "#888";
                const isAnomaly = sig.includes("⚡") ||
                  (!sig.includes("✓") && !sig.includes("passed") &&
                   !sig.includes("skipped") && !sig.includes("consistent"));
                return (
                  <div key={i} className="signal-row"
                    style={{ borderLeftColor: color }}>
                    <span className="signal-prefix" style={{ color }}>
                      {prefix || ""}
                    </span>
                    <span className="signal-text"
                      style={{ color: isAnomaly ? "#fff" : "#888" }}>
                      {sig.replace(prefix || "", "").trim()}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* ── Tab: Metadata ── */}
          {activeTab === "metadata" && result.metadata && (
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
          )}

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

          {/* ── AI Review (Gemini) — opt-in, supplementary, never part of the
              core 6-layer engine. Never runs automatically; only fires on
              click. Never touches result.verdict / result.combined_score. ── */}
          {!aiReview && !aiReviewLoading && (
            <div
              onClick={requestAiReview}
              style={{
                marginTop: 32,
                border: "1px dashed #cbd5e1",
                borderRadius: 10,
                padding: "18px 20px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                cursor: "pointer",
                background: "#f8fafc",
                transition: "background 0.15s, border-color 0.15s",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = "#f1f5f9"; e.currentTarget.style.borderColor = "#94a3b8"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "#f8fafc"; e.currentTarget.style.borderColor = "#cbd5e1"; }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: 15, color: "#1e293b" }}>
                  🤖 Ask AI — Get AI Explanation
                </div>
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>
                  Optional, supplementary Gemini review — explains this result in plain
                  English and computes a separate, clearly-labeled AI-adjusted score
                  (Layer 7). Never overwrites the deterministic score/verdict above.
                </div>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); requestAiReview(); }}
                style={{
                  background: "#1e293b", color: "#fff", border: "none",
                  borderRadius: 8, padding: "10px 20px", fontWeight: 700,
                  fontSize: 13, cursor: "pointer", whiteSpace: "nowrap",
                }}
              >
                ▶ Ask AI
              </button>
            </div>
          )}

          {aiReviewLoading && (
            <div style={{
              marginTop: 32, border: "1px dashed #cbd5e1", borderRadius: 10,
              padding: "18px 20px", textAlign: "center", color: "#64748b",
              background: "#f8fafc", fontSize: 13,
            }}>
              ⏳ Asking Gemini for a plain-English explanation, region review, and independent page scan…
            </div>
          )}

          {aiReviewError && !aiReviewLoading && (
            <div style={{
              marginTop: 32, border: "1px solid #fca5a5", borderRadius: 10,
              padding: "16px 20px", background: "#fef2f2", color: "#991b1b", fontSize: 13,
            }}>
              ❌ AI review failed: {aiReviewError}
              <button
                onClick={requestAiReview}
                style={{
                  marginLeft: 12, background: "#fff", color: "#991b1b",
                  border: "1px solid #fca5a5", borderRadius: 6,
                  padding: "4px 10px", fontSize: 12, fontWeight: 600, cursor: "pointer",
                }}
              >
                Retry
              </button>
            </div>
          )}

          {aiReview && !aiReviewLoading && (
            <div style={{
              marginTop: 32,
              border: "2px solid #6366f1",
              borderRadius: 12,
              overflow: "hidden",
              boxShadow: "0 4px 12px rgba(99,102,241,0.15)",
            }}>
              <div style={{
                background: "#312e81", color: "#fff", padding: "14px 20px",
                display: "flex", alignItems: "center", gap: 10,
              }}>
                <span style={{ fontSize: 20 }}>🤖</span>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 15 }}>AI REVIEW (Gemini) — Layer 7</div>
                  <div style={{ fontSize: 11.5, opacity: 0.85 }}>
                    AI-generated · contributes a separate, clearly-labeled AI-adjusted score ·
                    never overwrites the deterministic score/verdict above
                    {aiReview.from_cache && " · cached result (not a fresh Gemini call)"}
                  </div>
                </div>
              </div>

              <div style={{ padding: "18px 20px" }}>
                {!aiReview.available ? (
                  <div style={{ color: "#64748b", fontSize: 13 }}>
                    ℹ️ {aiReview.reason || "AI Review is not available in this environment."}
                  </div>
                ) : (
                  <>
                    {aiReview.ai_disagreement_flag && (
                      <div style={{
                        border: "1px solid #fca5a5", background: "#fef2f2", color: "#991b1b",
                        borderRadius: 8, padding: "12px 14px", marginBottom: 16, fontSize: 13, fontWeight: 600,
                      }}>
                        {aiReview.ai_disagreement_message}
                      </div>
                    )}

                    {aiReview.combined_score_with_ai != null && (
                      <div style={{
                        display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap",
                      }}>
                        <div style={{
                          flex: "1 1 160px", border: "1px solid #e5e7eb", borderRadius: 8,
                          padding: "10px 14px", background: "#f8fafc",
                        }}>
                          <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700 }}>
                            DETERMINISTIC (6 layers)
                          </div>
                          <div style={{ fontSize: 22, fontWeight: 800, color: "#1e293b" }}>
                            {aiReview.combined_score.toFixed(1)}
                          </div>
                        </div>
                        <div style={{
                          flex: "1 1 160px", border: "1px solid #c7d2fe", borderRadius: 8,
                          padding: "10px 14px", background: "#eef2ff",
                        }}>
                          <div style={{ fontSize: 11, color: "#4338ca", fontWeight: 700 }}>
                            WITH AI REVIEW (Layer 7, weight {aiReview.layer7_weight})
                          </div>
                          <div style={{ fontSize: 22, fontWeight: 800, color: "#312e81" }}>
                            {aiReview.combined_score_with_ai.toFixed(1)}
                          </div>
                        </div>
                        <div style={{
                          flex: "1 1 160px", border: "1px solid #e5e7eb", borderRadius: 8,
                          padding: "10px 14px", background: "#f8fafc",
                        }}>
                          <div style={{ fontSize: 11, color: "#64748b", fontWeight: 700 }}>
                            LAYER 7 AI ANOMALY SCORE
                          </div>
                          <div style={{ fontSize: 22, fontWeight: 800, color: "#1e293b" }}>
                            {aiReview.layer7_score}
                          </div>
                        </div>
                      </div>
                    )}

                    <div style={{ fontWeight: 700, fontSize: 13, color: "#312e81", marginBottom: 8 }}>
                      Plain-English Explanation
                    </div>
                    {aiReview.explanation ? (
                      <div style={{ fontSize: 13.5, color: "#334155", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                        {aiReview.explanation}
                      </div>
                    ) : (
                      <div style={{ fontSize: 13, color: "#991b1b" }}>
                        ⚠️ {aiReview.explanation_error || "No explanation was returned."}
                      </div>
                    )}

                    {aiReview.regions && aiReview.regions.length > 0 && (
                      <>
                        <div style={{ fontWeight: 700, fontSize: 13, color: "#312e81", marginTop: 20, marginBottom: 8 }}>
                          Flagged-Region Visual Check ({aiReview.regions.length})
                        </div>
                        {aiReview.regions.map((r, i) => {
                          const labelColor = r.label === "possible-edit" ? "#dc2626"
                                           : r.label === "template-element" ? "#16a34a"
                                           : r.label === "unavailable" ? "#b45309"
                                           : r.label === "error" ? "#b45309"
                                           : "#64748b";
                          return (
                            <div key={i} style={{
                              border: "1px solid #e5e7eb", borderRadius: 8,
                              padding: "10px 14px", marginBottom: 8, fontSize: 12.5,
                            }}>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                <span style={{ color: "#475569" }}>
                                  Page {r.page} · engine layer: {r.source_layer}
                                </span>
                                <span style={{ color: labelColor, fontWeight: 700 }}>
                                  {r.label}
                                </span>
                              </div>
                              <div style={{ color: "#94a3b8", marginBottom: 4 }}>
                                Engine finding: {r.engine_description}
                              </div>
                              <div style={{ color: "#334155" }}>
                                Gemini: {r.reasoning}
                              </div>
                            </div>
                          );
                        })}
                      </>
                    )}
                    {aiReview.regions_error && (
                      <div style={{ fontSize: 12, color: "#991b1b", marginTop: 10 }}>
                        ⚠️ {aiReview.regions_error}
                      </div>
                    )}

                    {/* ── Job C — cross-examination: per-finding verification of the
                        engine's OWN findings, visually distinct from Job B's
                        region-crop review above ── */}
                    {aiReview.per_finding_verification && aiReview.per_finding_verification.length > 0 && (
                      <>
                        <div style={{
                          fontWeight: 700, fontSize: 13, color: "#7c2d12",
                          marginTop: 24, marginBottom: 8, paddingTop: 16,
                          borderTop: "1px dashed #cbd5e1",
                        }}>
                          🔍 Cross-Examination — Verifying the Engine's Own Findings
                          ({aiReview.per_finding_verification.length})
                        </div>
                        {aiReview.per_finding_verification.map((v, i) => {
                          const verdictColor = v.gemini_verdict === "contradicted" ? "#16a34a"
                                             : v.gemini_verdict === "supported" ? "#dc2626"
                                             : "#64748b";
                          return (
                            <div key={i} style={{
                              border: "1px solid #e5e7eb", borderRadius: 8,
                              padding: "10px 14px", marginBottom: 8, fontSize: 12.5,
                            }}>
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                <span style={{ color: "#475569" }}>layer: {v.layer}</span>
                                <span style={{ color: verdictColor, fontWeight: 700 }}>
                                  {v.gemini_verdict.toUpperCase()}
                                </span>
                              </div>
                              <div style={{ color: "#94a3b8", marginBottom: 4 }}>
                                Engine claimed: {v.engine_finding}
                              </div>
                              <div style={{ color: "#334155" }}>Gemini: {v.reasoning}</div>
                            </div>
                          );
                        })}
                      </>
                    )}

                    {aiReview.additional_findings && aiReview.additional_findings.length > 0 && (
                      <>
                        <div style={{
                          fontWeight: 700, fontSize: 13, color: "#7c2d12",
                          marginTop: 20, marginBottom: 8,
                        }}>
                          ➕ Additional Findings — Not Flagged by the 6-Layer Engine
                          ({aiReview.additional_findings.length})
                        </div>
                        {aiReview.additional_findings.map((f, i) => (
                          <div key={i} style={{
                            border: "1px solid #fdba74", background: "#fff7ed",
                            borderRadius: 8, padding: "10px 14px", marginBottom: 8, fontSize: 12.5,
                          }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                              <span style={{ color: "#475569" }}>Page {f.page} · confidence: {f.confidence}</span>
                              <span style={{ color: "#c2410c", fontWeight: 700, fontSize: 11.5 }}>
                                AI-ONLY
                              </span>
                            </div>
                            <div style={{ color: "#334155" }}>{f.description}</div>
                          </div>
                        ))}
                      </>
                    )}

                    {aiReview.overall_assessment && (
                      <div style={{
                        marginTop: 20, border: "1px solid #c7d2fe", borderRadius: 8,
                        padding: "12px 14px", background: "#eef2ff", fontSize: 12.5,
                      }}>
                        <div style={{ fontWeight: 700, color: "#312e81", marginBottom: 4 }}>
                          Overall AI Assessment: {
                            aiReview.overall_assessment.agrees_with_engine_verdict === true ? "Agrees with verdict"
                            : aiReview.overall_assessment.agrees_with_engine_verdict === false ? "Disagrees with verdict"
                            : "Inconclusive"
                          }
                        </div>
                        <div style={{ color: "#334155" }}>{aiReview.overall_assessment.reasoning}</div>
                      </div>
                    )}
                    {aiReview.job_c_error && (
                      <div style={{ fontSize: 12, color: "#991b1b", marginTop: 10 }}>
                        ⚠️ {aiReview.job_c_error}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
