// Builds the downloadable PDF forensic report with jsPDF. Pure output
// module: takes the analysis result + hidden-text report, saves the PDF.

import jsPDF from "jspdf";
import { applyPlugin } from "jspdf-autotable";

import { API } from "./constants";

applyPlugin(jsPDF);

export async function buildForensicReportPdf(result, hiddenTextForReport) {
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
    doc.text("Fully deterministic: no AI/ML - pure statistical and structural analysis", pageW / 2, 234, { align: "center" });

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
      numeric:  "Layer 3 - Numeric Outlier Detection",
      ela:      "Layer 4 - ELA Visual Analysis",
      pymupdf:  "Layer 5 - PyMuPDF Deep Analysis",
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
      "METHODOLOGY: The engine uses 5 independent detection layers:",
      "  Layer 1 - Metadata Analysis: Checks producer, creator, timestamps, XMP consistency",
      "  Layer 2 - Content Analysis: Analyzes font consistency, CIDFont sessions, color per line",
      "  Layer 3 - Numeric Analysis: Statistical outlier detection using z-scores",
      "  Layer 4 - ELA Analysis: Error Level Analysis at multiple DPI scales",
      "  Layer 5 - PyMuPDF Analysis: Hidden overlays, ghost text, character spacing",
      "",
      "LIMITATIONS:",
      "  - This engine cannot guarantee 100% accuracy on all document types",
      "  - Print-scan-retype attacks may not be detectable",
      "  - Same-font, same-color edits may not produce statistical anomalies",
      "  - Results should be verified by a qualified document examiner",
      "  - This report is for investigative purposes only",
      "",
      "NO AI/ML: This engine is fully deterministic. All layers use",
      "statistical and structural analysis only — no machine learning models",
      "or AI systems participate anywhere in the analysis, the verdict, or",
      "the combined_score.",
    ];

    disclaimer.forEach(line => {
      addText(line, 8.5, line.startsWith("Layer") || line.startsWith("LIMITATIONS") || line.startsWith("METHODOLOGY") || line.startsWith("NO AI"), [50, 50, 50]);
    });

    // ── Save ──────────────────────────────────────────────────
    const safeName = result.filename.replace(/[^a-zA-Z0-9._-]/g, "_");
    doc.save(`ForensicReport_${safeName}_${Date.now()}.pdf`);
}
