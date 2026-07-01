import { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";

const API = "http://localhost:8000";
const RENDER_SCALE = 150 / 72;

const COLORS = {
  deposit:    "#3b82f6",
  withdrawal: "#ef4444",
  balance:    "#d97706",
  end:        "#8b5cf6",
};

const META = {
  deposit:    { icon: "📥", label: "Deposit / Credit (Cr.)" },
  withdrawal: { icon: "📤", label: "Withdrawal / Debit (Dr.)" },
  balance:    { icon: "⚖️",  label: "Balance" },
  end:        { icon: "🏁", label: "End of Table" },
};

const OPERATIONS = [
  { key: "+-", label: "Dep − Wdl" },
  { key: "+",  label: "A + B"     },
  { key: "-",  label: "A − B"     },
  { key: "*",  label: "A × B"     },
  { key: "/",  label: "A ÷ B"     },
];

function fmtNum(val) {
  if (val === null || val === undefined) return "—";
  const n = Number(val);
  if (isNaN(n)) return String(val);
  if (Math.abs(n) >= 1_000_000)
    return "₹" + (n / 1_000_000).toFixed(2) + "M";
  if (Math.abs(n) >= 1_000)
    return n.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return n.toFixed(2);
}

const KEYFRAMES = `
  @keyframes fc-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.45); }
    50%       { box-shadow: 0 0 0 7px rgba(99,102,241,0); }
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  @keyframes flashRed {
    0%   { background: #fca5a5; }
    100% { background: #fef2f2; }
  }
  @keyframes rowPulse {
    0%, 100% { opacity: 0.7; }
    50%       { opacity: 1; }
  }
  @keyframes progressGlow {
    0%, 100% { box-shadow: 0 0 4px rgba(34,197,94,0.4); }
    50%       { box-shadow: 0 0 10px rgba(34,197,94,0.8); }
  }
  .fc-sel-active { animation: fc-pulse 1.4s ease-in-out infinite; }
  .fc-row-new    { animation: slideIn 0.2s ease-out forwards; }
  .fc-row-flash  { animation: flashRed 0.6s ease-out forwards; }
  .fc-row-active { animation: rowPulse 0.9s ease-in-out infinite; }
`;

export default function ForensicCalculator({ fileId, result }) {
  // Column selection
  const [activeSelection, setActiveSelection] = useState(null);
  const [selectedCols, setSelectedCols] = useState({ deposit: null, withdrawal: null, balance: null });
  const [operation, setOperation]         = useState("+-");
  const [openingOverride, setOpeningOverride] = useState("");
  const [showOpeningOverride, setShowOpeningOverride] = useState(false);
  const [tolerance, setTolerance]         = useState("1.0");

  // "End of table" marker — { page (1-indexed), y (PDF points) }. Rows at
  // or after this point are excluded from the calculation on the backend.
  const [endMarker, setEndMarker]         = useState(null);
  // Which marker (deposit/withdrawal/balance/end) is currently being
  // dragged, so a window-level mousemove listener can live-update it.
  const [draggingType, setDraggingType]   = useState(null);
  // Vertical position of the Cr./Dr./Balance handles — { page, y (PDF pts) }
  // per marker. X position still determines which detected column is
  // selected; Y position (this) marks WHERE the table's first row is —
  // the earliest of the three placed handles becomes the "start of table"
  // boundary sent to the backend, so header/title rows above it get
  // excluded and the marked row's own balance anchors the calculation.
  const [handlePos, setHandlePos]         = useState({
    deposit:    { page: 1, y: 40 },
    withdrawal: { page: 1, y: 40 },
    balance:    { page: 1, y: 40 },
  });

  // API data
  const [columns, setColumns]             = useState([]);
  const [colsLoading, setColsLoading]     = useState(false);
  const [colsError, setColsError]         = useState(null);
  const [pageWidthPts, setPageWidthPts]   = useState(595);

  // Streaming state
  const [phase, setPhase]                 = useState("selector"); // selector | streaming | done
  const [streamRows, setStreamRows]       = useState([]);
  const [streamProgress, setStreamProgress] = useState({ current: 0, total: 0 });
  const [streamSummary, setStreamSummary] = useState(null);
  const [streamError, setStreamError]     = useState(null);
  const [activeRowY, setActiveRowY]       = useState(null);
  const [activeRowPage, setActiveRowPage] = useState(null);
  const [hasMismatch, setHasMismatch]     = useState(false);
  const [mismatchAlert, setMismatchAlert] = useState(null);
  const [flashRowNum, setFlashRowNum]     = useState(null);

  // Table filter/sort
  const [filterMode, setFilterMode]       = useState("all"); // all | ok | mismatch
  const [sortCol, setSortCol]             = useState(null);
  const [sortDir, setSortDir]             = useState("asc");
  const [hoveredRow, setHoveredRow]       = useState(null);
  const [selectedRow, setSelectedRow]     = useState(null);

  // Image display
  const [imageNaturalSize, setImageNaturalSize] = useState({ w: 0, h: 0 });
  const [displayedPage, setDisplayedPage] = useState(1);
  const imageContainerRef = useRef(null);
  const tableBodyRef = useRef(null);
  const esRef = useRef(null);

  const pageImageUrl = `${API}/annotated-image/${fileId}?page=${displayedPage}`;

  // Load columns
  useEffect(() => {
    if (!fileId) return;
    setColsLoading(true);
    setColsError(null);
    axios.get(`${API}/calculator/columns/${fileId}`)
      .then((res) => {
        const cols = res.data.columns || [];
        setColumns(cols);
        const balIdx  = cols.findIndex((c) => c.likely_type === "balance");
        const txnIdxs = cols.map((c, i) => c.likely_type === "transaction" ? i : -1).filter(i => i >= 0);
        setSelectedCols({
          deposit:    txnIdxs[0] >= 0 ? cols[txnIdxs[0]].col_index : null,
          withdrawal: txnIdxs[1] >= 0 ? cols[txnIdxs[1]].col_index : null,
          balance:    balIdx >= 0 ? cols[balIdx].col_index : null,
        });
      })
      .catch((err) => setColsError(err.response?.data?.detail || "Failed to load columns"))
      .finally(() => setColsLoading(false));
  }, [fileId]);

  // Cleanup SSE on unmount
  useEffect(() => () => esRef.current?.close(), []);

  const handleImageLoad = useCallback((e) => {
    const { naturalWidth: w, naturalHeight: h } = e.target;
    if (w > 0) {
      setPageWidthPts(w / RENDER_SCALE);
      setImageNaturalSize({ w, h });
    }
  }, []);

  // PDF-point page height, derived from the loaded image's natural size —
  // needed to convert a click's Y-fraction into a Y position in points.
  const pageHeightPts = imageNaturalSize.h ? imageNaturalSize.h / RENDER_SCALE : 842;

  const handleImageClick = useCallback((e) => {
    if (!activeSelection) return;

    const rect = e.currentTarget.getBoundingClientRect();

    if (activeSelection === "end") {
      const yFraction = (e.clientY - rect.top) / rect.height;
      const yPts = Math.max(0, Math.min(pageHeightPts, yFraction * pageHeightPts));
      setEndMarker({ page: displayedPage, y: Math.round(yPts * 10) / 10 });
      setActiveSelection(null);
      return;
    }

    if (columns.length === 0) return;
    const fraction = (e.clientX - rect.left) / rect.width;
    const clickXPts = fraction * pageWidthPts;
    const yFraction  = (e.clientY - rect.top) / rect.height;
    const clickYPts  = Math.max(0, Math.min(pageHeightPts, yFraction * pageHeightPts));

    const nearest = columns.reduce((best, col) =>
      Math.abs(col.x_center - clickXPts) < Math.abs(best.x_center - clickXPts) ? col : best
    );

    const newCols = { ...selectedCols, [activeSelection]: nearest.col_index };
    setSelectedCols(newCols);
    setHandlePos((prev) => ({
      ...prev,
      [activeSelection]: { page: displayedPage, y: Math.round(clickYPts * 10) / 10 },
    }));

    if (activeSelection === "deposit" && newCols.withdrawal === null) {
      setActiveSelection("withdrawal");
    } else if (activeSelection === "withdrawal" && newCols.balance === null) {
      setActiveSelection("balance");
    } else {
      setActiveSelection(null);
    }
  }, [activeSelection, columns, pageWidthPts, pageHeightPts, selectedCols, displayedPage]);

  // Live drag — once a marker handle is grabbed (mousedown sets
  // draggingType), track window-level mouse movement so the column
  // assignment / end-of-table Y position follows the pointer until
  // mouseup, instead of requiring a fresh click each time.
  useEffect(() => {
    if (!draggingType || !imageContainerRef.current) return;

    const handleMove = (e) => {
      const rect = imageContainerRef.current.getBoundingClientRect();

      if (draggingType === "end") {
        const yFraction = (e.clientY - rect.top) / rect.height;
        const yPts = Math.max(0, Math.min(pageHeightPts, yFraction * pageHeightPts));
        setEndMarker((prev) => ({ page: prev?.page ?? displayedPage, y: Math.round(yPts * 10) / 10 }));
        return;
      }

      // Vertical position always follows the pointer — this is what marks
      // where the table's first row is (see handlePos comment above).
      const yFraction = (e.clientY - rect.top) / rect.height;
      const yPts = Math.max(0, Math.min(pageHeightPts, yFraction * pageHeightPts));
      setHandlePos((prev) => ({
        ...prev,
        [draggingType]: { page: displayedPage, y: Math.round(yPts * 10) / 10 },
      }));

      if (columns.length === 0) return;
      const fraction = (e.clientX - rect.left) / rect.width;
      const clickXPts = Math.max(0, Math.min(pageWidthPts, fraction * pageWidthPts));
      const nearest = columns.reduce((best, col) =>
        Math.abs(col.x_center - clickXPts) < Math.abs(best.x_center - clickXPts) ? col : best
      );
      setSelectedCols((prev) => ({ ...prev, [draggingType]: nearest.col_index }));
    };

    const handleUp = () => setDraggingType(null);

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [draggingType, columns, pageWidthPts, pageHeightPts, displayedPage]);

  // Start SSE stream
  const startStream = useCallback(() => {
    const { deposit, withdrawal, balance } = selectedCols;
    if (balance === null) return;

    esRef.current?.close();
    setPhase("streaming");
    setStreamRows([]);
    setStreamProgress({ current: 0, total: 0 });
    setStreamSummary(null);
    setStreamError(null);
    setHasMismatch(false);
    setMismatchAlert(null);
    setActiveRowY(null);
    setFlashRowNum(null);
    setFilterMode("all");
    setSortCol(null);

    const body = {
      col_a_index:       deposit !== null ? deposit : balance,
      col_b_index:       withdrawal !== null ? withdrawal : balance,
      operation,
      balance_col_index: balance,
      starting_balance:  openingOverride !== "" ? parseFloat(openingOverride) : null,
      tolerance:         parseFloat(tolerance) || 1.0,
      page_filter:       null,
      end_page:          endMarker?.page ?? null,
      end_y:             endMarker?.y ?? null,
      start_page:        startMarker?.page ?? null,
      start_y:           startMarker?.y ?? null,
    };

    // SSE via fetch (EventSource doesn't support POST)
    fetch(`${API}/calculator/run-stream/${fileId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((res) => {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      function pump() {
        reader.read().then(({ done, value }) => {
          if (done) return;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop();
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const evt = JSON.parse(line.slice(6));
              if (evt.type === "row") {
                const row = evt.data;
                setStreamProgress({ current: evt.current, total: evt.total });
                setActiveRowY(row.y_position);
                setActiveRowPage(row.page);
                setDisplayedPage(row.page);
                setStreamRows((prev) => [...prev, { ...row, _new: true }]);
                if (row.is_mismatch) {
                  setHasMismatch(true);
                  setFlashRowNum(row.row_num);
                  setMismatchAlert(`⚠️ Mismatch at row ${row.row_num}!`);
                  setTimeout(() => setMismatchAlert(null), 2000);
                  setTimeout(() => setFlashRowNum(null), 700);
                }
              } else if (evt.type === "done") {
                setStreamSummary(evt.data);
                setPhase("done");
                setActiveRowY(null);
              } else if (evt.type === "error") {
                setStreamError(evt.data);
                setPhase("done");
              }
            } catch {}
          }
          pump();
        });
      }
      pump();
    }).catch((err) => {
      setStreamError(String(err));
      setPhase("done");
    });
  }, [fileId, selectedCols, operation, openingOverride, tolerance, endMarker, handlePos]);


  const exportCSV = useCallback(() => {
    if (!streamRows.length) return;
    const hdr = ["Row","Page","Col A","Col B","Delta","Expected","Printed","Diff","Severity"].join(",");
    const rows = streamRows.map((r) =>
      [r.row_num, r.page, r.val_a, r.val_b, r.delta, r.expected_balance, r.printed_balance ?? "", r.difference, r.severity].join(",")
    );
    const blob = new Blob([[hdr, ...rows].join("\n")], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    Object.assign(document.createElement("a"), { href: url, download: "forensic_calc_results.csv" }).click();
    URL.revokeObjectURL(url);
  }, [streamRows]);

  const getCol = (idx) => columns.find((c) => c.col_index === idx);
  const balCol = getCol(selectedCols.balance);
  const rowCount = balCol?.value_count ?? "?";
  const canRun = selectedCols.balance !== null;

  // Derived "start of table" marker — the earliest (topmost, then
  // lowest-page) of the three placed Cr./Dr./Balance handles. Sent to the
  // backend so rows above it (headers, titles) are excluded and that
  // row's own balance anchors the opening balance.
  const placedHandles = ["deposit", "withdrawal", "balance"].filter((t) => selectedCols[t] !== null);
  const startMarker = placedHandles.length > 0
    ? placedHandles
        .map((t) => handlePos[t])
        .reduce((earliest, h) =>
          (h.page < earliest.page || (h.page === earliest.page && h.y < earliest.y)) ? h : earliest
        )
    : null;

  // Column stripe positions
  const highlights = ["deposit", "withdrawal", "balance"].map((type) => {
    const idx = selectedCols[type];
    if (idx === null) return null;
    const col = getCol(idx);
    if (!col) return null;
    return { type, col, xPct: (col.x_center / pageWidthPts) * 100 };
  }).filter(Boolean);

  // Moving highlight box position on image
  const getRowHighlightStyle = useCallback((yPos, page) => {
    if (!imageContainerRef.current || !imageNaturalSize.h) return null;
    const containerH = imageContainerRef.current.offsetHeight;
    // yPos is in PDF points; convert to image pixel fraction
    // For multi-page PDFs the y is within-page, so show on correct page
    const pageHeightPts = imageNaturalSize.h / RENDER_SCALE;
    const yFraction = yPos / pageHeightPts;
    const topPx = yFraction * containerH;
    return {
      position: "absolute",
      left: 0,
      right: 0,
      top: Math.max(0, topPx - 12),
      height: 22,
      border: "2px solid #eab308",
      background: "rgba(254,252,232,0.35)",
      boxShadow: "0 0 8px 2px rgba(234,179,8,0.6)",
      pointerEvents: "none",
      transition: "top 0.15s ease-out",
      borderRadius: 2,
      zIndex: 10,
    };
  }, [imageNaturalSize, imageContainerRef]);

  // Sorted/filtered rows
  const displayRows = (() => {
    let rows = streamRows;
    if (filterMode === "ok")       rows = rows.filter((r) => !r.is_mismatch);
    if (filterMode === "mismatch") rows = rows.filter((r) => r.is_mismatch);
    if (sortCol) {
      rows = [...rows].sort((a, b) => {
        const av = a[sortCol] ?? 0;
        const bv = b[sortCol] ?? 0;
        return sortDir === "asc" ? av - bv : bv - av;
      });
    }
    return rows;
  })();

  const progressPct = streamProgress.total > 0
    ? Math.round((streamProgress.current / streamProgress.total) * 100)
    : 0;

  const progressColor = hasMismatch ? "#ef4444" : "#22c55e";

  const handleRowClick = (row) => {
    setSelectedRow(row.row_num === selectedRow?.row_num ? null : row);
    setDisplayedPage(row.page);
    setActiveRowY(row.y_position);
  };

  const toggleSort = (col) => {
    if (sortCol === col) {
      setSortDir((d) => d === "asc" ? "desc" : "asc");
    } else {
      setSortCol(col);
      setSortDir("asc");
    }
  };

  const rowHighlightStyle = (activeRowY && activeRowPage === displayedPage)
    ? getRowHighlightStyle(activeRowY, activeRowPage)
    : null;

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div style={{
      marginTop: 32,
      border: "1px solid #e2e8f0",
      borderRadius: 10,
      overflow: "hidden",
      boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
      background: "#fff",
    }}>
      <style>{KEYFRAMES}</style>

      {/* Header */}
      <div style={{
        background: "#1e293b", color: "#fff",
        padding: "14px 20px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: 0.4 }}>
          🧮 Forensic Calculator
        </span>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {phase !== "selector" && (
            <button onClick={() => { setPhase("selector"); setStreamRows([]); setStreamSummary(null); }}
              style={{ background: "#334155", border: "1px solid #475569", color: "#e2e8f0", borderRadius: 6, padding: "4px 12px", fontSize: 12, cursor: "pointer" }}>
              🔄 Start Over
            </button>
          )}
          <span style={{ fontSize: 12, color: "#94a3b8" }}>Running Balance Verifier</span>
        </div>
      </div>

      <div style={{ padding: 20 }}>

        {/* ── PHASE: selector ─────────────────────────────────────────────────── */}
        {phase === "selector" && (
          <>
            <p style={{ margin: "0 0 14px", fontSize: 13, color: "#64748b" }}>
              Click a marker, then click on the document image to place it.
              Drag a placed handle left/right to change its column, or up/down
              to the table's first row — the topmost Cr./Dr./Balance handle marks
              where the table starts, and its row's own balance anchors the calculation.
            </p>

            {/* 4 marker buttons: Deposit/Credit, Withdrawal/Debit, Balance, End of Table */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
              {["deposit", "withdrawal", "balance", "end"].map((type) => {
                const isActive = activeSelection === type;
                const isEnd    = type === "end";
                const colIdx   = isEnd ? null : selectedCols[type];
                const col      = colIdx !== null ? getCol(colIdx) : null;
                const isPlaced = isEnd ? !!endMarker : !!col;
                const color    = COLORS[type];
                const { icon, label } = META[type];
                return (
                  <button key={type}
                    className={isActive ? "fc-sel-active" : ""}
                    onClick={() => setActiveSelection(isActive ? null : type)}
                    style={{
                      position: "relative",
                      background: isActive ? color + "18" : isPlaced ? color + "0d" : "#f8fafc",
                      border: isActive ? `2px solid ${color}` : isPlaced ? `2px solid ${color}` : "2px dashed #cbd5e1",
                      borderRadius: 8, padding: "12px 10px", cursor: "pointer",
                      textAlign: "left", transition: "all 0.15s",
                    }}>
                    {isActive && (
                      <span style={{ position: "absolute", top: 7, right: 8, width: 8, height: 8, borderRadius: "50%", background: color, display: "block" }} />
                    )}
                    <div style={{ fontSize: 12, fontWeight: 700, color: isActive || isPlaced ? color : "#94a3b8", marginBottom: 4 }}>
                      {icon} {label}
                    </div>
                    {isEnd ? (
                      endMarker ? (
                        <div style={{ fontSize: 11, color: "#374151" }}>
                          Page {endMarker.page} · y={endMarker.y}
                          <span role="button" onClick={(e) => { e.stopPropagation(); setEndMarker(null); }}
                            style={{ marginLeft: 8, color: "#9ca3af", cursor: "pointer", fontWeight: 700 }}>✕</span>
                        </div>
                      ) : (
                        <div style={{ fontSize: 11, color: isActive ? color : "#9ca3af" }}>
                          {isActive ? "👆 Click the last row on the image" : "Optional — marks last row"}
                        </div>
                      )
                    ) : col ? (
                      <div style={{ fontSize: 11, color: "#374151" }}>
                        Column {colIdx} · {col.value_count} values
                        <span role="button" onClick={(e) => { e.stopPropagation(); setSelectedCols((p) => ({ ...p, [type]: null })); }}
                          style={{ marginLeft: 8, color: "#9ca3af", cursor: "pointer", fontWeight: 700 }}>✕</span>
                      </div>
                    ) : (
                      <div style={{ fontSize: 11, color: isActive ? color : "#9ca3af" }}>
                        {isActive ? "👆 Click on the image below" : "Click here, then click image"}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Page navigator — needed to place markers (especially End of
                Table) on documents where the table spans multiple pages. */}
            {(result?.total_pages ?? 1) > 1 && (
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10, fontSize: 12, color: "#475569" }}>
                <button
                  onClick={() => setDisplayedPage((p) => Math.max(1, p - 1))}
                  disabled={displayedPage <= 1}
                  style={{ background: "#f8fafc", border: "1px solid #d1d5db", borderRadius: 6, padding: "3px 10px", cursor: displayedPage <= 1 ? "not-allowed" : "pointer", opacity: displayedPage <= 1 ? 0.5 : 1 }}>
                  ◀ Prev
                </button>
                <span>Page {displayedPage} of {result.total_pages}</span>
                <button
                  onClick={() => setDisplayedPage((p) => Math.min(result.total_pages, p + 1))}
                  disabled={displayedPage >= result.total_pages}
                  style={{ background: "#f8fafc", border: "1px solid #d1d5db", borderRadius: 6, padding: "3px 10px", cursor: displayedPage >= result.total_pages ? "not-allowed" : "pointer", opacity: displayedPage >= result.total_pages ? 0.5 : 1 }}>
                  Next ▶
                </button>
              </div>
            )}

            {startMarker && (
              <div style={{
                display: "flex", alignItems: "center", gap: 8, marginBottom: 12,
                background: COLORS.balance + "0d", border: `1px solid ${COLORS.balance}55`,
                borderRadius: 6, padding: "6px 12px", fontSize: 12, color: "#78350f",
              }}>
                🚩 Table starts at Page {startMarker.page}, y={startMarker.y} — rows above this are ignored,
                and that row's own balance anchors the calculation.
              </div>
            )}

            {colsLoading && <p style={{ color: "#94a3b8", fontSize: 13, textAlign: "center", padding: "16px 0" }}>Detecting columns…</p>}
            {colsError && (
              <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 7, color: "#b91c1c", padding: "10px 14px", marginBottom: 12, fontSize: 13 }}>
                {colsError}
              </div>
            )}

            {/* Document image with stripes */}
            <div ref={imageContainerRef} onClick={handleImageClick}
              style={{
                position: "relative",
                border: activeSelection ? `2px solid ${COLORS[activeSelection]}` : "1px solid #e2e8f0",
                borderRadius: 8, overflow: "hidden",
                cursor: activeSelection ? "crosshair" : "default",
                transition: "border 0.2s", marginBottom: 16, userSelect: "none",
              }}>
              <img src={pageImageUrl} alt={`Document page ${displayedPage}`} onLoad={handleImageLoad} draggable={false}
                style={{ display: "block", width: "100%", pointerEvents: "none" }} />
              {highlights.map(({ type, xPct }) => (
                <div key={type} style={{
                  position: "absolute", top: 0, bottom: 0,
                  left: `${Math.max(0, xPct - 4)}%`, width: "8%",
                  backgroundColor: COLORS[type], opacity: 0.22, pointerEvents: "none",
                  borderLeft: `2px solid ${COLORS[type]}`, borderRight: `2px solid ${COLORS[type]}`,
                }} />
              ))}
              {/* Draggable handles — drag left/right to reassign which
                  column this marker points to; drag up/down to mark WHERE
                  the table's first row is (only shown on the page it was
                  placed on — same as the End-of-Table marker below). */}
              {highlights.filter(({ type }) => handlePos[type].page === displayedPage).map(({ type, xPct }) => (
                <div key={`handle-${type}`}
                  onMouseDown={(e) => { e.stopPropagation(); setDraggingType(type); }}
                  title={`Drag to move ${META[type].label}`}
                  style={{
                    position: "absolute", top: `${(handlePos[type].y / pageHeightPts) * 100}%`,
                    left: `${xPct}%`, transform: "translate(-50%, -50%)",
                    width: 22, height: 22, borderRadius: "50%",
                    background: COLORS[type], border: "2px solid #fff",
                    boxShadow: "0 1px 4px rgba(0,0,0,0.4)",
                    cursor: "move", zIndex: 5,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 11,
                  }}>
                  {META[type].icon}
                </div>
              ))}
              {/* End-of-table marker — full-width line + draggable handle,
                  only rendered on the page it was placed on. */}
              {endMarker && endMarker.page === displayedPage && imageNaturalSize.h > 0 && (() => {
                const topPct = (endMarker.y / pageHeightPts) * 100;
                return (
                  <>
                    <div style={{
                      position: "absolute", left: 0, right: 0,
                      top: `${topPct}%`, height: 0,
                      borderTop: `2px dashed ${COLORS.end}`, pointerEvents: "none", zIndex: 4,
                    }} />
                    <div
                      onMouseDown={(e) => { e.stopPropagation(); setDraggingType("end"); }}
                      title="Drag to move End of Table"
                      style={{
                        position: "absolute", top: `${topPct}%`, right: 8,
                        transform: "translateY(-50%)",
                        background: COLORS.end, color: "#fff",
                        borderRadius: 14, padding: "2px 10px", fontSize: 11, fontWeight: 700,
                        cursor: "ns-resize", zIndex: 5, whiteSpace: "nowrap",
                        boxShadow: "0 1px 4px rgba(0,0,0,0.4)",
                      }}>
                      🏁 End
                    </div>
                  </>
                );
              })()}
              {activeSelection && (
                <div style={{
                  position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)",
                  background: COLORS[activeSelection], color: "#fff",
                  padding: "5px 16px", borderRadius: 20, fontSize: 12, fontWeight: 700,
                  pointerEvents: "none", whiteSpace: "nowrap", boxShadow: "0 2px 8px rgba(0,0,0,0.25)",
                }}>
                  {activeSelection === "end"
                    ? "Click the last row of the table"
                    : `Click to mark ${META[activeSelection].label} column`}
                </div>
              )}
            </div>

            {/* Operation */}
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 12, color: "#6b7280", fontWeight: 600, marginBottom: 7 }}>Operation</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {OPERATIONS.map((op) => {
                  const active = operation === op.key;
                  return (
                    <button key={op.key} onClick={() => setOperation(op.key)} style={{
                      background: active ? "#1e293b" : "#f8fafc",
                      color: active ? "#fff" : "#374151",
                      border: `1px solid ${active ? "#1e293b" : "#d1d5db"}`,
                      borderRadius: 6, padding: "6px 14px", fontSize: 12,
                      fontWeight: active ? 700 : 400, cursor: "pointer", transition: "all 0.15s",
                    }}>
                      {op.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Opening balance + tolerance */}
            <div style={{ display: "flex", gap: 16, marginBottom: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
              <div style={{ fontSize: 13, color: "#475569" }}>
                Opening balance: <b>Auto-detect</b>
                <button onClick={() => setShowOpeningOverride((v) => !v)}
                  style={{ background: "none", border: "none", color: "#3b82f6", cursor: "pointer", fontSize: 12, marginLeft: 8, textDecoration: "underline" }}>
                  {showOpeningOverride ? "Hide ▲" : "Override ▼"}
                </button>
                {showOpeningOverride && (
                  <input type="number" placeholder="Enter opening balance" value={openingOverride}
                    onChange={(e) => setOpeningOverride(e.target.value)}
                    style={{ display: "block", marginTop: 8, padding: "7px 10px", border: "1px solid #d1d5db", borderRadius: 6, fontSize: 13, width: 220, background: "#f9fafb" }} />
                )}
              </div>
              <div style={{ fontSize: 13, color: "#475569" }}>
                Tolerance:
                <input type="number" step="0.01" min="0" value={tolerance}
                  onChange={(e) => setTolerance(e.target.value)}
                  style={{ marginLeft: 8, padding: "5px 8px", border: "1px solid #d1d5db", borderRadius: 6, fontSize: 13, width: 80, background: "#f9fafb" }} />
              </div>
            </div>

            {/* Run button */}
            <button onClick={startStream} disabled={!canRun}
              style={{
                background: canRun ? "#1e293b" : "#94a3b8", color: "#fff",
                border: "none", borderRadius: 8, padding: "12px 28px",
                fontWeight: 700, fontSize: 14, cursor: canRun ? "pointer" : "not-allowed",
                width: "100%", letterSpacing: 0.3, transition: "background 0.15s",
              }}>
              {canRun ? `▶ Verify ${rowCount} Rows` : "Select a balance column first"}
            </button>
          </>
        )}

        {/* ── PHASE: streaming + done ──────────────────────────────────────────── */}
        {(phase === "streaming" || phase === "done") && (
          <>
            {/* Progress bar */}
            {phase === "streaming" && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#64748b", marginBottom: 6 }}>
                  <span>⚡ Verifying arithmetic… Row {streamProgress.current} of {streamProgress.total}</span>
                  <span>{progressPct}%</span>
                </div>
                <div style={{ height: 8, background: "#f1f5f9", borderRadius: 4, overflow: "hidden" }}>
                  <div style={{
                    height: "100%", borderRadius: 4,
                    background: progressColor,
                    width: `${progressPct}%`,
                    transition: "width 0.1s ease-out, background 0.3s",
                    animation: phase === "streaming" ? "progressGlow 1.4s ease-in-out infinite" : "none",
                  }} />
                </div>
              </div>
            )}

            {/* Mismatch alert toast */}
            {mismatchAlert && (
              <div style={{
                position: "fixed", top: 24, right: 24, zIndex: 9999,
                background: "#dc2626", color: "#fff", padding: "10px 20px",
                borderRadius: 8, fontWeight: 700, fontSize: 13,
                boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
                animation: "slideIn 0.2s ease-out",
              }}>
                {mismatchAlert}
              </div>
            )}

            {streamError && (
              <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 7, color: "#b91c1c", padding: "10px 14px", marginBottom: 12, fontSize: 13 }}>
                {streamError}
              </div>
            )}

            {/* Split view: image + table */}
            <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>

              {/* Left: document image with row highlight */}
              <div style={{ flex: "0 0 45%", minWidth: 0 }}>
                <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 6 }}>
                  Page {displayedPage}
                </div>
                <div ref={imageContainerRef} style={{ position: "relative", border: "1px solid #e2e8f0", borderRadius: 8, overflow: "hidden" }}>
                  <img
                    src={pageImageUrl}
                    alt={`Page ${displayedPage}`}
                    onLoad={handleImageLoad}
                    draggable={false}
                    style={{ display: "block", width: "100%", pointerEvents: "none" }}
                  />
                  {/* Column stripes */}
                  {highlights.map(({ type, xPct }) => (
                    <div key={type} style={{
                      position: "absolute", top: 0, bottom: 0,
                      left: `${Math.max(0, xPct - 4)}%`, width: "8%",
                      backgroundColor: COLORS[type], opacity: 0.18, pointerEvents: "none",
                      borderLeft: `2px solid ${COLORS[type]}`, borderRight: `2px solid ${COLORS[type]}`,
                    }} />
                  ))}
                  {/* Moving row highlight */}
                  {rowHighlightStyle && <div style={rowHighlightStyle} />}
                  {/* Selected/hovered row highlight (permanent) */}
                  {selectedRow && selectedRow.page === displayedPage && (() => {
                    const s = getRowHighlightStyle(selectedRow.y_position, selectedRow.page);
                    if (!s) return null;
                    return <div style={{ ...s, border: "2px solid #dc2626", background: "rgba(254,226,226,0.4)", boxShadow: "0 0 8px 2px rgba(220,38,38,0.5)", transition: "none" }} />;
                  })()}
                </div>
              </div>

              {/* Right: results table */}
              <div style={{ flex: 1, minWidth: 0 }}>
                {/* Filter + export controls */}
                <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap", alignItems: "center" }}>
                  {[["all","All rows"],["ok","✅ OK"],["mismatch","🔴 Mismatches"]].map(([val, lbl]) => (
                    <button key={val} onClick={() => setFilterMode(val)} style={{
                      background: filterMode === val ? "#1e293b" : "#f8fafc",
                      color: filterMode === val ? "#fff" : "#374151",
                      border: `1px solid ${filterMode === val ? "#1e293b" : "#d1d5db"}`,
                      borderRadius: 6, padding: "5px 12px", fontSize: 11, fontWeight: 600, cursor: "pointer",
                    }}>{lbl}</button>
                  ))}
                  {phase === "done" && streamRows.length > 0 && (
                    <button onClick={exportCSV} style={{ marginLeft: "auto", background: "#fff", color: "#1e293b", border: "1px solid #cbd5e1", borderRadius: 6, padding: "5px 12px", fontSize: 11, fontWeight: 700, cursor: "pointer" }}>
                      📥 Export CSV
                    </button>
                  )}
                </div>

                <div style={{ overflowX: "auto", overflowY: "auto", maxHeight: 480, borderRadius: 7, border: "1px solid #e5e7eb" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 440 }}>
                    <thead>
                      <tr>
                        {[
                          ["row_num",          "#",       "left" ],
                          ["page",             "Pg",      "left" ],
                          ["val_a",            "Dep",     "right"],
                          ["val_b",            "Wdl",     "right"],
                          ["expected_balance", "Expected","right"],
                          ["printed_balance",  "Printed", "right"],
                          ["difference",       "Diff",    "right"],
                          [null,               "Status",  "left" ],
                        ].map(([col, hdr, align]) => (
                          <th key={hdr}
                            onClick={() => col && toggleSort(col)}
                            style={{
                              background: "#f1f5f9", padding: "7px 8px", textAlign: align,
                              fontWeight: 700, color: "#374151", borderBottom: "2px solid #e5e7eb",
                              whiteSpace: "nowrap", cursor: col ? "pointer" : "default",
                              userSelect: "none",
                            }}>
                            {hdr}{col && sortCol === col ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody ref={tableBodyRef}>
                      {displayRows.map((row, i) => {
                        const isFlash  = row.row_num === flashRowNum;
                        const isActive = phase === "streaming" && i === displayRows.length - 1;
                        const isSelected = selectedRow?.row_num === row.row_num;
                        const isHovered  = hoveredRow === row.row_num;
                        const rowBg =
                          isSelected ? "#fef9c3" :
                          isHovered  ? "#f8fafc" :
                          row.severity === "HIGH"   ? "#fef2f2" :
                          row.severity === "MEDIUM" ? "#fff7ed" :
                          i % 2 === 0 ? "#fff" : "#f9fafb";
                        const badge =
                          row.severity === "HIGH"
                            ? <span style={{ background: "#dc2626", color: "#fff", borderRadius: 4, padding: "1px 6px", fontSize: 10, fontWeight: 700 }}>HIGH</span>
                            : row.severity === "MEDIUM"
                            ? <span style={{ background: "#d97706", color: "#fff", borderRadius: 4, padding: "1px 6px", fontSize: 10, fontWeight: 700 }}>MED</span>
                            : <span style={{ background: "#d1fae5", color: "#065f46", borderRadius: 4, padding: "1px 6px", fontSize: 10 }}>OK</span>;
                        const td = (children, align = "right") => (
                          <td style={{ padding: "5px 8px", textAlign: align, fontFamily: "monospace", borderBottom: "1px solid #f3f4f6" }}>
                            {children}
                          </td>
                        );
                        return (
                          <tr key={row.row_num}
                            className={isFlash ? "fc-row-flash" : isActive ? "fc-row-active" : row._new ? "fc-row-new" : ""}
                            onClick={() => handleRowClick(row)}
                            onMouseEnter={() => { setHoveredRow(row.row_num); if (phase === "done") { setActiveRowY(row.y_position); setActiveRowPage(row.page); } }}
                            onMouseLeave={() => setHoveredRow(null)}
                            style={{ background: rowBg, cursor: "pointer", transition: "background 0.1s" }}>
                            {td(row.row_num, "left")}
                            {td(row.page, "left")}
                            {td(fmtNum(row.val_a))}
                            {td(fmtNum(row.val_b))}
                            {td(fmtNum(row.expected_balance))}
                            {td(fmtNum(row.printed_balance))}
                            {td(row.is_mismatch ? fmtNum(row.difference) : "—")}
                            <td style={{ padding: "5px 8px", textAlign: "left", borderBottom: "1px solid #f3f4f6" }}>{badge}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Row detail tooltip on selection */}
                {selectedRow && (
                  <div style={{
                    marginTop: 10, background: "#f8fafc", border: "1px solid #e2e8f0",
                    borderRadius: 7, padding: "10px 14px", fontSize: 12,
                  }}>
                    <b>Row {selectedRow.row_num} detail:</b>
                    <div style={{ marginTop: 6, display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px", fontFamily: "monospace" }}>
                      <span style={{ color: "#6b7280" }}>Expected:</span><span>{fmtNum(selectedRow.expected_balance)}</span>
                      <span style={{ color: "#6b7280" }}>Printed:</span><span>{fmtNum(selectedRow.printed_balance)}</span>
                      <span style={{ color: "#6b7280" }}>Difference:</span>
                      <span style={{ color: selectedRow.is_mismatch ? "#dc2626" : "#16a34a", fontWeight: 700 }}>
                        {selectedRow.is_mismatch ? fmtNum(selectedRow.difference) : "✓ Match"}
                      </span>
                      <span style={{ color: "#6b7280" }}>Page:</span><span>{selectedRow.page}</span>
                      <span style={{ color: "#6b7280" }}>Severity:</span><span>{selectedRow.severity}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ── Final summary card (done phase) ─────────────────────────────── */}
            {phase === "done" && streamSummary && (
              <div style={{ marginTop: 20 }}>
                <div style={{
                  border: `1px solid ${streamSummary.mismatch_count > 0 ? "#fca5a5" : "#86efac"}`,
                  borderRadius: 8,
                  background: streamSummary.mismatch_count > 0 ? "#fef2f2" : "#f0fdf4",
                  padding: "14px 18px",
                }}>
                  <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 10, color: streamSummary.mismatch_count > 0 ? "#b91c1c" : "#166534" }}>
                    {streamSummary.mismatch_count > 0 ? "⚠️ Verification Complete — Mismatches Found" : "✅ Verification Complete — All Rows Match"}
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 10 }}>
                    {[
                      { label: "Rows Checked",      value: streamSummary.total_rows },
                      { label: "Mismatches",         value: `${streamSummary.mismatch_count} / ${streamSummary.total_rows}` },
                      { label: "Opening Balance",    value: fmtNum(streamSummary.summary?.opening_balance), sub: streamSummary.opening_balance_method },
                      { label: "Expected Closing",   value: fmtNum(streamSummary.summary?.expected_closing) },
                      { label: "Printed Closing",    value: fmtNum(streamSummary.summary?.printed_closing) },
                      { label: "Closing Mismatch",   value: fmtNum(streamSummary.summary?.closing_mismatch) },
                    ].map(({ label, value, sub }) => (
                      <div key={label} style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 6, padding: "8px 12px", textAlign: "center" }}>
                        <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 3 }}>{label}</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: "#1e293b", fontFamily: "monospace" }}>{value}</div>
                        {sub && <div style={{ fontSize: 10, color: "#9ca3af", marginTop: 2 }}>{sub}</div>}
                      </div>
                    ))}
                  </div>

                  {streamSummary.opening_balance_anomaly && (
                    <div style={{ marginTop: 12, background: "#fff7ed", border: "1px solid #fb923c", borderRadius: 6, padding: "8px 12px", fontSize: 12, color: "#92400e" }}>
                      ⚠️ <b>Opening balance anomaly:</b> {streamSummary.opening_balance_anomaly_reason}
                    </div>
                  )}
                </div>

                {/* Mismatch detail list */}
                {streamSummary.mismatch_count > 0 && (
                  <div style={{ marginTop: 12, border: "1px solid #fca5a5", borderRadius: 8, overflow: "hidden" }}>
                    <div style={{ background: "#dc2626", color: "#fff", padding: "8px 14px", fontSize: 13, fontWeight: 700 }}>
                      🔴 {streamSummary.mismatch_count} Suspicious Row{streamSummary.mismatch_count !== 1 ? "s" : ""} Detected
                    </div>
                    <div style={{ maxHeight: 240, overflowY: "auto" }}>
                      {streamSummary.mismatch_rows.map((row) => (
                        <div key={row.row_num}
                          onClick={() => handleRowClick(row)}
                          style={{ padding: "10px 14px", borderBottom: "1px solid #fee2e2", cursor: "pointer", background: selectedRow?.row_num === row.row_num ? "#fef9c3" : "#fff" }}>
                          <div style={{ fontWeight: 700, fontSize: 12, color: "#b91c1c", marginBottom: 3 }}>
                            Row {row.row_num} (Page {row.page}) — {row.severity}
                          </div>
                          <div style={{ fontSize: 11, color: "#374151", fontFamily: "monospace" }}>
                            Expected: {fmtNum(row.expected_balance)} · Printed: {fmtNum(row.printed_balance)} · Diff: {fmtNum(row.difference)}
                          </div>
                          <div style={{ fontSize: 11, color: "#3b82f6", marginTop: 3 }}>Click to highlight in document ↑</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
