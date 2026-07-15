// Shared UI constants: backend base URL, per-layer labels and colors.

export const API = "http://localhost:8000";

export const LAYER_LABELS = {
  metadata: "Layer 1 — Metadata",
  content:  "Layer 2 — Content",
  numeric:  "Layer 3 — Numeric",
  ela:      "Layer 4 — ELA",
  pymupdf:  "Layer 5 — PyMuPDF",
};

export const SIGNAL_COLORS = {
  // [INCREMENTAL] is checked before [ELA] since it appears NESTED inside an
  // "[ELA]      [INCREMENTAL] ..." signal string — Object.keys(...).find()
  // below takes the first key whose substring matches, so the more specific
  // inner prefix has to come first or it'd always resolve to ELA's color.
  "[INCREMENTAL]": "#00e0d0",
  "[METADATA]": "#ff9944",
  "[CONTENT]":  "#ff4466",
  "[NUMERIC]":  "#ffdd00",
  "[ELA]":      "#cc44ff",
  "[PYMUPDF]":  "#00ffcc",
  "[TEXT_STACKING]": "#ff00ff",
};

// ── Image-document pipeline report (POST /analyze-image) ──────────────────
// Card accent colors match the backend's annotated-overlay box colors per
// evidence check (api/image_analysis_routes.py _CHECK_COLORS, BGR → hex).
export const IMAGE_CHECK_STYLE = {
  check1_local_variance: { color: "#ff4444", label: "Inpaint smoothing" },
  check5_edge_sharpness: { color: "#ffa500", label: "Overlay text edges" },
  check6_copy_move:      { color: "#ff00ff", label: "Copy-move clone" },
  check8_stamp_texture:  { color: "#4488ff", label: "Flat ink fill" },
  check9_stamp_boundary: { color: "#ffdd00", label: "Cutout boundary" },
};

// "not_applicable" is deliberately NOT rendered like a clean result: the
// engine's honesty design distinguishes "checked and found nothing" from
// "this technique had nothing it could check" (e.g. a PNG with no JPEG
// history to examine).
export const COMPRESSION_STYLE = {
  single_compression:           { text: "Single compression — no resave detected", color: "#00cc66", checked: true },
  double_compression_suspected: { text: "Double compression suspected — image was re-saved", color: "#ff4444", checked: true },
  uncertain:                    { text: "Uncertain — signal too weak to call", color: "#ffaa00", checked: true },
  not_applicable:               { text: "Couldn't check — no JPEG compression history in this container", color: "#8890a0", checked: false },
};
