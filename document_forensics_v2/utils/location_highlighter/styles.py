"""Box colors, render DPI, and per-layer label helpers for the
annotated-page renderer."""


def _age_color_intensity(age_days: int) -> tuple:
    """
    Returns (red, green, blue) intensity multiplier based on edit age.
    Recent edits = brighter/redder. Old edits = faded.
    """
    if age_days is None:
        return (1.0, 1.0, 1.0)  # default full intensity
    if age_days == 0:
        return (1.0, 0.3, 0.3)  # bright red — edited today
    elif age_days < 7:
        return (1.0, 0.5, 0.3)  # orange-red — this week
    elif age_days < 30:
        return (1.0, 0.7, 0.4)  # orange — this month
    elif age_days < 180:
        return (0.9, 0.8, 0.5)  # yellow — within 6 months
    elif age_days < 365:
        return (0.8, 0.8, 0.6)  # faded yellow — this year
    else:
        return (0.6, 0.6, 0.6)  # gray — old edit


RENDER_DPI = 150  # page render resolution, matches the DPI used for box-coordinate scaling

# Box colors per signal source, RGB. Each layer gets a distinct color so a
# page with multiple anomaly types is still visually distinguishable.
# COLOR = which layer flagged it; the box LABEL states the specific finding
# (derived from the finding's own reason/type, not a generic layer name) —
# see the _*_label helpers below.
COLOR_CONTENT = (255, 50, 50)    # red    — font/spacing anomaly (content_analyzer)
COLOR_NUMERIC = (255, 220, 0)    # yellow — numeric outlier
COLOR_ELA     = (180, 0, 255)    # purple — ELA outlier (image edit)
COLOR_WHITE_RECT    = (0, 200, 255)   # cyan    — white rect overlay (pymupdf_analyzer)
COLOR_IMAGE_OVERLAY = (255, 0, 200)   # magenta — image overlay (pymupdf_analyzer)
COLOR_GHOST         = (255, 200, 0)   # gold    — ghost text / overlapping text layers (pymupdf_analyzer)
# Coordinate-collision text stacking (utils/hidden_text_extractor.detect_stacked_text)
# — 2+ DIFFERENT text values at the same coordinates. Drawn in bright magenta
# with a DASHED border so it reads as its own category even when it lands on
# top of pymupdf's gold "ghost text" box (the two frequently co-locate on a
# genuine paste-over) — the dash pattern lets the underlying box show through
# rather than one solidly hiding the other.
COLOR_TEXT_STACKING = (255, 0, 255)   # magenta (dashed) — hidden text stacking
# Embedded-image forensics (utils/embedded_image_forensics) — the image
# pipeline's checks run on an image OBJECT extracted from the PDF, with
# findings mapped into page space. Green: no other drawn layer uses it.
COLOR_EMBEDDED_IMAGE = (0, 190, 90)


# ── Box-label helpers ───────────────────────────────────────────────────────
# Each label states WHAT was found, short enough for the one-line label
# strip; the full detail lives in the findings list / signals, not here.

# Content-layer anomaly strings (content_analyzer) → short specific labels.
# Matched by prefix against the finding's own first anomaly reason.
_CONTENT_LABEL_PREFIXES = [
    ("font size",            "Font Size Mismatch"),
    ("font:",                "Font Mismatch"),
    ("char spacing",         "Char Spacing Anomaly"),
    ("word spacing",         "Word Spacing Anomaly"),
    ("line height",          "Line Height Anomaly"),
    ("visual noise",         "Visual Noise Outlier"),
    ("sharpness",            "Sharpness Outlier"),
    ("unnaturally uniform",  "Uniform Spacing (Retyped?)"),
    ("replacement character", "Font Encoding Mismatch"),
    ("[line_gap]",           "Abnormal Line Gap"),
]


def _content_label(sl) -> str:
    reason = (sl.anomalies[0] if getattr(sl, "anomalies", None) else "").strip()
    low = reason.lower()
    for prefix, label in _CONTENT_LABEL_PREFIXES:
        if low.startswith(prefix):
            return f"Line {sl.line_num + 1}: {label}"
    # Unknown anomaly type — fall back to the finding's own reason text.
    return f"Line {sl.line_num + 1}: {reason[:34] if reason else 'Content Anomaly'}"


def _numeric_label(r) -> str:
    ctx = (getattr(r, "context", "") or "")
    if ctx.startswith("running_balance"):
        return "Balance Mismatch"
    if ctx.startswith("arithmetic_"):
        return "Arithmetic Inconsistency"
    return f"Statistical Outlier (z={r.z_score:.1f})"


# PyMuPDF overlay_type → specific label.
_OVERLAY_LABELS = {
    "covering_rect": "White-Out Cover-Up",
    "image_overlay": "Hidden Image Overlay",
    "ghost_text":    "Ghost Text (Layered)",
}


def _flat_zone_label(r) -> str:
    # ELA-layer flat/pasted-patch regions (ela_analyzer flat-zone check).
    if getattr(r, "stamp_associated", False):
        return "Pasted Stamp: Flat Background"
    return "Flat Region: Texture Mismatch"


def _bbox_overlaps(b1, b2) -> bool:
    """TRUE rectangle intersection (any shared area) — the same semantics as
    signal_fusion.SignalFusion._bbox_overlaps. Used to decide "is this the SAME
    location" for authoritative-box absorption; deliberately NOT a proximity or
    same-row test (this is identity, not corroboration-by-nearness)."""
    if not b1 or not b2:
        return False
    x0_1, y0_1, x1_1, y1_1 = b1
    x0_2, y0_2, x1_2, y1_2 = b2
    return (x0_1 < x1_2 and x1_1 > x0_2 and
            y0_1 < y1_2 and y1_1 > y0_2)


BOX_PADDING        = 4   # px padding added around each drawn box
LABEL_HEIGHT        = 16  # px height of the label background strip
LABEL_CHAR_WIDTH     = 6   # px width estimate per label character (monospace-ish)
LABEL_VERTICAL_OFFSET = 18  # px the label sits above the box's top edge
