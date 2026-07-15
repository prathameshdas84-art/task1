# Scoring constants
WHITE_RECT_SCORE_PER_REGION  = 25
WHITE_RECT_SCORE_CAP         = 70
IMAGE_OVERLAY_SCORE_PER_ITEM = 20
IMAGE_OVERLAY_SCORE_CAP      = 60
CHAR_ANOMALY_SCORE_PER_ITEM  = 5
CHAR_ANOMALY_SCORE_CAP       = 40
CHAR_SPACING_Z_THRESHOLD     = 4.0
WHITE_FILL_THRESHOLD         = 0.85  # RGB values above this = white/near-white
MIN_CHARS_FOR_SPACING_CHECK  = 4
MIN_WIDTHS_FOR_SPACING_CHECK = 3

# A vector rectangle or raster image covering most of the page is a
# background fill / letterhead template, not a cover-and-retype edit —
# without this guard, any document with a plain white page background
# (extremely common — Word, LibreOffice, Canva all draw one) would have
# its background rect intersect every text block and always max out
# WHITE_RECT_SCORE_CAP. Only small, targeted overlays count as suspicious.
MAX_OVERLAY_PAGE_AREA_FRACTION = 0.5

# Decorative panel/card backgrounds (common on bank statements, payslips)
# are well under 50% of the page but still far larger than a targeted
# cover-and-retype box — real-world testing found a 283x148pt summary-panel
# background alone produced 37 false-positive "white rect" hits on a clean
# bank statement. A genuine cover-up box is sized to the field/line it
# hides (roughly one to a few lines of text), not a multi-line panel.
MAX_OVERLAY_ABS_AREA_PT2  = 6000  # ~a 200x30pt box, generous for one field+value
MIN_OVERLAY_DIMENSION_PT  = 6     # excludes hairline table border/gutter rects

# Ghost-text detection: two different, non-empty text blocks occupying the
# same physical space is impossible in a legitimately laid-out document.
# A low-effort forgery pastes replacement text directly over the original
# without removing the original run, leaving both in the content stream.
# GHOST_TEXT_MAX_BLOCK_AREA_FRACTION excludes diagonal "CONFIDENTIAL"/
# "DRAFT" watermark stamps and background labels, which legitimately span
# a large fraction of the page and aren't a targeted paste-over.
GHOST_TEXT_OVERLAP_FRACTION         = 0.3
GHOST_TEXT_MAX_BLOCK_AREA_FRACTION  = 0.3
GHOST_TEXT_SCORE_PER_REGION         = 35
GHOST_TEXT_SCORE_CAP                = 70
MIN_GHOST_TEXT_LEN                  = 2

# Coordinate-overwrite detection: two different span texts at the exact same
# bbox (1pt precision) — a very precise paste-over that block-level ghost_text
# detection misses when the injected span is shorter than MIN_GHOST_TEXT_LEN
# or falls within the same block.
COORD_OVERWRITE_SCORE_PER_FINDING = 25
COORD_OVERWRITE_SCORE_CAP          = 50

# A cover-and-retype edit doesn't have to use a white box — on a colored
# letterhead/panel background, an editor will fill with whatever color
# matches the surrounding page so the patch is invisible. We detect that by
# sampling pixels just outside the rectangle's own edges and comparing.
LOCAL_BG_SAMPLE_DPI               = 150
LOCAL_BG_COLOR_DISTANCE_THRESHOLD = 0.15  # euclidean distance in 0-1 RGB space
LOCAL_BG_MARGIN_PT                = 10    # how far outside the rect to probe

# Z-order check: a table row background is painted BEFORE the text that
# sits on it (lower content-stream position = earlier seqno), so the text
# remains visible. A cover-and-retype edit pastes its box AFTER the text
# it's hiding (higher seqno), burying it. Geometric overlap alone can't
# tell these apart — only stream order can.
ROW_PATTERN_SIZE_TOLERANCE_PT = 2.0   # rects within this size delta count as "same size"
ROW_PATTERN_MIN_COUNT         = 3     # need this many same-sized rects to call it a pattern
ROW_PATTERN_INTERVAL_CV_MAX   = 0.3   # coefficient of variation of vertical spacing
COVERAGE_OVERLAP_FRACTION_MIN = 0.4   # fraction of a text span's own area that must be
                                       # inside the rect to count as "covered" (excludes
                                       # adjacent-row font ascender/descender edge-bleed)
