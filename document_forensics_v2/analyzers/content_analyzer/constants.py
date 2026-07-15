"""Content-layer tuning constants: PDF-type classification thresholds,
scoring weights, and structural-line heuristics."""

import re


# ── Shared constants ────────────────────────────────────────────────────────────
# Scanner brand/keyword fingerprints used to detect "scanned_native" PDFs
# (a native-text PDF whose producer/creator metadata identifies a physical
# scanner). Kept module-level so any consumer shares one keyword set.
SCANNER_KEYWORDS = [
    "scan", "canon", "epson", "hp", "fujitsu", "brother", "xerox",
    "ricoh", "sharp", "kodak", "ij scan", "scansnap", "twain", "wia",
]

# Thresholds for classifying a PDF as native_text / mixed / scanned, based
# on the fraction of pages with a substantial embedded text layer.
NATIVE_TEXT_RATIO_THRESHOLD = 0.7   # >=70% of pages have text -> native_text
MIXED_TEXT_RATIO_THRESHOLD  = 0.3   # >=30% of pages have text -> mixed
NATIVE_TEXT_MIN_CHARS       = 30    # min chars extracted to count a page as "has text"

# Z-score cutoff for flagging a line's font size / spacing / visual feature
# as an outlier relative to the rest of the document. 2.5 sigma is the same
# cutoff used by every per-line statistical check in _score_lines() below.
Z_OUTLIER_THRESHOLD = 2.5

# A font used on more than this fraction of lines is treated as deliberate
# document styling (e.g. a Canva template mixing two or three font
# families) rather than a tamper signal — see _build_profile().
DESIGN_FONT_RATIO_THRESHOLD = 0.15

# Keywords that mark a value line as high-stakes (payroll/identity fields).
# Lines matching these are never treated as "structural" (always scored),
# and a font mismatch on one of the CRITICAL subset scores higher.
# NOTE: this list is English-language and includes India-specific payroll/
# identity terms (ctc, pan, aadhaar) — extend with local-language/regional
# equivalents before using this analyzer on non-Indian or non-English
# documents.
ALWAYS_CHECK_KEYWORDS = [
    "salary", "amount", "balance", "total",
    "net pay", "gross", "income", "compensation",
    "remuneration", "stipend", "payment",
    "account number", "aadhaar",
    "date of birth",
]
# Short keywords collide with ordinary words when matched as bare
# substrings — "pan" matches "company"/"Japan"/"expand", "dob" matches
# "Adobe", "wage" matches "sewage", "ctc" is short enough to risk the
# same. Require word boundaries so they only match the actual abbreviation.
ALWAYS_CHECK_KEYWORDS_WORD_BOUNDARY = [r"\bctc\b", r"\bpan\b", r"\bdob\b", r"\bwage\b"]
CRITICAL_VALUE_KEYWORDS = [
    "salary", "ctc", "amount", "balance", "total",
    "net pay", "gross", "income", "compensation",
]

# Payslip table-header rows (e.g. "Total Days in Month: 31.00 Days Paid:
# 31.00 LWP/Absent: Arrears Days Paid:") naturally have irregular spacing
# because they span multiple table columns — that's a layout artifact, not
# evidence of editing. English/India-payroll-specific terms; extend for
# other locales' payslip formats.
NEVER_FLAG_PATTERNS = [
    "total days",
    "days paid",
    "lwp",
    "lop",
    "absent",
    "arrears days",
    "working days",
]

# Regex patterns identifying letterhead/address/contact lines, which
# legitimately use a different font from the document body.
# India-specific: the city/state name list and "taluk"/"mandal" (Indian
# administrative-division terms) only match Indian addresses — extend this
# list before relying on it for documents from other countries.
ADDRESS_PATTERNS = [
    r'\d+[\/\-]\d+',           # address numbers like "A/1" "7-32"
    r'\b\d{6}\b',              # 6-digit pincode (India-specific format)
    r'@\w+\.\w+',              # email address
    r'\+?\d[\d\s\-]{8,}',     # phone number
    r'\b(road|street|nagar|colony|compound|post|village|district|'
    r'taluk|mandal|state|india|maharashtra|karnataka|gujarat|'
    r'delhi|mumbai|bangalore|bengaluru|hyderabad|chennai|pune)\b',
]

# _is_structural_line() heuristic thresholds (see that method for context).
LETTERHEAD_LINE_COUNT        = 3     # first N lines of page 0 = letterhead
ALL_CAPS_RATIO_THRESHOLD     = 0.85  # fraction of uppercase alpha chars = header
SHORT_LINE_MAX_WORDS         = 3     # lines with <= N words = field label
NUMERIC_LINE_RATIO_THRESHOLD = 0.7   # fraction of digits = purely numeric/date line
LABEL_PATTERN_MAX_WORDS      = 8     # "Label: Value" lines up to N words
RULE7_MAX_WORDS              = 5     # short lines with measured line height
SEPARATOR_MIN_LENGTH         = 5     # min length for a "----"/"====" divider line

# Character-spacing uniformity check (_score_lines()): genuine typed text
# has natural variation in per-character width; retyped/edited text often
# has unnaturally uniform spacing because it was set with fixed character
# advances rather than the original font's natural kerning.
CHAR_SPACING_CV_THRESHOLD  = 0.05  # coefficient of variation below this = too uniform
CHAR_SPACING_CV_MIN_CHARS  = 8     # only evaluate lines with more than this many chars
CHAR_SPACING_CV_SCORE      = 0.25  # anomaly score contribution when flagged

# Unicode replacement/placeholder glyphs that show up when a font can't
# render a character it was asked to — e.g. a currency symbol (₹, €, $)
# typed in a font/encoding that doesn't have that glyph after editing.
# Always checked regardless of line type: an encoding-failure glyph isn't
# something a letterhead/label can legitimately contain.
REPLACEMENT_CHARS = [
    '■',  # ■ BLACK SQUARE
    '□',  # □ WHITE SQUARE
    '▪',  # ▪ BLACK SMALL SQUARE
    '▫',  # ▫ WHITE SMALL SQUARE
    '●',  # ● BLACK CIRCLE
    '○',  # ○ WHITE CIRCLE
    '�',  # � UNICODE REPLACEMENT CHARACTER
]
REPLACEMENT_CHAR_SCORE = 0.60

# ── Upgrade 4: glyph consistency filter ─────────────────────────────────────────
# Canva/Figma/InDesign/Puppeteer/wkhtmltopdf all subset-embed fonts with
# custom glyph IDs ("AAAAAA+Helvetica") — when PyMuPDF can't map one of those
# IDs back to a real character it reads as U+FFFD/"?"/NBSP. That's the
# EXPORT TOOL's behavior, repeated identically everywhere that glyph is used
# in that subset (e.g. every Rs./currency symbol on a Canva payslip) — not a
# one-off edit. GLYPH_WATCH_CHARS is tracked for the registry/ratio
# computation; only chars also in REPLACEMENT_CHARS above actually get
# flagged in _score_lines — a bare "?" is far too common in legitimate text
# (questions, "N/A?", etc.) to safely treat as a tamper signal on its own,
# so it's tracked for ratio purposes but never itself score-flagged.
GLYPH_WATCH_CHARS = ('�', '?', ' ')

# -- Upgrade 5: form field suppression -----------------------------------------
# Form lines (Date:____, Sign:____, table cells separated by tabs) have wide,
# deliberately irregular spacing by design -- not an edit. Suppresses ONLY the
# spacing-related checks (char/word spacing, line height) for these lines;
# font size and color checks still run, since a genuine edit on a form field
# would still show up there.
FORM_FIELD_PATTERNS = [
    r'_{3,}',                  # three or more underscores
    r'\t{2,}',                 # multiple tabs
    r':\s{5,}',                # colon followed by 5+ spaces
    r'^(date|sign|signature|name|place|witness|designation|stamp|seal)\s*:?\s*$',
]
FORM_FIELD_SHORT_LINE_MAX_WORDS = 2
FORM_FIELD_SHORT_LINE_MIN_LEN   = 30  # 1-2 words spanning this much horizontal width = signature block, not a sentence
GLYPH_CONSISTENCY_RATIO_THRESHOLD = 0.02  # char recurring on >2% of a subset font's chars = platform behavior, not an edit

# _score_lines() per-anomaly score contributions. Each "outlier" check adds
# min(CAP, z * MULT) so a borderline z-score (just above Z_OUTLIER_THRESHOLD)
# contributes little while a very extreme one saturates at CAP.
FONT_SIZE_SCORE_CAP,    FONT_SIZE_SCORE_MULT    = 0.25, 0.05
CHAR_SPACING_SCORE_CAP, CHAR_SPACING_SCORE_MULT = 0.20, 0.04
WORD_SPACING_SCORE_CAP, WORD_SPACING_SCORE_MULT = 0.20, 0.04
LINE_HEIGHT_SCORE_CAP,  LINE_HEIGHT_SCORE_MULT  = 0.15, 0.03
NOISE_SCORE_CAP,        NOISE_SCORE_MULT        = 0.25, 0.05
SHARPNESS_SCORE_CAP,    SHARPNESS_SCORE_MULT    = 0.20, 0.04

# Font-mismatch score tiers — a CIDFont-subset mismatch (different
# embedded-subset fonts claiming the same role) is the strongest signal
# since it indicates two separate editing sessions; a mismatch on a
# critical value line (salary/total/etc.) is next; a mismatch on the
# letterhead is weighted lowest since letterheads legitimately use
# different fonts from the body even in unmodified documents.
FONT_MISMATCH_CIDFONT_SCORE   = 0.90
FONT_MISMATCH_CRITICAL_SCORE  = 0.70
FONT_MISMATCH_LETTERHEAD_SCORE = 0.15
FONT_MISMATCH_DEFAULT_SCORE   = 0.40

# Document-level signal (_build_signals()): the same base font family
# present both embedded and non-embedded (different subset tags) means the
# document went through two separate save/edit sessions with different font
# handling — one session embedded its subset, the other didn't. On the
# document's 0-100 anomaly_score scale, not the 0.0-1.0 per-line scale.
MIXED_FONT_EMBEDDING_SCORE = 25

# Per-line font-color consistency: spans within one PyMuPDF text LINE were
# drawn by the same renderer in the same pass, so they should share the
# exact RGB. Raising COLOR_DIFF_MIN alone can't separate a sloppy edit
# (ink color that's close-but-not-quite a match) from deliberate "Label:
# Value" styling (gray label + black filled-in value on one visual row —
# extremely common on payslips, bank statements, certificates), because
# the latter often has an even BIGGER RGB distance than a careless forgery.
# The actual distinguishing fact is repetition: a label/value color pair
# recurs on many lines throughout the document (it's the template's
# style), while an edited span's slightly-off color appears once. So a
# color is only a candidate anomaly if it's RARE document-wide
# (frequency-clustering: styles recur, edits are one-offs).
COLOR_DIFF_MIN                    = 15   # filters anti-aliasing/rounding noise, not real styling
COLOR_CLUSTER_MIN_SHARE           = 0.03  # a color on >=3% of spans is a deliberate document style
COLOR_CONSISTENCY_SCORE_PER_SPAN  = 10
COLOR_CONSISTENCY_SCORE_CAP       = 40

# Government ID cards (Aadhaar, PAN, driving licence, passport, voter ID)
# intentionally mix multiple ink colors on the same visual line by template
# design (e.g. Aadhaar's blue/black/orange) — the per-line color-consistency
# check below would otherwise flag that as tampering on every single one.
ID_CARD_KEYWORDS = [
    "aadhaar", "aadhar", "uid", "uidai", "unique identification",
    "permanent account number", "pan card",
    "driving licence", "driving license",
    "passport", "voter id", "epic no",
    "date of birth", "dob", "s/o", "d/o",
    "government of india",
]

# ── Upgrade 1: vertical line-gap density check ──────────────────────────────────
# Text injected into empty page space keeps the surrounding font/color, so the
# font/spacing checks above miss it — but it almost always breaks the page's
# vertical rhythm (the gap above/below it doesn't match the rest of the page).
# Computed PER PAGE (not document-wide) since different pages can legitimately
# have different line spacing.
LINE_GAP_MIN_LINES_PER_PAGE = 5     # need this many lines on a page for meaningful gap stats
LINE_GAP_LARGE_MULTIPLIER   = 3.0   # gaps > this many x the median = section/paragraph break, excluded from baseline
LINE_GAP_Z_THRESHOLD        = 3.5
LINE_GAP_MIN_WORDS          = 3     # don't flag short lines (labels/headers)
LINE_GAP_REPEAT_EXCLUDE     = 3     # a gap size recurring this many+ times = deliberate paragraph spacing
LINE_GAP_SCORE_PER_ANOMALY  = 20
LINE_GAP_SCORE_CAP          = 60
LINE_GAP_SCORE_WEIGHT       = 0.5

# Step 5 form-field exclusion for the gap check specifically — a line that's
# just a form field (date/signature line, underscores) legitimately sits in
# wide empty space and isn't an injection target.
LINE_GAP_FORM_FIELD_PATTERNS = [
    r'_{3,}',                                              # "____"
    r'^\s*(date|sign|signature)\s*:?\s*$',                 # "Date:", "Sign:", "Signature" alone
    r'\t{2,}',                                             # tab-separated single words (form fields)
    r'^\s*for\s+[\w\s.,&]+,\s*$',                          # "For Acme Technologies Pvt Ltd," — letter closing/signature block opener, always preceded by a deliberate gap reserved for the signature
]

# ── Upgrade 3: page-level baseline isolation ────────────────────────────────────
# A merged/compiled PDF legitimately has different fonts/sizes/colors per
# source page — scoring page 2 against page 1's document-wide baseline is
# what causes false positives on compilations. Below this many lines, a
# page's own stats are too unstable to score against (falls back to the
# document-wide profile instead).
MIN_LINES_FOR_PAGE_PROFILE = 8


# ── Data structures ────────────────────────────────────────────────────────────
