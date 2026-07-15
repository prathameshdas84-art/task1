"""Human-readable classification of hidden-text findings: replacement
type, field type, plain-English explanations, and the report conclusion."""

import re


class HiddenTextNarrativeMixin:
    # ── Missing-vs-replaced classification ──────────────────────────────

    def _classify_replacement_type(self, covering_text: str) -> str:
        """Classify a hidden-text finding by whether anything visible was put
        in place of the hidden original.

        "missing"  — covering_text is empty/whitespace after normalization (the
                     legacy "unknown" sentinel is treated as missing too, for
                     any finding produced before this field existed). The
                     original was removed/covered with nothing visibly typed
                     over it.
        "replaced" — covering_text carries actual content (the existing,
                     already-working case)."""
        norm = re.sub(r"\s+", "", covering_text or "")
        if not norm or norm.lower() == "unknown":
            return "missing"
        return "replaced"

    def _compose_hidden_text_description(self, f: "HiddenTextFinding") -> str:
        """Human-readable description that reads clearly for each case."""
        orig = f.original_text[:60]
        if f.replacement_type == "missing":
            return (
                f"Original data hidden — no replacement text visible: "
                f"'{orig}' (content was removed, nothing put in its place)"
            )
        return (
            f"Original data hidden and replaced with different visible text: "
            f"'{orig}' → '{f.covering_text[:60]}'"
        )

    # ── Field-type classification & plain-English explanations ──────────

    def _classify_field_type(self, text) -> str:
        text_lower = text.lower().strip()

        # Amount/number fields
        if re.match(r'^[\d,\.\s₹$€£]+$', text):
            return "amount"

        # Date fields
        if re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', text):
            return "date"

        # ID/Reference numbers
        if re.match(r'^[A-Z]{1,4}[\d\/\-]+$', text.upper()):
            return "id_number"

        # Name fields (words only, mixed case)
        if re.match(r'^[A-Za-z\s\.]+$', text) and len(text) > 3:
            return "name"

        # Address
        if any(w in text_lower for w in
               ['street', 'road', 'avenue', 'city',
                'state', 'country', 'pin', 'postal']):
            return "address"

        return "unknown"

    def _get_plain_explanation(self, method, field_type) -> str:
        explanations = {
            "white_rectangle_cover": (
                "A white box was placed over the original "
                "text and new content was typed on top. "
                "Visual PDF readers only show the top layer, "
                "but the original text remains hidden in "
                "the file's data."
            ),
            "text_overlap": (
                "New text was placed directly over the "
                "original text without using a white box. "
                "Both versions exist in the file — the newer "
                "text appears on top when the document is "
                "opened normally."
            ),
            "incremental_update": (
                "The document was edited and re-saved. "
                "The original version is preserved in the "
                "file's edit history, revealing what the "
                "content looked like before it was changed."
            ),
        }

        return explanations.get(
            method,
            "Original content was found beneath "
            "the visible text in this document.",
        )

    def _generate_conclusion(self, findings) -> str:
        if not findings:
            return "No hidden content detected. The visible text appears to be the original."

        n = len(findings)
        pages = sorted(set(f.page for f in findings))
        methods = set(f.method for f in findings)

        method_descriptions = {
            "white_rectangle_cover": "white boxes placed over original text",
            "text_overlap": "new text layered over original text",
            "incremental_update": "content changed between saved versions",
        }

        method_text = " and ".join(
            method_descriptions.get(m, m) for m in methods
        )

        page_text = (
            f"page {pages[0]}" if len(pages) == 1
            else f"pages {', '.join(str(p) for p in pages)}"
        )

        return (
            f"{n} hidden text region{'s' if n > 1 else ''} "
            f"found on {page_text}. "
            f"The document appears to have been altered "
            f"using {method_text}. "
            f"The original content shown above was present "
            f"in the document before it was modified."
        )

