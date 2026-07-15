"""Multi-scale confirmation and document-type filters: bbox matching
across DPI scales, raster-content restriction, cluster filtering, and
high-DPI refinement."""

import fitz
import numpy as np
from PIL import Image

from .constants import *


class PageFilterMixin:
    # ── Multi-scale helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _bbox_overlaps(a: tuple, b: tuple, padding: float = SCALE_MATCH_PADDING_PT) -> bool:
        """Rectangle-intersection test with a small tolerance — block grids
        from different DPIs/crops don't land on identical point boundaries,
        so an exact-coordinate match would miss the same physical block."""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (ax0 - padding < bx1 and ax1 + padding > bx0 and
                ay0 - padding < by1 and ay1 + padding > by0)

    def _region_confirmed_by(self, bbox: tuple, other_regions: list) -> bool:
        """Does `bbox` (a phase-N candidate) overlap any region found in a
        different scale's independent pass over the same page?"""
        return any(self._bbox_overlaps(bbox, r.bbox) for r in other_regions)

    @staticmethod
    def _get_image_block_bboxes(page) -> list:
        """Bounding boxes (PDF points) of raster image blocks on this page."""
        try:
            blocks = page.get_text("dict").get("blocks", [])
            return [tuple(b["bbox"]) for b in blocks if b.get("type") == 1]
        except Exception:
            return []

    def _restrict_to_raster_content(self, page, regions: list) -> list:
        """
        ELA's JPEG-recompression model only means something for RASTER
        content — vector text is rasterized fresh at whatever DPI it's
        rendered at, so it has no real prior "compression history" to
        violate. A block landing on dense vector text (a bold header, a
        busy paragraph) reads as a recompression outlier purely because
        it's denser than the surrounding blank page — and that gap gets
        WORSE, not better, at higher DPI (crisper edges read as more
        anomalous), which is the opposite of what multi-scale confirmation
        assumes for a genuine logo/raster artifact. So: keep a flagged
        block only if it overlaps actual embedded image content, unless
        the page has no extractable text at all (a fully scanned/photo
        page, where the entire page IS raster and ELA fully applies).
        """
        has_text = bool((page.get_text("text") or "").strip())
        if not has_text:
            return regions
        image_bboxes = self._get_image_block_bboxes(page)
        if not image_bboxes:
            return []
        return [r for r in regions if any(
            self._bbox_overlaps(r.bbox, b) for b in image_bboxes
        )]

    def _is_image_based_document(self, doc, pdf_type: str) -> bool:
        """
        True if this document is a photograph/scan of pages rather than a
        native-text PDF — ELA's JPEG-recompression model doesn't apply to
        such documents the same way; noise-pattern analysis does instead.
        """
        if pdf_type == "scanned":
            return True
        if len(doc) == 0:
            return False
        image_pages = 0
        for page in doc:
            text = page.get_text("text") or ""
            if len(text.strip()) < 20 and len(page.get_images()) >= 1:
                image_pages += 1
        return image_pages >= max(1, len(doc) // 2)

    def _is_compiled_document(self, pdf_path: str, pdf_type: str) -> bool:
        """
        True when a multi-page scanned PDF is likely a portfolio of several
        independently-scanned source documents merged together.  Each source
        page carries its own JPEG compression baseline; ELA sees page-boundary
        differences as "edits" and produces false positives on every page.
        Detection heuristic: scanned type + at least COMPILED_MIN_PAGES pages.
        """
        if pdf_type not in ("scanned", "scanned_native"):
            return False
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
        except Exception:
            return False
        return total_pages >= COMPILED_MIN_PAGES

    def _filter_to_significant_clusters(self, regions: list, min_size: int = None) -> list:
        """
        Group regions into connected components by bbox proximity (touching
        or near-touching), then keep only blocks belonging to a component
        with >= min_size members. See the SCANNED_* constant comment above
        for why this separates a real edit from scan noise.
        """
        if min_size is None:
            min_size = SCANNED_MIN_CLUSTER_SIZE
        n = len(regions)
        if n == 0:
            return []
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        tol = SCANNED_CLUSTER_GAP_TOLERANCE_PT
        for i in range(n):
            xi0, yi0, xi1, yi1 = regions[i].bbox
            for j in range(i + 1, n):
                xj0, yj0, xj1, yj1 = regions[j].bbox
                gap_x = max(xi0, xj0) - min(xi1, xj1)
                gap_y = max(yi0, yj0) - min(yi1, yj1)
                if gap_x <= tol and gap_y <= tol:
                    union(i, j)

        component_sizes = {}
        for i in range(n):
            root = find(i)
            component_sizes[root] = component_sizes.get(root, 0) + 1

        return [
            regions[i] for i in range(n)
            if component_sizes[find(i)] >= min_size
        ]

    def _refine_region_at_high_dpi(self, page, bbox_pts: tuple, high_dpi: float):
        """
        Render ONLY a padded crop around a phase-2-confirmed block at
        high_dpi (not the whole page) and re-run the block-grid check inside
        that crop. This gets an exact-location refinement and a third,
        independent confirmation at high resolution for the price of a
        small render instead of a full-page 600 DPI pass.

        Returns (refined_bbox_or_None, confirmed_at_high_dpi: bool).
        """
        try:
            x0, y0, x1, y1 = bbox_pts
            pad = BLOCK_SIZE_PT * HIGH_DPI_CROP_PADDING_BLOCKS
            page_rect = page.rect
            crop = fitz.Rect(
                max(page_rect.x0, x0 - pad), max(page_rect.y0, y0 - pad),
                min(page_rect.x1, x1 + pad), min(page_rect.y1, y1 + pad),
            )
            if crop.width < 8 or crop.height < 8:
                return None, False

            mat = fitz.Matrix(high_dpi / 72, high_dpi / 72)
            pix = page.get_pixmap(matrix=mat, clip=crop, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.w, pix.h], pix.samples)
            gray = np.asarray(img.convert("L"), dtype=np.float64)

            block_px = self._block_px_for_dpi(high_dpi)
            z, _errors, n_rows, n_cols = self._ela_block_grid(img, gray, ELA_QUALITY, block_px)
            if z is None:
                return None, False

            flagged_idx = [i for i in range(z.size) if z[i] > Z_THRESHOLD]
            if not flagged_idx:
                return None, False

            pts_scale = 72 / high_dpi
            xs0 = [(i % n_cols) * block_px * pts_scale for i in flagged_idx]
            ys0 = [(i // n_cols) * block_px * pts_scale for i in flagged_idx]
            refined_bbox = (
                crop.x0 + min(xs0),
                crop.y0 + min(ys0),
                crop.x0 + max(xs0) + block_px * pts_scale,
                crop.y0 + max(ys0) + block_px * pts_scale,
            )
            return refined_bbox, True
        except Exception:
            return None, False

