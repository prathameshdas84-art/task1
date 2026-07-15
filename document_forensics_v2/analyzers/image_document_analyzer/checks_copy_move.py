"""Check 6: copy-move detection with offset-vector consensus + NCC
verification (lattice/template-text families rejected)."""

import cv2
import numpy as np

from .constants import (
    CM_ORB_FEATURES, CM_MIN_SPATIAL_DIST, CM_MAX_HAMMING, CM_OFFSET_BIN,
    CM_MERGE_BIN_DIST, CM_MIN_PAIRS, CM_NCC_VERIFY, CM_NCC_PATCH,
    CM_MIN_REGION_DIM, CM_HARMONIC_ANGLE_COS, CM_HARMONIC_RATIO_TOL,
)
from .report import ImageAnomaly


class CopyMoveCheckMixin:
    # ── Check 6 ──────────────────────────────────────────────────────────

    @staticmethod
    def _copy_move_check(gray):
        """Copy-move with OFFSET-VECTOR CONSENSUS: a genuine clone is many
        keypoint pairs sharing ONE displacement vector; scattered matches
        (repeated glyphs in any normal document) never converge on a single
        offset with this much support. Fired clusters are then verified by
        raw patch correlation before being reported."""
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        orb = cv2.ORB_create(nfeatures=CM_ORB_FEATURES)
        kps, des = orb.detectAndCompute(g8, None)
        metrics = {"keypoints": 0 if kps is None else len(kps)}
        if des is None or len(kps) < 20:
            return [], metrics

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf.knnMatch(des, des, k=4)
        pairs = []
        for ms in knn:
            for mtc in ms:
                if mtc.queryIdx == mtc.trainIdx or mtc.distance > CM_MAX_HAMMING:
                    continue
                p1 = np.array(kps[mtc.queryIdx].pt)
                p2 = np.array(kps[mtc.trainIdx].pt)
                if np.hypot(*(p2 - p1)) < CM_MIN_SPATIAL_DIST:
                    continue
                dx, dy = p2 - p1
                if dx < 0 or (dx == 0 and dy < 0):   # canonical direction
                    dx, dy, p1, p2 = -dx, -dy, p2, p1
                pairs.append((dx, dy, tuple(p1), tuple(p2)))
                break  # one non-self match per keypoint is enough

        metrics["candidate_pairs"] = len(pairs)
        if not pairs:
            return [], metrics

        raw_clusters = {}
        for dx, dy, p1, p2 in pairs:
            key = (round(dx / CM_OFFSET_BIN), round(dy / CM_OFFSET_BIN))
            raw_clusters.setdefault(key, []).append((p1, p2))

        # Merge clusters in adjacent bins (the same physical offset lands in
        # neighboring bins through keypoint jitter) so a real clone is ONE
        # cluster — otherwise the harmonic filter below would see its own
        # bin-split halves as a "lattice" and reject it.
        merged = []   # list of [sum_key(px), members]
        for key, members in sorted(raw_clusters.items(), key=lambda kv: -len(kv[1])):
            placed = False
            for mc in merged:
                mk = mc["key"]
                if (abs(mk[0] - key[0]) <= CM_MERGE_BIN_DIST
                        and abs(mk[1] - key[1]) <= CM_MERGE_BIN_DIST):
                    mc["members"].extend(members)
                    placed = True
                    break
            if not placed:
                merged.append({"key": key, "members": list(members)})
        metrics["largest_offset_cluster"] = (
            max(len(mc["members"]) for mc in merged) if merged else 0
        )

        candidates = [mc for mc in merged if len(mc["members"]) >= CM_MIN_PAIRS]

        # Lattice/harmonic rejection: repeated template text (line pitch,
        # column grid) produces a FAMILY of near-parallel offsets at integer
        # multiples of one base vector. A genuine clone is a single offset.
        def _is_lattice(a, b):
            va = np.array(a["key"], float) * CM_OFFSET_BIN
            vb = np.array(b["key"], float) * CM_OFFSET_BIN
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1 or nb < 1:
                return False
            cos = abs(float(va @ vb) / (na * nb))
            if cos < CM_HARMONIC_ANGLE_COS:
                return False
            ratio = max(na, nb) / min(na, nb)
            return abs(ratio - round(ratio)) < CM_HARMONIC_RATIO_TOL
        lattice = set()
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if _is_lattice(candidates[i], candidates[j]):
                    lattice.add(i)
                    lattice.add(j)
        candidates = [c for i, c in enumerate(candidates) if i not in lattice]
        metrics["lattice_clusters_rejected"] = len(lattice)

        anomalies = []
        for mc in candidates[:3]:
            members = mc["members"]
            key = mc["key"]
            src = np.array([m[0] for m in members])
            dst = np.array([m[1] for m in members])
            sx0, sy0 = src.min(axis=0); sx1, sy1 = src.max(axis=0)
            dx0, dy0 = dst.min(axis=0); dx1, dy1 = dst.max(axis=0)

            # Thin regions = a repeated glyph run / substring, not a clone.
            if (min(sx1 - sx0, sy1 - sy0) < CM_MIN_REGION_DIM
                    or min(dx1 - dx0, dy1 - dy0) < CM_MIN_REGION_DIM):
                continue

            # Source/dest overlap = periodic structure matching itself one
            # period over (text lines, table rows) — a clone's source and
            # destination are disjoint regions.
            ox = min(sx1, dx1) - max(sx0, dx0)
            oy = min(sy1, dy1) - max(sy0, dy0)
            if ox > 0 and oy > 0:
                inter = ox * oy
                smaller = min((sx1 - sx0) * (sy1 - sy0), (dx1 - dx0) * (dy1 - dy0))
                if smaller > 0 and inter / smaller > 0.2:
                    continue

            # NCC verification on sample keypoint patches
            ok = 0
            checked = 0
            half = CM_NCC_PATCH // 2
            for p1, p2 in members[:6]:
                x1, y1 = int(p1[0]), int(p1[1])
                x2, y2 = int(p2[0]), int(p2[1])
                if (min(x1, x2) < half or min(y1, y2) < half or
                        max(x1, x2) >= g8.shape[1] - half or
                        max(y1, y2) >= g8.shape[0] - half):
                    continue
                a = g8[y1 - half:y1 + half, x1 - half:x1 + half].astype(np.float64)
                b = g8[y2 - half:y2 + half, x2 - half:x2 + half].astype(np.float64)
                a -= a.mean(); b -= b.mean()
                denom = np.sqrt((a * a).sum() * (b * b).sum())
                checked += 1
                if denom > 1e-9 and (a * b).sum() / denom > CM_NCC_VERIFY:
                    ok += 1
            if checked == 0 or ok / checked < 0.5:
                continue

            conf = float(np.clip(len(members) / (3.0 * CM_MIN_PAIRS), 0.3, 1.0))
            for (x0, y0, x1, y1), tag in (((sx0, sy0, sx1, sy1), "source"),
                                          ((dx0, dy0, dx1, dy1), "clone")):
                anomalies.append(ImageAnomaly(
                    type="copy_move_region",
                    bbox=(int(x0), int(y0), int(max(8, x1 - x0)), int(max(8, y1 - y0))),
                    confidence=round(conf, 2),
                    evidence_check="check6_copy_move",
                    detail=(f"{tag} of {len(members)} keypoint pairs sharing "
                            f"offset ~({key[0] * CM_OFFSET_BIN},{key[1] * CM_OFFSET_BIN})px, "
                            f"NCC-verified"),
                ))
        return anomalies, metrics

