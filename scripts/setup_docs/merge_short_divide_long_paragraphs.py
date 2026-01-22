#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-process paragraphs.jsonl:
1) Short paragraph merge:
   - Merge "lead-in" short paragraphs like "又曰：" / "诗曰：" / "词曰：" / "题曰：" into the following paragraph,
     because they are semantically incomplete and hurt retrieval.
   - Optionally merge ultra-short "title-like" lines into the next paragraph (configurable).
2) Long paragraph split:
   - Split very long prose paragraphs into smaller segments using punctuation boundaries,
     targeting ~600 chars per segment (configurable).
   - Preserve poem paragraphs (multi-line) as-is by default.

Input:  /mnt/data/paragraphs.jsonl
Output: /mnt/data/paragraphs_v2.jsonl
Stats:  /mnt/data/stats_v2.json
Log:    /mnt/data/merge_log_v2.jsonl
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    # Short paragraph handling
    short_len_threshold: int = 10          # paragraphs with char_len <= this are considered "very short"
    leadin_len_threshold: int = 20         # lead-in markers within this length will be merged into next paragraph
    merge_title_like_short: bool = True    # merge ultra-short title-like lines into next paragraph

    # Long paragraph handling
    long_len_threshold: int = 1200         # paragraphs longer than this will be split (prose only)
    target_segment_len: int = 600          # desired segment length after splitting
    min_segment_len: int = 200             # avoid producing tiny segments when splitting
    max_segment_len: int = 800             # soft upper bound; if no boundary found, we may exceed slightly

    # Safety
    keep_poem_as_is: bool = True           # do not split poems; poems already multi-line

    # Output formatting
    merged_joiner: str = "\n"              # joiner when merging lead-in + next paragraph


CFG = Config()


# -------------------------
# IO helpers
# -------------------------
def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# Utilities
# -------------------------
LEADIN_PATTERNS = [
    # Common literary lead-ins
    r"^(又)?曰[:：]?$",
    r"^(诗|词|题|赋|赞|偈|歌|曲)曰[:：]?$",
    r"^(其诗|其词|其文)曰[:：]?$",
    r"^（?(诗|词)曰）?[:：]?$",
    r"^《.*》$",
    r"^【.*】$",
]

LEADIN_RE = re.compile("|".join(LEADIN_PATTERNS))

# "Title-like" short lines: very short, mostly non-punctuation, often used as plaque/name headings.
# We keep this conservative to avoid merging meaningful short dialogue.
TITLE_LIKE_RE = re.compile(r"^[\u4e00-\u9fff]{1,6}$")  # 1-6 Chinese characters only


def is_leadin(text: str, char_len: int) -> bool:
    """Return True if paragraph looks like a lead-in marker that should be merged into next paragraph."""
    t = text.strip()
    if char_len > CFG.leadin_len_threshold:
        return False
    return LEADIN_RE.match(t) is not None


def is_title_like_short(text: str, char_len: int) -> bool:
    """
    Heuristic: ultra-short, Chinese-only line (1-6 chars) such as plaques / scene headings.
    If you find it merges too aggressively, set CFG.merge_title_like_short=False.
    """
    if not CFG.merge_title_like_short:
        return False
    t = text.strip()
    if char_len > 6:
        return False
    return TITLE_LIKE_RE.match(t) is not None


def recompute_char_len(text: str) -> int:
    return len(text)


def ensure_type(row: Dict) -> str:
    # Some pipelines may not have 'type'; default to prose.
    return row.get("type") or "prose"


# -------------------------
# Short merge pass
# -------------------------
def merge_short_paragraphs(rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Merge within each chapter only, preserving para order.
    We do NOT merge across chapter boundaries.

    Merge rules:
    - If a paragraph is lead-in (e.g., "诗曰：") -> merge into the immediate next paragraph.
    - If a paragraph is title-like ultra-short (e.g., "迎春") -> merge into the immediate next paragraph.
    - If paragraph is very short but not lead-in/title-like, we keep it (conservative).
    """
    out: List[Dict] = []
    logs: List[Dict] = []

    # Group by chapter while preserving order
    rows_sorted = sorted(rows, key=lambda r: (int(r["chapter"]), int(r["para_idx"])))

    i = 0
    while i < len(rows_sorted):
        cur = rows_sorted[i]
        ch = int(cur["chapter"])
        cur_text = cur["text"]
        cur_len = int(cur.get("char_len", len(cur_text)))
        cur_type = ensure_type(cur)

        # By default, keep poems as-is for merging logic (but lead-ins are usually prose/meta)
        # We can still merge lead-in into poem if the next paragraph is poem.
        should_merge_into_next = (
            is_leadin(cur_text, cur_len) or is_title_like_short(cur_text, cur_len)
        )

        # Ensure next exists and in same chapter
        if should_merge_into_next and (i + 1 < len(rows_sorted)):
            nxt = rows_sorted[i + 1]
            if int(nxt["chapter"]) == ch:
                # Merge cur into nxt
                merged_text = cur_text.strip() + CFG.merged_joiner + nxt["text"].lstrip()
                merged_type = ensure_type(nxt)  # keep next's type
                merged_row = dict(nxt)
                merged_row["text"] = merged_text
                merged_row["type"] = merged_type
                merged_row["char_len"] = recompute_char_len(merged_text)

                logs.append({
                    "action": "merge_short_into_next",
                    "chapter": ch,
                    "from_para_idx": int(cur["para_idx"]),
                    "into_para_idx": int(nxt["para_idx"]),
                    "from_text": cur_text,
                    "into_text_preview_before": nxt["text"][:80],
                    "merged_text_preview_after": merged_text[:120],
                    "reason": "leadin" if is_leadin(cur_text, cur_len) else "title_like_short",
                })

                # Skip cur and nxt, output merged_row as a single unit
                out.append(merged_row)
                i += 2
                continue

        # Otherwise keep current paragraph
        out.append(cur)
        i += 1

    # Re-number para_idx within each chapter to keep them contiguous after merges
    out = renumber_para_idx(out)

    return out, logs


def renumber_para_idx(rows: List[Dict]) -> List[Dict]:
    """
    After merges/splits, para_idx may have gaps.
    Renumber para_idx starting from 1 for each chapter, preserving order.
    """
    rows_sorted = sorted(rows, key=lambda r: (int(r["chapter"]), int(r["para_idx"])))
    out: List[Dict] = []
    current_ch = None
    counter = 0
    for r in rows_sorted:
        ch = int(r["chapter"])
        if current_ch is None or ch != current_ch:
            current_ch = ch
            counter = 1
        else:
            counter += 1
        rr = dict(r)
        rr["para_idx"] = counter
        out.append(rr)
    return out


# -------------------------
# Long split pass
# -------------------------
SPLIT_PUNCT = set("。！？!?；;")  # strong boundaries
WEAK_PUNCT = set("，,：:、")      # weaker boundaries, use only if needed


def split_long_prose(rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Split long prose paragraphs using punctuation boundaries.

    - Only split if type == prose (or missing => prose)
    - If CFG.keep_poem_as_is is True, poem paragraphs won't be split.
    - Splitting aims for CFG.target_segment_len, with safety bounds.
    """
    out: List[Dict] = []
    logs: List[Dict] = []

    for r in sorted(rows, key=lambda x: (int(x["chapter"]), int(x["para_idx"]))):
        t = ensure_type(r)
        text = r["text"]
        L = int(r.get("char_len", len(text)))

        # Keep poems intact (recommended)
        if CFG.keep_poem_as_is and t == "poem":
            out.append(r)
            continue

        # Only split long prose
        if t != "prose" or L <= CFG.long_len_threshold:
            out.append(r)
            continue

        # Perform splitting
        segments = smart_split_text(text)

        if len(segments) <= 1:
            out.append(r)
            continue

        # Create new rows; para_idx will be renumbered later anyway
        for seg_i, seg in enumerate(segments, start=1):
            nr = dict(r)
            nr["text"] = seg
            nr["char_len"] = len(seg)
            nr["type"] = "prose"
            # Temporary para_idx: keep original order; renumber later
            nr["para_idx"] = int(r["para_idx"]) * 1000 + seg_i
            out.append(nr)

        logs.append({
            "action": "split_long_prose",
            "chapter": int(r["chapter"]),
            "orig_para_idx": int(r["para_idx"]),
            "orig_len": L,
            "segments": len(segments),
            "seg_lens": [len(s) for s in segments],
            "preview": text[:160],
        })

    out = renumber_para_idx(out)
    return out, logs


def smart_split_text(text: str) -> List[str]:
    """
    Split text into segments around punctuation boundaries.
    Strategy:
    - Walk through text; pick the closest strong boundary near target length.
    - If none found, allow weak boundary; if still none, force cut at max_segment_len.
    - Ensure min_segment_len by merging tiny tail back.

    Note:
    - This is intentionally deterministic and rule-based (good for reproducibility).
    """
    s = text.strip()
    if len(s) <= CFG.long_len_threshold:
        return [s]

    segments: List[str] = []
    start = 0
    n = len(s)

    while start < n:
        # If remaining is small enough, take it
        remaining = n - start
        if remaining <= CFG.max_segment_len:
            segments.append(s[start:].strip())
            break

        # Ideal cut point near target
        target = start + CFG.target_segment_len
        search_left = max(start + CFG.min_segment_len, target - 120)
        search_right = min(n - 1, target + 120)

        # 1) find strong boundary in [search_left, search_right]
        cut = find_boundary(s, search_left, search_right, strong=True)

        # 2) if not found, find weak boundary
        if cut is None:
            cut = find_boundary(s, search_left, search_right, strong=False)

        # 3) if still not found, force cut at start+max_segment_len
        if cut is None:
            cut = start + CFG.max_segment_len

        # Make segment
        seg = s[start:cut].strip()
        if seg:
            segments.append(seg)
        start = cut

    # Post-fix: if last segment too short, merge into previous
    segments = [seg for seg in segments if seg]
    if len(segments) >= 2 and len(segments[-1]) < CFG.min_segment_len:
        segments[-2] = (segments[-2].rstrip() + "\n" + segments[-1].lstrip()).strip()
        segments.pop()

    return segments


def find_boundary(s: str, left: int, right: int, strong: bool) -> Optional[int]:
    """
    Return an index to cut (exclusive), preferring boundary closest to 'right' (i.e., later),
    to keep segments near target length.

    We cut AFTER the punctuation, so the segment ends at boundary+1.
    """
    punct_set = SPLIT_PUNCT if strong else (SPLIT_PUNCT | WEAK_PUNCT)

    best = None
    for i in range(right, left - 1, -1):
        if s[i] in punct_set:
            best = i + 1
            break
    return best


# -------------------------
# Stats
# -------------------------
def compute_stats(rows: List[Dict]) -> Dict:
    total = len(rows)
    by_type = {"poem": 0, "prose": 0, "other": 0}
    lens = []
    short_le_5 = 0
    short_le_10 = 0
    long_gt_1200 = 0
    max_len = 0

    chapters = set()

    for r in rows:
        chapters.add(int(r["chapter"]))
        t = ensure_type(r)
        L = int(r.get("char_len", len(r["text"])))
        lens.append(L)
        max_len = max(max_len, L)
        if L <= 5:
            short_le_5 += 1
        if L <= 10:
            short_le_10 += 1
        if L > 1200:
            long_gt_1200 += 1

        if t in by_type:
            by_type[t] += 1
        else:
            by_type["other"] += 1

    lens_sorted = sorted(lens)
    def percentile(p: float) -> int:
        if not lens_sorted:
            return 0
        idx = int(round((len(lens_sorted) - 1) * p))
        return lens_sorted[idx]

    return {
        "chapters": len(chapters),
        "total": total,
        "by_type": by_type,
        "len_avg": sum(lens) / total if total else 0,
        "len_p50": percentile(0.50),
        "len_p90": percentile(0.90),
        "len_max": max_len,
        "short_le_5": short_le_5,
        "short_le_10": short_le_10,
        "long_gt_1200": long_gt_1200,
    }


# -------------------------
# Main
# -------------------------
def main(
    in_path: str = "../data_clean/paragraphs.jsonl",
    out_path: str = "../data_clean/paragraphs_v2.jsonl",
    stats_path: str = "../data_clean/stats_v2.json",
    log_path: str = "../data_clean/merge_log_v2.jsonl",
) -> None:
    rows = read_jsonl(in_path)
    stats_before = compute_stats(rows)

    # Pass 1: merge short lead-ins / title-like shorts
    rows_merged, merge_logs = merge_short_paragraphs(rows)

    # Pass 2: split long prose paragraphs
    rows_final, split_logs = split_long_prose(rows_merged)

    stats_after = compute_stats(rows_final)

    # Write outputs
    write_jsonl(out_path, rows_final)
    write_json(stats_path, {"before": stats_before, "after": stats_after, "config": CFG.__dict__})
    write_jsonl(log_path, merge_logs + split_logs)

    print("OK")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Stats:  {stats_path}")
    print(f"Log:    {log_path}")
    print("\nSTATS BEFORE:", stats_before)
    print("STATS AFTER: ", stats_after)


if __name__ == "__main__":
    main()
