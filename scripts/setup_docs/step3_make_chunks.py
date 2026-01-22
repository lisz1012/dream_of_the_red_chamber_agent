#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: Chunking
Input:  paragraphs_v2.jsonl  (paragraph-level, with chapter/para_idx/type/text/char_len)
Output: chunks.jsonl         (chunk-level, ready for embedding + Milvus)
        chunks_stats.json    (basic distribution statistics)

Design goals:
- Keep poems intact: each poem paragraph becomes its own chunk.
- For prose: accumulate consecutive paragraphs within the same chapter to reach ~target_len.
- Do NOT cross chapter boundary.
- Preserve traceability: chunk references start_para/end_para and paragraph count.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Tuple


# -------------------------
# Config
# -------------------------
@dataclass
class ChunkConfig:
    # Target length for prose chunks
    target_len: int = 500

    # Hard-ish bounds for prose chunks
    min_len: int = 350
    max_len: int = 650

    # If a single prose paragraph is longer than max_len, still keep it as a single chunk
    # (because it is already semantically coherent, and we don't want to split again here)
    allow_single_over_max: bool = True

    # Join paragraphs inside a chunk
    joiner: str = "\n"

    # Output paths (defaults to /mnt/data, adjust as needed)
    in_path: str = "../../data_clean/paragraphs_v2.jsonl"
    out_chunks: str = "../data_clean/chunks.jsonl"
    out_stats: str = "../data_clean/chunks_stats.json"


CFG = ChunkConfig()


# -------------------------
# IO
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
# Helpers
# -------------------------
def ensure_type(row: Dict) -> str:
    return row.get("type") or "prose"


def make_chunk_id(chapter: int, idx_in_chapter: int) -> str:
    # Stable, readable chunk_id
    # Example: c001_0001
    return f"c{chapter:03d}_{idx_in_chapter:04d}"


def flush_prose_chunk(
    chapter: int,
    chunk_idx_in_chapter: int,
    buf_paras: List[Dict],
) -> Dict:
    """
    Convert buffered prose paragraphs into a chunk row.
    """
    start_para = int(buf_paras[0]["para_idx"])
    end_para = int(buf_paras[-1]["para_idx"])
    text = CFG.joiner.join(p["text"].strip() for p in buf_paras if p.get("text"))
    char_len = len(text)

    return {
        "chunk_id": make_chunk_id(chapter, chunk_idx_in_chapter),
        "chapter": chapter,
        "type": "prose",
        "start_para": start_para,
        "end_para": end_para,
        "para_count": len(buf_paras),
        "char_len": char_len,
        "text": text,
    }


def build_chunks(rows: List[Dict]) -> List[Dict]:
    """
    Main chunking logic.
    """
    # Ensure correct order
    rows_sorted = sorted(rows, key=lambda r: (int(r["chapter"]), int(r["para_idx"])))

    chunks: List[Dict] = []

    current_ch = None
    chunk_idx_in_chapter = 0

    # Prose buffer
    buf: List[Dict] = []
    buf_len = 0

    def flush_buf_if_any():
        nonlocal buf, buf_len, chunk_idx_in_chapter
        if not buf:
            return
        chunk_idx_in_chapter += 1
        chunks.append(flush_prose_chunk(current_ch, chunk_idx_in_chapter, buf))
        buf = []
        buf_len = 0

    for r in rows_sorted:
        ch = int(r["chapter"])
        para_idx = int(r["para_idx"])
        t = ensure_type(r)
        text = (r.get("text") or "").strip()
        if not text:
            continue
        L = int(r.get("char_len", len(text)))

        # Chapter boundary: flush buffer, reset per-chapter counter
        if current_ch is None or ch != current_ch:
            # flush previous chapter
            flush_buf_if_any()
            current_ch = ch
            chunk_idx_in_chapter = 0

        # Poems: flush prose buffer first, then emit poem chunk as single unit
        if t == "poem":
            flush_buf_if_any()
            chunk_idx_in_chapter += 1
            chunks.append({
                "chunk_id": make_chunk_id(current_ch, chunk_idx_in_chapter),
                "chapter": current_ch,
                "type": "poem",
                "start_para": para_idx,
                "end_para": para_idx,
                "para_count": 1,
                "char_len": len(text),
                "text": text,
            })
            continue

        # Prose: accumulate within bounds
        # If a single paragraph is already very long, keep it alone (optional)
        if CFG.allow_single_over_max and L > CFG.max_len:
            flush_buf_if_any()
            chunk_idx_in_chapter += 1
            chunks.append({
                "chunk_id": make_chunk_id(current_ch, chunk_idx_in_chapter),
                "chapter": current_ch,
                "type": "prose",
                "start_para": para_idx,
                "end_para": para_idx,
                "para_count": 1,
                "char_len": len(text),
                "text": text,
            })
            continue

        # Normal prose accumulation
        if not buf:
            buf = [r]
            buf_len = len(text)
            continue

        # If adding this paragraph exceeds max_len:
        # - If current buf already >= min_len, flush current buf, start new buf with this paragraph.
        # - Else (buf too short), accept overflow and then flush (to avoid tiny chunks).
        new_len = buf_len + 1 + len(text)  # +1 approximates joiner cost
        if new_len > CFG.max_len:
            if buf_len >= CFG.min_len:
                flush_buf_if_any()
                buf = [r]
                buf_len = len(text)
            else:
                # buf too short; take this paragraph, overflow allowed once
                buf.append(r)
                buf_len = new_len
                flush_buf_if_any()
        else:
            buf.append(r)
            buf_len = new_len
            # If we've reached target, flush for tighter chunks (optional behavior)
            if buf_len >= CFG.target_len:
                flush_buf_if_any()

    # flush last chapter
    flush_buf_if_any()

    return chunks


def compute_stats(chunks: List[Dict]) -> Dict:
    total = len(chunks)
    by_type = {"poem": 0, "prose": 0, "other": 0}
    lens = []
    max_len = 0

    for c in chunks:
        t = c.get("type", "other")
        L = int(c.get("char_len", len(c.get("text", ""))))
        lens.append(L)
        max_len = max(max_len, L)
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
        "total_chunks": total,
        "by_type": by_type,
        "len_avg": (sum(lens) / total) if total else 0,
        "len_p50": percentile(0.50),
        "len_p90": percentile(0.90),
        "len_max": max_len,
        "config": CFG.__dict__,
    }


def main() -> None:
    rows = read_jsonl(CFG.in_path)
    chunks = build_chunks(rows)
    stats = compute_stats(chunks)

    write_jsonl(CFG.out_chunks, chunks)
    write_json(CFG.out_stats, stats)

    print("OK")
    print(f"Input:  {CFG.in_path}")
    print(f"Output: {CFG.out_chunks}")
    print(f"Stats:  {CFG.out_stats}")
    print("Summary:", stats)

    # Small preview
    print("\nFIRST 3 CHUNKS PREVIEW:")
    for c in chunks[:3]:
        preview = c["text"][:120].replace("\n", "\\n")
        print(f"- {c['chunk_id']} ch={c['chapter']} type={c['type']} "
              f"paras={c['start_para']}-{c['end_para']} len={c['char_len']} text={preview}...")


if __name__ == "__main__":
    main()
