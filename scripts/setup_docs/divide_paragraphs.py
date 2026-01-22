# scripts/step2_ch1_paragraphs.py
import json
from pathlib import Path
from typing import List, Tuple

FULLWIDTH_INDENT = "\u3000\u3000"  # "　　"
_POEM_LINE_END = ("。", "？", "！", "…", "?", "!", "，", "、", ",")


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def parse_title_and_body(text: str) -> Tuple[str, str]:
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    title = lines[0].strip() if lines else ""
    body = "\n".join(lines[1:]).strip()
    return title, body


def is_poem_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # 过长一般不是诗句（可按实际微调）
    if len(s) > 40:
        return False
    # 诗句常见句末标点
    return s.endswith(_POEM_LINE_END)


def split_paragraphs_by_indent_merge_poem(body: str) -> List[str]:
    paras: List[str] = []
    buf_lines: List[str] = []
    poem_mode = False

    def flush():
        nonlocal buf_lines, poem_mode
        if buf_lines:
            text = "\n".join(buf_lines).strip()  # 保留诗行换行
            if text:
                paras.append(text)
        buf_lines = []
        poem_mode = False

    lines = [ln.rstrip("\n") for ln in body.split("\n")]

    for raw in lines:
        stripped = raw.strip()

        if not stripped:
            flush()
            continue

        if stripped in {"(本章完)", "（本章完）"}:
            flush()
            continue

        is_new_para_candidate = raw.startswith(FULLWIDTH_INDENT)
        content = stripped  # 去掉缩进后的内容

        if is_new_para_candidate:
            # 诗词模式下，连续诗句不新开段落
            if poem_mode and is_poem_line(content):
                buf_lines.append(content)
                continue

            # 连续两行诗句 => 进入诗词模式，不 flush，直接并入
            if buf_lines and (not poem_mode):
                prev_line = buf_lines[-1]
                if is_poem_line(prev_line) and is_poem_line(content):
                    poem_mode = True
                    buf_lines.append(content)
                    continue

            # 正常新段落
            flush()
            buf_lines.append(content)
        else:
            # 非缩进行：并入当前段落（续行）
            if not buf_lines:
                buf_lines.append(content)
            else:
                buf_lines.append(content)

            # 覆盖部分版式：非缩进行也可能组成连续诗句
            if len(buf_lines) >= 2 and (not poem_mode):
                if is_poem_line(buf_lines[-2]) and is_poem_line(buf_lines[-1]):
                    poem_mode = True

    flush()
    return paras


def write_jsonl(path: str, rows: List[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_txt_preview(path: str, title: str, paras: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        for i, para in enumerate(paras, start=1):
            f.write(f"[{i}] {para}\n\n")


def main(
    in_path: str = "data_raw/hlm_1.txt",
    out_jsonl: str = "data_clean/paragraphs_ch1.jsonl",
    out_preview: str = "data_clean/paragraphs_ch1.txt",
) -> None:
    text = read_text(in_path)
    title, body = parse_title_and_body(text)
    paras = split_paragraphs_by_indent_merge_poem(body)

    rows = []
    for i, p in enumerate(paras, start=1):
        rows.append(
            {
                "chapter": 1,
                "title": title,
                "para_idx": i,
                "text": p,
                "char_len": len(p),
            }
        )

    write_jsonl(out_jsonl, rows)
    write_txt_preview(out_preview, title, paras)

    print(f"OK: chapter=1 title={title}")
    print(f"OK: paragraphs={len(paras)}")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_preview}")
    print("\nFIRST 3 PARAS PREVIEW:")
    for i, p in enumerate(paras[:3], start=1):
        print(f"[{i}] {p[:120]}{'...' if len(p) > 120 else ''}")


if __name__ == "__main__":
    main()
