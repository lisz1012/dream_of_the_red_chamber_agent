import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

FULLWIDTH_INDENT = "\u3000\u3000"  # "　　"
_POEM_LINE_END = ("。", "？", "！", "…", "?", "!", "，", "、", ",")


# -------------------------
# IO helpers
# -------------------------
def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


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
# Chapter parsing
# -------------------------
def normalize_lines(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in text.split("\n")]
    # 去掉文件头尾多余空行
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def is_chapter_title(line: str) -> bool:
    """
    适配你可能出现的两种标题形式：
    - 第1章 ...
    - 第1回 ...
    以及中文数字：
    - 第一回 ...
    """
    s = line.strip()
    if not s:
        return False

    # 常见：第1章 / 第78章 / 第1回 / 第八十回
    # 允许标题后面跟空格或直接跟标题内容
    pattern = r"^第[0-9一二三四五六七八九十百千两]+[章].+"
    return re.match(pattern, s) is not None


def chinese_or_arabic_to_int(s: str) -> int:
    """
    支持阿拉伯数字直接转；中文数字只支持到 100+ 的常用形式（足够覆盖 80）。
    """
    s = s.strip()
    if s.isdigit():
        return int(s)

    mapping = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000}
    # 极简中文数字解析（覆盖 1-99/100+常见写法）
    total = 0
    num = 0
    unit = 1
    for ch in s:
        if ch in ("十", "百", "千"):
            u = mapping[ch]
            if num == 0:
                num = 1
            total += num * u
            num = 0
        else:
            num = num * 10 + mapping.get(ch, 0) if ch.isdigit() else mapping.get(ch, 0)
    total += num
    return total


def extract_chapter_number(title_line: str) -> int:
    """
    从标题行提取章节号：第XX章/回
    """
    s = title_line.strip()
    m = re.match(r"^第([0-9一二三四五六七八九十百千两]+)[章回]", s)
    if not m:
        return -1
    return chinese_or_arabic_to_int(m.group(1))


def split_chapters(full_text: str, expected_max: int = 80) -> List[Dict]:
    lines = normalize_lines(full_text)

    # 找到所有标题行的位置
    title_indices = [i for i, ln in enumerate(lines) if is_chapter_title(ln)]
    if not title_indices:
        raise ValueError("未检测到任何回目/章节标题行（形如 '第X章' 或 '第X回'）。请检查原文格式。")

    chapters: List[Dict] = []
    for idx, start in enumerate(title_indices):
        end = title_indices[idx + 1] if idx + 1 < len(title_indices) else len(lines)
        title = lines[start].strip()

        ch_no = extract_chapter_number(title)
        # 将标题行之后到下一标题行之前的内容作为正文
        body_lines = lines[start + 1:end]
        body = "\n".join(body_lines).strip()

        chapters.append({
            "chapter": ch_no,
            "title": title,
            "text": body
        })

    # 只保留 1..expected_max（防止文件里有后 40 回）
    chapters = [c for c in chapters if 1 <= c["chapter"] <= expected_max]

    # 按 chapter 排序，避免原文顺序异常
    chapters.sort(key=lambda x: x["chapter"])

    if len(chapters) != expected_max:
        # 不直接报错，先给提示，方便你排查缺回/标题不匹配
        print(f"WARNING: 解析到 {len(chapters)} 个章节（期望 {expected_max}）。请检查标题格式或是否缺回。")

    return chapters


# -------------------------
# Paragraph segmentation (your format)
# -------------------------
def is_poem_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if len(s) > 40:
        return False
    return s.endswith(_POEM_LINE_END)


def split_paragraphs_by_indent_merge_poem(body: str) -> List[str]:
    """
    适配你文本的段落切分：
    - 行首全角缩进（　　）视为“新段落候选”
    - 连续诗句（连续两行短句）进入 poem_mode：后续诗句继续合并到同一段落
    - 空行 flush
    - 忽略 (本章完)/(本回完)
    """
    paras: List[str] = []
    buf_lines: List[str] = []
    poem_mode = False

    def flush():
        nonlocal buf_lines, poem_mode
        if buf_lines:
            text = "\n".join(buf_lines).strip()  # 诗词保留换行
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

        if stripped in {"(本章完)", "（本章完）", "(本回完)", "（本回完）"}:
            flush()
            continue

        is_new_para_candidate = raw.startswith(FULLWIDTH_INDENT)
        content = stripped

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


def classify_type(text: str) -> str:
    """
    MVP 类型分类（先粗分即可）：
    - poem：多行且多数行像诗句
    - prose：其他
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2:
        poem_like = sum(1 for ln in lines if is_poem_line(ln))
        if poem_like / len(lines) >= 0.6:
            return "poem"
    return "prose"


# -------------------------
# Main pipeline
# -------------------------
def main(
    in_path: str = "../data_raw/hlm_1_80.txt",
    out_chapters: str = "../data_clean/chapters.jsonl",
    out_paragraphs: str = "../data_clean/paragraphs.jsonl",
    out_stats: str = "../data_clean/stats.json",
    expected_max: int = 80,
) -> None:
    full_text = read_text(in_path)
    chapters = split_chapters(full_text, expected_max=expected_max)

    # 写 chapters.jsonl
    write_jsonl(out_chapters, chapters)

    # 写 paragraphs.jsonl
    para_rows: List[Dict] = []
    stats = {
        "chapters_parsed": len(chapters),
        "expected_max": expected_max,
        "paragraphs_total": 0,
        "poem_paragraphs_total": 0,
        "prose_paragraphs_total": 0,
        "by_chapter": {}
    }

    for ch in chapters:
        ch_no = int(ch["chapter"])
        title = ch.get("title", "")
        body = ch.get("text", "")

        paras = split_paragraphs_by_indent_merge_poem(body)

        poem_cnt = 0
        for i, p in enumerate(paras, start=1):
            t = classify_type(p)
            if t == "poem":
                poem_cnt += 1

            para_rows.append({
                "chapter": ch_no,
                "title": title,
                "para_idx": i,
                "type": t,
                "text": p,
                "char_len": len(p),
            })

        stats["by_chapter"][str(ch_no)] = {
            "title": title,
            "paragraphs": len(paras),
            "poem_paragraphs": poem_cnt,
        }
        stats["paragraphs_total"] += len(paras)
        stats["poem_paragraphs_total"] += poem_cnt

    stats["prose_paragraphs_total"] = stats["paragraphs_total"] - stats["poem_paragraphs_total"]

    write_jsonl(out_paragraphs, para_rows)
    write_json(out_stats, stats)

    print("OK")
    print(f"Input: {in_path}")
    print(f"Wrote chapters: {out_chapters}  (count={len(chapters)})")
    print(f"Wrote paragraphs: {out_paragraphs}  (count={stats['paragraphs_total']})")
    print(f"Wrote stats: {out_stats}")
    # 额外提示：若章节数不等于 80，先排查标题行
    if len(chapters) != expected_max:
        print("WARNING: chapters_parsed != expected_max. 请检查原文是否缺回或标题格式不匹配。")


if __name__ == "__main__":
    main()
