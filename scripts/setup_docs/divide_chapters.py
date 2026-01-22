import json
import re

INPUT_FILE = "../../data_raw/hlm_1_80.txt"
OUTPUT_FILE = "../../data_clean/chapters.jsonl"

CHAPTER_PATTERN = re.compile(r'^第\s*(\d+)\s*章')

chapters = []

current_chapter = None
current_title = None
current_lines = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过纯空行（可选）

        m = CHAPTER_PATTERN.match(line)
        if m:
            # 遇到新章，先收尾上一章
            if current_chapter is not None:
                chapters.append({
                    "chapter": current_chapter,
                    "title": current_title,
                    "text": "\n".join(current_lines)
                })

            # 开启新章
            current_chapter = int(m.group(1))
            current_title = line
            current_lines = []
        else:
            # 普通正文行
            if current_chapter is not None:
                current_lines.append(line)
            else:
                # 章标题之前的杂项（前言、版权说明等），直接忽略
                pass

# 文件结束后，别忘了最后一章
if current_chapter is not None:
    chapters.append({
        "chapter": current_chapter,
        "title": current_title,
        "text": "\n".join(current_lines)
    })

# 写出 jsonl
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ch in chapters:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

print(f"Parsed {len(chapters)} chapters.")
