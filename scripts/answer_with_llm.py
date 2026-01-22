def _format_blocks(title: str, docs: list[dict], max_items: int = 30) -> str:
    if not docs:
        return f"## {title}\n（无）"

    blocks = [f"## {title}（{len(docs)}条）"]
    for d in docs[:max_items]:
        blocks.append(
            f"【第{d['chapter']}回 段{d['start_para']}-{d['end_para']} | {d['chunk_id']}】\n"
            f"{d['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, hits: list[dict], neighbors: list[dict]) -> str:
    context = "\n\n".join([
        _format_blocks("Top Hits（直接检索命中）", hits, max_items=20),
        _format_blocks("Neighbor Expansion（邻近扩展补充）", neighbors, max_items=40),
    ])

    return f"""
你是一名研究《红楼梦》的学者，请严格依据【给定原文】回答问题。

问题：
{query}

说明：
- “Top Hits” 是通过检索直接命中的关键段落；
- “Neighbor Expansion” 是为补齐上下文，从命中段落的前后邻近段落中扩展得到。

要求：
1. 只使用给定原文，不得引入外部知识
2. 按“能力 / 手段 / 方面”分点总结
3. 每一点必须至少引用一处原文（标明第几回与段号）
4. 若原文不足以支持结论，请明确说明不确定

【给定原文】
{context}

请按如下格式回答：

一、总体结论（1–2句）

二、具体体现（分点）
- 方面一：
  - 引用：
  - 说明：
- 方面二：
  - 引用：
  - 说明：

三、不确定或争议之处（如有）
""".strip()

