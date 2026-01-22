import re
from pymilvus import WeightedRanker

def choose_ranker(query: str) -> WeightedRanker:
    q = (query or "").strip()

    # 1) 明确的“定位/事实”问法：优先 sparse
    fact_patterns = [
        r"^(谁|哪(个|位|一回|里|儿)|第.+回|几回|哪里|何处|是什么|出自|原文|哪句)",
        r"(谁写|谁作|谁题|谁说|谁给|谁的)",
        r"[“”\"《》]"  # 引号/书名号：通常要精确匹配
    ]
    if any(re.search(p, q) for p in fact_patterns):
        return WeightedRanker(0.7, 0.3)

    # 2) 总结/归纳/评价/能力/主题：优先 dense
    abstract_keywords = [
        "体现", "如何", "为什么", "分析", "评价", "概括", "总结",
        "归纳", "说明", "表现", "反映", "揭示", "展现", "揭露",
        "性格", "形象", "能力", "手腕", "理家", "管理", "才干",
        "主题", "意义", "象征", "伏线", "暗示", "表现", "能力",
        "风格", "特色", "特点", "写法", "艺术", "手法", "作用",
        "影响", "启示", "教训", "感受", "感想", "感情", "情感",
        "情绪", "氛围", "气氛", "外貌", "心理", "描写", "心态",
        "意味", "寓意", "象征", "隐喻", "讽刺", "反映", "描绘",
        "刻画", "意涵", "意象", "意境", "风采", "风貌", "风情",
        "风骨", "风流", "风范", "思想", "精神", "品格", "人品",
        "心理活动", "心理描写", "性格特点", "人物形象", "主题思想",
        "文学才华"
    ]
    if any(k in q for k in abstract_keywords):
        return WeightedRanker(0, 1)

    # 3) 兜底：看长度。短句更可能是锚点；长句更可能是抽象总结
    if len(q) <= 12:
        return WeightedRanker(0.65, 0.35)

    return WeightedRanker(0.35, 0.65)