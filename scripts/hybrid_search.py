import numpy as np
from pymilvus import (
    MilvusClient,
    AnnSearchRequest,
)
from sentence_transformers import SentenceTransformer

from scripts.ranker_router import choose_ranker

# -----------------------
# Embedding
# -----------------------
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EXPECTED_DIM = 1024
_model = None

def normalize_query_text(q) -> str:
    """
    兼容：
    - str
    - [{"text": "...", "type": "text"}]  (Gradio MultimodalTextbox)
    - ["..."]
    - 其他类型 -> str(q)
    """
    if q is None:
        return ""

    # 最常见：Gradio MultimodalTextbox 的 list[dict]
    if isinstance(q, list):
        if len(q) == 0:
            return ""
        # 只有一个 text item 的情况
        if len(q) == 1 and isinstance(q[0], dict) and "text" in q[0]:
            return str(q[0]["text"] or "")
        # 多个 item：把所有 text 拼起来
        parts = []
        for item in q:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"] or ""))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return " ".join([p for p in parts if p])

    # 纯字符串
    if isinstance(q, str):
        return q

    return str(q)

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_query(q: str):
    if not q:
        raise ValueError("Empty query")
    vec = _get_model().encode([q], normalize_embeddings=True)
    vec = np.asarray(vec[0], dtype=np.float32)
    assert vec.shape[0] == EXPECTED_DIM
    return vec.tolist()


# -----------------------
# Milvus
# -----------------------
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hlm_chunks_v2"
TOPK = 25

def main():
    client = MilvusClient(uri=MILVUS_URI)

    # query = "咏柳絮的《临江仙》是谁写的"
    # query = "金麒麟是谁的？"
    # query = "哪些地方体现了凤姐的理家能力？"
    query = "凤姐对于大观园里的各个女孩子的行事风格都有什么评价？"
    query_text = normalize_query_text(query).strip()
    q_dense = embed_query(query_text)

    expr = "chapter >= 1 && chapter <= 80"

    # 1) sparse（BM25）
    sparse_req = AnnSearchRequest(
        data=[query],                 # 注意：BM25 直接给文本
        anns_field="sparse",
        param={"metric_type": "BM25", "params": {}},
        limit=50,
        expr=expr,
    )

    # 2) dense（向量）
    dense_req = AnnSearchRequest(
        data=[q_dense],
        anns_field="dense",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=50,
        expr=expr,
    )

    # 3) 融合（锚点类问题，sparse 权重大）
    ranker = choose_ranker(query_text)  # ranker = WeightedRanker(0.7, 0.3)

    res = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[sparse_req, dense_req],
        ranker=ranker,
        limit=TOPK,
        output_fields=[
            "chunk_id", "chapter", "type",
            "start_para", "end_para", "char_len", "text"
        ],
    )

    def _normalize_hits(res):
        # 常见返回：res 是 list，res[0] 是 hits
        if isinstance(res, list) and res:
            return res[0]
        return res

    def _get_entity_and_score(hit):
        if isinstance(hit, dict):
            if "entity" in hit and isinstance(hit["entity"], dict):
                return hit["entity"], hit.get("score", hit.get("distance"))
            # hit 本身就是字段字典
            return hit, hit.get("score", hit.get("distance"))
        # 兼容老版本 Hit 对象
        e = getattr(hit, "entity", None)
        s = getattr(hit, "score", None)
        if e is None and hasattr(hit, "__dict__"):
            # 有些版本字段直接挂在 hit 上
            e = hit.__dict__
        return e, s

    hits = _normalize_hits(res)
    for i, hit in enumerate(hits, start=1):
        e, score = _get_entity_and_score(hit)
        print("=" * 80)
        print(f"#{i} score={score}")
        print(
            f"{e.get('chunk_id')}  第{e.get('chapter')}回  {e.get('type')}  "
            f"paras={e.get('start_para')}-{e.get('end_para')} len={e.get('char_len')}"
        )
        print((e.get("text") or "")[:1300])


if __name__ == "__main__":
    main()
