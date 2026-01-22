import numpy as np
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EXPECTED_DIM = 1024
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_query(q: str) -> np.ndarray:
    q = (q or "").strip()
    if not q:
        raise ValueError("Query is empty.")
    model = _get_model()
    vec = model.encode([q], normalize_embeddings=True, show_progress_bar=False)
    vec = np.asarray(vec[0], dtype=np.float32)
    if vec.shape[0] != EXPECTED_DIM:
        raise ValueError(f"Embedding dim mismatch: got {vec.shape[0]}, expected {EXPECTED_DIM}.")
    return vec


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "hlm_chunks_v1"

TOPK = 10


def main():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)
    col.load()

    query = "金麒麟是谁的？"
    qv = embed_query(query).astype(np.float32).tolist()

    # 示例：优先查 poem 或不限定都可以；你可以先 type=='poem' 再放宽
    expr = "chapter >= 1 && chapter <= 80"  # 可再加: && type == 'poem'
    res = col.search(
        data=[qv],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=TOPK,
        expr=expr,
        output_fields=["chunk_id", "chapter", "type", "start_para", "end_para", "char_len", "text"],
    )

    hits = res[0]
    for i, h in enumerate(hits, start=1):
        f = h.entity
        print(f"\n#{i} score={h.score}")
        print(f"chunk_id={f.get('chunk_id')} chapter={f.get('chapter')} type={f.get('type')} "
              f"paras={f.get('start_para')}-{f.get('end_para')} len={f.get('char_len')}")
        print(f.get("text")[:600])

if __name__ == "__main__":
    main()
