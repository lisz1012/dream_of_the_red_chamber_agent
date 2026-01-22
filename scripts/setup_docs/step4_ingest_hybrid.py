import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# -----------------------
# Embedding config
# -----------------------
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EXPECTED_DIM = 1024
BATCH_SIZE = 64

# normalize_embeddings=True:
# - 如果你 dense 索引用 metric=IP，归一化后 IP 排序会更像 cosine，检索更稳
NORMALIZE = True

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EXPECTED_DIM), dtype=np.float32)

    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=NORMALIZE,
    )

    vecs = np.asarray(vecs, dtype=np.float32)

    if vecs.ndim != 2 or vecs.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Embedding dim mismatch: got shape {vecs.shape}, expected (*, {EXPECTED_DIM}). "
            f"Model={MODEL_NAME}"
        )
    return vecs


# -----------------------
# Milvus config
# -----------------------
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "hlm_chunks_v2"   # 注意：改成你新建的 hybrid collection

# -----------------------
# Input
# -----------------------
INPUT_JSONL = "../data_clean/chunks.jsonl"


def read_chunks(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows

def batch(iterable: List, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def main():
    # 1) connect
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)

    # 2) load data
    rows = read_chunks(INPUT_JSONL)
    print(f"Loaded chunks: {len(rows)} from {INPUT_JSONL}")

    # 3) insert in batches
    inserted = 0
    for rows_b in batch(rows, BATCH_SIZE):
        texts = [r["text"] for r in rows_b]

        vecs = embed_texts(texts)  # float32 (n, 1024)
        dense = vecs.tolist()

        # schema fields (must match your hlm_chunks_v2):
        # chunk_id (PK), chapter, type, start_para, end_para, char_len, text, dense
        chunk_ids  = [r["chunk_id"] for r in rows_b]
        chapters   = [int(r["chapter"]) for r in rows_b]
        types      = [r["type"] for r in rows_b]
        start_paras= [int(r["start_para"]) for r in rows_b]
        end_paras  = [int(r["end_para"]) for r in rows_b]
        char_lens  = [int(r["char_len"]) for r in rows_b]
        texts_col  = texts

        # 注意：不要传 sparse。BM25 Function 会由 text 自动生成 sparse。
        entities = [
            chunk_ids,
            chapters,
            types,
            start_paras,
            end_paras,
            char_lens,
            texts_col,
            dense,
        ]

        col.insert(entities)
        inserted += len(rows_b)
        if inserted % (BATCH_SIZE * 10) == 0 or inserted == len(rows):
            print(f"Inserted: {inserted}/{len(rows)}")

    col.flush()
    print("Done. Flushed.")

    # load for search
    col.load()
    print("Collection loaded.")


if __name__ == "__main__":
    main()
