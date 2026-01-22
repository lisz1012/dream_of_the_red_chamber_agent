import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# 你已选定的模型
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EXPECTED_DIM = 1024

# 全局单例：避免每个 batch 重复加载模型
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # device: 默认自动（有 GPU 会用 GPU，没有则 CPU）
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a batch of texts into embeddings.

    Returns:
        np.ndarray of shape (n, EXPECTED_DIM), dtype float32

    Notes:
    - normalize_embeddings=True makes cosine similarity meaningful and stable.
    - For Milvus metric_type="COSINE", it's recommended to normalize.
    """
    if not texts:
        return np.zeros((0, EXPECTED_DIM), dtype=np.float32)

    model = _get_model()

    # sentence-transformers 会返回 np.ndarray（通常 float32）
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    vecs = np.asarray(vecs, dtype=np.float32)

    # Dimension check (fail fast)
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
COLLECTION_NAME = "hlm_chunks_v1"

# -----------------------
# Embedding config
# -----------------------
# 方案A：sentence-transformers
# pip install -U sentence-transformers
# MODEL_NAME = "BAAI/bge-m3"  # 示例：你要换成你真正使用的模型（并确认输出维度）

BATCH_SIZE = 64
INPUT_JSONL = "../data_clean/chunks.jsonl"


def read_chunks(path: str) -> List[Dict]:
    rows = []
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
    # 1) connect milvus
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)

    # 2) load data
    rows = read_chunks(INPUT_JSONL)
    print(f"Loaded chunks: {len(rows)}")

    # 3) insert in batches
    inserted = 0
    for rows_b in batch(rows, BATCH_SIZE):
        texts = [r["text"] for r in rows_b]

        vecs = embed_texts(texts)
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)

        # 组装列数据（按 schema 字段顺序）
        chunk_ids = [r["chunk_id"] for r in rows_b]
        chapters = [int(r["chapter"]) for r in rows_b]
        types = [r["type"] for r in rows_b]
        start_paras = [int(r["start_para"]) for r in rows_b]
        end_paras = [int(r["end_para"]) for r in rows_b]
        char_lens = [int(r["char_len"]) for r in rows_b]
        texts_col = texts
        embeddings = vecs.tolist()  # pymilvus 接受 list[list[float]]

        entities = [
            chunk_ids,
            chapters,
            types,
            start_paras,
            end_paras,
            char_lens,
            texts_col,
            embeddings,
        ]

        mr = col.insert(entities)
        inserted += len(rows_b)
        print(f"Inserted: {inserted}/{len(rows)}")

    col.flush()
    print("Done. Flushed.")

    # load for search
    col.load()
    print("Collection loaded.")

if __name__ == "__main__":
    main()
