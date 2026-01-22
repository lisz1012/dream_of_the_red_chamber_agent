from pymilvus import MilvusClient, AnnSearchRequest
from ranker_router import choose_ranker

MILVUS_URI = "http://localhost:19530"
COLLECTION = "hlm_chunks_v2"

def hybrid_retrieve(query: str, topk: int = 30):
    client = MilvusClient(uri=MILVUS_URI)

    expr = "chapter >= 1 && chapter <= 80"

    # dense embedding 由你已有函数提供
    from hybrid_search import embed_query
    q_dense = embed_query(query)

    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse",
        param={"metric_type": "BM25", "params": {}},
        limit=50,
        expr=expr,
    )

    dense_req = AnnSearchRequest(
        data=[q_dense],
        anns_field="dense",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=50,
        expr=expr,
    )

    ranker = choose_ranker(query)

    res = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[sparse_req, dense_req],
        ranker=ranker,
        limit=topk,
        output_fields=[
            "chunk_id", "chapter", "type",
            "start_para", "end_para", "text"
        ],
    )

    hits = res[0]
    docs = []
    for h in hits:
        e = h.get("entity", h)
        docs.append(e)

    return docs


def neighbor_expand(docs, window: int = 2):
    """
    对每个命中 chunk，补充同回目前后段落, 提供更完整的上下文语境
    """
    client = MilvusClient(uri=MILVUS_URI)
    expanded = {}
    for d in docs:
        ch = d["chapter"]
        sp = d["start_para"]
        ep = d["end_para"]

        expr = (
            f"chapter == {ch} "
            f"&& start_para >= {sp - window} "
            f"&& end_para <= {ep + window}"
        )

        neighbors = client.query(
            collection_name=COLLECTION,
            filter=expr,
            output_fields=[
                "chunk_id", "chapter", "type",
                "start_para", "end_para", "text"
            ],
        )

        for n in neighbors:
            expanded[n["chunk_id"]] = n

    return list(expanded.values())

def neighbor_expand_with_trace(hits, window: int = 2):
    client = MilvusClient(uri=MILVUS_URI)

    hit_ids = {d["chunk_id"] for d in hits}
    neighbors_map = {}

    for d in hits:
        ch = d["chapter"]
        sp = d["start_para"]
        ep = d["end_para"]

        expr = (
            f"chapter == {ch} "
            f"&& start_para >= {sp - window} "
            f"&& end_para <= {ep + window}"
        )

        rows = client.query(
            collection_name=COLLECTION,
            filter=expr,
            output_fields=["chunk_id", "chapter", "type", "start_para", "end_para", "text"],
        )

        for r in rows:
            cid = r["chunk_id"]
            if cid in hit_ids:
                continue
            neighbors_map[cid] = r

    neighbors = list(neighbors_map.values())
    return hits, neighbors
