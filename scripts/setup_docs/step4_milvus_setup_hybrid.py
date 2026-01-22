from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    IndexType,
)
from pymilvus.client.types import MetricType

MILVUS_URI = "http://localhost:19530"   # 如果你用的是 tcp，可改成 "tcp://localhost:19530"
COLLECTION_NAME = "hlm_chunks_v2"
DIM = 1024  # 必须与你 embedding 维度一致

# BM25 参数：红楼梦/短诗词较多，建议 b 略低一点
BM25_K1 = 1.6
BM25_B = 0.7


def main(drop_if_exists: bool = False):
    client = MilvusClient(uri=MILVUS_URI)

    # 1) 若已存在：可选择删除重建
    if COLLECTION_NAME in client.list_collections():
        if not drop_if_exists:
            print(f"Collection already exists: {COLLECTION_NAME}")
            return
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        client.drop_collection(collection_name=COLLECTION_NAME)

    # 2) Schema
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # 你的主键：chunk_id（和你现在 v1 一致）
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=32,
    )

    # 你现有的标量字段（用于 filter + 邻近扩展）
    schema.add_field(field_name="chapter", datatype=DataType.INT16)
    schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=8)
    schema.add_field(field_name="start_para", datatype=DataType.INT32)
    schema.add_field(field_name="end_para", datatype=DataType.INT32)
    schema.add_field(field_name="char_len", datatype=DataType.INT32)

    # text：必须启用 analyzer，否则 BM25 sparse 生成/召回不靠谱
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=8192,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
    )

    # sparse / dense
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=DIM)

    # 3) BM25 Function：text -> sparse（插入时只给 text，sparse 自动生成）
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    # 4) Index params
    index_params = client.prepare_index_params()

    # sparse: SPARSE_INVERTED_INDEX + BM25
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
        },
    )

    # dense: 你若是 Milvus Lite/某些本地模式不支持 HNSW，可把 IndexType.HNSW 改成 IndexType.FLAT
    index_params.add_index(
        field_name="dense",
        index_name="dense_vector_index",
        index_type=IndexType.HNSW,
        metric_type=MetricType.IP,   # 内积（如果你向量已归一化，也可用 COSINE）
        params={
            "M": 16,
            "efConstruction": 200,
        },
    )

    # 5) Create
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )

    print(f"Created collection: {COLLECTION_NAME}")
    print("Indexes:", client.list_indexes(collection_name=COLLECTION_NAME))


if __name__ == "__main__":
    # drop_if_exists=True 会删库重建
    main(drop_if_exists=False)
