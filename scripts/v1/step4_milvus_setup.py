from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

COLLECTION_NAME = "hlm_chunks_v1"
DIM = 1024  # 你embedding模型的维度；务必与实际一致

def main():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection already exists: {COLLECTION_NAME}")
        return

    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=32),
        FieldSchema(name="chapter", dtype=DataType.INT16),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name="start_para", dtype=DataType.INT32),
        FieldSchema(name="end_para", dtype=DataType.INT32),
        FieldSchema(name="char_len", dtype=DataType.INT32),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),  # 如有超长可再增
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]

    schema = CollectionSchema(fields, description="Dream of the Red Chamber (1-80), chunks v1")
    col = Collection(name=COLLECTION_NAME, schema=schema)

    # 为标量字段建索引不是必须，但你会经常 filter chapter/type
    # Milvus 会处理过滤；此处省略标量索引

    # 向量索引（HNSW：小规模最省心）
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",   # 或 "L2"
        "params": {"M": 16, "efConstruction": 200},
    }
    col.create_index(field_name="embedding", index_params=index_params)

    print("Created collection and index:", COLLECTION_NAME)

if __name__ == "__main__":
    main()
