# magic-vectorstore

模块化向量存储与相似性检索库，专为 RAG 应用设计。

## 功能特性

| 策略 | 类 | 说明 |
|------|-----|------|
| ChromaDB | `ChromaVectorStore` | 持久化/内存模式，支持 cosine/euclidean 距离 |
| FAISS | `FAISSVectorStore` | flat/ivf/hnsw 索引，支持持久化和加载 |

## 安装

```bash
pip install chromadb faiss-cpu  # 或 faiss-gpu
```

## 快速开始

```python
from magic_vectorstore import ChromaVectorStore, VectorEntry

store = ChromaVectorStore(collection_name="my_collection")
store.add([
    VectorEntry(id="1", embedding=[0.1, 0.2, 0.3], text="Hello world")
])
result = store.search([0.1, 0.2, 0.3], top_k=5)
print(f"找到 {len(result.entries)} 条结果")
```

## 模块结构

```
magic_vectorstore/
    __init__.py          # 统一导出
    run.py               # CLI 入口
    core/                # VectorEntry, QueryResult, BaseVectorStore
    strategies/
        chroma/          # ChromaDB 实现
        faiss/           # FAISS 实现
    utils/
```

## CLI 用法

```bash
python -m magic_vectorstore.run \
    --input entries.json \
    --strategy chroma \
    --action add
```

## 设计原则

1. **统一接口**：`add()`, `search()`, `delete()`, `persist()`, `stats()`
2. **ChromaDB**：通过 `include=["embeddings"]` 返回存储向量，支持 MMR 多样性计算
3. **FAISS**：支持 `persist()`/`load()` 持久化，flat/ivf/hnsw 多种索引类型
