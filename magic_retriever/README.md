# magic-retriever

模块化检索策略库，专为 RAG 应用设计。

## 功能特性

| 策略 | 类 | 说明 |
|------|-----|------|
| 相似度检索 | `SimilarityRetriever` | 嵌入查询向量 + 余弦相似度搜索 |
| MMR 检索 | `MMRRetriever` | 最大边际相关性，同时平衡相关性与多样性 |

## 安装

```bash
pip install sentence-transformers
```

## 快速开始

```python
from magic_retriever.strategies import SimilarityRetriever
from magic_embedder.strategies import SentenceTransformerEmbedder
from magic_vectorstore import ChromaVectorStore

embedder = SentenceTransformerEmbedder()
store = ChromaVectorStore(collection_name="my_collection")
retriever = SimilarityRetriever(embedder=embedder, vector_store=store)

result = retriever.retrieve("公司营收是多少？", top_k=5)
print(f"检索到 {len(result.chunks)} 条结果")
```

## MMR 检索

MMR（最大边际相关性）在保持相关性的同时增加结果多样性，避免重复内容：

```python
from magic_retriever.strategies import MMRRetriever

mmr = MMRRetriever(
    embedder=embedder,
    vector_store=store,
    top_k=5,
    fetch_k=20,
    lambda_mult=0.5,  # 0=最大多样性, 1=最大相关性
)
result = mmr.retrieve("公司营收是多少？", top_k=5)
```

## 模块结构

```
magic_retriever/
    __init__.py          # 统一导出
    run.py               # CLI 入口
    core/                # RetrievedChunk, RetrievalResult, BaseRetriever
    strategies/
        similarity_retriever.py   # 相似度检索
        mmr_retriever.py         # MMR 多样性检索
    utils/
```

## 设计原则

1. **基于向量检索**：查询文本经 embedder 转为向量，在 vector_store 中搜索
2. **MMR 原理**：`score = lambda * rel - (1-lambda) * div`，平衡相关性与多样性
3. **embedding 返回**：ChromaDB 需使用 `include=["embeddings"]` 才能做 MMR 多样性计算
