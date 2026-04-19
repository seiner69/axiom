# rag_pipeline

模块化 RAG（检索增强生成）流水线，整合分块、向量化、存储、检索、生成五大模块。

## 架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│magic_chunker│ -> │magic_embedder│ -> │magic_vectorstore│ -> │magic_retriever│ -> │magic_generator│
│   分块       │    │   向量化     │    │   向量存储    │    │    检索      │    │    生成      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 包含模块

| 模块 | 功能 | 策略 |
|------|------|------|
| [magic_chunker](https://github.com/seiner69/magic_chunker) | 文档分块 | SemanticChunker、MarkdownChunker |
| [magic_embedder](https://github.com/seiner69/magic_embedder) | 文本/图像向量化 | OpenAI、SentenceTransformer、CLIP |
| [magic_vectorstore](https://github.com/seiner69/magic_vectorstore) | 向量存储检索 | ChromaDB、FAISS |
| [magic_retriever](https://github.com/seiner69/magic_retriever) | 相似度/MMR 检索 | SimilarityRetriever、MMRRetriever |
| [magic_generator](https://github.com/seiner69/magic_generator) | LLM 生成 | OpenAI GPT、Anthropic Claude |

## 快速开始

### Python API

```python
from rag_pipeline import RAGPipeline, RAGPipelineConfig

config = RAGPipelineConfig(
    embedder_type="sentence_transformer",
    embedder_model="all-MiniLM-L6-v2",
    vectorstore_type="chroma",
    retriever_type="mmr",
    top_k=5,
    generator_type="openai",
    generator_model="gpt-4o-mini",
)

pipeline = RAGPipeline(config)

# MinerU 文件直接加载分块
pipeline.chunk(files=["/path/to/content_list.json", "/path/to/doc.md"])

# 向量化
pipeline.embed()

# 存储
pipeline.store()

# 完整流水线
result = pipeline.run("公司营收是多少？")
print(result)
```

### CLI

```bash
# 检索
python -m rag_pipeline.run \
    --action retrieve \
    --query "公司营收是多少？" \
    --top-k 5

# 生成
python -m rag_pipeline.run \
    --action generate \
    --query "公司营收是多少？" \
    --generator openai \
    --model gpt-4o-mini

# 完整流水线
python -m rag_pipeline.run \
    --action run \
    --query "公司营收是多少？"
```

## 支持的组件

### 向量化 (embedder_type)

| 类型 | 模型 |
|------|------|
| `sentence_transformer` | all-MiniLM-L6-v2, all-mpnet-base-v2 |
| `openai` | text-embedding-3-small, text-embedding-3-large, ada-002 |
| `clip` | openai/clip-vit-base-patch32 |

### 向量存储 (vectorstore_type)

| 类型 | 说明 |
|------|------|
| `chroma` | ChromaDB，持久化/内存模式 |
| `faiss` | FAISS，flat/ivf/hnsw 索引 |

### 检索 (retriever_type)

| 类型 | 说明 |
|------|------|
| `similarity` | 相似度检索，余弦相似度 |
| `mmr` | 最大边际相关性，平衡相关性与多样性 |

### 生成 (generator_type)

| 类型 | 模型 |
|------|------|
| `openai` | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| `anthropic` | claude-sonnet, claude-haiku, claude-opus |

## 安装依赖

```bash
pip install sentence-transformers chromadb faiss-cpu openai anthropic transformers torch torchvision Pillow
```

## 模块结构

```
rag_pipeline/
    __init__.py
    run.py               # CLI 入口
    pipeline.py          # 流水线主类
    magic_chunker/       # 分块模块
    magic_embedder/      # 向量化模块
    magic_vectorstore/   # 向量存储模块
    magic_retriever/     # 检索模块
    magic_generator/     # 生成模块
```
