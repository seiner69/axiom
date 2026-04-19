# magic-embedder

模块化文本和图像向量化库，专为 RAG 应用设计。

## 功能特性

| 策略 | 类 | 说明 |
|------|-----|------|
| OpenAI 文本 | `OpenAITextEmbedder` | text-embedding-3-small/large, ada-002 |
| SentenceTransformer | `SentenceTransformerEmbedder` | all-MiniLM-L6-v2 等多模型 |
| CLIP 图像 | `CLIPImageEmbedder` | openai/clip-vit-base-patch32（基于 transformers） |

## 安装

```bash
pip install sentence-transformers openai torch torchvision Pillow transformers
```

## 快速开始

### 文本嵌入

```python
from magic_embedder.strategies import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
result = embedder.embed(["Hello world", "这是测试文本"])
print(f"维度: {result.dimension}, 数量: {len(result.embeddings)}")
```

### 图像嵌入

```python
from magic_embedder.strategies import CLIPImageEmbedder

embedder = CLIPImageEmbedder(model_name="openai/clip-vit-base-patch32")
result = embedder.embed_images(["/path/to/image.jpg"])
print(f"图像嵌入维度: {result.dimension}")
```

## 模块结构

```
magic_embedder/
    __init__.py          # 统一导出
    run.py               # CLI 入口
    core/                # BaseEmbedder, EmbeddingResult
    strategies/
        text/            # OpenAI, SentenceTransformer
        image/           # CLIP（基于 transformers）
    utils/
```

## CLI 用法

```bash
# 文本嵌入
python -m magic_embedder.run \
    --input input.json \
    --output embeddings.json \
    --strategy sentence_transformer \
    --model all-MiniLM-L6-v2

# 图像嵌入
python -m magic_embedder.run \
    --input images.json \
    --output embeddings.json \
    --strategy clip \
    --type image
```

## 设计原则

1. **接口统一**：`BaseEmbedder.embed(texts) -> TextEmbeddingResult`
2. **可选依赖**：CLIP 图像嵌入使用 transformers，`clip` 包未安装时模块仍可加载
3. **批量处理**：支持 batch_size 控制批量嵌入
