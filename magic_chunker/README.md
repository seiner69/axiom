# magic-chunker

模块化文档分块库，为 RAG 应用将长文档拆分为语义连贯的小块。

## 功能概览

| 类别 | 策略 | 说明 |
|------|------|------|
| **文本分块** | `FixedSizeChunker` | 固定字符数 + overlap |
| | `SentenceChunker` | 按句子分块，支持重叠 |
| | `SemanticChunker` | 按段落/标题语义边界分块 |
| **文件分块** | `MarkdownChunker` | 保留 Markdown 结构（标题层级/代码块/表格） |
| | `PDFChunker` | 按 PDF 页面分块，可嵌套文本 chunker |
| | `HTMLChunker` | 按 DOM 结构分块 |
| **数据加载** | `MinerUContentListLoader` | 加载 MinerU content_list.json |
| | `MinerUMarkdownLoader` | 加载 MinerU markdown 输出 |

## 安装

```bash
pip install sentence-transformers  # 用于语义分块的嵌入参考
# 其他依赖按需安装
```

## 快速开始

### 基础用法

```python
from magic_chunker.core import Node, ChunkType
from magic_chunker.strategies import SemanticChunker

# 1. 将文档转为 Node 列表
nodes = [
    Node(id="1", content="长文本内容...", metadata={"source": "doc.pdf"}),
]

# 2. 选择分块策略
chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=1000, merge_small_chunks=True)

# 3. 分块
result = chunker.chunk(nodes)
print(f"生成了 {len(result.chunks)} 个 chunks")

for chunk in result.chunks:
    print(f"  [{chunk.index}] {len(chunk.content)} chars")
```

### 与 MinerU 数据配合

```python
from magic_chunker.loaders import MinerUContentListLoader
from magic_chunker.strategies import SemanticChunker

# 1. 加载 MinerU 输出的 JSON
loader = MinerUContentListLoader("annual_report.pdf_content_list.json")
nodes = loader.load()
# → 2171 nodes (text: 1793, header: 143, table: 235)

# 2. 分块
chunker = SemanticChunker(min_chunk_size=200, max_chunk_size=1000, merge_small_chunks=True)
result = chunker.chunk(nodes)
# → 337 chunks (text: 157, table: 180)
```

## 核心接口

### Node 与 Chunk

```python
from magic_chunker.core import Node, Chunk, ChunkingResult, ChunkType

# Node — 文档解析后的最小单元
node = Node(
    id="node_1",
    content="文本内容",
    node_type=ChunkType.TEXT,  # TEXT / TABLE / IMAGE / CODE
    metadata={"page_idx": 0, "bbox": "100,200,300,400"},
)

# Chunk — 分块后的输出单元
chunk = Chunk(
    id="chunk_0",
    content="分块后的文本",
    chunk_type=ChunkType.TEXT,
    node_ids=["node_1"],     # 来源节点
    index=0,                  # 块序号
    metadata={},
)
```

### BaseChunker 接口

```python
from magic_chunker.core import BaseChunker

class MyChunker(BaseChunker):
    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        # 实现分块逻辑
        return ChunkingResult(chunks=[...], metadata={})

    @property
    def name(self) -> str:
        return "my_chunker"
```

## 分块策略详解

### SemanticChunker（推荐）

按自然语义边界分块：段落换行、标题行。合并过小的块。

```python
chunker = SemanticChunker(
    min_chunk_size=100,        # 最小字符数（过小则合并）
    max_chunk_size=1000,      # 最大字符数（强制截断）
    split_on_headers=True,    # 在标题行处分割
    merge_small_chunks=True,  # 合并过小的相邻块
)
```

### FixedSizeChunker

固定字符数分块，适合结构不规则的纯文本。

```python
chunker = FixedSizeChunker(
    chunk_size=500,    # 每块目标字符数
    overlap=50,        # 相邻块重叠字符数
)
```

### SentenceChunker

按句子边界分块，适合中文/英文正式文档。

```python
chunker = SentenceChunker(
    chunk_size=5,      # 每块目标句子数
    overlap=1,        # 相邻块重叠句子数
    min_sentences=1,  # 最小句子数
)
```

### MarkdownChunker

保留 Markdown 标题层级，代码块和表格保持独立。可嵌套文本 chunker。

```python
chunker = MarkdownChunker(
    max_header_level=2,        # 按 h1/h2 分割
    preserve_code_blocks=True, # 代码块单独成块
    preserve_tables=True,      # 表格单独成块
    text_chunker=SemanticChunker(),  # 嵌套文本分块策略
)
```

### PDFChunker

按 PDF 页面结构分块，适用于 PDF 直接解析后的数据。

```python
chunker = PDFChunker(
    chunk_by_page=True,     # 尊重页面边界
    max_page_gap=1,        # 允许的页间隔
    text_chunker=SemanticChunker(),
)
```

### HTMLChunker

按 HTML DOM 结构分块，标题、段落、div 区块均可作为分割点。

```python
chunker = HTMLChunker(
    max_heading_level=2,
    remove_scripts=True,
    text_chunker=SemanticChunker(),
)
```

## 数据加载器

### MinerUContentListLoader

加载 MinerU 的结构化 JSON 输出，适合年报、研报等格式规范的文档。

```python
from magic_chunker.loaders import MinerUContentListLoader

loader = MinerUContentListLoader("path/to/content_list.json")
nodes = loader.load()

# 获取图片节点
image_nodes = loader.get_image_nodes()
for node in image_nodes:
    print(node.id, node.metadata.get("img_path"))

# 解析图片绝对路径
full_path = loader.resolve_image_path("images/xxx.jpg")
print(full_path)  # → E:/project/output/images/xxx.jpg
```

### MinerUMarkdownLoader

加载 MinerU 输出的 `.md` 文件，按 section 分组输出节点。

```python
from magic_chunker.loaders import MinerUMarkdownLoader

loader = MinerUMarkdownLoader("path/to/report.pdf.md")
nodes = loader.load()
```

### 图片嵌入

```python
from magic_chunker.loaders import MinerUContentListLoader, embed_mineru_images
from magic_embedder.strategies import CLIPImageEmbedder

loader = MinerUContentListLoader("content_list.json")
embedder = CLIPImageEmbedder(model_name="openai/clip-vit-base-patch32")

# 返回 VectorEntry 列表，可直接存入向量库
image_entries = embed_mineru_images(loader, embedder)
# → [VectorEntry(id="mineru_52_841", embedding=[...], text="xxx.jpg", metadata={}), ...]
```

## 模块结构

```
magic_chunker/
    __init__.py              # 统一导出
    run.py                   # CLI 入口
    core/
        __init__.py          # Node, Chunk, ChunkType, ChunkingResult, BaseChunker
    strategies/
        __init__.py
        text/                # 文本分块策略
            fixed_size_chunker.py
            sentence_chunker.py
            semantic_chunker.py
        file/                # 文件类型分块策略
            markdown_chunker.py
            pdf_chunker.py
            html_chunker.py
    loaders/                 # 数据加载器
        __init__.py
        mineru_loader.py      # MinerUContentListLoader, MinerUMarkdownLoader
        image_utils.py        # embed_mineru_images
    utils/
        __init__.py           # generate_id, truncate_text
```

## CLI 用法

```bash
# 从 JSON 文件分块
python -m magic_chunker.run \
    --input input.json \
    --output chunks.json \
    --strategy semantic \
    --chunk-size 200

# 支持的策略：fixed, sentence, semantic, markdown, pdf, html
```

## 设计原则

1. **每个策略独立**：所有 chunker 继承 `BaseChunker`，接口统一 `chunk(nodes) -> ChunkingResult`
2. **可组合**：file 类型 chunker 可嵌套 text 类型 chunker
3. **非文本保留**：表格、图片、代码默认保留为独立块，不拆分
4. **可插拔**：loader 和策略均为可替换，可对接任意数据源
