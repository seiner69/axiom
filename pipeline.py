"""
RAG 流水线主模块
将 magic_chunker、magic_embedder、magic_vectorstore、magic_retriever、magic_generator 串联起来
"""

from dataclasses import dataclass
from typing import Optional

from magic_chunker.core import ChunkingResult
from magic_embedder.core import EmbeddingResult
from magic_vectorstore.core import QueryResult
from magic_retriever.core import RetrievalResult
from magic_generator.core import GenerationPrompt, GenerationResult


@dataclass
class RAGPipelineConfig:
    """RAG 流水线配置"""
    # 向量化配置
    embedder_type: str = "sentence_transformer"  # openai | sentence_transformer | clip
    embedder_model: str = "all-MiniLM-L6-v2"

    # 向量存储配置
    vectorstore_type: str = "chroma"  # chroma | faiss
    collection_name: str = "rag_collection"
    persist_dir: Optional[str] = "./chroma_db"

    # 检索配置
    retriever_type: str = "similarity"  # similarity | mmr
    top_k: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.5  # MMR 参数，0=最大多样性，1=最大相关性

    # 生成配置
    generator_type: str = "openai"  # openai | anthropic
    generator_model: str = "gpt-4o-mini"


class RAGPipeline:
    """
    模块化 RAG 流水线

    使用方式:
        pipeline = RAGPipeline(config)
        pipeline.chunk(files=[...])
        pipeline.embed()
        pipeline.store()
        result = pipeline.retrieve("查询问题")
        response = pipeline.generate("查询问题")
    """

    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        self.config = config or RAGPipelineConfig()
        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self._generator = None
        self._chunks = None
        self._embeddings = None

    def _get_embedder(self):
        if self._embedder is None:
            if self.config.embedder_type == "sentence_transformer":
                from magic_embedder.strategies import SentenceTransformerEmbedder
                self._embedder = SentenceTransformerEmbedder(model_name=self.config.embedder_model)
            elif self.config.embedder_type == "openai":
                from magic_embedder.strategies import OpenAITextEmbedder
                self._embedder = OpenAITextEmbedder(model_name=self.config.embedder_model)
            elif self.config.embedder_type == "clip":
                from magic_embedder.strategies import CLIPImageEmbedder
                self._embedder = CLIPImageEmbedder(model_name=self.config.embedder_model)
            else:
                raise ValueError(f"Unknown embedder type: {self.config.embedder_type}")
        return self._embedder

    def _get_vector_store(self):
        if self._vector_store is None:
            if self.config.vectorstore_type == "chroma":
                from magic_vectorstore import ChromaVectorStore
                self._vector_store = ChromaVectorStore(
                    collection_name=self.config.collection_name,
                    persist_dir=self.config.persist_dir,
                )
            elif self.config.vectorstore_type == "faiss":
                from magic_vectorstore import FAISSVectorStore
                self._vector_store = FAISSVectorStore(
                    collection_name=self.config.collection_name,
                    persist_dir=self.config.persist_dir,
                )
            else:
                raise ValueError(f"Unknown vectorstore type: {self.config.vectorstore_type}")
        return self._vector_store

    def _get_retriever(self):
        if self._retriever is None:
            embedder = self._get_embedder()
            vector_store = self._get_vector_store()
            if self.config.retriever_type == "similarity":
                from magic_retriever.strategies import SimilarityRetriever
                self._retriever = SimilarityRetriever(
                    embedder=embedder,
                    vector_store=vector_store,
                )
            elif self.config.retriever_type == "mmr":
                from magic_retriever.strategies import MMRRetriever
                self._retriever = MMRRetriever(
                    embedder=embedder,
                    vector_store=vector_store,
                    top_k=self.config.top_k,
                    fetch_k=self.config.fetch_k,
                    lambda_mult=self.config.lambda_mult,
                )
            else:
                raise ValueError(f"Unknown retriever type: {self.config.retriever_type}")
        return self._retriever

    def _get_generator(self):
        if self._generator is None:
            if self.config.generator_type == "openai":
                from magic_generator.strategies import OpenAIGenerator
                self._generator = OpenAIGenerator(model=self.config.generator_model)
            elif self.config.generator_type == "anthropic":
                from magic_generator.strategies import AnthropicGenerator
                self._generator = AnthropicGenerator(model=self.config.generator_model)
            else:
                raise ValueError(f"Unknown generator type: {self.config.generator_type}")
        return self._generator

    def chunk(self, files=None, nodes=None) -> ChunkingResult:
        """
        分块阶段：加载文件并切分为块

        Args:
            files: 文件路径列表，支持 MinerU 的 content_list.json 和 .md 文件
            nodes: 或者直接传入预处理的 Node 列表
        """
        from magic_chunker.loaders import MinerUContentListLoader, MinerUMarkdownLoader

        all_chunks = []
        all_nodes = []

        if files:
            for file_path in files:
                if file_path.endswith('content_list.json'):
                    loader = MinerUContentListLoader(file_path)
                    nodes = loader.load()
                    all_nodes.extend(nodes)
                elif file_path.endswith('.md'):
                    loader = MinerUMarkdownLoader(file_path)
                    nodes = loader.load()
                    all_nodes.extend(nodes)

        if all_nodes:
            from magic_chunker.core import BaseChunker
            from magic_chunker.strategies import SemanticChunker
            chunker = SemanticChunker()
            result = chunker.chunk(all_nodes)
            self._chunks = result
            return result

        from magic_chunker.core import ChunkingResult
        return ChunkingResult(chunks=[], total_nodes=0)

    def embed(self, texts: list[str] = None) -> EmbeddingResult:
        """
        向量化阶段：对文本进行嵌入
        """
        if texts is None and self._chunks:
            texts = [chunk.content for chunk in self._chunks.chunks]

        embedder = self._get_embedder()
        self._embeddings = embedder.embed(texts)
        return self._embeddings

    def store(self, texts: list[str] = None, embeddings: list = None):
        """
        存储阶段：将向量存入向量数据库
        """
        if texts is None and self._chunks:
            texts = [chunk.content for chunk in self._chunks.chunks]

        if embeddings is None:
            embeddings = self._embeddings.embeddings if self._embeddings else None

        vector_store = self._get_vector_store()

        if embeddings is None:
            raise ValueError("No embeddings provided. Call embed() first or provide embeddings.")

        from magic_vectorstore.core import VectorEntry
        entries = [
            VectorEntry(
                id=f"chunk_{i}",
                embedding=emb,
                text=text,
            )
            for i, (emb, text) in enumerate(zip(embeddings, texts))
        ]
        vector_store.add(entries)

        if hasattr(vector_store, 'persist'):
            vector_store.persist()

    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        检索阶段：根据查询检索相关块
        """
        if top_k is None:
            top_k = self.config.top_k

        retriever = self._get_retriever()
        return retriever.retrieve(query, top_k=top_k)

    def generate(self, query: str, context: list[str] = None) -> GenerationResult:
        """
        生成阶段：基于检索结果生成回答
        """
        if context is None:
            retrieval_result = self.retrieve(query)
            context = [chunk.content for chunk in retrieval_result.chunks]

        generator = self._get_generator()
        prompt = GenerationPrompt(
            user_prompt=query,
            context=context,
        )
        return generator.generate(prompt)

    def run(self, query: str) -> str:
        """
        完整流水线：检索 + 生成
        """
        result = self.generate(query)
        return result.response
