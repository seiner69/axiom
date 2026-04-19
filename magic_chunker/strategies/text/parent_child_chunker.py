"""Parent-Child chunking strategy for hierarchical document retrieval."""

import uuid
from magic_chunker.core import BaseChunker, Chunk, ChunkType, ChunkingResult, Node


class ParentChildChunker(BaseChunker):
    """
    Parent-Child 双层切块策略。

    父文档：较大块 (默认 1000 字符)，保持完整上下文
    子文档：较小块 (默认 200 字符)，用于精准检索，支持 overlap

    每一个子文档的 metadata 中包含 parent_id，指向其归属的父文档。
    """

    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_overlap: int = 100,
        child_chunk_size: int = 200,
        child_overlap: int = 50,
        min_parent_size: int = 200,
    ):
        if parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be positive")
        if child_chunk_size <= 0:
            raise ValueError("child_chunk_size must be positive")
        if parent_overlap < 0 or parent_overlap >= parent_chunk_size:
            raise ValueError("parent_overlap must be non-negative and smaller than parent_chunk_size")
        if child_overlap < 0 or child_overlap >= child_chunk_size:
            raise ValueError("child_overlap must be non-negative and smaller than child_chunk_size")

        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.min_parent_size = min_parent_size

    @property
    def name(self) -> str:
        return "parent_child"

    @property
    def description(self) -> str:
        return (
            f"Parent-Child chunker "
            f"(parent={self.parent_chunk_size}, child={self.child_chunk_size}, "
            f"child_overlap={self.child_overlap})"
        )

    def _split_into_parent_chunks(self, text: str) -> list[str]:
        """按父文档大小切分文本，使用带 overlap 的滑动窗口。"""
        parents = []
        start = 0
        while start < len(text):
            end = start + self.parent_chunk_size
            chunk = text[start:end]
            parents.append(chunk)
            if end >= len(text):
                break
            start = end - self.parent_overlap
        return parents

    def _split_into_child_chunks(self, parent_text: str, parent_id: str) -> list[Chunk]:
        """将父文档切分为子文档。"""
        children = []
        start = 0
        child_idx = 0
        while start < len(parent_text):
            end = start + self.child_chunk_size
            chunk_text = parent_text[start:end]
            child = Chunk(
                id=f"child_{parent_id}_{child_idx}",
                content=chunk_text,
                chunk_type=ChunkType.TEXT,
                metadata={
                    "parent_id": parent_id,
                    "child_index": child_idx,
                    "char_start": start,
                    "char_end": end,
                },
                node_ids=[parent_id],
                index=child_idx,
            )
            children.append(child)
            child_idx += 1
            if end >= len(parent_text):
                break
            start = end - self.child_overlap
        return children

    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        """
        对节点列表进行父子切分。

        Returns:
            ChunkingResult:
                - chunks: 所有子文档（用于向量检索）
                - metadata["parent_documents"]: 所有父文档原始文本（用于 DocumentStore）
                - metadata["parent_id_to_children"]: 父ID -> 子文档ID 列表
        """
        all_children: list[Chunk] = []
        parent_documents: list[dict] = []
        parent_id_to_children: dict[str, list[str]] = {}
        child_counter = 0

        for node in nodes:
            # 非文本节点直接作为父文档处理
            if node.node_type != ChunkType.TEXT:
                parent_id = str(uuid.uuid4())
                parent_doc = {
                    "id": parent_id,
                    "content": node.content,
                    "metadata": node.metadata.copy(),
                }
                parent_documents.append(parent_doc)

                child = Chunk(
                    id=f"child_{parent_id}_0",
                    content=node.content,
                    chunk_type=node.node_type,
                    metadata={
                        "parent_id": parent_id,
                        "child_index": 0,
                        "char_start": 0,
                        "char_end": len(node.content),
                    },
                    node_ids=[node.id],
                    index=child_counter,
                )
                all_children.append(child)
                parent_id_to_children[parent_id] = [child.id]
                child_counter += 1
                continue

            # 文本节点：先切父文档，再切子文档
            parent_chunks = self._split_into_parent_chunks(node.content)
            for p_idx, p_text in enumerate(parent_chunks):
                if len(p_text) < self.min_parent_size and p_idx > 0:
                    # 合并到上一个父文档
                    if parent_documents:
                        parent_documents[-1]["content"] += "\n" + p_text
                        old_parent_id = parent_documents[-1]["id"]
                        # 为合并的块创建子文档
                        extra_children = self._split_into_child_chunks(p_text, old_parent_id)
                        for ec in extra_children:
                            ec.index = child_counter
                            all_children.append(ec)
                            parent_id_to_children[old_parent_id].append(ec.id)
                            child_counter += 1
                    continue

                parent_id = str(uuid.uuid4())
                parent_doc = {
                    "id": parent_id,
                    "content": p_text,
                    "metadata": node.metadata.copy(),
                }
                parent_documents.append(parent_doc)

                # 切分子文档
                children = self._split_into_child_chunks(p_text, parent_id)
                parent_id_to_children[parent_id] = [c.id for c in children]

                for child in children:
                    child.index = child_counter
                    all_children.append(child)
                    child_counter += 1

        return ChunkingResult(
            chunks=all_children,
            metadata={
                "chunker": self.name,
                "parent_chunk_size": self.parent_chunk_size,
                "child_chunk_size": self.child_chunk_size,
                "child_overlap": self.child_overlap,
                "total_nodes": len(nodes),
                "total_children": len(all_children),
                "total_parents": len(parent_documents),
                "parent_documents": parent_documents,
                "parent_id_to_children": parent_id_to_children,
            },
        )
