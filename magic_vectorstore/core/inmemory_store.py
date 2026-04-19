"""In-memory key-value store for parent documents."""

from typing import Any


class InMemoryStore:
    """
    简单的内存键值存储，用于存放父文档。

    每一个父文档以 UUID 为 key 存储，支持增删改查。
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    def add(self, documents: list[dict]) -> None:
        """
        添加文档。

        Args:
            documents: 文档列表，每个文档需包含 'id' 字段。
        """
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                raise ValueError("Document must have an 'id' field")
            self._store[doc_id] = doc

    def get(self, doc_id: str) -> dict | None:
        """根据 ID 获取文档。"""
        return self._store.get(doc_id)

    def get_multi(self, doc_ids: list[str]) -> list[dict]:
        """根据多个 ID 获取文档列表。"""
        return [self._store[doc_id] for doc_id in doc_ids if doc_id in self._store]

    def delete(self, doc_id: str) -> bool:
        """删除文档。返回是否成功删除。"""
        if doc_id in self._store:
            del self._store[doc_id]
            return True
        return False

    def exists(self, doc_id: str) -> bool:
        """检查文档是否存在。"""
        return doc_id in self._store

    def list_ids(self) -> list[str]:
        """列出所有文档 ID。"""
        return list(self._store.keys())

    def count(self) -> int:
        """返回文档数量。"""
        return len(self._store)

    def clear(self) -> None:
        """清空所有文档。"""
        self._store.clear()

    def persist(self, path: str) -> None:
        """持久化到本地 JSON 文件。"""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "InMemoryStore":
        """从本地 JSON 文件加载。"""
        import json
        store = cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                store._store = json.load(f)
        except FileNotFoundError:
            pass
        return store
