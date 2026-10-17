import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("LifeAgent.VectorStore")


class VectorStore:
    """FAISS-backed vector store with JSON serialization (secure, no pickle)."""

    def __init__(self, dimension: int, index_path: Optional[Path] = None):
        self.dimension = dimension
        self.index_path = index_path
        self._chunks: List[str] = []
        self._metadatas: List[dict] = []

        try:
            import faiss
            self.index = faiss.IndexFlatIP(dimension)
            self._faiss_available = True
        except ImportError:
            logger.warning("faiss-cpu not installed, using numpy-based search")
            self.index = None
            self._faiss_available = False
            self._vectors: List[np.ndarray] = []

    def add(self, vectors: List[List[float]], chunks: List[str], metadatas: Optional[List[dict]] = None) -> None:
        if not vectors:
            return
        np_vectors = np.array(vectors, dtype=np.float32)
        if self._faiss_available:
            self.index.add(np_vectors)
        else:
            for v in np_vectors:
                self._vectors.append(v)
        self._chunks.extend(chunks)
        self._metadatas.extend(metadatas or [{} for _ in chunks])
        logger.info("Added %d vectors, total: %d", len(vectors), len(self._chunks))

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[str, float, dict]]:
        if not self._chunks:
            return []
        qv = np.array([query_vector], dtype=np.float32)
        if self._faiss_available:
            scores, indices = self.index.search(qv, min(top_k, len(self._chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self._chunks):
                    results.append((self._chunks[idx], float(score), self._metadatas[idx]))
            return results
        else:
            similarities = []
            for i, vec in enumerate(self._vectors):
                sim = np.dot(qv[0], vec) / (np.linalg.norm(qv[0]) * np.linalg.norm(vec) + 1e-8)
                similarities.append((i, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [
                (self._chunks[idx], float(score), self._metadatas[idx])
                for idx, score in similarities[:top_k]
            ]

    def save(self, path: Optional[Path] = None) -> None:
        save_path = path or self.index_path
        if not save_path:
            return
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._faiss_available:
            import faiss
            # faiss.write_index 的 C++ 层不支持非 ASCII 路径（中文用户名会崩），
            # 改为内存序列化后用 Python 写文件
            index_bytes = faiss.serialize_index(self.index)
            with open(str(save_path) + ".faiss", "wb") as f:
                f.write(index_bytes.tobytes())

        # Save chunks and metadata as JSON (secure, no pickle)
        meta_data = {
            "dimension": self.dimension,
            "count": len(self._chunks),
            "chunks": self._chunks,
            "metadatas": self._metadatas,
        }
        with open(str(save_path) + ".json", "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False)
        logger.info("Saved %d vectors to %s", len(self._chunks), save_path)

    def load(self, path: Optional[Path] = None) -> bool:
        load_path = path or self.index_path
        if not load_path:
            return False

        json_file = str(load_path) + ".json"
        faiss_file = str(load_path) + ".faiss"

        if not os.path.exists(json_file):
            return False

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._chunks = data["chunks"]
        self._metadatas = data["metadatas"]
        self.dimension = data.get("dimension", self.dimension)

        if self._faiss_available and os.path.exists(faiss_file):
            import faiss
            with open(faiss_file, "rb") as f:
                raw = f.read()
            self.index = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))

        logger.info("Loaded %d vectors from %s", len(self._chunks), load_path)
        return True

    @property
    def size(self) -> int:
        return len(self._chunks)
