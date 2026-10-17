import logging
import os
from typing import List, Optional, Tuple

from .document_processor import DocumentProcessor
from .embedding import EmbeddingService
from .vector_store import VectorStore

logger = logging.getLogger("LifeAgent.Retriever")


class RAGRetriever:
    def __init__(self, embedding: EmbeddingService, vector_store: VectorStore,
                 processor: DocumentProcessor, top_k: int = 3):
        self.embedding = embedding
        self.vector_store = vector_store
        self.processor = processor
        self.top_k = top_k

    def build_knowledge_base(self, docs_path: str) -> int:
        if not os.path.isdir(docs_path):
            return 0
        documents = self.processor.load_directory(docs_path)
        if not documents:
            return 0
        chunks = self.processor.process_documents(documents)
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = self.embedding.embed([c.content for c in batch])
            self.vector_store.add(vectors, [c.content for c in batch],
                                  [c.metadata for c in batch])
        logger.info("Knowledge base built: %d chunks", len(chunks))
        return len(chunks)

    def query(self, question: str, top_k: Optional[int] = None) -> Tuple[str, List[dict]]:
        k = top_k or self.top_k
        if self.vector_store.size == 0:
            return "", []
        query_vec = self.embedding.embed_single(question)
        results = self.vector_store.search(query_vec, k)
        sources = [{"content": t[:200], "score": s, "source": m.get("filename", "unknown")}
                   for t, s, m in results]
        context = "\n\n---\n\n".join(t for t, _, _ in results)
        return context, sources

    def load(self, load_path) -> bool:
        return self.vector_store.load(load_path)

    def save(self, save_path) -> None:
        self.vector_store.save(save_path)

    @property
    def is_ready(self) -> bool:
        return self.vector_store.size > 0
