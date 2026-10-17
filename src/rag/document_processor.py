import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("LifeAgent.DocumentProcessor")


@dataclass
class Document:
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    content: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log"}

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str) -> Optional[Document]:
        if not os.path.isfile(file_path):
            return None
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return Document(content=content, source=file_path,
                          metadata={"filename": os.path.basename(file_path), "extension": ext})
        except Exception as e:
            logger.error("Failed to read %s: %s", file_path, e)
            return None

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[Document]:
        documents = []
        if not os.path.isdir(dir_path):
            return documents
        for root, _, files in os.walk(dir_path):
            for filename in files:
                doc = self.load_file(os.path.join(root, filename))
                if doc:
                    documents.append(doc)
            if not recursive:
                break
        logger.info("Loaded %d documents from %s", len(documents), dir_path)
        return documents

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.content
        if not text.strip():
            return []
        chunks = []
        start, idx = 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(
                content=text[start:end], source=document.source, chunk_index=idx,
                metadata={**document.metadata, "chunk_start": start, "chunk_end": end},
            ))
            idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def process_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Split %d documents into %d chunks", len(documents), len(all_chunks))
        return all_chunks
