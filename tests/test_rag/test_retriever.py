"""RAGRetriever 集成测试：建库 -> 查询 全链路（Embedding 用确定性假实现，不碰网络）。"""
import pytest

from src.rag.document_processor import DocumentProcessor
from src.rag.retriever import RAGRetriever
from src.rag.vector_store import VectorStore


class FakeEmbedding:
    """确定性 embedding：同文本同向量，不同文本向量不同。"""

    dimension = 16

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_single(self, text):
        return self._vec(text)

    def _vec(self, text: str) -> list:
        v = [0.0] * self.dimension
        for i, ch in enumerate(text[: self.dimension]):
            v[i] = (ord(ch) % 17) / 17 + 0.05
        return v


@pytest.fixture
def retriever(tmp_path):
    store = VectorStore(dimension=16)
    return RAGRetriever(
        embedding=FakeEmbedding(),
        vector_store=store,
        processor=DocumentProcessor(chunk_size=50, chunk_overlap=10),
    )


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "documents"
    d.mkdir()
    (d / "laptop.md").write_text("续航测试：X Pro 型号在省电模式下可连续使用 12 小时。" * 3, encoding="utf-8")
    (d / "warranty.txt").write_text("保修政策：整机保修一年，电池保修六个月。", encoding="utf-8")
    (d / "virus.exe").write_bytes(b"\x00")  # 不支持的后缀，应被过滤
    return d


class TestBuildKnowledgeBase:
    def test_build_returns_chunk_count_and_ready(self, retriever, docs_dir):
        n = retriever.build_knowledge_base(str(docs_dir))
        assert n > 0
        assert retriever.vector_store.size == n
        assert retriever.is_ready is True

    def test_build_missing_dir_returns_zero(self, retriever, tmp_path):
        assert retriever.build_knowledge_base(str(tmp_path / "ghost")) == 0
        assert retriever.is_ready is False

    def test_build_empty_dir_returns_zero(self, retriever, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert retriever.build_knowledge_base(str(empty)) == 0


class TestQuery:
    def test_query_before_build_returns_empty(self, retriever):
        context, sources = retriever.query(" anything ")
        assert context == ""
        assert sources == []

    def test_query_returns_context_and_sources(self, retriever, docs_dir):
        retriever.build_knowledge_base(str(docs_dir))
        context, sources = retriever.query("保修政策是什么")
        assert context != ""
        assert len(sources) == retriever.top_k
        for s in sources:
            assert set(s.keys()) == {"content", "score", "source"}
            assert len(s["content"]) <= 200
            assert s["score"] >= 0  # IndexFlatIP 点积分数，未归一化向量可 > 1
        assert any(s["source"] in ("laptop.md", "warranty.txt") for s in sources)

    def test_query_top_k_override(self, retriever, docs_dir):
        retriever.build_knowledge_base(str(docs_dir))
        _, sources = retriever.query("续航", top_k=1)
        assert len(sources) == 1

    def test_save_and_reload_keeps_answers(self, retriever, docs_dir, tmp_path):
        retriever.build_knowledge_base(str(docs_dir))
        path = tmp_path / "kb"
        retriever.save(path)

        fresh = RAGRetriever(
            embedding=FakeEmbedding(),
            vector_store=VectorStore(dimension=16, index_path=path),
            processor=DocumentProcessor(chunk_size=50, chunk_overlap=10),
        )
        assert fresh.load(path) is True
        assert fresh.is_ready is True
        context, sources = fresh.query("保修政策")
        assert context != "" and len(sources) > 0
