"""DocumentProcessor 单元测试：文件加载、滑动窗口分块、元数据传递。"""
import pytest

from src.rag.document_processor import Document, DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


class TestChunkDocument:
    def test_short_text_single_chunk(self, processor):
        doc = Document(content="短文本", source="a.md")
        chunks = processor.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "短文本"
        assert chunks[0].chunk_index == 0

    def test_long_text_multi_chunk_with_overlap(self, processor):
        text = "x" * 250  # chunk_size=100, overlap=20 -> 步长80
        doc = Document(content=text, source="a.md")
        chunks = processor.chunk_document(doc)
        assert len(chunks) == 4  # start: 0, 80, 160, 240
        assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
        # 相邻块重叠 chunk_overlap 个字符
        assert chunks[1].content[:20] == chunks[0].content[80:100]
        assert chunks[2].content[:20] == chunks[1].content[80:100]

    def test_last_chunk_covers_tail(self, processor):
        text = "y" * 150
        doc = Document(content=text, source="a.md")
        chunks = processor.chunk_document(doc)
        assert chunks[-1].content.endswith("y" * 10)
        joined = "".join(c.content for c in chunks)
        # 所有字符都应被覆盖（允许重叠，不允许遗漏）
        assert text[:100] in joined

    def test_empty_and_whitespace_only(self, processor):
        assert processor.chunk_document(Document(content="", source="a")) == []
        assert processor.chunk_document(Document(content="   \n\t ", source="a")) == []

    def test_metadata_propagated_with_offsets(self, processor):
        doc = Document(content="z" * 120, source="docs/note.md",
                       metadata={"filename": "note.md", "extension": ".md"})
        chunks = processor.chunk_document(doc)
        first = chunks[0]
        assert first.source == "docs/note.md"
        assert first.metadata["filename"] == "note.md"
        assert first.metadata["chunk_start"] == 0
        assert first.metadata["chunk_end"] == 100


class TestLoadFile:
    def test_supported_extension(self, processor, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# hello", encoding="utf-8")
        doc = processor.load_file(str(f))
        assert doc is not None
        assert doc.content == "# hello"
        assert doc.metadata["extension"] == ".md"
        assert doc.metadata["filename"] == "readme.md"

    def test_unsupported_extension_returns_none(self, processor, tmp_path):
        f = tmp_path / "binary.exe"
        f.write_bytes(b"\x00\x01")
        assert processor.load_file(str(f)) is None

    def test_missing_file_returns_none(self, processor, tmp_path):
        assert processor.load_file(str(tmp_path / "nope.md")) is None


class TestLoadDirectory:
    def test_recursive_and_extension_filter(self, processor, tmp_path):
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.md").write_text("B", encoding="utf-8")
        (sub / "skip.exe").write_bytes(b"\x00")

        docs = processor.load_directory(str(tmp_path), recursive=True)
        names = {d.metadata["filename"] for d in docs}
        assert names == {"a.txt", "b.md"}

    def test_non_recursive(self, processor, tmp_path):
        (tmp_path / "top.txt").write_text("T", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("D", encoding="utf-8")

        docs = processor.load_directory(str(tmp_path), recursive=False)
        assert [d.metadata["filename"] for d in docs] == ["top.txt"]

    def test_missing_directory_returns_empty(self, processor, tmp_path):
        assert processor.load_directory(str(tmp_path / "ghost")) == []


def test_process_documents_merges_chunks(processor):
    docs = [
        Document(content="a" * 250, source="1.md"),
        Document(content="b" * 30, source="2.md"),
        Document(content="   ", source="3.md"),  # 空文档产生 0 块
    ]
    chunks = processor.process_documents(docs)
    assert len(chunks) == 4 + 1  # 250字符->4块, 30字符->1块, 空->0块
    sources = {c.source for c in chunks}
    assert sources == {"1.md", "2.md"}
