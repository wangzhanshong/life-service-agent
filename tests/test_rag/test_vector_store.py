"""VectorStore 单元测试：FAISS 增删查、top_k 边界、JSON 持久化 roundtrip。"""
import numpy as np
import pytest

from src.rag.vector_store import VectorStore


def one_hot(dim: int, idx: int) -> list:
    v = [0.0] * dim
    v[idx] = 1.0
    return v


@pytest.fixture
def store():
    return VectorStore(dimension=8)


class TestAddAndSearch:
    def test_empty_store_returns_empty(self, store):
        assert store.size == 0
        assert store.search(one_hot(8, 0), top_k=3) == []

    def test_add_empty_vectors_is_noop(self, store):
        store.add([], [])
        assert store.size == 0

    def test_search_finds_identical_vector_first(self, store):
        vecs = [one_hot(8, 0), one_hot(8, 1), one_hot(8, 2)]
        store.add(vecs, ["chunk-0", "chunk-1", "chunk-2"],
                  [{"n": 0}, {"n": 1}, {"n": 2}])
        results = store.search(one_hot(8, 1), top_k=2)
        assert len(results) == 2
        top_content, top_score, top_meta = results[0]
        assert top_content == "chunk-1"
        assert top_score == pytest.approx(1.0, abs=1e-5)
        assert top_meta == {"n": 1}
        # 分数单调不增
        scores = [s for _, s, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_store(self, store):
        store.add([one_hot(8, i) for i in range(3)], [f"c{i}" for i in range(3)])
        results = store.search(one_hot(8, 0), top_k=10)
        assert len(results) == 3

    def test_add_without_metadata_fills_defaults(self, store):
        store.add([one_hot(8, 0)], ["only-chunk"])
        results = store.search(one_hot(8, 0), top_k=1)
        assert results[0][2] == {}

    def test_multiple_adds_accumulate(self, store):
        store.add([one_hot(8, 0)], ["first"])
        store.add([one_hot(8, 1)], ["second"])
        assert store.size == 2
        results = store.search(one_hot(8, 1), top_k=1)
        assert results[0][0] == "second"


class TestPersistence:
    def test_save_and_load_roundtrip(self, store, tmp_path):
        vecs = [one_hot(8, 0), one_hot(8, 1)]
        store.add(vecs, ["alpha", "beta"], [{"k": 1}, {"k": 2}])
        path = tmp_path / "index"
        store.save(path)

        restored = VectorStore(dimension=8, index_path=path)
        assert restored.load(path) is True
        assert restored.size == 2
        results = restored.search(one_hot(8, 1), top_k=1)
        assert results[0][0] == "beta"
        assert results[0][2] == {"k": 2}

    def test_load_missing_file_returns_false(self, store, tmp_path):
        assert store.load(tmp_path / "ghost") is False

    def test_save_without_path_is_safe(self, store):
        store.save()  # 无 index_path 时静默返回，不抛异常


def test_numpy_cosine_fallback_matches_faiss():
    """无 faiss 时的 numpy 余弦分支与主逻辑一致（防退化）。"""
    s = VectorStore(dimension=4)
    if s._faiss_available:
        pytest.skip("faiss 可用，跳过 fallback 分支")
    s.add([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], ["x", "y"])
    results = s.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == "x"
