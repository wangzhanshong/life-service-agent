import hashlib
import logging
from typing import List

from src.core.config import get_settings

logger = logging.getLogger("LifeAgent.Embedding")


class EmbeddingService:
    """Text embedding service using DashScope API with fallback."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.dashscope_api_key
        self.model = settings.embedding_model
        self._dimension = settings.embedding_dimension
        self._fallback_dim = 256  # Fallback will be padded to match config

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts. Falls back to hash embeddings if API unavailable."""
        if not self.api_key:
            logger.warning("No API key set, using fallback hash embeddings")
            return self._fallback_embed(texts)

        try:
            import dashscope
            from http import HTTPStatus

            resp = dashscope.TextEmbedding.call(
                model=self.model, input=texts, api_key=self.api_key
            )
            if resp.status_code == HTTPStatus.OK:
                embeddings = [item["embedding"] for item in resp.output["embeddings"]]
                actual_dim = len(embeddings[0])
                if actual_dim != self._dimension:
                    self._dimension = actual_dim
                    logger.info("Embedding dimension updated to %d", actual_dim)
                return embeddings
            else:
                logger.error("DashScope embedding failed: %s - %s", resp.status_code, resp.message)
                return self._fallback_embed(texts)
        except ImportError:
            logger.warning("dashscope not installed, using fallback embeddings")
            return self._fallback_embed(texts)
        except Exception as e:
            logger.error("Embedding error: %s", e)
            return self._fallback_embed(texts)

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        """Hash-based fallback, padded to configured dimension."""
        results = []
        for text in texts:
            vec = [0.0] * self._fallback_dim
            words = text.lower().split()
            for i, word in enumerate(words):
                h = hashlib.md5(word.encode()).digest()
                idx = int.from_bytes(h[:2], "big") % self._fallback_dim
                vec[idx] += 1.0 / (i + 1)
            # Normalize
            norm = sum(v**2 for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            # Pad to configured dimension
            if self._dimension > self._fallback_dim:
                vec.extend([0.0] * (self._dimension - self._fallback_dim))
            results.append(vec)
        return results
