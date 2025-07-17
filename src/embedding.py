import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for sentence-transformers embedding."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str = "../data/embeddings/cache"
    normalize_embeddings: bool = True
    batch_size: int = 32


class EmbeddingModel:
    """Simplified embedding model using sentence-transformers only."""

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info(f"Loading sentence-transformers model: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded, embedding dimension: {self.embedding_dim}")

    def encode_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a batch of texts to embeddings.

        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings
        """
        if not texts:
            raise ValueError("No texts provided for encoding")

        logger.info(f"Encoding {len(texts)} texts...")

        # Process in batches if needed
        if len(texts) > self.config.batch_size:
            return self._encode_large_batch(texts, show_progress)

        # Single batch processing
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )

        logger.info(f"Generated {len(embeddings)} embeddings (dim: {embeddings.shape[1]})")
        return embeddings

    def _encode_large_batch(self, texts: List[str], show_progress: bool) -> np.ndarray:
        """Process large batches in chunks to avoid memory issues."""
        embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            if show_progress:
                logger.info(
                    f"Processing batch {i // self.config.batch_size + 1}/{(len(texts) - 1) // self.config.batch_size + 1}")

            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,  # Don't show progress for individual batches
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding.

        Args:
            text: Text to encode

        Returns:
            numpy array embedding
        """
        if not text.strip():
            raise ValueError("Empty text provided")

        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )[0]

        return embedding

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'model_name': self.config.model_name,
            'embedding_dim': self.embedding_dim,
            'normalize_embeddings': self.config.normalize_embeddings,
            'batch_size': self.config.batch_size
        }

    def save_model_config(self, path: str):
        """Save model configuration."""
        import json

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        config_data = {
            'model_name': self.config.model_name,
            'embedding_dim': self.embedding_dim,
            'normalize_embeddings': self.config.normalize_embeddings,
            'batch_size': self.config.batch_size
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Model configuration saved to {save_path}")


def test_embedding_model():
    """Test the embedding model."""
    print("Testing sentence-transformers embedding model...")

    config = EmbeddingConfig()
    model = EmbeddingModel(config)

    # Test single encoding
    test_text = "France is located in Western Europe."
    embedding = model.encode_single(test_text)
    print(f"Single embedding shape: {embedding.shape}")

    # Test batch encoding
    test_texts = [
        "France has diverse geographical features.",
        "The Alps are France's highest mountain range.",
        "France has a temperate climate."
    ]

    embeddings = model.encode_batch(test_texts)
    print(f"Batch embeddings shape: {embeddings.shape}")

    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embedding], embeddings)[0]
    print(f"Similarities: {similarity}")

    print("✅ Embedding model test completed successfully!")


if __name__ == "__main__":
    test_embedding_model()
