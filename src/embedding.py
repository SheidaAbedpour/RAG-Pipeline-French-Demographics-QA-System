import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
import time
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    embedding_type: str = "tfidf"  # tfidf, sentence-transformers
    model_name: str = "all-MiniLM-L6-v2"  # for sentence-transformers
    max_features: int = 5000  # for TF-IDF
    ngram_range: tuple = (1, 2)  # for TF-IDF
    cache_dir: str = "data/embeddings/cache"


class EmbeddingModel:
    """Handles embedding generation with multiple backend support."""

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.vectorizer = None
        self.sentence_model = None
        self.embedding_dim = None
        self.is_fitted = False

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized embedding model: {self.config.embedding_type}")

    def _setup_tfidf_vectorizer(self, texts: List[str]):
        """Setup and fit TF-IDF vectorizer."""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
            )

            # Fit vectorizer
            logger.info("Fitting TF-IDF vectorizer...")
            self.vectorizer.fit(texts)
            self.embedding_dim = len(self.vectorizer.vocabulary_)
            self.is_fitted = True

            logger.info(f"TF-IDF fitted with {self.embedding_dim} features")

    def _setup_sentence_transformer(self):
        """Setup sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            if self.sentence_model is None:
                logger.info(f"Loading sentence transformer: {self.config.model_name}")
                self.sentence_model = SentenceTransformer(self.config.model_name)
                self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
                self.is_fitted = True

                logger.info(f"Sentence transformer loaded, dimension: {self.embedding_dim}")

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def encode_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode a batch of texts to embeddings."""
        if not texts:
            raise ValueError("No texts provided for encoding")

        start_time = time.time()

        if self.config.embedding_type == "tfidf":
            self._setup_tfidf_vectorizer(texts)
            embeddings = self.vectorizer.transform(texts).toarray()

        elif self.config.embedding_type == "sentence-transformers":
            self._setup_sentence_transformer()
            embeddings = self.sentence_model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

        else:
            raise ValueError(f"Unsupported embedding type: {self.config.embedding_type}")

        end_time = time.time()

        if show_progress:
            logger.info(
                f"Generated {len(embeddings)} embeddings "
                f"(dim: {embeddings.shape[1]}) in {end_time - start_time:.2f}s"
            )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call encode_batch first.")

        if self.config.embedding_type == "tfidf":
            if self.vectorizer is None:
                raise RuntimeError("TF-IDF vectorizer not initialized")
            return self.vectorizer.transform([text]).toarray()[0]

        elif self.config.embedding_type == "sentence-transformers":
            if self.sentence_model is None:
                raise RuntimeError("Sentence transformer not loaded")
            return self.sentence_model.encode([text], convert_to_numpy=True)[0]

        else:
            raise ValueError(f"Unsupported embedding type: {self.config.embedding_type}")

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'embedding_type': self.config.embedding_type,
            'embedding_dim': self.embedding_dim,
            'model_name': self.config.model_name,
            'is_fitted': self.is_fitted,
            'max_features': self.config.max_features if self.config.embedding_type == "tfidf" else None
        }

    def save_model(self, path: str):
        """Save the fitted model to disk."""
        import pickle

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.config.embedding_type == "tfidf" and self.vectorizer:
            with open(save_path / "tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"Saved TF-IDF model to {save_path}")

        # Save config
        import json
        with open(save_path / "config.json", 'w') as f:
            json.dump({
                'embedding_type': self.config.embedding_type,
                'model_name': self.config.model_name,
                'embedding_dim': self.embedding_dim,
                'max_features': self.config.max_features
            }, f, indent=2)

    def load_model(self, path: str):
        """Load a fitted model from disk."""
        import pickle
        import json

        load_path = Path(path)

        # Load config
        with open(load_path / "config.json", 'r') as f:
            saved_config = json.load(f)

        self.config.embedding_type = saved_config['embedding_type']
        self.embedding_dim = saved_config['embedding_dim']

        if self.config.embedding_type == "tfidf":
            with open(load_path / "tfidf_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.is_fitted = True
            logger.info(f"Loaded TF-IDF model from {load_path}")

        elif self.config.embedding_type == "sentence-transformers":
            self._setup_sentence_transformer()
            logger.info(f"Loaded sentence transformer model")
