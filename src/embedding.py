import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
from dataclasses import dataclass
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    embedding_type: str = "tfidf"  # tfidf, openai, huggingface
    api_key: Optional[str] = None
    model_name: str = "text-embedding-3-small"
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    cache_embeddings: bool = True
    cache_dir: str = "data/embeddings"


class EmbeddingModel:
    """Handles embedding generation with caching and batching"""

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.vectorizer = None
        self.embedding_dim = None
        self.is_fitted = False

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Embedding model initialized with {self.config.embedding_type}")

    def _setup_tfidf_vectorizer(self, texts: List[str]):
        """Setup TF-IDF vectorizer"""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )

            # Fit vectorizer on all texts
            self.vectorizer.fit(texts)
            self.embedding_dim = len(self.vectorizer.vocabulary_)
            self.is_fitted = True

            logger.info(f"TF-IDF vectorizer fitted with {self.embedding_dim} features")

    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        if not self.config.api_key:
            raise ValueError("OpenAI API key required for OpenAI embeddings")

        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }

        embeddings = []
        batch_size = 100  # OpenAI limit

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            data = {
                'input': batch,
                'model': self.config.model_name
            }

            response = requests.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")

            result = response.json()
            batch_embeddings = [item['embedding'] for item in result['data']]
            embeddings.extend(batch_embeddings)

            # Rate limiting
            time.sleep(0.1)

        return np.array(embeddings)

    def _get_huggingface_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Hugging Face API"""
        # This is a fallback using requests to HF inference API
        if not self.config.api_key:
            logger.warning("No HF API key provided, using free tier (rate limited)")

        headers = {}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'

        embeddings = []

        for text in texts:
            response = requests.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers=headers,
                json={"inputs": text},
                timeout=30
            )

            if response.status_code == 200:
                embedding = response.json()
                embeddings.append(embedding)
            else:
                logger.warning(f"HF API error for text: {text[:50]}...")
                # Fallback to zero vector
                embeddings.append([0.0] * 384)

            time.sleep(0.2)  # Rate limiting

        return np.array(embeddings)

    def encode_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode a batch of texts"""
        if self.config.embedding_type == "tfidf":
            self._setup_tfidf_vectorizer(texts)
            embeddings = self.vectorizer.transform(texts).toarray()

        elif self.config.embedding_type == "openai":
            embeddings = self._get_openai_embeddings(texts)

        elif self.config.embedding_type == "huggingface":
            embeddings = self._get_huggingface_embeddings(texts)

        else:
            raise ValueError(f"Unsupported embedding type: {self.config.embedding_type}")

        if show_progress:
            logger.info(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode_batch([text], show_progress=False)[0]

    def get_model_info(self) -> Dict:
        """Get information about the model"""
        return {
            'embedding_type': self.config.embedding_type,
            'embedding_dim': self.embedding_dim,
            'model_name': self.config.model_name,
            'is_fitted': self.is_fitted
        }

