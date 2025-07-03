import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict
    section: str
    subsection: Optional[str] = None
    chunk_index: int = 0


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system"""
    vector_store_type: str = "faiss"  # faiss, chromadb, memory
    index_type: str = "flat"  # flat, ivf, hnsw
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    nprobe: int = 10  # for IVF indices
    ef_search: int = 128  # for HNSW indices
    storage_path: str = "data/embeddings"


class VectorStore:
    """Base class for vector stores"""

    def __init__(self, storage_path: str = "data/embeddings"):
        self.storage_path = Path(storage_path)
        self.embeddings = None
        self.metadata_list = []
        self.embedding_dim = None

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings with metadata"""
        if self.embeddings is None:
            self.embeddings = embeddings
            self.metadata_list = metadata
            self.embedding_dim = embeddings.shape[1]
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.metadata_list.extend(metadata)

        logger.info(f"Added {len(embeddings)} embeddings to vector store")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        """Search for similar embeddings"""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Add embeddings first.")

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            metadata = self.metadata_list[idx]

            result = RetrievalResult(
                chunk_id=metadata.get('chunk_id', f'chunk_{idx}'),
                text=metadata.get('text', ''),
                score=float(similarities[idx]),
                metadata=metadata,
                section=metadata.get('section', ''),
                subsection=metadata.get('subsection'),
                chunk_index=metadata.get('chunk_index', 0)
            )
            results.append(result)

        return results

    def save(self, path: str):
        """Save vector store"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(save_path / "embeddings.npy", self.embeddings)

        # Save metadata
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(self.metadata_list, f, indent=2)

        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'num_embeddings': len(self.embeddings) if self.embeddings is not None else 0
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: str):
        """Load vector store"""
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Vector store path {path} does not exist")

        # Load embeddings
        self.embeddings = np.load(load_path / "embeddings.npy")

        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            self.metadata_list = json.load(f)

        # Load config
        with open(load_path / "config.json", 'r') as f:
            config = json.load(f)
            self.embedding_dim = config['embedding_dim']

        logger.info(f"Loaded vector store from {path}")
