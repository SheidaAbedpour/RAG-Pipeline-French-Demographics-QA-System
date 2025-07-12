import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a vector similarity search."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict
    section: str
    subsection: Optional[str] = None
    chunk_index: int = 0


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    distance_metric: str = "cosine"
    storage_path: str = "data/embeddings"


class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self, storage_path: str = "data/embeddings"):
        self.storage_path = Path(storage_path)
        self.embeddings = None
        self.metadata_list = []
        self.embedding_dim = None

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings with their metadata to the store."""
        if self.embeddings is None:
            self.embeddings = embeddings.copy()
            self.metadata_list = metadata.copy()
            self.embedding_dim = embeddings.shape[1]
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.metadata_list.extend(metadata)

        logger.info(f"Added {len(embeddings)} embeddings to vector store")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        """Search for the most similar embeddings using cosine similarity."""
        if self.embeddings is None:
            logger.warning("No embeddings in store")
            return []

        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        # Build results
        results = []
        for idx in top_indices:
            if idx >= len(self.metadata_list):
                continue

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
        """Save the vector store to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings is not None:
            # Save embeddings
            np.save(save_path / "embeddings.npy", self.embeddings)

            # Save metadata
            with open(save_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.metadata_list, f, indent=2, ensure_ascii=False)

        # Save configuration
        config_data = {
            'embedding_dim': self.embedding_dim,
            'num_embeddings': len(self.embeddings) if self.embeddings is not None else 0,
            'storage_path': str(self.storage_path)
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved vector store to {save_path}")

    def load(self, path: str):
        """Load the vector store from disk."""
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Vector store path {path} does not exist")

        # Load embeddings
        embeddings_file = load_path / "embeddings.npy"
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

        # Load metadata
        metadata_file = load_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata_list = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load configuration
        config_file = load_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                self.embedding_dim = config_data.get('embedding_dim')

        logger.info(f"Loaded vector store from {load_path}")
        logger.info(f"  Embeddings: {self.embeddings.shape}")
        logger.info(f"  Metadata items: {len(self.metadata_list)}")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if self.embeddings is None:
            return {"status": "empty"}

        return {
            "status": "loaded",
            "num_embeddings": len(self.embeddings),
            "embedding_dim": self.embedding_dim,
            "num_metadata": len(self.metadata_list),
            "sections": list(set(m.get('section', '') for m in self.metadata_list)),
            "storage_path": str(self.storage_path)
        }
