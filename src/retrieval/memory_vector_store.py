import json
from pathlib import Path
import numpy as np
from typing import List,Dict, Optional
import logging
from torch import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.retrieval.vector_store import VectorStore, RetrievalResult, RetrievalConfig


class MemoryVectorStore(VectorStore):
    """In-memory vector store for simple cosine similarity"""

    def __init__(self, config: RetrievalConfig):
        super().__init__(config)
        self.embeddings = None
        self.metadata_list = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings to memory store"""
        if self.embeddings is None:
            self.embeddings = embeddings
            self.metadata_list = metadata
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.metadata_list.extend(metadata)

        logger.info(f"Added {len(embeddings)} embeddings to memory store")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        """Search using cosine similarity"""
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
        """Save memory store"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(save_path / "embeddings.npy", self.embeddings)

        # Save metadata
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(self.metadata_list, f, indent=2)

        logger.info(f"Saved memory store to {path}")

    def load(self, path: str):
        """Load memory store"""
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Store path {path} does not exist")

        # Load embeddings
        self.embeddings = np.load(load_path / "embeddings.npy")

        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            self.metadata_list = json.load(f)

        logger.info(f"Loaded memory store from {path}")
