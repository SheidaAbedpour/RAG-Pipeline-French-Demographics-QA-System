import faiss
import logging
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.retrieval.vector_store import VectorStore, RetrievalConfig, RetrievalResult


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation"""

    def __init__(self, config: RetrievalConfig):
        super().__init__(config)
        self.index = None
        self.id_to_metadata = {}
        self.next_id = 0

    def _create_index(self, embedding_dim: int):
        """Create FAISS index based on configuration"""
        self.embedding_dim = embedding_dim

        if self.config.index_type == "flat":
            if self.config.distance_metric == "cosine":
                # For cosine similarity, we normalize embeddings and use inner product
                self.index = faiss.IndexFlatIP(embedding_dim)
            elif self.config.distance_metric == "euclidean":
                self.index = faiss.IndexFlatL2(embedding_dim)
            else:  # dot_product
                self.index = faiss.IndexFlatIP(embedding_dim)

        elif self.config.index_type == "ivf":
            # IVF index for larger datasets
            nlist = min(100, max(1, int(np.sqrt(1000))))  # Default nlist

            if self.config.distance_metric == "cosine":
                quantizer = faiss.IndexFlatIP(embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)

        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")

        logger.info(f"Created FAISS index: {self.config.index_type}, metric: {self.config.distance_metric}")

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings with metadata to FAISS index"""
        if self.index is None:
            self._create_index(embeddings.shape[1])

        # Normalize embeddings for cosine similarity
        if self.config.distance_metric == "cosine":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add embeddings to index
        ids = np.arange(self.next_id, self.next_id + len(embeddings))

        if self.config.index_type == "ivf" and not self.is_trained:
            # Train IVF index
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            self.is_trained = True

        self.index.add_with_ids(embeddings.astype('float32'), ids)

        # Store metadata
        for i, metadata_item in enumerate(metadata):
            self.id_to_metadata[self.next_id + i] = metadata_item

        self.next_id += len(embeddings)

        logger.info(f"Added {len(embeddings)} embeddings to FAISS index")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        """Search for similar embeddings"""
        if self.index is None:
            raise ValueError("No index available. Add embeddings first.")

        # Normalize query embedding for cosine similarity
        if self.config.distance_metric == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Set nprobe for IVF indices
        if self.config.index_type == "ivf":
            self.index.nprobe = self.config.nprobe

        # Search
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)

        # Convert to RetrievalResult objects
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            metadata = self.id_to_metadata.get(idx, {})

            result = RetrievalResult(
                chunk_id=metadata.get('chunk_id', f'chunk_{idx}'),
                text=metadata.get('text', ''),
                score=float(score),
                metadata=metadata,
                section=metadata.get('section', ''),
                subsection=metadata.get('subsection'),
                chunk_index=metadata.get('chunk_index', 0)
            )
            results.append(result)

        return results

    def save(self, path: str):
        """Save FAISS index and metadata"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "faiss_index.bin"))

        # Save metadata
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(self.id_to_metadata, f, indent=2)

        # Save config
        config_dict = {
            'index_type': self.config.index_type,
            'distance_metric': self.config.distance_metric,
            'embedding_dim': self.embedding_dim,
            'next_id': self.next_id,
            'is_trained': self.is_trained
        }

        with open(save_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str):
        """Load FAISS index and metadata"""
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Index path {path} does not exist")

        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "faiss_index.bin"))

        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            self.id_to_metadata = {int(k): v for k, v in json.load(f).items()}

        # Load config
        with open(load_path / "config.json", 'r') as f:
            config_dict = json.load(f)
            self.embedding_dim = config_dict['embedding_dim']
            self.next_id = config_dict['next_id']
            self.is_trained = config_dict['is_trained']

        logger.info(f"Loaded FAISS index from {path}")
