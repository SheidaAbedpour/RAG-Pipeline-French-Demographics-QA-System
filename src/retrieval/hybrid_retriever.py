from typing import List, Optional

from src.embedding import EmbeddingModel
from src.retrieval.vector_store import VectorStore, RetrievalResult


class HybridRetriever:
    """Hybrid retrieval combining vector search with metadata filtering"""

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def search(self,
               query: str,
               k: int = 5,
               section_filter: Optional[str] = None,
               subsection_filter: Optional[str] = None,
               min_score: float = 0.0) -> List[RetrievalResult]:
        """Search with optional filtering"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)

        # Get initial results
        initial_k = min(k * 3, 50)
        initial_results = self.vector_store.search(query_embedding, initial_k)

        # Apply filters
        filtered_results = []

        for result in initial_results:
            # Score threshold
            if result.score < min_score:
                continue

            # Section filter
            if section_filter and result.section.lower() != section_filter.lower():
                continue

            # Subsection filter
            if subsection_filter and (
                    not result.subsection or
                    result.subsection.lower() != subsection_filter.lower()
            ):
                continue

            filtered_results.append(result)

            if len(filtered_results) >= k:
                break

        return filtered_results[:k]

    def get_available_sections(self) -> List[str]:
        """Get all available sections"""
        sections = set()
        for metadata in self.vector_store.metadata_list:
            if 'section' in metadata:
                sections.add(metadata['section'])
        return sorted(sections)

    def get_available_subsections(self, section: str = None) -> List[str]:
        """Get all available subsections"""
        subsections = set()
        for metadata in self.vector_store.metadata_list:
            if section and metadata.get('section', '').lower() != section.lower():
                continue
            if 'subsection' in metadata and metadata['subsection']:
                subsections.add(metadata['subsection'])
        return sorted(subsections)

