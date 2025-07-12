import logging
from typing import List, Optional

from src.embedding import EmbeddingModel
from src.retrieval.vector_store import VectorStore, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines vector similarity search with metadata-based filtering."""

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # Ensure the embedding model is fitted
        if self.vector_store.metadata_list:
            texts = [metadata.get('text', '') for metadata in self.vector_store.metadata_list]
            self.embedding_model.encode_batch(texts)  # Fit the model with vector store data

        logger.info("Initialized hybrid retriever")

    def search(self,
               query: str,
               k: int = 5,
               section_filter: Optional[str] = None,
               subsection_filter: Optional[str] = None,
               min_score: float = 0.0) -> List[RetrievalResult]:
        """
        Search for relevant chunks using hybrid approach.

        Args:
            query: Search query text
            k: Number of results to return
            section_filter: Filter by section name (case-insensitive)
            subsection_filter: Filter by subsection name (case-insensitive)
            min_score: Minimum similarity score threshold

        Returns:
            List of retrieval results, ranked by similarity score
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode_single(query)

            # Get more initial results to allow for filtering
            initial_k = min(k * 3, 50)  # Get 3x more results initially
            initial_results = self.vector_store.search(query_embedding, initial_k)

            # Apply filters
            filtered_results = []

            for result in initial_results:
                # Apply score threshold
                if result.score < min_score:
                    continue

                # Apply section filter
                if section_filter:
                    if not result.section or result.section.lower() != section_filter.lower():
                        continue

                # Apply subsection filter
                if subsection_filter:
                    if not result.subsection or result.subsection.lower() != subsection_filter.lower():
                        continue

                filtered_results.append(result)

                # Stop when we have enough results
                if len(filtered_results) >= k:
                    break

            # Return top-k results
            final_results = filtered_results[:k]

            logger.debug(f"Retrieved {len(final_results)} results for query: '{query[:50]}...'")

            return final_results

        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return []

    def get_available_sections(self) -> List[str]:
        """Get all available section names from the vector store."""
        if not self.vector_store.metadata_list:
            return []

        sections = set()
        for metadata in self.vector_store.metadata_list:
            section = metadata.get('section')
            if section:
                sections.add(section)

        return sorted(list(sections))

    def get_available_subsections(self, section: Optional[str] = None) -> List[str]:
        """
        Get all available subsection names.

        Args:
            section: If provided, only return subsections for this section

        Returns:
            List of subsection names
        """
        if not self.vector_store.metadata_list:
            return []

        subsections = set()
        for metadata in self.vector_store.metadata_list:
            # Filter by section if specified
            if section:
                metadata_section = metadata.get('section', '').lower()
                if metadata_section != section.lower():
                    continue

            subsection = metadata.get('subsection')
            if subsection:
                subsections.add(subsection)

        return sorted(list(subsections))

    def get_content_stats(self) -> dict:
        """Get statistics about the content in the vector store."""
        if not self.vector_store.metadata_list:
            return {"total_chunks": 0}

        stats = {
            "total_chunks": len(self.vector_store.metadata_list),
            "sections": {},
            "total_sections": 0,
            "total_subsections": 0
        }

        # Count chunks per section and subsection
        for metadata in self.vector_store.metadata_list:
            section = metadata.get('section', 'Unknown')
            subsection = metadata.get('subsection')

            if section not in stats["sections"]:
                stats["sections"][section] = {
                    "chunk_count": 0,
                    "subsections": set()
                }

            stats["sections"][section]["chunk_count"] += 1

            if subsection:
                stats["sections"][section]["subsections"].add(subsection)

        # Convert sets to counts
        for section_data in stats["sections"].values():
            section_data["subsections"] = len(section_data["subsections"])

        stats["total_sections"] = len(stats["sections"])
        stats["total_subsections"] = sum(
            section_data["subsections"] for section_data in stats["sections"].values()
        )

        return stats
