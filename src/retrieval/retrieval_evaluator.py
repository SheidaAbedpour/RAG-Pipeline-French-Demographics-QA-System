import time
from typing import List,Dict, Optional
import numpy as np

from src.retrieval.faiss_vector_store import FAISSVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.memory_vector_store import MemoryVectorStore
from src.retrieval.vector_store import RetrievalConfig, VectorStore


class RetrievalEvaluator:
    """Evaluates retrieval performance"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def evaluate_precision_at_k(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """
        Evaluate precision@k for a set of test queries

        Args:
            test_queries: List of dicts with 'query' and 'relevant_chunk_ids'
            k: Number of results to consider
        """
        precisions = []

        for test_query in test_queries:
            query = test_query['query']
            relevant_ids = set(test_query['relevant_chunk_ids'])

            results = self.retriever.search(query, k=k)
            retrieved_ids = {result.chunk_id for result in results}

            # Calculate precision@k
            if results:
                precision = len(relevant_ids.intersection(retrieved_ids)) / len(results)
                precisions.append(precision)

        return {
            'precision_at_k': np.mean(precisions) if precisions else 0.0,
            'num_queries': len(test_queries),
            'individual_precisions': precisions
        }

    def evaluate_recall_at_k(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """Evaluate recall@k for a set of test queries"""
        recalls = []

        for test_query in test_queries:
            query = test_query['query']
            relevant_ids = set(test_query['relevant_chunk_ids'])

            results = self.retriever.search(query, k=k)
            retrieved_ids = {result.chunk_id for result in results}

            # Calculate recall@k
            if relevant_ids:
                recall = len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids)
                recalls.append(recall)

        return {
            'recall_at_k': np.mean(recalls) if recalls else 0.0,
            'num_queries': len(test_queries),
            'individual_recalls': recalls
        }

    def benchmark_retrieval_speed(self, num_queries: int = 100) -> Dict:
        """Benchmark retrieval speed"""
        # Generate random queries
        sample_texts = ["France geography", "mountain regions", "climate patterns",
                        "river systems", "soil types", "vegetation", "temperature", "rainfall"]

        start_time = time.time()

        for i in range(num_queries):
            query = sample_texts[i % len(sample_texts)]
            self.retriever.search(query, k=5)

        end_time = time.time()
        total_time = end_time - start_time

        return {
            'total_time': total_time,
            'avg_query_time': total_time / num_queries,
            'queries_per_second': num_queries / total_time
        }


def create_vector_store(config: RetrievalConfig) -> VectorStore:
    """Factory function to create vector store"""
    if config.vector_store_type == "faiss":
        return FAISSVectorStore(config)
    elif config.vector_store_type == "memory":
        return MemoryVectorStore(config)
    else:
        raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")
