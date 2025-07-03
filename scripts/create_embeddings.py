import sys
import os
import json
import argparse
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.embedding import EmbeddingModel, EmbeddingConfig
from src.retrieval.retrieval_evaluator import create_vector_store, RetrievalEvaluator
from src.retrieval.vector_store import RetrievalConfig, RetrievalResult
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_chunks(data_dir: str, chunking_strategy: str = "fixed") -> List[Dict]:
    """Load processed chunks from JSON file"""
    chunk_file = "C:\\Users\\Sonat\\Desktop\\RAG\\RAG-Pipeline\\data\\processed\\chunks_fixed.json"

    with open(chunk_file, 'r') as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {chunk_file}")
    return chunks


def create_embeddings(chunks: List[Dict], config: EmbeddingConfig) -> Tuple[np.ndarray, List[Dict]]:
    """Create embeddings using simple methods"""
    logger.info("Creating embeddings...")

    # Initialize embedding model
    embedding_model = EmbeddingModel(config)

    # Extract texts
    texts = [chunk['text'] for chunk in chunks]

    # Generate embeddings
    start_time = time.time()
    embeddings = embedding_model.encode_batch(texts, show_progress=True)
    end_time = time.time()

    logger.info(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
    logger.info(f"Embedding shape: {embeddings.shape}")

    # Prepare metadata
    metadata = []
    for chunk in chunks:
        metadata.append({
            'chunk_id': chunk['chunk_id'],
            'text': chunk['text'],
            'section': chunk['section'],
            'subsection': chunk.get('subsection'),
            'source_url': chunk['source_url'],
            'chunk_index': chunk['chunk_index'],
            'chunk_size': chunk['chunk_size'],
            'metadata': chunk['metadata']
        })

    return embeddings, metadata, embedding_model


def test_retrieval(vector_store: VectorStore, embedding_model: EmbeddingModel):
    """Test the simple retrieval system"""
    logger.info("Testing simple retrieval system...")

    test_queries = [
        "What are the main mountain ranges in France?",
        "Tell me about France's climate patterns",
        "What rivers flow through France?",
        "Describe the soil types in France",
        "What is the vegetation like in France?"
    ]

    retriever = HybridRetriever(vector_store, embedding_model)

    print("\n" + "=" * 60)
    print("SIMPLE RETRIEVAL TEST RESULTS")
    print("=" * 60)

    for i, query in enumerate(test_queries):
        print(f"\nQuery {i + 1}: '{query}'")
        print("-" * 40)

        results = retriever.search(query, k=3)

        for j, result in enumerate(results):
            print(f"  {j + 1}. [{result.chunk_id}] Score: {result.score:.3f}")
            print(f"     Section: {result.section}")
            if result.subsection:
                print(f"     Subsection: {result.subsection}")
            print(f"     Text: {result.text[:150]}...")
            print()

    print(f"\nAvailable sections: {retriever.get_available_sections()}")


def main():
    # parser = argparse.ArgumentParser(description='Create embeddings using simple methods')
    # parser.add_argument('--data-dir', default="C:\\Users\\Sonat\\Desktop\\RAG\\RAG-Pipeline\\data",
    #                     help='Data directory')
    # parser.add_argument('--chunking-strategy', default='fixed',
    #                     choices=['fixed', 'sentence', 'semantic'],
    #                     help='Chunking strategy to use')
    # parser.add_argument('--embedding-type', default='tfidf',
    #                     choices=['tfidf', 'openai', 'huggingface'],
    #                     help='Embedding type to use')
    # parser.add_argument('--api-key', default=None,
    #                     help='API key for OpenAI or Hugging Face')
    # parser.add_argument('--max-features', type=int, default=5000,
    #                     help='Max features for TF-IDF')
    # parser.add_argument('--test-only', action='store_true',
    #                     help='Only test existing embeddings')

    # args = parser.parse_args()

    embedding_type = 'tfidf'
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    chunking_strategy = 'semantic'
    max_features = 5000

    output_dir = "C:\\Users\\Sonat\\Desktop\\RAG\\RAG-Pipeline\\data\\embeddings"
    data_dir = "C:\\Users\\Sonat\\Desktop\\RAG\\RAG-Pipeline\\data\\processed"

    try:
        # Load chunks
        chunks = load_processed_chunks(data_dir, chunking_strategy)

        # Configure embedding model
        embedding_config = EmbeddingConfig(
            embedding_type=embedding_type,
            api_key=api_key,
            max_features=max_features,
            cache_dir=str(output_dir + "\\cache")
        )

        # Create embeddings
        embeddings, metadata, embedding_model = create_embeddings(chunks, embedding_config)

        # Save embeddings
        np.save(output_dir + "\\embeddings.npy", embeddings)
        with open(output_dir + "\\metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Setup vector store
        vector_store = VectorStore(str(output_dir))
        vector_store.add_embeddings(embeddings, metadata)
        vector_store.save(str(output_dir + "\\vector_store"))

        logger.info(f"Saved embeddings to {output_dir}")

        # Test retrieval
        test_retrieval(vector_store, embedding_model)

        print("\n" + "=" * 60)
        print("SIMPLE EMBEDDING CREATION COMPLETE!")
        print("=" * 60)
        print(f"✅ Embeddings saved to: {output_dir}")
        print(f"✅ Embedding type: {embedding_type}")
        print(f"✅ Chunking strategy: {chunking_strategy}")

    except Exception as e:
        logger.error(f"Error during embedding creation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
