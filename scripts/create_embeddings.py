"""
Simplified embedding creation script.
Focused on sentence-transformers/all-MiniLM-L6-v2 embedding model.
"""
import sys
import os
import json
import time
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import config
from src.embedding import EmbeddingModel, EmbeddingConfig
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_chunks() -> List[Dict]:
    """Load processed chunks from JSON file."""
    chunk_file = config.processed_dir / "chunks_fixed.json"

    if not chunk_file.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunk_file}\n"
            "❌ Please run data preprocessing first:\n"
            "   python scripts/data_preprocessing.py"
        )

    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {chunk_file}")
    return chunks


def create_embeddings(chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict], EmbeddingModel]:
    """Create embeddings using sentence-transformers."""
    logger.info("Creating embeddings using sentence-transformers...")

    # Configure embedding model
    embedding_config = EmbeddingConfig(
        model_name=config.embedding_model,
        cache_dir=str(config.cache_dir),
        normalize_embeddings=config.normalize_embeddings,
        batch_size=config.embedding_batch_size
    )

    # Initialize model
    embedding_model = EmbeddingModel(embedding_config)

    # Extract texts
    texts = [chunk['text'] for chunk in chunks]

    print(f"📊 Processing {len(texts)} text chunks...")
    print(f"🧮 Embedding model: {config.embedding_model}")
    print(f"📐 Expected dimension: {embedding_model.embedding_dim}")

    # Generate embeddings
    start_time = time.time()
    embeddings = embedding_model.encode_batch(texts, show_progress=True)
    end_time = time.time()

    print(f"✅ Generated {len(embeddings)} embeddings in {end_time - start_time:.2f}s")
    print(f"📐 Embedding shape: {embeddings.shape}")

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


def save_embeddings(embeddings: np.ndarray, metadata: List[Dict], embedding_model: EmbeddingModel):
    """Save embeddings and metadata to disk."""
    print("💾 Saving embeddings and metadata...")

    # Save embeddings
    embeddings_file = config.embeddings_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)

    # Save metadata
    metadata_file = config.embeddings_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save embedding model configuration
    embedding_model.save_model_config(str(config.embeddings_dir))

    print(f"✅ Saved to {config.embeddings_dir}")
    print(f"   📄 embeddings.npy ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   📄 metadata.json ({len(metadata)} items)")
    print(f"   📄 config.json (model configuration)")


def setup_vector_store(embeddings: np.ndarray, metadata: List[Dict]) -> VectorStore:
    """Setup and save vector store."""
    print("🔍 Setting up vector store...")

    vector_store = VectorStore(str(config.embeddings_dir))
    vector_store.add_embeddings(embeddings, metadata)

    # Save vector store
    store_path = config.embeddings_dir / "vector_store"
    vector_store.save(str(store_path))

    print(f"✅ Vector store saved to {store_path}")
    return vector_store


def test_retrieval(vector_store: VectorStore, embedding_model: EmbeddingModel):
    """Test the retrieval system with sample queries."""
    print("\n🧪 Testing retrieval system...")

    test_queries = [
        "What are the main mountain ranges in France?",
        "Tell me about France's climate patterns",
        "What rivers flow through France?",
        "Describe the soil types in France"
    ]

    retriever = HybridRetriever(vector_store, embedding_model)

    print("\n" + "=" * 60)
    print("🔍 RETRIEVAL TEST RESULTS")
    print("=" * 60)

    for i, query in enumerate(test_queries):
        print(f"\n🔎 Query {i + 1}: '{query}'")
        print("-" * 40)

        try:
            results = retriever.search(query, k=3)

            if results:
                for j, result in enumerate(results):
                    print(f"  {j + 1}. [{result.chunk_id}] Score: {result.score:.3f}")
                    print(f"     📂 Section: {result.section}")
                    if result.subsection:
                        print(f"     📁 Subsection: {result.subsection}")
                    preview = result.text[:120].replace('\n', ' ')
                    print(f"     📄 Text: {preview}...")
                    print()
            else:
                print("  ⚠️ No results found")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Show available sections
    sections = retriever.get_available_sections()
    print(f"📚 Available sections ({len(sections)}): {', '.join(sections)}")

    # Show content stats
    stats = retriever.get_content_stats()
    print(f"📊 Content statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total sections: {stats['total_sections']}")
    print(f"   Total subsections: {stats['total_subsections']}")


def verify_setup():
    """Verify that all required files exist."""
    required_files = [
        config.embeddings_dir / "embeddings.npy",
        config.embeddings_dir / "metadata.json",
        config.embeddings_dir / "config.json",
        config.embeddings_dir / "vector_store" / "embeddings.npy",
        config.embeddings_dir / "vector_store" / "metadata.json",
        config.embeddings_dir / "vector_store" / "config.json"
    ]

    print("\n🔍 Verifying setup...")
    all_good = True

    for file_path in required_files:
        if file_path.exists():
            size = file_path.stat().st_size / 1024 / 1024  # MB
            print(f"✅ {file_path.name} ({size:.1f} MB)")
        else:
            print(f"❌ {file_path.name} - Missing!")
            all_good = False

    return all_good


def main():
    """Main execution function."""
    try:
        print("🚀 Starting embedding creation process...")
        print(f"📁 Data directory: {config.data_dir}")
        print(f"🧮 Embedding model: {config.embedding_model}")
        print(f"📊 Chunk size: {config.chunk_size} (overlap: {config.chunk_overlap})")

        # Validate configuration
        config.validate()
        print("✅ Configuration valid")

        # Check if data preprocessing was done
        if not config.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed data directory not found: {config.processed_dir}\n"
                "❌ Please run data preprocessing first:\n"
                "   python scripts/data_preprocessing.py"
            )

        # Load chunks
        chunks = load_chunks()

        # Create embeddings
        embeddings, metadata, embedding_model = create_embeddings(chunks)

        # Save embeddings
        save_embeddings(embeddings, metadata, embedding_model)

        # Setup vector store
        vector_store = setup_vector_store(embeddings, metadata)

        # Test retrieval
        test_retrieval(vector_store, embedding_model)

        # Verify setup
        if verify_setup():
            print("\n" + "=" * 60)
            print("🎉 EMBEDDING CREATION COMPLETE!")
            print("=" * 60)
            print(f"✅ Embeddings saved to: {config.embeddings_dir}")
            print(f"✅ Vector store ready for retrieval")
            print(f"✅ Processed {len(chunks)} chunks successfully")
            print(f"✅ System ready for API deployment")
            print(f"✅ Embedding model: {config.embedding_model}")
            print(f"✅ Embedding dimension: {embedding_model.embedding_dim}")

            print("\n🔄 Next steps:")
            print("   1. Start API: python scripts/run_api.py")
            print("   2. Test system: python test_system.py")
            print("   3. Launch UI: streamlit run frontend/france_rag_ui.py")
        else:
            print("\n⚠️ Setup verification failed - some files missing")

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("💡 Make sure to set your TOGETHER_API_KEY environment variable")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during embedding creation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n❌ Embedding creation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
