import sys
import os
import json
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_status(message, status="INFO"):
    """Print a status message"""
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{icons.get(status, 'ℹ️')} {message}")


def check_api_key():
    """Check if TogetherAI API key is available"""
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print_status("TogetherAI API key not found!", "ERROR")
        print("Please set your API key:")
        print("  export TOGETHER_API_KEY=your_api_key_here")
        print("  (or on Windows: set TOGETHER_API_KEY=your_api_key_here)")
        print("\nGet your API key from: https://api.together.xyz/settings/api-keys")
        return False

    print_status(f"API key found: {api_key[:10]}...{api_key[-4:]}", "SUCCESS")
    return True


def check_prerequisites():
    """Check if required files exist"""
    data_dir = Path("C:\\Users\\Sonat\\Desktop\\RAG\\RAG-Pipeline\\data")

    required_files = [
        data_dir / "processed" / "chunks_fixed.json",
        data_dir / "embeddings" / "embeddings.npy",
        data_dir / "embeddings" / "metadata.json"
    ]

    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        print_status("Missing required files:", "ERROR")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run data processing first:")
        print("1. python scripts/data_preprocessing.py --test-mode")
        print("2. python scripts/create_embeddings_simple.py --embedding-type tfidf")
        return False

    print_status("All required files found", "SUCCESS")
    return True


def setup_simple_retriever():
    """Setup the simple retriever system"""
    print_status("Setting up retriever system...")

    try:
        # Import simple components
        from src.embedding import EmbeddingModel, EmbeddingConfig
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.hybrid_retriever import HybridRetriever

        # Configure embedding model
        embedding_config = EmbeddingConfig(embedding_type="tfidf")
        embedding_model = EmbeddingModel(embedding_config)

        # Load vector store
        vector_store = VectorStore(str(Path("C:\\Users\\Sonat\\Desktop\\RAG\\RAG-Pipeline\\data") / "embeddings"))
        vector_store.load(str(Path("C:\\Users\\Sonat\\Desktop\\RAG\RAG-Pipeline\\data") / "embeddings" / "vector_store"))

        # Create retriever
        retriever = HybridRetriever(vector_store, embedding_model)

        print_status("Retriever system setup complete", "SUCCESS")
        return retriever

    except Exception as e:
        print_status(f"Error setting up retriever: {str(e)}", "ERROR")
        return None


def create_rag_generator(retriever):
    """Create RAG generator"""
    print_status("Creating RAG generator...")

    try:
        from src.generation import RAGGenerator, GenerationConfig

        # Configure generation
        config = GenerationConfig(
            api_key=os.getenv('TOGETHER_API_KEY'),
            model_name='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
            temperature=0.3,
            max_tokens=512
        )

        # Create generator
        generator = RAGGenerator(config, retriever)

        print_status("RAG generator created successfully", "SUCCESS")
        return generator

    except Exception as e:
        print_status(f"Error creating RAG generator: {str(e)}", "ERROR")
        return None


def save_generator_config(generator):
    """Save generator configuration for FastAPI"""
    print_status("Saving configuration...")

    try:
        # Create output directory
        output_dir = Path("data") / "generation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_data = {
            "model_name": generator.config.model_name,
            "temperature": generator.config.temperature,
            "max_tokens": generator.config.max_tokens,
            "setup_timestamp": time.time(),
            "status": "ready"
        }

        with open(output_dir / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)

        # Save setup status
        with open(output_dir / "setup_status.json", 'w') as f:
            json.dump({
                "generation_ready": True,
                "retriever_ready": True,
                "api_key_configured": True,
                "setup_complete": True,
                "setup_time": time.time()
            }, f, indent=2)

        print_status(f"Configuration saved to {output_dir}", "SUCCESS")

    except Exception as e:
        print_status(f"Error saving configuration: {str(e)}", "ERROR")


def main():
    """Main setup function"""
    print_header("RAG GENERATION SYSTEM SETUP")

    # Step 1: Check API key
    if not check_api_key():
        return False

    # Step 2: Check prerequisites
    if not check_prerequisites():
        return False

    # Step 3: Setup retriever
    retriever = setup_simple_retriever()
    if not retriever:
        return False

    # Step 4: Create RAG generator
    generator = create_rag_generator(retriever)
    if not generator:
        return False

    # Step 5: Save configuration
    save_generator_config(generator)

    # Success message
    print_header("SETUP COMPLETE")
    print_status("RAG generation system is ready!", "SUCCESS")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)