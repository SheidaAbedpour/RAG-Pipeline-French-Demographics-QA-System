import sys
import uvicorn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import config


def check_prerequisites():
    """Check if required files exist before starting server."""
    print("🔍 Checking prerequisites...")

    required_files = [
        (config.processed_dir / "chunks_fixed.json", "Processed chunks"),
        (config.embeddings_dir / "embeddings.npy", "Vector embeddings"),
        (config.embeddings_dir / "metadata.json", "Embedding metadata"),
        (config.embeddings_dir / "vector_store" / "config.json", "Vector store config")
    ]

    missing_files = []
    for file_path, description in required_files:
        if file_path.exists():
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description}: {file_path}")
            missing_files.append((file_path, description))

    if missing_files:
        print("\n🔧 Missing required files! Please run:")
        if any("chunks" in str(f[0]) for f in missing_files):
            print("   1. python scripts/data_preprocessing.py")
        if any("embeddings" in str(f[0]) or "vector_store" in str(f[0]) for f in missing_files):
            print("   2. python scripts/create_embeddings.py")

        print("\n❌ Cannot start API server without these files.")
        return False

    print("✅ All prerequisites satisfied!")
    return True


def print_startup_info():
    """Print startup information."""
    print("\n🚀 Starting France RAG API Server...")
    print("=" * 50)
    print(f"📁 Data directory: {config.data_dir}")
    print(f"🧮 Embedding type: {config.embedding_type}")
    print(f"🤖 LLM model: {config.model_name}")
    print(f"🌡️ Temperature: {config.temperature}")
    print(f"📏 Max tokens: {config.max_tokens}")
    print("=" * 50)
    print(f"🌐 Server URL: http://{config.host}:{config.port}")
    print(f"📚 API docs: http://{config.host}:{config.port}/docs")
    print(f"🏥 Health check: http://{config.host}:{config.port}/health")
    print("=" * 50)
    print("🔥 Server is starting...")
    print("⚡ Ready to process your geography questions about France!")
    print("\n💡 Test the API:")
    print(f"   curl http://{config.host}:{config.port}/health")
    print("\n🛑 Stop server: Press Ctrl+C")


def main():
    """Main function to start the API server."""
    try:
        # Validate configuration first
        config.validate()
        print("✅ Configuration valid")

        # Check prerequisites
        if not check_prerequisites():
            return

        # Print startup info
        print_startup_info()

        # Start the server
        uvicorn.run(
            "main:app",
            host=config.host,
            port=config.port,
            reload=False,  # Set to True for development
            log_level=config.log_level,
            access_log=True
        )

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("💡 Make sure to set your TOGETHER_API_KEY environment variable")
        print("   Get your key from: https://api.together.xyz/settings/api-keys")
        print("   Add it to your .env file:")
        print("   TOGETHER_API_KEY=your_actual_api_key_here")

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("💡 Make sure you've run the data processing scripts")

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user. Goodbye!")

    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
