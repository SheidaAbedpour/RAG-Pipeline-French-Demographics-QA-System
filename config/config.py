import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Simplified application configuration."""

    # API Configuration
    together_api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Paths
    data_dir: Path = Path("data")

    # Generation (TogetherAI)
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    temperature: float = 0.3
    max_tokens: int = 512

    # Embedding (sentence-transformers)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    embedding_batch_size: int = 32

    # Text Processing
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    default_k: int = 5
    min_score_threshold: float = 0.0

    def __post_init__(self):
        """Load from environment variables and validate."""
        # Load from environment
        self.together_api_key = os.getenv("TOGETHER_API_KEY", self.together_api_key)
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", str(self.port)))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

        # Paths
        data_dir_env = os.getenv("DATA_DIR", str(self.data_dir))
        self.data_dir = Path(data_dir_env)

        # Generation
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.temperature = float(os.getenv("TEMPERATURE", str(self.temperature)))
        self.max_tokens = int(os.getenv("MAX_TOKENS", str(self.max_tokens)))

        # Embedding
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.normalize_embeddings = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(self.embedding_batch_size)))

        # Text Processing
        self.chunk_size = int(os.getenv("CHUNK_SIZE", str(self.chunk_size)))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(self.chunk_overlap)))

        # Create directories
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "embeddings",
            self.data_dir / "embeddings" / "cache"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def raw_dir(self) -> Path:
        """Raw data directory."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Processed data directory."""
        return self.data_dir / "processed"

    @property
    def embeddings_dir(self) -> Path:
        """Embeddings directory."""
        return self.data_dir / "embeddings"

    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        return self.embeddings_dir / "cache"

    def validate(self):
        """Validate required configuration."""
        if not self.together_api_key:
            raise ValueError(
                "TOGETHER_API_KEY is required. "
                "Get it from https://api.together.xyz/settings/api-keys"
            )

        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")

        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")

    def get_embedding_config(self):
        """Get embedding configuration."""
        return {
            'model_name': self.embedding_model,
            'cache_dir': str(self.cache_dir),
            'normalize_embeddings': self.normalize_embeddings,
            'batch_size': self.embedding_batch_size
        }

    def get_generation_config(self):
        """Get generation configuration."""
        return {
            'model_name': self.model_name,
            'api_key': self.together_api_key,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

    def get_chunking_config(self):
        """Get chunking configuration."""
        return {
            'chunk_size': self.chunk_size,
            'overlap': self.chunk_overlap
        }

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'api': {
                'host': self.host,
                'port': self.port,
                'log_level': self.log_level
            },
            'paths': {
                'data_dir': str(self.data_dir),
                'raw_dir': str(self.raw_dir),
                'processed_dir': str(self.processed_dir),
                'embeddings_dir': str(self.embeddings_dir)
            },
            'embedding': {
                'model': self.embedding_model,
                'normalize': self.normalize_embeddings,
                'batch_size': self.embedding_batch_size
            },
            'generation': {
                'model': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            },
            'processing': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            },
            'retrieval': {
                'default_k': self.default_k,
                'min_score_threshold': self.min_score_threshold
            }
        }

    def __str__(self):
        """String representation of config."""
        return f"""
France RAG Configuration:
========================
🔧 API: {self.host}:{self.port}
📁 Data: {self.data_dir}
🧠 LLM: {self.model_name}
🔍 Embedding: {self.embedding_model}
📊 Chunk Size: {self.chunk_size} (overlap: {self.chunk_overlap})
🎯 Default K: {self.default_k}
        """.strip()


# Global configuration instance
config = AppConfig()


def main():
    """Test configuration."""
    print("Testing France RAG Configuration...")

    try:
        config.validate()
        print("✅ Configuration is valid!")
        print(config)

        # Print configuration details
        print("\n📊 Configuration Details:")
        import json
        print(json.dumps(config.to_dict(), indent=2))

    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 To fix:")
        print("1. Set TOGETHER_API_KEY environment variable")
        print("2. Check other environment variables if needed")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
