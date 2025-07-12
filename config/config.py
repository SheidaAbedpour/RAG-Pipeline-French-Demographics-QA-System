"""
Central configuration management for the RAG pipeline.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Application configuration with environment variable support."""

    # API Configuration
    together_api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Paths
    data_dir: Path = Path("data")

    # Generation
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    temperature: float = 0.3
    max_tokens: int = 512

    # Retrieval
    embedding_type: str = "tfidf"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_features: int = 5000

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

        # Retrieval
        self.embedding_type = os.getenv("EMBEDDING_TYPE", self.embedding_type)
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
            self.data_dir / "embeddings"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    def validate(self):
        """Validate required configuration."""
        if not self.together_api_key:
            raise ValueError(
                "TOGETHER_API_KEY is required. "
                "Get it from https://api.together.xyz/settings/api-keys"
            )


# Global configuration instance
config = AppConfig()
