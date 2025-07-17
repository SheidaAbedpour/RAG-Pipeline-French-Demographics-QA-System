import sys
import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.config import config
from src.schemas import (
    RetrievalRequest, RetrievalResponse, RetrievalSource,
    GenerationRequest, GenerationResponse, HealthResponse, ErrorResponse
)
from src.generation import RAGGenerator, GenerationConfig
from src.embedding import EmbeddingModel, EmbeddingConfig
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
rag_generator: Optional[RAGGenerator] = None
app_metrics = {
    "total_requests": 0,
    "retrieval_requests": 0,
    "generation_requests": 0,
    "errors": 0,
    "start_time": time.time()
}


async def initialize_rag_system():
    """Initialize the RAG system components."""
    global rag_generator

    try:
        logger.info("Initializing RAG system...")

        # Validate configuration
        config.validate()
        logger.info("✅ Configuration validated")

        # Setup embedding model with sentence-transformers
        embedding_config = EmbeddingConfig(
            model_name=config.embedding_model,
            cache_dir=str(config.cache_dir),
            normalize_embeddings=config.normalize_embeddings,
            batch_size=config.embedding_batch_size
        )

        logger.info(f"Loading embedding model: {config.embedding_model}")
        embedding_model = EmbeddingModel(embedding_config)
        logger.info(f"✅ Embedding model loaded (dim: {embedding_model.embedding_dim})")

        # Load vector store
        vector_store = VectorStore(str(config.embeddings_dir))
        store_path = config.embeddings_dir / "vector_store"

        if not store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {store_path}. "
                "Run scripts/create_embeddings.py first."
            )

        vector_store.load(str(store_path))
        logger.info("✅ Vector store loaded")

        # Setup retriever
        retriever = HybridRetriever(vector_store, embedding_model)
        logger.info("✅ Hybrid retriever initialized")

        # Setup generation
        generation_config = GenerationConfig(
            api_key=config.together_api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        rag_generator = RAGGenerator(generation_config, retriever)
        logger.info("✅ RAG generator initialized")

        # Test the system
        test_results = retriever.search("test", k=1)
        logger.info(f"✅ System test: {len(test_results)} results found")

        logger.info("🎉 RAG system initialized successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("🚀 Starting France RAG API...")
    await initialize_rag_system()

    yield

    # Shutdown
    logger.info("👋 Shutting down France RAG API...")


# Initialize FastAPI
app = FastAPI(
    title="France RAG API",
    description="Retrieval-Augmented Generation API for France Geography using sentence-transformers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_requests(request, call_next):
    """Track request metrics."""
    app_metrics["total_requests"] += 1
    start_time = time.time()

    try:
        response = await call_next(request)
        return response
    except Exception as e:
        app_metrics["errors"] += 1
        logger.error(f"Request error: {str(e)}")
        raise
    finally:
        process_time = time.time() - start_time
        logger.debug(f"Request processed in {process_time:.2f}s")


def get_rag_generator() -> RAGGenerator:
    """Dependency to get RAG generator."""
    if rag_generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    return rag_generator


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "France RAG API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation for France Geography",
        "embedding_model": config.embedding_model,
        "generation_model": config.model_name,
        "endpoints": {
            "health": "/health",
            "retrieve": "/retrieve",
            "generate": "/generate",
            "sections": "/sections",
            "metrics": "/metrics"
        },
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {"api": "healthy"}

    if rag_generator:
        try:
            # Quick retrieval test
            test_results = rag_generator.retriever.search("test", k=1)
            components["rag_system"] = "healthy"
            components["retrieval"] = "healthy" if test_results else "no_data"
            components["embedding_model"] = config.embedding_model
            components["generation_model"] = config.model_name
        except Exception as e:
            components["rag_system"] = f"unhealthy: {str(e)}"
    else:
        components["rag_system"] = "not_initialized"

    status = "healthy" if all(
        "healthy" in comp for comp in components.values()
    ) else "unhealthy"

    return HealthResponse(status=status, components=components)


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_sources(
        request: RetrievalRequest,
        generator: RAGGenerator = Depends(get_rag_generator)
):
    """Retrieve relevant sources for a query."""
    app_metrics["retrieval_requests"] += 1

    try:
        logger.info(f"Retrieving for: '{request.query[:50]}...'")

        # Search for relevant chunks
        results = generator.retriever.search(
            request.query,
            k=request.k,
            section_filter=request.section_filter,
            min_score=request.min_score
        )

        # Convert to response format
        sources = [
            RetrievalSource(
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                section=result.section,
                subsection=result.subsection,
                metadata=result.metadata
            )
            for result in results
        ]

        logger.info(f"Found {len(sources)} relevant sources")

        return RetrievalResponse(
            query=request.query,
            sources=sources,
            metadata={
                "total_sources": len(sources),
                "embedding_model": config.embedding_model,
                "search_parameters": request.dict()
            }
        )

    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate_answer(
        request: GenerationRequest,
        generator: RAGGenerator = Depends(get_rag_generator)
):
    """Generate an answer using RAG."""
    app_metrics["generation_requests"] += 1

    try:
        logger.info(f"Generating answer for: '{request.query[:50]}...'")

        # Generate response
        result = generator.generate_response(
            request.query,
            k=request.k,
            section_filter=request.section_filter,
            min_score=request.min_score,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Convert sources
        sources = [
            RetrievalSource(
                chunk_id=source["chunk_id"],
                text=source["text_preview"],
                score=source["score"],
                section=source["section"],
                subsection=source.get("subsection"),
                metadata={"preview": True}
            )
            for source in result["sources"]
        ]

        logger.info(f"Generated {len(result['answer'])} character answer")

        return GenerationResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            metadata={
                **result["metadata"],
                "embedding_model": config.embedding_model,
                "generation_model": config.model_name
            }
        )

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sections")
async def get_sections(generator: RAGGenerator = Depends(get_rag_generator)):
    """Get available content sections."""
    try:
        sections = generator.retriever.get_available_sections()
        return {
            "sections": sections,
            "total_sections": len(sections),
            "embedding_model": config.embedding_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    metrics = app_metrics.copy()
    metrics["uptime_seconds"] = time.time() - app_metrics["start_time"]
    metrics["embedding_model"] = config.embedding_model
    metrics["generation_model"] = config.model_name

    if rag_generator:
        rag_metrics = rag_generator.get_performance_metrics()
        metrics["rag_performance"] = rag_metrics

    return metrics


@app.get("/config")
async def get_config():
    """Get system configuration."""
    return {
        "embedding": {
            "model": config.embedding_model,
            "normalize": config.normalize_embeddings,
            "batch_size": config.embedding_batch_size
        },
        "generation": {
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        },
        "processing": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        },
        "retrieval": {
            "default_k": config.default_k,
            "min_score_threshold": config.min_score_threshold
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            details=f"HTTP {exc.status_code}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details="An unexpected error occurred"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting France RAG API Server...")
    print(f"🧮 Embedding model: {config.embedding_model}")
    print(f"🤖 Generation model: {config.model_name}")
    print(f"🌐 Server: http://{config.host}:{config.port}")
    print(f"📚 Docs: http://{config.host}:{config.port}/docs")

    uvicorn.run(
        "fastapi_run:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level=config.log_level
    )
