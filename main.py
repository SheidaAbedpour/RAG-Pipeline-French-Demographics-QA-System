from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import asyncio
from typing import Optional
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.schemas import (
    RetrievalRequest, RetrievalResponse, RetrievalSource,
    GenerationRequest, GenerationResponse, HealthResponse, ErrorResponse
)
from src.generation import RAGGenerator, GenerationConfig
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for dependency injection
rag_generator: Optional[RAGGenerator] = None
app_metrics = {
    "total_requests": 0,
    "retrieval_requests": 0,
    "generation_requests": 0,
    "errors": 0,
    "start_time": time.time()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app"""
    # Startup
    logger.info("Starting up RAG API...")
    await startup_event()

    yield

    # Shutdown
    logger.info("Shutting down RAG API...")
    await shutdown_event()


async def startup_event():
    """Initialize RAG system on startup"""
    global rag_generator

    try:
        logger.info("Initializing RAG system...")

        # Get API key
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable required")

        # Setup retriever
        data_dir = os.getenv('DATA_DIR', 'data')

        from src.embedding import EmbeddingModel, EmbeddingConfig
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.hybrid_retriever import HybridRetriever

        # Load retriever
        embedding_config = EmbeddingConfig(embedding_type="tfidf")
        embedding_model = EmbeddingModel(embedding_config)

        vector_store = VectorStore(str(Path(data_dir) / "embeddings"))
        vector_store.load(str(Path(data_dir) / "embeddings" / "vector_store"))

        retriever = HybridRetriever(vector_store, embedding_model)

        # Setup generation
        generation_config = GenerationConfig(
            api_key=api_key,
            model_name=os.getenv('MODEL_NAME', 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'),
            temperature=float(os.getenv('TEMPERATURE', '0.3')),
            max_tokens=int(os.getenv('MAX_TOKENS', '512'))
        )

        rag_generator = RAGGenerator(generation_config, retriever)

        logger.info("RAG system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise


async def shutdown_event():
    """Cleanup on shutdown"""
    global rag_generator

    if rag_generator:
        # Export metrics and history
        try:
            output_dir = Path("data") / "api_logs"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Export generation history
            rag_generator.export_history(str(output_dir / "generation_history.json"))

            # Export app metrics
            with open(output_dir / "app_metrics.json", 'w') as f:
                json.dump(app_metrics, f, indent=2)

            logger.info("Metrics and history exported successfully")

        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="France RAG API",
    description="Retrieval-Augmented Generation API for France Geography Data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get RAG generator
async def get_rag_generator() -> RAGGenerator:
    """Dependency to get RAG generator instance"""
    global rag_generator
    if rag_generator is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_generator


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Middleware to track requests"""
    global app_metrics

    start_time = time.time()
    app_metrics["total_requests"] += 1

    try:
        response = await call_next(request)
        return response
    except Exception as e:
        app_metrics["errors"] += 1
        logger.error(f"Request error: {str(e)}")
        raise
    finally:
        process_time = time.time() - start_time
        logger.info(f"Request processed in {process_time:.2f}s")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global rag_generator, app_metrics

    components = {
        "rag_generator": "healthy" if rag_generator else "unhealthy",
        "api": "healthy"
    }

    # Test RAG generator if available
    if rag_generator:
        try:
            # Quick test
            test_results = rag_generator.retriever.search("test", k=1)
            components["retrieval"] = "healthy" if test_results else "unhealthy"
        except Exception as e:
            components["retrieval"] = f"unhealthy: {str(e)}"

    status = "healthy" if all(comp == "healthy" for comp in components.values()) else "unhealthy"

    return HealthResponse(
        status=status,
        components=components
    )


# Retrieval endpoint
@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_sources(
        request: RetrievalRequest,
        rag_generator: RAGGenerator = Depends(get_rag_generator)
):
    """Retrieve relevant sources for a query"""
    global app_metrics
    app_metrics["retrieval_requests"] += 1

    try:
        logger.info(f"Processing retrieval request: {request.query[:50]}...")

        # Retrieve sources
        results = rag_generator.retriever.search(
            request.query,
            k=request.k,
            section_filter=request.section_filter,
            min_score=request.min_score
        )

        # Convert to response format
        sources = []
        for result in results:
            source = RetrievalSource(
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                section=result.section,
                subsection=result.subsection,
                metadata=result.metadata
            )
            sources.append(source)

        response = RetrievalResponse(
            query=request.query,
            sources=sources,
            metadata={
                "total_sources": len(sources),
                "parameters": request.dict()
            }
        )

        logger.info(f"Retrieval completed: {len(sources)} sources found")
        return response

    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Generation endpoint
@app.post("/generate", response_model=GenerationResponse)
async def generate_answer(
        request: GenerationRequest,
        rag_generator: RAGGenerator = Depends(get_rag_generator)
):
    """Generate an answer using RAG"""
    global app_metrics
    app_metrics["generation_requests"] += 1

    try:
        logger.info(f"Processing generation request: {request.query[:50]}...")

        # Generate response
        result = rag_generator.generate_response(
            request.query,
            k=request.k,
            section_filter=request.section_filter,
            min_score=request.min_score,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Convert sources to response format
        sources = []
        for source_data in result["sources"]:
            source = RetrievalSource(
                chunk_id=source_data["chunk_id"],
                text=source_data["text_preview"],
                score=source_data["score"],
                section=source_data["section"],
                subsection=source_data.get("subsection"),
                metadata={"full_text_available": True}
            )
            sources.append(source)

        response = GenerationResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            metadata=result["metadata"]
        )

        logger.info(f"Generation completed: {len(result['answer'])} characters generated")
        return response

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    global app_metrics, rag_generator

    metrics = app_metrics.copy()
    metrics["uptime"] = time.time() - app_metrics["start_time"]

    if rag_generator:
        rag_metrics = rag_generator.get_performance_metrics()
        metrics["rag_metrics"] = rag_metrics

    return metrics


# Available sections endpoint
@app.get("/sections")
async def get_available_sections(
        rag_generator: RAGGenerator = Depends(get_rag_generator)
):
    """Get available sections for filtering"""
    try:
        sections = rag_generator.retriever.get_available_sections()
        return {"sections": sections}
    except Exception as e:
        logger.error(f"Error getting sections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Available subsections endpoint
@app.get("/subsections")
async def get_available_subsections(
        section: Optional[str] = None,
        rag_generator: RAGGenerator = Depends(get_rag_generator)
):
    """Get available subsections for filtering"""
    try:
        subsections = rag_generator.retriever.get_available_subsections(section)
        return {"subsections": subsections, "section": section}
    except Exception as e:
        logger.error(f"Error getting subsections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            details=f"HTTP {exc.status_code}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc)
        ).dict()
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "France RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "retrieve": "/retrieve",
            "generate": "/generate",
            "sections": "/sections",
            "subsections": "/subsections",
            "metrics": "/metrics"
        }
    }


# Run the app
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Run the app
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info")
    )
