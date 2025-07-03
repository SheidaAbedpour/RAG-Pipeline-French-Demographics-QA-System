from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RetrievalRequest(BaseModel):
    """Request schema for retrieval endpoint"""
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    section_filter: Optional[str] = Field(default=None, description="Filter by section name")
    subsection_filter: Optional[str] = Field(default=None, description="Filter by subsection name")
    min_score: float = Field(default=0.0, description="Minimum similarity score threshold", ge=0.0, le=1.0)


class RetrievalSource(BaseModel):
    """Schema for a retrieval source"""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Content of the text chunk")
    score: float = Field(..., description="Similarity score")
    section: str = Field(..., description="Section name")
    subsection: Optional[str] = Field(default=None, description="Subsection name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievalResponse(BaseModel):
    """Response schema for retrieval endpoint"""
    query: str = Field(..., description="Original search query")
    sources: List[RetrievalSource] = Field(..., description="Retrieved sources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class GenerationRequest(BaseModel):
    """Request schema for generation endpoint"""
    query: str = Field(..., description="Question to answer", min_length=1, max_length=1000)
    k: int = Field(default=5, description="Number of sources to retrieve", ge=1, le=20)
    section_filter: Optional[str] = Field(default=None, description="Filter by section name")
    min_score: float = Field(default=0.0, description="Minimum similarity score threshold", ge=0.0, le=1.0)
    temperature: float = Field(default=0.3, description="Generation temperature", ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, description="Maximum tokens to generate", ge=50, le=2048)


class GenerationResponse(BaseModel):
    """Response schema for generation endpoint"""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[RetrievalSource] = Field(..., description="Sources used for generation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    components: Dict[str, str] = Field(default_factory=dict, description="Component status")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
