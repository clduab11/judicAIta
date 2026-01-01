"""
Document-related API models.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadataResponse(BaseModel):
    """Metadata about a processed document."""

    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    word_count: int | None = None
    created_date: str | None = None
    modified_date: str | None = None
    file_type: str | None = None


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="Detected file type")
    status: str = Field(default="uploaded", description="Processing status")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    message: str = Field(default="Document uploaded successfully")


class DocumentResponse(BaseModel):
    """Response containing processed document data."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    text: str = Field(..., description="Extracted text content")
    text_length: int = Field(..., description="Length of extracted text")
    metadata: DocumentMetadataResponse = Field(
        default_factory=DocumentMetadataResponse,
        description="Document metadata",
    )
    sections: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Document sections with page numbers",
    )
    tables: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted tables",
    )
    citations_found: int = Field(
        default=0, description="Number of citations detected"
    )
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentAnalyzeRequest(BaseModel):
    """Request to analyze a document."""

    extract_citations: bool = Field(
        default=True,
        description="Extract and map legal citations",
    )
    generate_summary: bool = Field(
        default=True,
        description="Generate plain-English summary",
    )
    summary_level: str = Field(
        default="medium",
        description="Summary detail level: brief, short, medium, detailed",
    )
    reading_level: str = Field(
        default="high_school",
        description="Target reading level for summary",
    )
    generate_reasoning_trace: bool = Field(
        default=False,
        description="Generate reasoning trace (requires query)",
    )
    query: str | None = Field(
        default=None,
        description="Legal query for reasoning trace generation",
    )


class DocumentAnalyzeResponse(BaseModel):
    """Response from document analysis."""

    document_id: str = Field(..., description="Document identifier")
    status: str = Field(default="completed", description="Analysis status")
    citations_count: int | None = Field(
        default=None, description="Number of citations found"
    )
    summary_generated: bool = Field(default=False)
    reasoning_trace_generated: bool = Field(default=False)
    analysis_time_ms: float = Field(
        default=0.0, description="Analysis time in milliseconds"
    )
    message: str = Field(default="Analysis completed")
