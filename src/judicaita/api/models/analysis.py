"""
Analysis-related API models.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ReasoningStepResponse(BaseModel):
    """A single step in a reasoning trace."""

    step_id: str
    step_type: str
    description: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float
    sources: list[str] = Field(default_factory=list)


class ReasoningTraceRequest(BaseModel):
    """Request to generate a reasoning trace."""

    query: str = Field(..., description="Legal question or query to analyze")
    context: str = Field(..., description="Relevant context (case facts, statutes)")
    citations: list[str] | None = Field(
        default=None, description="Optional list of relevant citations"
    )
    checkpoint_path: str | None = Field(
        default=None, description="Optional path to GRPO-tuned checkpoint"
    )


class ReasoningTraceResponse(BaseModel):
    """Response containing a reasoning trace."""

    trace_id: str = Field(..., description="Unique trace identifier")
    query: str = Field(..., description="Original query")
    steps: list[ReasoningStepResponse] = Field(
        default_factory=list, description="Reasoning steps"
    )
    final_conclusion: str = Field(..., description="Final conclusion")
    overall_confidence: float = Field(..., description="Overall confidence score")
    citations_used: list[str] = Field(default_factory=list)
    model_info: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CitationResponse(BaseModel):
    """A single extracted citation."""

    citation_id: str = Field(..., description="Unique citation identifier")
    raw_citation: str = Field(..., description="Original citation text")
    citation_type: str = Field(..., description="Type of citation (case, statute, etc)")
    jurisdiction: str | None = Field(default=None, description="Legal jurisdiction")
    is_valid: bool = Field(default=False, description="Whether citation was validated")
    url: str | None = Field(default=None, description="Link to citation source")
    context: str | None = Field(
        default=None, description="Surrounding text context"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationExtractionRequest(BaseModel):
    """Request to extract citations from text."""

    text: str = Field(..., description="Text containing legal citations")
    validate_citations: bool = Field(
        default=True,
        description="Validate citations against databases",
        alias="validate",
    )


class CitationExtractionResponse(BaseModel):
    """Response containing extracted citations."""

    citations: list[CitationResponse] = Field(
        default_factory=list, description="Extracted citations"
    )
    total_count: int = Field(default=0, description="Total citations found")
    validated_count: int = Field(default=0, description="Number validated")
    extraction_time_ms: float = Field(default=0.0)


class SummaryRequest(BaseModel):
    """Request to generate a summary."""

    text: str = Field(..., description="Legal text to summarize")
    summary_level: str = Field(
        default="medium",
        description="Summary detail: brief, short, medium, detailed",
    )
    reading_level: str = Field(
        default="high_school",
        description="Target reading level: elementary, middle_school, high_school, college, professional",
    )
    include_sections: bool = Field(
        default=True, description="Include detailed sections"
    )


class SummarySectionResponse(BaseModel):
    """A section within a summary."""

    title: str
    content: str
    key_points: list[str] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    """Response containing a generated summary."""

    summary: str = Field(..., description="Generated summary text")
    summary_level: str = Field(..., description="Summary detail level used")
    reading_level: str = Field(..., description="Reading level used")
    sections: list[SummarySectionResponse] = Field(
        default_factory=list, description="Summary sections"
    )
    key_terms: dict[str, str] = Field(
        default_factory=dict, description="Key terms and definitions"
    )
    key_takeaways: list[str] = Field(
        default_factory=list, description="Key takeaways"
    )
    original_length: int = Field(default=0, description="Original text length")
    summary_length: int = Field(default=0, description="Summary text length")
    compression_ratio: float = Field(default=0.0, description="Compression ratio")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
