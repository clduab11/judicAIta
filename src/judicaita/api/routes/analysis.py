"""
Analysis endpoints for reasoning traces, citations, and summaries.
"""

import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from judicaita.api.models.analysis import (
    CitationExtractionRequest,
    CitationExtractionResponse,
    CitationResponse,
    ReasoningStepResponse,
    ReasoningTraceRequest,
    ReasoningTraceResponse,
    SummaryRequest,
    SummaryResponse,
    SummarySectionResponse,
)

router = APIRouter()


@router.post("/analysis/reasoning-trace", response_model=ReasoningTraceResponse)
async def generate_reasoning_trace(
    request: ReasoningTraceRequest,
) -> ReasoningTraceResponse:
    """
    Generate a reasoning trace for a legal query.

    Creates a step-by-step explainable reasoning trace showing how
    the AI arrives at its conclusions.

    Args:
        request: Query, context, and optional citations

    Returns:
        Complete reasoning trace with steps and conclusion
    """
    from judicaita.reasoning_trace import ReasoningTraceGenerator

    try:
        generator = ReasoningTraceGenerator(checkpoint_path=request.checkpoint_path)
        await generator.initialize()

        trace = await generator.generate_trace(
            query=request.query,
            context=request.context,
            citations=request.citations,
        )

        # Convert steps to response format
        steps = [
            ReasoningStepResponse(
                step_id=step.step_id,
                step_type=step.step_type.value,
                description=step.description,
                input_data=step.input_data,
                output_data=step.output_data,
                confidence_score=step.confidence_score,
                sources=step.sources,
            )
            for step in trace.steps
        ]

        return ReasoningTraceResponse(
            trace_id=trace.trace_id,
            query=trace.query,
            steps=steps,
            final_conclusion=trace.final_conclusion,
            overall_confidence=trace.overall_confidence,
            citations_used=trace.citations_used,
            model_info=trace.model_info,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating reasoning trace: {e}") from e


@router.get("/analysis/reasoning-trace/stream")
async def stream_reasoning_trace(
    query: str,
    context: str,
) -> StreamingResponse:
    """
    Stream a reasoning trace using Server-Sent Events (SSE).

    Provides real-time updates as each reasoning step is generated.

    Args:
        query: Legal question to analyze
        context: Relevant context

    Returns:
        SSE stream of reasoning steps
    """

    async def generate_sse() -> AsyncGenerator[str, None]:
        """Generate SSE events for reasoning trace."""
        from judicaita.reasoning_trace import ReasoningTraceGenerator

        try:
            generator = ReasoningTraceGenerator()
            await generator.initialize()

            # Send start event
            yield _format_sse("start", {"status": "generating", "query": query})

            # Generate trace
            trace = await generator.generate_trace(
                query=query,
                context=context,
            )

            # Send each step
            for i, step in enumerate(trace.steps):
                step_data = {
                    "step_number": i + 1,
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "description": step.description,
                    "confidence": step.confidence_score,
                }
                yield _format_sse("step", step_data)

            # Send completion
            yield _format_sse(
                "complete",
                {
                    "trace_id": trace.trace_id,
                    "conclusion": trace.final_conclusion,
                    "overall_confidence": trace.overall_confidence,
                    "total_steps": len(trace.steps),
                },
            )

        except Exception as e:
            yield _format_sse("error", {"message": str(e)})

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/analysis/citations", response_model=CitationExtractionResponse)
async def extract_citations(
    request: CitationExtractionRequest,
) -> CitationExtractionResponse:
    """
    Extract and validate legal citations from text.

    Identifies citations to cases, statutes, regulations, and other
    legal sources within the provided text.

    Args:
        request: Text to analyze and validation options

    Returns:
        List of extracted citations with metadata
    """
    from judicaita.citation_mapping import CitationMappingService

    start_time = time.time()

    try:
        service = CitationMappingService()
        matches = await service.extract_and_map_citations(
            text=request.text,
            validate=request.validate_citations,
        )

        citations = [
            CitationResponse(
                citation_id=str(uuid.uuid4()),
                raw_citation=m.citation.raw_citation,
                citation_type=m.citation.citation_type.value,
                jurisdiction=(m.citation.jurisdiction.value if m.citation.jurisdiction else None),
                is_valid=m.citation.is_valid,
                url=m.citation.url,
                context=m.context,
                metadata=m.citation.metadata or {},
            )
            for m in matches
        ]

        validated_count = sum(1 for c in citations if c.is_valid)
        elapsed_ms = (time.time() - start_time) * 1000

        return CitationExtractionResponse(
            citations=citations,
            total_count=len(citations),
            validated_count=validated_count,
            extraction_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting citations: {e}") from e


@router.get("/citations/{citation_id}", response_model=CitationResponse)
async def get_citation(citation_id: str) -> CitationResponse:
    """
    Retrieve details about a specific citation.

    Note: Citations are currently not persisted to a database.
    This endpoint returns 404 until database integration is implemented.

    Args:
        citation_id: Unique citation identifier

    Returns:
        Citation details

    Raises:
        HTTPException: 404 if citation not found
    """
    # Citations are extracted on-demand and not currently persisted.
    # When database integration is added, this will query the citations table.
    raise HTTPException(
        status_code=404,
        detail=f"Citation with ID '{citation_id}' not found. "
        "Use POST /analysis/citations to extract citations from text.",
    )


@router.post("/analysis/summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest) -> SummaryResponse:
    """
    Generate a plain-English summary of legal text.

    Creates an accessible summary at the specified reading level
    with key terms, takeaways, and optional sections.

    Args:
        request: Text and summary configuration

    Returns:
        Generated summary with metadata
    """
    from judicaita.summary_generator import SummaryGenerator
    from judicaita.summary_generator.models import ReadingLevel, SummaryLevel

    try:
        generator = SummaryGenerator()
        await generator.initialize()

        # Parse levels from strings
        summary_level = SummaryLevel(request.summary_level.lower())
        reading_level = ReadingLevel(request.reading_level.lower())

        summary = await generator.generate_summary(
            text=request.text,
            summary_level=summary_level,
            reading_level=reading_level,
            include_sections=request.include_sections,
        )

        # Convert sections
        sections = [
            SummarySectionResponse(
                title=s.title,
                content=s.content,
                key_points=s.key_points,
            )
            for s in summary.sections
        ]

        return SummaryResponse(
            summary=summary.summary,
            summary_level=summary.summary_level.value,
            reading_level=summary.reading_level.value,
            sections=sections,
            key_terms=summary.key_terms,
            key_takeaways=summary.key_takeaways,
            original_length=summary.metadata.get("original_length", 0),
            summary_length=summary.metadata.get("summary_length", 0),
            compression_ratio=summary.metadata.get("compression_ratio", 0.0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}") from e


def _format_sse(event: str, data: dict) -> str:
    """
    Format data as a Server-Sent Event.

    Args:
        event: Event type name
        data: Event data dictionary

    Returns:
        Formatted SSE string
    """
    import json

    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
