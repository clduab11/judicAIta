"""
Document processing endpoints.
"""

import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from judicaita.api.models.documents import (
    DocumentAnalyzeRequest,
    DocumentResponse,
    DocumentUploadResponse,
)
from judicaita.core.exceptions import DocumentProcessingError, UnsupportedDocumentFormatError
from judicaita.document_input import DocumentInputService

router = APIRouter()

# In-memory storage for demo purposes
# In production, this would use a database
_documents: dict[str, dict[str, Any]] = {}


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Upload a document for processing.

    Accepts PDF, Word (DOCX), and other supported formats.
    Returns a document ID for subsequent operations.

    Args:
        file: The document file to upload

    Returns:
        Document upload confirmation with ID
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Generate document ID
    document_id = str(uuid.uuid4())

    # Get file extension
    file_ext = Path(file.filename).suffix.lower().lstrip(".")

    # Check supported formats
    service = DocumentInputService()
    if not service.supports_format(file_ext):
        supported = service.get_supported_formats()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {supported}",
        )

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Process the document
    try:
        # Write to temp file for processing
        with NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Process document
        document_content = await service.process_document(tmp_path)

        # Store processed document
        _documents[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "file_type": file_ext,
            "file_size": file_size,
            "content": document_content,
            "status": "processed",
        }

        # Cleanup temp file
        tmp_path.unlink(missing_ok=True)

    except UnsupportedDocumentFormatError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except DocumentProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {e}") from e

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        file_size=file_size,
        file_type=file_ext,
        status="processed",
        message="Document uploaded and processed successfully",
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str) -> DocumentResponse:
    """
    Retrieve a processed document by ID.

    Args:
        document_id: The unique document identifier

    Returns:
        Processed document data
    """
    if document_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_data = _documents[document_id]
    content = doc_data["content"]

    return DocumentResponse(
        document_id=document_id,
        filename=doc_data["filename"],
        text=content.text,
        text_length=len(content.text),
        metadata={
            "title": content.metadata.title if content.metadata else None,
            "author": content.metadata.author if content.metadata else None,
            "page_count": content.metadata.page_count if content.metadata else None,
            "word_count": content.metadata.word_count if content.metadata else None,
            "file_type": doc_data["file_type"],
        },
        sections=content.sections,
        tables=content.tables,
        citations_found=len(content.citations) if content.citations else 0,
    )


@router.post("/documents/{document_id}/analyze")
async def analyze_document(
    document_id: str,
    request: DocumentAnalyzeRequest,
) -> dict[str, Any]:
    """
    Trigger analysis on a document.

    Performs citation extraction, summary generation, and/or reasoning trace
    generation based on request parameters.

    Args:
        document_id: The document to analyze
        request: Analysis configuration

    Returns:
        Analysis results
    """
    if document_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_data = _documents[document_id]
    content = doc_data["content"]
    text = content.text

    start_time = time.time()
    result: dict[str, Any] = {
        "document_id": document_id,
        "status": "completed",
    }

    # Extract citations
    if request.extract_citations:
        from judicaita.citation_mapping import CitationMappingService

        citation_service = CitationMappingService()
        citations = await citation_service.extract_and_map_citations(text)
        result["citations"] = [
            {
                "raw_citation": m.citation.raw_citation,
                "type": m.citation.citation_type.value,
                "is_valid": m.citation.is_valid,
                "context": m.context,
            }
            for m in citations
        ]
        result["citations_count"] = len(citations)

    # Generate summary
    if request.generate_summary:
        from judicaita.summary_generator import SummaryGenerator
        from judicaita.summary_generator.models import ReadingLevel, SummaryLevel

        summary_gen = SummaryGenerator()
        await summary_gen.initialize()

        summary_level = SummaryLevel(request.summary_level.lower())
        reading_level = ReadingLevel(request.reading_level.lower())

        summary = await summary_gen.generate_summary(
            text=text,
            summary_level=summary_level,
            reading_level=reading_level,
        )
        result["summary"] = {
            "text": summary.summary,
            "key_takeaways": summary.key_takeaways,
            "key_terms": summary.key_terms,
        }
        result["summary_generated"] = True

    # Generate reasoning trace
    if request.generate_reasoning_trace and request.query:
        from judicaita.reasoning_trace import ReasoningTraceGenerator

        trace_gen = ReasoningTraceGenerator()
        await trace_gen.initialize()

        citations_list = (
            [m.citation.raw_citation for m in citations] if request.extract_citations else None
        )

        trace = await trace_gen.generate_trace(
            query=request.query,
            context=text,
            citations=citations_list,
        )
        result["reasoning_trace"] = {
            "trace_id": trace.trace_id,
            "conclusion": trace.final_conclusion,
            "confidence": trace.overall_confidence,
            "steps_count": len(trace.steps),
        }
        result["reasoning_trace_generated"] = True

    elapsed_ms = (time.time() - start_time) * 1000
    result["analysis_time_ms"] = round(elapsed_ms, 2)
    result["message"] = "Analysis completed successfully"

    return result
