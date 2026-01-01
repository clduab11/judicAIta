"""
API models package.
"""

from judicaita.api.models.analysis import (
    CitationExtractionRequest,
    CitationExtractionResponse,
    CitationResponse,
    ReasoningTraceRequest,
    ReasoningTraceResponse,
    SummaryRequest,
    SummaryResponse,
)
from judicaita.api.models.documents import (
    DocumentAnalyzeRequest,
    DocumentResponse,
    DocumentUploadResponse,
)

__all__ = [
    "DocumentUploadResponse",
    "DocumentResponse",
    "DocumentAnalyzeRequest",
    "ReasoningTraceRequest",
    "ReasoningTraceResponse",
    "CitationExtractionRequest",
    "CitationExtractionResponse",
    "CitationResponse",
    "SummaryRequest",
    "SummaryResponse",
]
