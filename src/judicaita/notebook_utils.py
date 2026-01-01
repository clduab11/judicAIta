"""
Notebook-friendly utilities for JudicAIta.

This module provides synchronous wrappers for async operations and helper classes
for seamless use in Jupyter notebooks and Kaggle environments.

Example usage:
    >>> from judicaita.notebook_utils import NotebookHelper
    >>> helper = NotebookHelper()
    >>> result = helper.upload_and_analyze("document.pdf")
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from judicaita.citation_mapping import CitationMappingService
from judicaita.citation_mapping.models import Citation, CitationMatch
from judicaita.document_input import DocumentInputService
from judicaita.document_input.base import DocumentContent
from judicaita.reasoning_trace import ReasoningTraceGenerator
from judicaita.reasoning_trace.models import ReasoningTrace
from judicaita.summary_generator import SummaryGenerator
from judicaita.summary_generator.models import LegalSummary, ReadingLevel, SummaryLevel


def _is_notebook() -> bool:
    """
    Detect if running in a Jupyter/IPython notebook environment.

    Returns:
        bool: True if running in a notebook, False otherwise
    """
    try:
        # Check for IPython shell
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        # Check if it's a ZMQ shell (Jupyter notebook)
        if shell.__class__.__name__ == "ZMQInteractiveShell":
            return True
        # Check for Google Colab
        if "google.colab" in sys.modules:
            return True
    except ImportError:
        pass
    return False


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create an event loop compatible with notebooks.

    Returns:
        Event loop instance
    """
    try:
        # Try to get existing loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in a notebook with a running loop, use nest_asyncio
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                logger.warning("nest_asyncio not installed. Install with: pip install nest_asyncio")
        return loop
    except RuntimeError:
        # No running event loop, create a new one
        return asyncio.new_event_loop()


def _run_async(coro: Any) -> Any:
    """
    Run an async coroutine in a notebook-compatible way.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    if _is_notebook():
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new loop in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def process_document_sync(file_path: str | Path, max_size_mb: int = 50) -> DocumentContent:
    """
    Synchronous wrapper for document processing.

    Args:
        file_path: Path to the document file
        max_size_mb: Maximum file size in megabytes

    Returns:
        DocumentContent: Processed document content

    Example:
        >>> content = process_document_sync("brief.pdf")
        >>> print(f"Extracted {len(content.text)} characters")
    """
    service = DocumentInputService(max_size_bytes=max_size_mb * 1024 * 1024)
    return _run_async(service.process_document(Path(file_path)))


def generate_trace_sync(
    query: str,
    context: str,
    citations: list[str] | None = None,
    checkpoint_path: str | None = None,
) -> ReasoningTrace:
    """
    Synchronous wrapper for reasoning trace generation.

    Args:
        query: Legal question or query
        context: Relevant context (case facts, statutes)
        citations: Optional list of relevant citations
        checkpoint_path: Optional path to GRPO-tuned checkpoint

    Returns:
        ReasoningTrace: Generated reasoning trace

    Example:
        >>> trace = generate_trace_sync(
        ...     query="What is the precedent for this case?",
        ...     context=document_text
        ... )
        >>> print(f"Confidence: {trace.overall_confidence:.2%}")
    """

    async def _generate() -> ReasoningTrace:
        generator = ReasoningTraceGenerator(checkpoint_path=checkpoint_path)
        await generator.initialize()
        return await generator.generate_trace(query, context, citations)

    return _run_async(_generate())


def extract_citations_sync(text: str, validate: bool = True) -> list[CitationMatch]:
    """
    Synchronous wrapper for citation extraction.

    Args:
        text: Text containing legal citations
        validate: Whether to validate citations against databases

    Returns:
        List of CitationMatch objects with extracted citations

    Example:
        >>> citations = extract_citations_sync(document_text)
        >>> for match in citations:
        ...     print(f"Found: {match.citation.raw_citation}")
    """
    service = CitationMappingService()
    return _run_async(service.extract_and_map_citations(text, validate=validate))


def generate_summary_sync(
    text: str,
    summary_level: SummaryLevel | str = SummaryLevel.MEDIUM,
    reading_level: ReadingLevel | str = ReadingLevel.HIGH_SCHOOL,
    include_sections: bool = True,
) -> LegalSummary:
    """
    Synchronous wrapper for summary generation.

    Args:
        text: Legal text to summarize
        summary_level: Level of detail (brief, short, medium, detailed)
        reading_level: Target reading level
        include_sections: Whether to include detailed sections

    Returns:
        LegalSummary: Generated summary with metadata

    Example:
        >>> summary = generate_summary_sync(
        ...     text=document_text,
        ...     reading_level="high_school"
        ... )
        >>> print(summary.summary)
    """
    # Convert string to enum if necessary
    if isinstance(summary_level, str):
        summary_level = SummaryLevel(summary_level.lower())
    if isinstance(reading_level, str):
        reading_level = ReadingLevel(reading_level.lower())

    async def _generate() -> LegalSummary:
        generator = SummaryGenerator()
        await generator.initialize()
        return await generator.generate_summary(
            text=text,
            summary_level=summary_level,
            reading_level=reading_level,
            include_sections=include_sections,
        )

    return _run_async(_generate())


def validate_citation_sync(citation_str: str) -> Citation | None:
    """
    Synchronous wrapper for citation validation.

    Args:
        citation_str: Citation string to validate

    Returns:
        Validated Citation object or None if invalid

    Example:
        >>> citation = validate_citation_sync("347 U.S. 483")
        >>> if citation:
        ...     print(f"Valid: {citation.is_valid}")
    """
    service = CitationMappingService()
    return _run_async(service.validate_citation(citation_str))


class NotebookHelper:
    """
    High-level helper class for notebook workflows.

    Provides convenient methods for common JudicAIta operations with
    progress bars and notebook-optimized output.

    Example:
        >>> helper = NotebookHelper()
        >>> result = helper.upload_and_analyze("document.pdf")
        >>> print(result["summary"].summary)
    """

    def __init__(
        self,
        show_progress: bool = True,
        checkpoint_path: str | None = None,
    ) -> None:
        """
        Initialize the NotebookHelper.

        Args:
            show_progress: Show tqdm progress bars for long operations
            checkpoint_path: Path to GRPO-tuned checkpoint for reasoning
        """
        self.show_progress = show_progress
        self.checkpoint_path = checkpoint_path
        self._doc_service = DocumentInputService()
        self._citation_service = CitationMappingService()
        self._reasoning_generator: ReasoningTraceGenerator | None = None
        self._summary_generator: SummaryGenerator | None = None

    def _get_progress_bar(self, iterable: Any, desc: str, total: int | None = None) -> Any:
        """Get a progress bar if enabled."""
        if self.show_progress:
            try:
                from tqdm.auto import tqdm

                return tqdm(iterable, desc=desc, total=total)
            except ImportError:
                logger.warning("tqdm not installed, progress bars disabled")
        return iterable

    def upload_and_analyze(
        self,
        file_path: str | Path,
        generate_summary: bool = True,
        extract_citations: bool = True,
        create_reasoning_trace: bool = True,
        query: str | None = None,
        summary_level: SummaryLevel | str = SummaryLevel.MEDIUM,
        reading_level: ReadingLevel | str = ReadingLevel.HIGH_SCHOOL,
    ) -> dict[str, Any]:
        """
        Complete document analysis workflow.

        Processes a document and optionally generates summary, extracts citations,
        and creates a reasoning trace.

        Args:
            file_path: Path to the document
            generate_summary: Generate a plain-English summary
            extract_citations: Extract and validate citations
            create_reasoning_trace: Create a reasoning trace (requires query)
            query: Optional legal query for reasoning trace
            summary_level: Summary detail level
            reading_level: Target reading level

        Returns:
            Dictionary with analysis results:
            - document: DocumentContent
            - summary: LegalSummary (if generate_summary)
            - citations: List[CitationMatch] (if extract_citations)
            - reasoning_trace: ReasoningTrace (if create_reasoning_trace and query)

        Example:
            >>> result = helper.upload_and_analyze(
            ...     "case.pdf",
            ...     query="What are the key legal issues?"
            ... )
            >>> print(f"Found {len(result['citations'])} citations")
        """
        result: dict[str, Any] = {}

        # Step 1: Process document
        logger.info(f"Processing document: {file_path}")
        result["document"] = process_document_sync(file_path)
        document_text = result["document"].text

        # Step 2: Extract citations
        if extract_citations:
            logger.info("Extracting citations...")
            result["citations"] = extract_citations_sync(document_text)
            logger.info(f"Found {len(result['citations'])} citations")

        # Step 3: Generate summary
        if generate_summary:
            logger.info("Generating summary...")
            result["summary"] = generate_summary_sync(
                text=document_text,
                summary_level=summary_level,
                reading_level=reading_level,
            )

        # Step 4: Create reasoning trace
        if create_reasoning_trace and query:
            logger.info("Generating reasoning trace...")
            citations_list = []
            if "citations" in result:
                citations_list = [m.citation.raw_citation for m in result["citations"]]

            result["reasoning_trace"] = generate_trace_sync(
                query=query,
                context=document_text,
                citations=citations_list,
                checkpoint_path=self.checkpoint_path,
            )

        return result

    def batch_process_documents(
        self,
        file_paths: list[str | Path],
        generate_summary: bool = True,
        extract_citations: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Process multiple documents with progress tracking.

        Args:
            file_paths: List of document paths
            generate_summary: Generate summaries for each document
            extract_citations: Extract citations from each document

        Returns:
            List of analysis results for each document

        Example:
            >>> files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
            >>> results = helper.batch_process_documents(files)
            >>> for r in results:
            ...     print(f"Processed: {len(r['document'].text)} chars")
        """
        results = []
        file_paths_iter = self._get_progress_bar(
            file_paths,
            desc="Processing documents",
            total=len(file_paths),
        )

        for file_path in file_paths_iter:
            try:
                result = self.upload_and_analyze(
                    file_path=file_path,
                    generate_summary=generate_summary,
                    extract_citations=extract_citations,
                    create_reasoning_trace=False,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({"error": str(e), "file": str(file_path)})

        return results

    def export_results(
        self,
        result: dict[str, Any],
        output_dir: str | Path = "./output",
        format: str = "markdown",
    ) -> dict[str, Path]:
        """
        Export analysis results to files.

        Args:
            result: Analysis result from upload_and_analyze
            output_dir: Directory for output files
            format: Output format ("markdown" or "json")

        Returns:
            Dictionary mapping result types to output file paths

        Example:
            >>> result = helper.upload_and_analyze("document.pdf")
            >>> paths = helper.export_results(result, "./analysis")
            >>> print(f"Summary saved to: {paths['summary']}")
        """
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported: dict[str, Path] = {}

        # Export summary
        if "summary" in result:
            summary = result["summary"]
            if format == "markdown":
                summary_path = output_path / "summary.md"
                summary_path.write_text(summary.to_markdown())
            else:
                summary_path = output_path / "summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "summary": summary.summary,
                            "key_takeaways": summary.key_takeaways,
                            "key_terms": summary.key_terms,
                            "metadata": summary.metadata,
                        },
                        indent=2,
                    )
                )
            exported["summary"] = summary_path

        # Export citations
        if "citations" in result:
            citations = result["citations"]
            if format == "markdown":
                citations_path = output_path / "citations.md"
                lines = ["# Extracted Citations\n"]
                for i, match in enumerate(citations, 1):
                    cit = match.citation
                    lines.append(f"## {i}. {cit.raw_citation}")
                    lines.append(f"- **Type**: {cit.citation_type.value}")
                    lines.append(f"- **Valid**: {cit.is_valid}")
                    lines.append(f"- **Context**: {match.context}\n")
                citations_path.write_text("\n".join(lines))
            else:
                citations_path = output_path / "citations.json"
                citations_data = [
                    {
                        "citation": m.citation.raw_citation,
                        "type": m.citation.citation_type.value,
                        "is_valid": m.citation.is_valid,
                        "context": m.context,
                    }
                    for m in citations
                ]
                citations_path.write_text(json.dumps(citations_data, indent=2))
            exported["citations"] = citations_path

        # Export reasoning trace
        if "reasoning_trace" in result:
            trace = result["reasoning_trace"]
            if format == "markdown":
                trace_path = output_path / "reasoning_trace.md"
                trace_path.write_text(trace.to_markdown())
            else:
                trace_path = output_path / "reasoning_trace.json"
                trace_path.write_text(
                    json.dumps(
                        {
                            "trace_id": trace.trace_id,
                            "query": trace.query,
                            "conclusion": trace.final_conclusion,
                            "confidence": trace.overall_confidence,
                            "steps": [
                                {
                                    "type": s.step_type.value,
                                    "description": s.description,
                                    "confidence": s.confidence_score,
                                }
                                for s in trace.steps
                            ],
                        },
                        indent=2,
                    )
                )
            exported["reasoning_trace"] = trace_path

        logger.info(f"Exported {len(exported)} files to {output_path}")
        return exported


# Convenience exports for notebook users
__all__ = [
    "process_document_sync",
    "generate_trace_sync",
    "extract_citations_sync",
    "generate_summary_sync",
    "validate_citation_sync",
    "NotebookHelper",
]
