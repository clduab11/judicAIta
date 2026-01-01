# Judicaita API Reference

## Overview

Judicaita provides a comprehensive Python API for programmatic access to all functionality.

## Installation

```bash
pip install judicaita
```

## Quick Start

```python
import asyncio
from pathlib import Path
from judicaita.document_input import DocumentInputService
from judicaita.reasoning_trace import ReasoningTraceGenerator
from judicaita.summary_generator import SummaryGenerator

async def main():
    # Initialize services
    doc_service = DocumentInputService()
    
    # Process a document
    document = await doc_service.process_document(Path("case.pdf"))
    
    print(f"Extracted {len(document.text)} characters")
    print(f"Found {len(document.citations)} citations")

if __name__ == "__main__":
    asyncio.run(main())
```

## Modules

### Document Input

#### DocumentInputService

Process documents from various formats.

```python
from judicaita.document_input import DocumentInputService

service = DocumentInputService(max_size_bytes=50*1024*1024)

# Process a document
document = await service.process_document(Path("legal_brief.pdf"))

# Access content
print(document.text)
print(document.metadata.title)
print(document.metadata.author)

# Access extracted sections
for section in document.sections:
    print(f"Page {section['page']}: {section['text'][:100]}...")

# Access tables
for table in document.tables:
    print(f"Table on page {table['page']}")
    print(table['data'])
```

#### Supported Formats

```python
# Check supported formats
formats = service.get_supported_formats()
print(formats)  # ['pdf', 'docx', 'doc']

# Check if format is supported
if service.supports_format('pdf'):
    print("PDF is supported")
```

### Reasoning Trace

#### ReasoningTraceGenerator

Generate explainable reasoning traces.

```python
from judicaita.reasoning_trace import ReasoningTraceGenerator

generator = ReasoningTraceGenerator()
await generator.initialize()

# Generate a trace
trace = await generator.generate_trace(
    query="What is the legal precedent for this case?",
    context=document.text,
    citations=document.citations
)

# Access trace information
print(f"Trace ID: {trace.trace_id}")
print(f"Confidence: {trace.overall_confidence:.2%}")
print(f"Steps: {len(trace.steps)}")

# Iterate through reasoning steps
for step in trace.steps:
    print(f"\n{step.step_type.value.upper()}: {step.description}")
    print(f"Confidence: {step.confidence_score:.2%}")
    if step.sources:
        print(f"Sources: {', '.join(step.sources)}")

# Export to markdown
markdown = trace.to_markdown()
with open("reasoning_trace.md", "w") as f:
    f.write(markdown)
```

### Citation Mapping

#### CitationMappingService

Extract and validate legal citations.

```python
from judicaita.citation_mapping import CitationMappingService

service = CitationMappingService()

# Extract citations from text
citations = await service.extract_and_map_citations(
    text=document.text,
    validate=True
)

# Access citation details
for match in citations:
    citation = match.citation
    print(f"\nCitation: {citation.raw_citation}")
    print(f"Type: {citation.citation_type.value}")
    print(f"Jurisdiction: {citation.jurisdiction.value}")
    print(f"Valid: {citation.is_valid}")
    print(f"Context: {match.context}")

# Build citation graph
citations_list = [match.citation for match in citations]
graph = await service.build_citation_graph(citations_list)

print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

# Validate single citation
citation = await service.validate_citation("347 U.S. 483")
if citation:
    print(f"Valid citation: {citation.raw_citation}")
```

### Summary Generator

#### SummaryGenerator

Generate plain-English summaries.

```python
from judicaita.summary_generator import (
    SummaryGenerator,
    SummaryLevel,
    ReadingLevel
)

generator = SummaryGenerator()
await generator.initialize()

# Generate a summary
summary = await generator.generate_summary(
    text=document.text,
    summary_level=SummaryLevel.MEDIUM,
    reading_level=ReadingLevel.HIGH_SCHOOL,
    include_sections=True
)

# Access summary content
print(f"Summary: {summary.summary}")
print(f"Key Takeaways: {summary.key_takeaways}")

# Access key terms
for term, definition in summary.key_terms.items():
    print(f"\n{term}: {definition}")

# Export to markdown
markdown = summary.to_markdown()
with open("summary.md", "w") as f:
    f.write(markdown)

# Simplify specific text
simplified = await generator.simplify_text(
    text=complex_legal_text,
    target_reading_level=ReadingLevel.MIDDLE_SCHOOL
)
```

### Audit Logs

#### AuditLogger

Log events for compliance.

```python
from judicaita.audit_logs import (
    AuditLogger,
    AuditEventType,
    AuditSeverity,
    ComplianceStatus
)
from datetime import datetime, timedelta

logger = AuditLogger()

# Log an event
entry = await logger.log_event(
    event_type=AuditEventType.DOCUMENT_PROCESS,
    action="Processed legal brief",
    user_id="user123",
    session_id="session456",
    severity=AuditSeverity.INFO,
    status="success",
    compliance_status=ComplianceStatus.COMPLIANT,
    details={
        "document_name": "brief.pdf",
        "pages": 25
    }
)

print(f"Logged event: {entry.entry_id}")

# Query logs
recent_logs = await logger.query_logs(
    start_date=datetime.now() - timedelta(days=7),
    event_types=[AuditEventType.DOCUMENT_PROCESS],
    limit=100
)

print(f"Found {len(recent_logs)} recent logs")

# Generate report
report = await logger.generate_report(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

print(f"Report: {report.total_events} events")
print(f"Violations: {len(report.compliance_violations)}")

# Export report
markdown = report.to_markdown()
with open("audit_report.md", "w") as f:
    f.write(markdown)
```

## Configuration

### Using Environment Variables

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key"
os.environ["MODEL_TEMPERATURE"] = "0.3"

from judicaita.core import get_settings

settings = get_settings()
print(settings.google_api_key)
```

### Using .env File

```bash
# .env
GOOGLE_API_KEY=your-api-key
MODEL_TEMPERATURE=0.3
AUDIT_LOG_ENABLED=true
```

### Custom Settings

```python
from judicaita.core import Settings

settings = Settings(
    google_api_key="your-key",
    model_temperature=0.5,
    max_document_size_mb=100
)
```

## Error Handling

```python
from judicaita.core.exceptions import (
    DocumentProcessingError,
    ModelInferenceError,
    CitationError,
    AuditError
)

try:
    document = await doc_service.process_document(path)
except DocumentProcessingError as e:
    print(f"Error: {e.message}")
    print(f"Code: {e.error_code}")
    print(f"Details: {e.details}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### 1. Use Async/Await

All I/O operations are async for better performance:

```python
async def process_multiple_documents(paths):
    service = DocumentInputService()
    
    # Process concurrently
    tasks = [service.process_document(path) for path in paths]
    documents = await asyncio.gather(*tasks)
    
    return documents
```

### 2. Enable Audit Logging

Always log important operations:

```python
logger = AuditLogger()

try:
    result = await some_operation()
    await logger.log_event(
        event_type=AuditEventType.OPERATION_SUCCESS,
        action="Operation completed",
        status="success"
    )
except Exception as e:
    await logger.log_event(
        event_type=AuditEventType.ERROR,
        action="Operation failed",
        status="error",
        severity=AuditSeverity.ERROR,
        details={"error": str(e)}
    )
```

### 3. Cache Results

Use built-in caching for expensive operations:

```python
# Results are automatically cached
summary = await generator.generate_summary(text)

# Clear cache when needed
citation_service.clear_cache()
```

### 4. Handle Large Documents

Process large documents in chunks:

```python
from judicaita.utils import chunk_text

# Split large text
chunks = chunk_text(large_text, chunk_size=1000, overlap=100)

# Process each chunk
summaries = []
for chunk in chunks:
    summary = await generator.generate_summary(chunk)
    summaries.append(summary)
```

## Examples

See the `examples/` directory for complete examples:

- `examples/basic_usage.py`: Basic usage patterns
- `examples/document_processing.py`: Document processing examples
- `examples/reasoning_trace.py`: Reasoning trace examples
- `examples/citation_mapping.py`: Citation mapping examples
- `examples/notebooks/`: Jupyter notebooks with interactive examples

---

## Notebook Utilities

For use in Jupyter notebooks and Kaggle environments, JudicAIta provides synchronous wrappers:

### Quick Start (Notebook)

```python
from judicaita.notebook_utils import NotebookHelper

helper = NotebookHelper(show_progress=True)

# Complete workflow
result = helper.upload_and_analyze(
    file_path="document.pdf",
    query="What are the key legal issues?",
    summary_level="medium",
    reading_level="high_school"
)

print(result["summary"].summary)
print(f"Found {len(result['citations'])} citations")
```

### Sync Wrappers

```python
from judicaita.notebook_utils import (
    process_document_sync,
    generate_trace_sync,
    extract_citations_sync,
    generate_summary_sync,
)

# Process document synchronously
content = process_document_sync("brief.pdf")

# Extract citations synchronously
citations = extract_citations_sync(content.text)

# Generate summary synchronously
summary = generate_summary_sync(
    text=content.text,
    reading_level="high_school"
)
```

### Batch Processing

```python
helper = NotebookHelper(show_progress=True)

files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = helper.batch_process_documents(files)

for r in results:
    if "error" not in r:
        print(f"Processed: {len(r['document'].text)} chars")
```

### Export Results

```python
result = helper.upload_and_analyze("document.pdf")
paths = helper.export_results(result, "./output", format="markdown")
print(f"Summary saved to: {paths['summary']}")
```

---

## REST API

JudicAIta provides a FastAPI-based REST API for programmatic access.

### Starting the Server

```bash
# Using CLI
judicaita serve --host 0.0.0.0 --port 8000

# Or directly with uvicorn
uvicorn judicaita.api:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Core Endpoints

#### Health Check

```bash
# Basic health
GET /health

# Readiness (checks dependencies)
GET /ready
```

#### Documents

```bash
# Upload document
POST /api/v1/documents/upload
Content-Type: multipart/form-data

# Get processed document
GET /api/v1/documents/{document_id}

# Analyze document
POST /api/v1/documents/{document_id}/analyze
```

#### Analysis

```bash
# Generate reasoning trace
POST /api/v1/analysis/reasoning-trace
{
  "query": "What are the legal issues?",
  "context": "...",
  "citations": ["347 U.S. 483"]
}

# Stream reasoning trace (SSE)
GET /api/v1/analysis/reasoning-trace/stream?query=...&context=...

# Extract citations
POST /api/v1/analysis/citations
{
  "text": "Brown v. Board of Education, 347 U.S. 483 (1954)",
  "validate": true
}

# Generate summary
POST /api/v1/analysis/summary
{
  "text": "...",
  "summary_level": "medium",
  "reading_level": "high_school"
}
```

### Python Client Example

```python
import httpx

async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
    # Upload document
    with open("brief.pdf", "rb") as f:
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": f}
        )
    doc = response.json()
    
    # Analyze
    response = await client.post(
        f"/api/v1/documents/{doc['document_id']}/analyze",
        json={"extract_citations": True, "generate_summary": True}
    )
    analysis = response.json()
```

---

## Support

- Documentation: https://github.com/clduab11/judicAIta/docs
- Issues: https://github.com/clduab11/judicAIta/issues
- Discussions: https://github.com/clduab11/judicAIta/discussions
