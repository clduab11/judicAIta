# Judicaita Architecture Overview

## System Architecture

Judicaita follows a modular, layered architecture designed for scalability, maintainability, and extensibility.

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   CLI    │  │  Web UI  │  │   API    │  │  SDK     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Core Services Layer                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ Document   │  │ Reasoning  │  │ Citation   │           │
│  │ Input      │  │ Trace      │  │ Mapping    │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│  ┌────────────┐  ┌────────────┐                            │
│  │ Summary    │  │ Audit      │                            │
│  │ Generator  │  │ Logs       │                            │
│  └────────────┘  └────────────┘                            │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                  AI/ML Layer (Gemma 3n)                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │         Google Tunix & Gemma 3n Models             │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ PostgreSQL │  │   Redis    │  │  File      │           │
│  │ Database   │  │   Cache    │  │  Storage   │           │
│  └────────────┘  └────────────┘  └────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Document Input Module

**Purpose**: Process various document formats (PDF, Word, etc.) and extract structured content.

**Key Classes**:
- `DocumentProcessor`: Abstract base class for processors
- `PDFProcessor`: PDF document processing
- `WordProcessor`: Word document processing
- `DocumentInputService`: Unified interface for document processing

**Features**:
- Multi-format support (PDF, DOCX, DOC)
- Metadata extraction
- Text, table, and image extraction
- Citation detection
- Size and format validation

### 2. Reasoning Trace Module

**Purpose**: Generate explainable, step-by-step reasoning traces for legal analysis.

**Key Classes**:
- `ReasoningTraceGenerator`: Main generator class
- `ReasoningTrace`: Complete trace with steps
- `ReasoningStep`: Individual reasoning step

**Features**:
- Step-by-step reasoning
- Confidence scoring
- Source tracking
- Multiple step types (analysis, inference, conclusion)
- Markdown export

### 3. Citation Mapping Module

**Purpose**: Extract, parse, validate, and map legal citations.

**Key Classes**:
- `CitationParser`: Citation extraction and parsing
- `CitationMappingService`: Validation and enrichment
- `Citation`: Structured citation data
- `CitationGraph`: Citation relationships

**Features**:
- Multiple citation formats (cases, statutes, regulations)
- Jurisdiction detection
- Citation validation
- Relationship mapping
- Context extraction

### 4. Summary Generator Module

**Purpose**: Generate plain-English summaries at various reading levels.

**Key Classes**:
- `SummaryGenerator`: Main generator class
- `LegalSummary`: Summary with metadata
- `SummarySection`: Individual summary section

**Features**:
- Multiple summary levels (brief, short, medium, detailed)
- Reading level targeting (elementary to professional)
- Key term extraction and definition
- Structured sections
- Markdown export

### 5. Audit Logs Module

**Purpose**: Comprehensive audit logging for compliance and transparency.

**Key Classes**:
- `AuditLogger`: Main logging service
- `AuditLogEntry`: Individual log entry
- `AuditReport`: Aggregated reporting

**Features**:
- Event type classification
- Severity levels
- Compliance status tracking
- Query and filtering
- Report generation
- Long-term retention

## Data Flow

### Document Processing Flow

```
1. User uploads document
   ↓
2. DocumentInputService validates format and size
   ↓
3. Appropriate processor (PDF/Word) extracts content
   ↓
4. Content is structured into DocumentContent object
   ↓
5. Citations are detected and extracted
   ↓
6. Metadata is enriched
   ↓
7. Result is cached
   ↓
8. Audit log entry created
```

### Analysis Flow

```
1. User submits query with context
   ↓
2. ReasoningTraceGenerator creates trace
   ↓
3. For each reasoning step:
   - Call Gemma model
   - Record step details
   - Track sources
   - Calculate confidence
   ↓
4. CitationMappingService validates citations
   ↓
5. SummaryGenerator creates plain-English summary
   ↓
6. All results compiled and returned
   ↓
7. Audit log entry created
```

## Technology Stack

### Core Technologies
- **Python 3.10+**: Primary language
- **Pydantic**: Data validation and settings
- **FastAPI**: API framework
- **Typer**: CLI framework

### AI/ML
- **Google Tunix**: AI platform
- **Gemma 3n**: Language model
- **LangChain**: LLM framework
- **spaCy**: NLP processing

### Document Processing
- **pdfplumber**: PDF extraction
- **python-docx**: Word processing
- **pypdf**: PDF metadata

### Data Storage
- **PostgreSQL**: Primary database
- **Redis**: Caching layer
- **SQLite**: Development/testing

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Linting
- **mypy**: Type checking

## Design Principles

### 1. Modularity
Each component is self-contained and can be used independently.

### 2. Explainability
All AI decisions include transparent reasoning traces.

### 3. Type Safety
Comprehensive type hints and validation using Pydantic.

### 4. Async First
Async/await pattern for I/O operations.

### 5. Configuration
Environment-based configuration with validation.

### 6. Compliance
Audit logging built into every operation.

### 7. Extensibility
Easy to add new document formats, citation types, or analysis methods.

## Scalability Considerations

### Horizontal Scaling
- Stateless services enable easy scaling
- Redis for distributed caching
- PostgreSQL for shared state

### Performance
- Async operations for concurrency
- Caching for expensive operations
- Batch processing for large documents

### Reliability
- Comprehensive error handling
- Retry logic for external services
- Graceful degradation

## Security

### Authentication & Authorization
- JWT-based authentication (planned)
- Role-based access control (planned)
- API key management

### Data Protection
- Audit logging for all access
- Data retention policies
- Encryption at rest (planned)
- Encryption in transit (HTTPS)

### Compliance
- GDPR considerations
- Legal industry standards
- Audit trail maintenance

## Future Enhancements

1. **Real-time Collaboration**: Multi-user document annotation
2. **Advanced Citation Database**: Integration with legal databases
3. **Multi-language Support**: International legal systems
4. **API Rate Limiting**: Enhanced API protection
5. **Webhooks**: Event-driven integrations
6. **Advanced Analytics**: Usage patterns and insights
7. **Mobile Apps**: iOS and Android clients
8. **Plugin System**: Community extensions

---

## Detailed Diagrams

For detailed architecture diagrams using Mermaid, see:

- [System Architecture](diagrams/system_architecture.md) - Component relationships
- [Data Flow](diagrams/data_flow.md) - Request/response flows
- [Training Pipeline](diagrams/training_pipeline.md) - GRPO training architecture

---

## GRPO Training Integration

JudicAIta uses Group Relative Policy Optimization (GRPO) for fine-tuning:

### Training Pipeline

1. **Data Preparation**
   - LegalBench dataset for task-specific examples
   - Pile of Law for domain knowledge
   - Synthetic Chain-of-Thought generation

2. **Model Configuration**
   - Base: Gemma 3-1B-IT
   - LoRA adapters for parameter efficiency
   - TPU-optimized with JAX/Flax

3. **Reward Function**
   - XML format validation (40%)
   - Reasoning length (30%)
   - Citation quality (20%)
   - Output coherence (10%)

4. **Validation Phases**
   - Phase 1: Environment validation
   - Phase 2: Training setup verification
   - Phase 3: Output quality checks
   - Phase 4: Submission preparation

### Async Architecture

JudicAIta uses async/await patterns throughout:

```python
# Event loop management
async def process_document(path):
    content = await doc_service.process_document(path)
    citations = await citation_service.extract_and_map_citations(content.text)
    summary = await summary_gen.generate_summary(content.text)
    return content, citations, summary

# Concurrent processing
async def process_batch(paths):
    tasks = [process_document(p) for p in paths]
    return await asyncio.gather(*tasks)
```

### Caching Strategy

- **In-memory**: Citation cache in CitationMappingService
- **LRU Cache**: Settings via `@lru_cache` decorator
- **Redis**: Planned for distributed caching

---

## Troubleshooting

### Common Architectural Questions

**Q: How do I add a new document processor?**

```python
from judicaita.document_input.base import DocumentProcessor, DocumentContent

class MyProcessor(DocumentProcessor):
    async def process(self, file_path: Path) -> DocumentContent:
        # Implementation
        pass

# Register
service.register_processor("myformat", MyProcessor())
```

**Q: How do I extend reasoning steps?**

Modify `ReasoningTraceGenerator._perform_inference()` or add new step methods following the existing pattern.

**Q: How do I add custom citation types?**

Extend `CitationParser` with new regex patterns in `CITATION_PATTERNS`.
