# System Architecture Diagram

```mermaid
graph TB
    subgraph User["User Interfaces"]
        CLI[CLI - judicaita]
        API[REST API - FastAPI]
        SDK[Python SDK]
        NB[Notebook Utils]
    end

    subgraph Core["Core Services"]
        DI[Document Input Service]
        RT[Reasoning Trace Generator]
        CM[Citation Mapping Service]
        SG[Summary Generator]
        AL[Audit Logger]
    end

    subgraph AI["AI/ML Layer"]
        GM[Gemma 3n Model]
        GRPO[GRPO-Tuned Checkpoint]
        TX[Google Tunix]
    end

    subgraph Data["Data Layer"]
        DB[(PostgreSQL)]
        RD[(Redis Cache)]
        FS[File Storage]
    end

    CLI --> DI
    CLI --> RT
    API --> DI
    API --> RT
    API --> CM
    API --> SG
    SDK --> DI
    SDK --> RT
    SDK --> CM
    SDK --> SG
    NB --> SDK

    DI --> FS
    RT --> GM
    RT --> GRPO
    CM --> DB
    SG --> GM
    AL --> DB

    GM --> TX
    GRPO --> TX
```

## Component Descriptions

| Component | Description |
|-----------|-------------|
| CLI | Command-line interface using Typer |
| REST API | FastAPI server with OpenAPI docs |
| Python SDK | Direct Python API for all services |
| Notebook Utils | Sync wrappers for Jupyter/Kaggle |
| Document Input | PDF/Word processing and extraction |
| Reasoning Trace | Step-by-step legal reasoning |
| Citation Mapping | Legal citation extraction and validation |
| Summary Generator | Plain-English summary creation |
| Audit Logger | Compliance and audit logging |
| Gemma 3n | Base language model |
| GRPO Checkpoint | Fine-tuned reasoning model |
| Google Tunix | Training framework for TPU |
