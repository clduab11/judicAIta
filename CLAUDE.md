# JudicAIta - Claude Code Project Guide

## Project Overview

JudicAIta is an open-source agentic legal automation platform designed for lawyers, law firms, and legal educators. The project provides:

- **Multi-agent orchestration** for specialized legal document workflows
- **Legal document processing**: contracts, briefs, discovery materials, case law
- **Retrieval-augmented generation (RAG)** for legal research and citation mapping
- **Automated drafting**, compliance monitoring, and legal checklists
- **Explainable AI reasoning** with step-by-step legal analysis traces

**Target:** Google Tunix Hackathon on Kaggle - fine-tuning Gemma 2B/3 1B models on TPU for legal reasoning tasks

## Tech Stack

### Core Technologies
- **Languages**: Python 3.10+, TypeScript/JavaScript
- **LLM Framework**: LangChain, LangGraph (agentic workflows)
- **Models**:
  - Gemma 2B / Gemma 3 1B (fine-tuned via Tunix on Kaggle TPU V3-8)
  - Gemini API (for enhanced reasoning)
  - Claude API (for complex legal analysis)
- **Backend**: FastAPI, Python asyncio
- **Frontend**: React, TypeScript, Tailwind CSS
- **Infrastructure**:
  - Kaggle TPU V3-8 for training
  - JAX/Flax for model training
  - Keras 3.0 for high-level training API
  - PyTorch/Transformers for inference

### Data & Storage
- **Vector Database**: Chroma, FAISS (for RAG)
- **Document Parsing**: PyPDF2, python-docx, pdfplumber, Tesseract OCR
- **NLP**: spaCy (legal entity recognition), NLTK (text processing)

### Development Tools
- **Testing**: pytest, unittest, hypothesis (property-based testing)
- **Code Quality**: black, ruff, mypy, pylint
- **CI/CD**: GitHub Actions
- **Documentation**: MkDocs, Sphinx

## Repository Structure

```text
judicaita/
├── .github/                    # GitHub Actions workflows
│   └── workflows/
├── docs/                       # Documentation (MkDocs)
├── src/
│   ├── agents/                 # Multi-agent system
│   │   ├── base/              # Base agent classes
│   │   ├── parsing/           # Document parsing agents
│   │   ├── extraction/        # Entity/clause extraction agents
│   │   ├── drafting/          # Legal document drafting agents
│   │   ├── qa/                # QA and compliance agents
│   │   └── orchestration/     # Workflow orchestrators
│   ├── models/                 # Fine-tuned models & wrappers
│   │   ├── gemma/             # Gemma model integration
│   │   └── rag/               # RAG system components
│   ├── parsers/                # Document parsers (PDF, DOCX, etc.)
│   ├── legal_utils/            # Legal-specific utilities
│   │   ├── citation.py        # Bluebook citation formatting
│   │   ├── entities.py        # Legal entity schemas
│   │   └── compliance.py      # Compliance validation
│   ├── api/                    # FastAPI endpoints
│   └── frontend/               # React web interface
├── notebooks/                  # Kaggle/Jupyter notebooks
│   └── tunix_finetuning.ipynb # Gemma fine-tuning notebook
├── data/                       # Training/test datasets
│   ├── raw/                   # Raw legal documents
│   ├── processed/             # Processed datasets
│   └── synthetic/             # Synthetically generated legal texts
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/                    # Utility scripts
│   ├── setup_kaggle_tpu.sh
│   └── preprocess_legal_corpus.py
├── configs/                    # Configuration files
│   ├── agents/                # Agent configurations
│   └── models/                # Model configurations
├── CLAUDE.md                   # This file (project conventions)
├── README.md                   # Main project documentation
├── TUNIX_SETUP.md             # Kaggle TPU fine-tuning guide
├── CONTRIBUTING.md             # Contribution guidelines
├── pyproject.toml              # Python dependencies & tooling
├── requirements.txt            # Python dependencies
├── package.json                # Node.js dependencies (frontend)
└── LICENSE                     # AGPL-3.0 license
```

## Code Style & Conventions

### Python
- **Style Guide**: PEP 8 compliant
- **Formatter**: black (line length: 100)
- **Linter**: ruff (replaces flake8, isort, pyupgrade)
- **Type Checking**: mypy with strict mode
- **Imports**:
  - Absolute imports from `src/` root
  - Group: stdlib → third-party → local
  - Example: `from src.agents.base import BaseAgent`
- **Naming**:
  - `snake_case` for functions, variables, modules
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- **Docstrings**: Google-style docstrings with type hints
  ```python
  def extract_clauses(document: str, clause_types: list[str]) -> dict[str, list[str]]:
      """Extract specific clauses from a legal document.

      Args:
          document: The full text of the legal document.
          clause_types: List of clause types to extract (e.g., ['indemnity', 'termination']).

      Returns:
          Dictionary mapping clause types to extracted text segments.

      Raises:
          ValueError: If document is empty or clause_types is invalid.
      """
  ```

### TypeScript
- **Style Guide**: Airbnb TypeScript style
- **Formatter**: Prettier
- **Linter**: ESLint with TypeScript plugin
- **Naming**:
  - `camelCase` for functions, variables
  - `PascalCase` for components, classes, types
  - `UPPER_SNAKE_CASE` for constants
- **Documentation**: JSDoc comments for public APIs

### Legal Domain Standards
- **Citations**: Bluebook format (21st edition)
  - Example: `Brown v. Board of Educ., 347 U.S. 483 (1954)`
- **Entity Recognition**: Use standardized legal entity schemas
  - `PARTY`, `JUDGE`, `ATTORNEY`, `COURT`, `STATUTE`, `CASE_CITATION`
- **Document Schemas**: Define Pydantic models for all legal document types
  ```python
  from pydantic import BaseModel, Field

  class Contract(BaseModel):
      """Legal contract document schema."""
      parties: list[str] = Field(..., description="Contract parties")
      effective_date: str = Field(..., description="Contract effective date")
      clauses: dict[str, str] = Field(default_factory=dict)
      jurisdiction: str = Field(..., description="Governing jurisdiction")
  ```

## Development Patterns

### Agentic Architecture Patterns

#### 1. Hierarchical Orchestration
Use a coordinator agent to manage specialized subagents:
```python
from typing import TypedDict
from langgraph.graph import StateGraph
from src.agents.orchestration import CoordinatorAgent
from src.agents.parsing import ContractParsingAgent
from src.agents.extraction import ClauseExtractionAgent

# Define workflow state
class WorkflowState(TypedDict):
    document: str
    parsed_data: dict
    extracted_clauses: dict
    output: str

# Build workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("parse", ContractParsingAgent())
workflow.add_node("extract", ClauseExtractionAgent())
workflow.add_edge("parse", "extract")
```

#### 2. Agent Delegation
Each agent has a single responsibility:
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all legal agents."""

    def __init__(self, name: str, skills: list[str], tools: list[Tool]):
        self.name = name
        self.skills = skills
        self.tools = tools

    @abstractmethod
    async def execute(self, input_data: dict) -> dict:
        """Execute agent's primary task."""
        pass
```

#### 3. Tool Composition
Agents use composable tools:
```python
from langchain.tools import Tool

citation_validator = Tool(
    name="validate_citation",
    description="Validate legal citation format (Bluebook)",
    func=validate_bluebook_citation
)

entity_extractor = Tool(
    name="extract_entities",
    description="Extract legal entities (parties, judges, courts)",
    func=extract_legal_entities
)
```

### Test-Driven Development (TDD)
1. **Write tests first**: Define expected behavior before implementation
2. **Run tests** (they should fail initially)
3. **Implement** code to pass tests
4. **Refactor** while maintaining passing tests

Example:
```python
# tests/unit/test_citation.py
def test_validate_bluebook_citation():
    """Test Bluebook citation validation."""
    valid_citation = "Brown v. Board of Educ., 347 U.S. 483 (1954)"
    assert validate_bluebook_citation(valid_citation) is True

    invalid_citation = "Brown v Board of Education"
    assert validate_bluebook_citation(invalid_citation) is False
```

### Git Workflow
- **Branch Naming**:
  - `feature/description` for new features
  - `fix/description` for bug fixes
  - `refactor/description` for refactoring
  - `docs/description` for documentation
- **Commit Messages**: Conventional Commits format
  ```text
  feat(agents): add contract parsing agent with clause extraction

  - Implement ContractParsingAgent class
  - Add clause type detection using spaCy NER
  - Support PDF and DOCX input formats

  Closes #12
  ```
- **Pull Requests**: Use PR template with:
  - Summary of changes
  - Test plan checklist
  - Breaking changes (if any)
  - Documentation updates

## Legal Domain Rules

### Compliance & Privacy
- **GDPR Compliance**: Anonymize PII in training datasets
- **Attorney-Client Privilege**: Mark privileged documents with `confidential=True`
- **Audit Logging**: Log all document processing operations with timestamps
- **Data Retention**: Comply with jurisdiction-specific retention policies

### Citation Validation
All legal citations must:
1. Follow Bluebook format (21st edition)
2. Include case name, volume, reporter, page, and year
3. Be validated against citation corpus (if available)

### Output Standards
Legal document outputs must include:
- **Reasoning Traces**: Step-by-step explanation of analysis
- **Citations**: Source citations for all legal claims
- **Confidence Scores**: Model confidence for each output
- **Disclaimers**: "This is AI-generated output. Consult a licensed attorney."

## Testing & Quality

### Test Coverage Targets
- **Unit Tests**: >80% code coverage
- **Integration Tests**: All agent workflows end-to-end
- **Performance**: Document processing <5s for standard contracts

### Test Categories
1. **Unit Tests**: Individual functions/classes
2. **Integration Tests**: Multi-agent workflows
3. **E2E Tests**: Full application flows (API → agents → output)
4. **Property-Based Tests**: Use Hypothesis for legal text generation

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test category
pytest tests/unit/
```

### Performance Benchmarks
- **Document Parsing**: <2s for 50-page PDF
- **Clause Extraction**: <3s for standard contract
- **RAG Retrieval**: <500ms for top-k=10 results
- **Model Inference**: <1s per query (Gemma 2B on TPU)

## Tunix Hackathon Guidelines

### Model Training
- **Model**: Gemma 2B or Gemma 3 1B
- **Infrastructure**: Kaggle TPU V3-8
- **Framework**: JAX/Flax with Keras 3.0
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficiency
- **Dataset**: Legal corpus (cases, statutes, contracts)

### Submission Requirements
1. **README.md**: Comprehensive project overview
2. **Kaggle Notebook**: Fine-tuning + inference demo
3. **Video**: 2-minute demo of agentic workflow
4. **Model Weights**: Uploaded to Kaggle/Hugging Face
5. **Documentation**: Setup guide, API docs, usage examples

### Evaluation Criteria
- **Innovation**: Novel agentic architecture for legal domain
- **Explainability**: Clear reasoning traces and citations
- **Performance**: Speed, accuracy, scalability
- **Usability**: Clean UI, documentation, onboarding
- **Legal Accuracy**: Correct citation format, entity extraction

## Command Reference

### Development
```bash
# Install dependencies
pip install -e ".[dev]"

# Format code
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/

# Start API server
uvicorn src.api.main:app --reload

# Start frontend (from src/frontend/)
npm run dev
```

### Kaggle TPU Setup
```bash
# Setup TPU environment
bash scripts/setup_kaggle_tpu.sh

# Preprocess legal corpus
python scripts/preprocess_legal_corpus.py --input data/raw --output data/processed

# Fine-tune Gemma on TPU
python src/models/gemma/train.py --config configs/models/gemma_2b_lora.yaml
```

## Resources

### Legal Domain
- [Bluebook Citation Guide](https://www.legalbluebook.com/)
- [LegalBERT: Pre-trained Language Model for Legal Domain](https://arxiv.org/abs/2010.02559)
- [CUAD Dataset: Contract Understanding Atticus Dataset](https://github.com/TheAtticusProject/cuad)

### Multi-Agent Systems
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AutoGen: Multi-Agent Conversations](https://microsoft.github.io/autogen/)
- [CrewAI: Role-based Agent Framework](https://github.com/joaomdmoura/crewAI)

### Kaggle Tunix
- [Google Tunix Hackathon](https://www.kaggle.com/competitions/google-tunix-hackathon)
- [Gemma Model Documentation](https://ai.google.dev/gemma)
- [Kaggle TPU Guide](https://www.kaggle.com/docs/tpu)

## Contact & Support

For questions or issues:
- GitHub Issues: [judicaita/judicAIta/issues](https://github.com/judicaita/judicAIta/issues)
- Kaggle Discussion: Google Tunix Hackathon forum
- Documentation: [docs.judicaita.dev](https://docs.judicaita.dev) (coming soon)

---

**Last Updated**: 2025-11-13
**Maintainers**: JudicAIta Core Team
**License**: AGPL-3.0
