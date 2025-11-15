# JudicAIta: Multi-Agent Architecture Design

**Version**: 1.0
**Date**: 2025-11-13
**Target**: Google Tunix Hackathon - Gemma 3 1B Fine-tuning

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Agent Taxonomy](#agent-taxonomy)
3. [Workflow Orchestration](#workflow-orchestration)
4. [Agent Communication Protocol](#agent-communication-protocol)
5. [Model Integration (Gemma 3n)](#model-integration-gemma-3n)
6. [RAG System Design](#rag-system-design)
7. [Example Workflows](#example-workflows)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Architecture Overview

### Design Philosophy

JudicAIta employs a **hierarchical multi-agent architecture** where specialized agents collaborate to handle complex legal document workflows. The system is built on these principles:

1. **Single Responsibility**: Each agent has one well-defined task
2. **Composability**: Agents can be chained and combined
3. **Explainability**: All reasoning is traced and auditable
4. **Model Agnostic**: Core logic independent of LLM (supports Gemma 3n family)
5. **Fault Tolerance**: Graceful degradation when agents fail

### System Layers

```text
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  (React Frontend, CLI, API)                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                 Orchestration Layer                          │
│  (CoordinatorAgent, LangGraph Workflows)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Agent Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Ingestion  │  │  Analysis   │  │  Reasoning  │        │
│  │   Agents    │  │   Agents    │  │   Agents    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Tool Layer                                │
│  (Parsers, Extractors, Validators, Generators)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Model Layer                                │
│  (Gemma 3 1B + LoRA, RAG, Vector DB)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Agent Taxonomy

### Base Agent Class

All agents inherit from a common base class:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent purpose")
    model: str = Field(default="google/gemma-3-1b", description="LLM model to use")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=1024, description="Max output tokens")
    tools: List[str] = Field(default_factory=list, description="Available tools")


class BaseAgent(ABC):
    """Base class for all legal agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.description = config.description
        self.model = self._initialize_model(config.model)
        self.tools = self._load_tools(config.tools)
        self.trace = []  # Reasoning trace

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's primary task."""
        pass

    def _initialize_model(self, model_name: str):
        """Initialize the LLM model (Gemma 3n compatible)."""
        # Model initialization logic (supports Gemma 2.5/3)
        pass

    def _load_tools(self, tool_names: List[str]):
        """Load tools from registry."""
        pass

    def add_trace(self, step: str, reasoning: str, output: Any):
        """Add step to reasoning trace."""
        from datetime import datetime
        
        self.trace.append({
            "step": step,
            "reasoning": reasoning,
            "output": output,
            "timestamp": datetime.now().isoformat()
        })
```

### Agent Categories

#### 1. Document Ingestion Agents

**Purpose**: Parse and preprocess legal documents

##### PDFParsingAgent
- **Input**: PDF file path or bytes
- **Output**: Structured text, metadata, page layout
- **Tools**: PyPDF2, pdfplumber, PyMuPDF
- **Specialization**: Handle multi-column layouts, extract tables

```python
class PDFParsingAgent(BaseAgent):
    """Parse PDF legal documents."""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            input_data: {"file_path": str, "extract_tables": bool}

        Returns:
            {
                "text": str,
                "pages": List[str],
                "metadata": Dict,
                "tables": List[Dict] (optional)
            }
        """
        file_path = input_data["file_path"]
        self.add_trace("parse_pdf", f"Parsing PDF: {file_path}", None)

        # Parse with pdfplumber for robustness
        text, pages, tables = self._parse_pdf(file_path)

        output = {
            "text": text,
            "pages": pages,
            "metadata": self._extract_metadata(file_path),
            "tables": tables if input_data.get("extract_tables") else []
        }

        self.add_trace("parse_complete", "PDF parsing successful", output)
        return output
```

##### DOCXParsingAgent
- **Input**: DOCX file path or bytes
- **Output**: Structured text, styles, metadata
- **Tools**: python-docx
- **Specialization**: Preserve formatting, extract comments

##### OCRAgent
- **Input**: Image or scanned PDF
- **Output**: Extracted text
- **Tools**: Tesseract OCR, EasyOCR
- **Specialization**: Handle low-quality scans, multi-language

#### 2. Analysis Agents

**Purpose**: Extract structured information from documents

##### EntityExtractionAgent
- **Input**: Document text
- **Output**: Legal entities (parties, judges, courts, statutes)
- **Model**: Gemma 3 1B fine-tuned on legal NER
- **Tools**: spaCy, custom NER model

```python
class EntityExtractionAgent(BaseAgent):
    """Extract legal entities from documents."""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            input_data: {"text": str, "entity_types": List[str]}

        Returns:
            {
                "entities": {
                    "PARTY": List[str],
                    "JUDGE": List[str],
                    "COURT": List[str],
                    "STATUTE": List[str],
                    "CASE_CITATION": List[str]
                },
                "confidence": Dict[str, float]
            }
        """
        text = input_data["text"]
        entity_types = input_data.get("entity_types", ["PARTY", "JUDGE", "COURT"])

        self.add_trace("extract_entities", f"Extracting: {entity_types}", None)

        # Use Gemma 3 1B for entity extraction with prompt
        prompt = self._build_ner_prompt(text, entity_types)
        entities = await self._call_model(prompt)

        # Validate with spaCy legal NER
        entities = self._validate_entities(entities, text)

        output = {
            "entities": entities,
            "confidence": self._calculate_confidence(entities)
        }

        self.add_trace("extraction_complete", "Entities extracted", output)
        return output
```

##### ClauseExtractionAgent
- **Input**: Contract text
- **Output**: Identified clauses with types (termination, indemnity, liability, etc.)
- **Model**: Gemma 3 1B with RAG (CUAD dataset)
- **Tools**: CUAD classifier, regex patterns

##### CitationAnalysisAgent
- **Input**: Legal document text
- **Output**: Parsed citations (Bluebook format)
- **Model**: Gemma 3 1B fine-tuned on citation parsing
- **Tools**: Citation parser, Bluebook validator

```python
class CitationAnalysisAgent(BaseAgent):
    """Parse and analyze legal citations."""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            input_data: {"text": str, "validate_bluebook": bool}

        Returns:
            {
                "citations": List[{
                    "raw": str,
                    "parsed": {
                        "case_name": str,
                        "volume": str,
                        "reporter": str,
                        "page": str,
                        "year": str
                    },
                    "valid_bluebook": bool
                }]
            }
        """
        text = input_data["text"]
        validate = input_data.get("validate_bluebook", True)

        self.add_trace("find_citations", "Searching for citations", None)

        # Use regex + Gemma 3 1B for citation extraction
        raw_citations = self._find_citations(text)

        # Parse each citation
        citations = []
        for raw in raw_citations:
            parsed = await self._parse_citation(raw)
            valid = self._validate_bluebook(parsed) if validate else True

            citations.append({
                "raw": raw,
                "parsed": parsed,
                "valid_bluebook": valid
            })

        output = {"citations": citations}
        self.add_trace("citations_parsed", f"Found {len(citations)} citations", output)
        return output
```

#### 3. Reasoning Agents

**Purpose**: Perform legal analysis and generate insights

##### LegalQAAgent
- **Input**: Document + question
- **Output**: Answer with citations
- **Model**: Gemma 3 1B + RAG
- **Tools**: Vector DB retrieval, citation linker

```python
class LegalQAAgent(BaseAgent):
    """Answer questions about legal documents."""

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            input_data: {
                "document": str,
                "question": str,
                "retrieve_context": bool
            }

        Returns:
            {
                "answer": str,
                "citations": List[str],
                "confidence": float,
                "reasoning_trace": List[str]
            }
        """
        document = input_data["document"]
        question = input_data["question"]

        self.add_trace("receive_question", f"Q: {question}", None)

        # Retrieve relevant context from document (RAG)
        if input_data.get("retrieve_context", True):
            context = await self._retrieve_context(document, question)
        else:
            context = document

        self.add_trace("retrieve_context", f"Context: {len(context)} chars", None)

        # Generate answer with Gemma 3 1B
        prompt = self._build_qa_prompt(context, question)
        answer = await self._call_model(prompt)

        # Extract citations from answer
        citations = self._extract_citations(answer)

        output = {
            "answer": answer,
            "citations": citations,
            "confidence": self._calculate_confidence(answer),
            "reasoning_trace": self.trace
        }

        self.add_trace("answer_generated", f"A: {answer[:100]}...", output)
        return output
```

##### SummarizationAgent
- **Input**: Legal document
- **Output**: Plain-English summary
- **Model**: Gemma 3 1B fine-tuned on legal summarization
- **Tools**: BART/Pegasus fallback, readability scorer

##### ComplianceAgent
- **Input**: Document + compliance rules
- **Output**: Compliance report with violations
- **Model**: Gemma 3 1B + rule engine
- **Tools**: Regex patterns, regulatory knowledge base

#### 4. Output Agents

**Purpose**: Generate outputs and reports

##### DraftingAgent
- **Input**: Template + context
- **Output**: Generated legal document
- **Model**: Gemma 3 1B fine-tuned on legal drafting
- **Tools**: Template engine, clause library

##### ReportingAgent
- **Input**: Analysis results
- **Output**: Structured report (JSON, PDF, HTML)
- **Tools**: Jinja2 templates, WeasyPrint

##### ExplanationAgent
- **Input**: Agent traces
- **Output**: Human-readable explanation
- **Model**: Gemma 3 1B for natural language generation
- **Tools**: Trace formatter, visualization

---

## Workflow Orchestration

### LangGraph-Based Orchestration

JudicAIta uses **LangGraph** for declarative workflow orchestration:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List


class ContractAnalysisState(TypedDict):
    """State for contract analysis workflow."""
    # Input
    file_path: str

    # Intermediate
    parsed_text: str
    pages: List[str]
    entities: Dict[str, List[str]]
    clauses: Dict[str, str]
    citations: List[Dict]

    # Output
    summary: str
    compliance_report: Dict
    reasoning_trace: List[Dict]
    output_report: str


# Define workflow graph
workflow = StateGraph(ContractAnalysisState)

# Add nodes (agents)
workflow.add_node("parse_pdf", PDFParsingAgent(config))
workflow.add_node("extract_entities", EntityExtractionAgent(config))
workflow.add_node("extract_clauses", ClauseExtractionAgent(config))
workflow.add_node("analyze_citations", CitationAnalysisAgent(config))
workflow.add_node("summarize", SummarizationAgent(config))
workflow.add_node("check_compliance", ComplianceAgent(config))
workflow.add_node("generate_report", ReportingAgent(config))

# Define edges (workflow)
workflow.set_entry_point("parse_pdf")
workflow.add_edge("parse_pdf", "extract_entities")
workflow.add_edge("parse_pdf", "extract_clauses")  # Parallel execution
workflow.add_edge("extract_entities", "analyze_citations")
workflow.add_edge("extract_clauses", "summarize")
workflow.add_edge(["analyze_citations", "summarize"], "check_compliance")
workflow.add_edge("check_compliance", "generate_report")
workflow.add_edge("generate_report", END)

# Compile workflow
app = workflow.compile()

# Execute
result = await app.ainvoke({
    "file_path": "contract.pdf"
})
```

### Conditional Routing

For complex workflows, use conditional edges:

```python
def route_by_document_type(state: ContractAnalysisState) -> str:
    """Route to different agents based on document type."""
    text = state["parsed_text"]

    if "plaintiff" in text.lower() and "defendant" in text.lower():
        return "case_analysis_agent"
    elif "party a" in text.lower() or "agreement" in text.lower():
        return "contract_analysis_agent"
    else:
        return "generic_analysis_agent"


workflow.add_conditional_edges(
    "parse_pdf",
    route_by_document_type,
    {
        "case_analysis_agent": "case_analysis",
        "contract_analysis_agent": "contract_analysis",
        "generic_analysis_agent": "generic_analysis"
    }
)
```

### Parallel Agent Execution

Execute independent agents in parallel for efficiency:

```python
# These agents can run in parallel
workflow.add_edge("parse_pdf", "extract_entities")
workflow.add_edge("parse_pdf", "extract_clauses")
workflow.add_edge("parse_pdf", "analyze_citations")

# Synchronize before next stage
workflow.add_edge(
    ["extract_entities", "extract_clauses", "analyze_citations"],
    "summarize"
)
```

---

## Agent Communication Protocol

### Message Format

Agents communicate using typed dictionaries validated by Pydantic:

```python
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime


class AgentMessage(BaseModel):
    """Standard message format for agent communication."""
    sender: str = Field(..., description="Agent name sending message")
    receiver: str = Field(..., description="Target agent name")
    message_type: str = Field(..., description="Message type (request, response, error)")
    data: Dict[str, Any] = Field(..., description="Message payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentRequest(BaseModel):
    """Request to an agent."""
    task: str = Field(..., description="Task to perform")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    config: Optional[Dict[str, Any]] = Field(None, description="Task-specific config")


class AgentResponse(BaseModel):
    """Response from an agent."""
    success: bool = Field(..., description="Whether task succeeded")
    output: Dict[str, Any] = Field(..., description="Output data")
    error: Optional[str] = Field(None, description="Error message if failed")
    trace: List[Dict] = Field(default_factory=list, description="Reasoning trace")
```

### Error Handling

Agents implement graceful error handling:

```python
class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class AgentExecutionError(AgentError):
    """Error during agent execution."""
    pass


class AgentTimeoutError(AgentError):
    """Agent execution timeout."""
    pass


# In agent execute method
async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Agent logic
        result = await self._perform_task(input_data)
        return {
            "success": True,
            "output": result,
            "trace": self.trace
        }
    except AgentTimeoutError as e:
        self.add_trace("error", f"Timeout: {str(e)}", None)
        return {
            "success": False,
            "output": {},
            "error": f"Agent timeout: {str(e)}",
            "trace": self.trace
        }
    except Exception as e:
        self.add_trace("error", f"Execution failed: {str(e)}", None)
        return {
            "success": False,
            "output": {},
            "error": str(e),
            "trace": self.trace
        }
```

---

## Model Integration (Gemma 3n)

### Gemma 3 1B Integration

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class GemmaModelWrapper:
    """Wrapper for Gemma 3 1B with cross-compatibility."""

    def __init__(self, model_name: str = "google/gemma-3-1b", use_lora: bool = True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._load_model(model_name, use_lora)

        # Compatibility settings
        self.supports_gemma_25 = self._check_gemma_25_compatibility()
        self.supports_gemma_3 = True

    def _load_model(self, model_name: str, use_lora: bool):
        """Load model with optional LoRA."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        if use_lora:
            from peft import PeftModel
            # Load LoRA weights fine-tuned on legal corpus
            model = PeftModel.from_pretrained(model, "legal_lora_weights")

        return model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """Generate text with Gemma 3 1B."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            **kwargs
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]  # Return only new tokens

    def _check_gemma_25_compatibility(self) -> bool:
        """Test compatibility with Gemma 2.5 architecture."""
        try:
            # Load Gemma 2.5 tokenizer and compare vocab
            gemma_25_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2.5-1b")
            vocab_match = (
                self.tokenizer.vocab_size == gemma_25_tokenizer.vocab_size
            )
            return vocab_match
        except Exception:
            return False
```

### Cross-Model Compatibility Layer

```python
class CrossModelCompatibilityLayer:
    """Ensure compatibility across Gemma 2.5 and Gemma 3."""

    def __init__(self, primary_model: str = "google/gemma-3-1b"):
        self.primary_model = primary_model
        self.fallback_model = "google/gemma-2.5-1b"
        self.model = GemmaModelWrapper(primary_model)

    async def generate_with_fallback(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate with primary model, fallback to Gemma 2.5 if needed."""
        try:
            output = await self.model.generate(prompt, **kwargs)
            return {
                "output": output,
                "model_used": self.primary_model,
                "fallback": False
            }
        except Exception as e:
            # Fallback to Gemma 2.5
            fallback_model = GemmaModelWrapper(self.fallback_model)
            output = await fallback_model.generate(prompt, **kwargs)
            return {
                "output": output,
                "model_used": self.fallback_model,
                "fallback": True,
                "error": str(e)
            }

    def validate_cross_model_consistency(
        self,
        prompt: str,
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """Test output consistency between Gemma 2.5 and 3."""
        # Generate with both models
        gemma_3_output = self.model.generate(prompt)
        gemma_25_model = GemmaModelWrapper(self.fallback_model)
        gemma_25_output = gemma_25_model.generate(prompt)

        # Compare outputs (semantic similarity)
        similarity = self._calculate_semantic_similarity(
            gemma_3_output,
            gemma_25_output
        )

        return {
            "gemma_3_output": gemma_3_output,
            "gemma_25_output": gemma_25_output,
            "similarity": similarity,
            "compatible": similarity >= (1 - tolerance)
        }
```

---

## RAG System Design

### Vector Database for Legal Knowledge

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Dict


class LegalKnowledgeRAG:
    """RAG system for legal document retrieval."""

    def __init__(self, corpus_path: str, embedding_model: str = "BAAI/bge-base-en-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(
            collection_name="legal_corpus",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        self._load_corpus(corpus_path)

    def _load_corpus(self, corpus_path: str):
        """Load legal corpus into vector database."""
        # Load documents (cases, statutes, contracts)
        documents = self._read_legal_documents(corpus_path)
        self.vectorstore.add_documents(documents)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve relevant legal documents."""
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
            filter=filters
        )

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]

    async def retrieve_and_generate(
        self,
        query: str,
        model: GemmaModelWrapper,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """RAG: Retrieve context and generate answer."""
        # Retrieve relevant documents
        retrieved_docs = await self.retrieve(query, top_k=top_k)

        # Build prompt with context
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        prompt = f"""Context:
{context}

Question: {query}

Answer (cite sources using Bluebook format):"""

        # Generate answer with Gemma 3 1B
        answer = await model.generate(prompt, max_tokens=512)

        return {
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "query": query
        }
```

---

## Example Workflows

### Workflow 1: Contract Review Pipeline

```python
async def contract_review_workflow(contract_path: str) -> Dict[str, Any]:
    """
    Complete contract review workflow.

    Steps:
    1. Parse PDF contract
    2. Extract parties and key entities
    3. Extract and classify clauses
    4. Identify potential risks
    5. Generate plain-English summary
    6. Create compliance report
    """
    # Initialize agents
    pdf_agent = PDFParsingAgent(config)
    entity_agent = EntityExtractionAgent(config)
    clause_agent = ClauseExtractionAgent(config)
    qa_agent = LegalQAAgent(config)
    summary_agent = SummarizationAgent(config)
    compliance_agent = ComplianceAgent(config)

    # Step 1: Parse PDF
    parsed = await pdf_agent.execute({"file_path": contract_path})
    text = parsed["output"]["text"]

    # Step 2: Extract entities (parallel)
    entities_task = entity_agent.execute({"text": text})
    clauses_task = clause_agent.execute({"text": text})

    entities, clauses = await asyncio.gather(entities_task, clauses_task)

    # Step 3: Risk analysis
    risks = await qa_agent.execute({
        "document": text,
        "question": "What are the main legal risks in this contract?",
        "retrieve_context": True
    })

    # Step 4: Summary
    summary = await summary_agent.execute({
        "text": text,
        "max_length": 500
    })

    # Step 5: Compliance check
    compliance = await compliance_agent.execute({
        "text": text,
        "rules": ["GDPR", "CCPA", "SOX"]
    })

    # Aggregate results
    return {
        "contract_path": contract_path,
        "parties": entities["output"]["entities"].get("PARTY", []),
        "key_clauses": clauses["output"]["clauses"],
        "risks": risks["output"]["answer"],
        "summary": summary["output"]["summary"],
        "compliance_report": compliance["output"]["report"],
        "reasoning_traces": {
            "entities": entities["trace"],
            "clauses": clauses["trace"],
            "risks": risks["trace"]
        }
    }
```

### Workflow 2: Legal Research Assistant

```python
async def legal_research_workflow(
    research_question: str,
    jurisdiction: str = "federal"
) -> Dict[str, Any]:
    """
    Legal research workflow with RAG.

    Steps:
    1. Retrieve relevant case law and statutes
    2. Analyze citations and precedents
    3. Generate research memo
    4. Provide Bluebook citations
    """
    # Initialize RAG and agents
    rag = LegalKnowledgeRAG(corpus_path="data/legal_corpus")
    model = GemmaModelWrapper("google/gemma-3-1b")
    citation_agent = CitationAnalysisAgent(config)

    # Step 1: RAG retrieval and generation
    research_result = await rag.retrieve_and_generate(
        query=research_question,
        model=model,
        top_k=5
    )

    # Step 2: Extract and validate citations
    citations = await citation_agent.execute({
        "text": research_result["answer"],
        "validate_bluebook": True
    })

    # Step 3: Format research memo
    memo = f"""LEGAL RESEARCH MEMO

QUESTION:
{research_question}

JURISDICTION:
{jurisdiction}

ANALYSIS:
{research_result['answer']}

CITATIONS:
{_format_citations(citations['output']['citations'])}

SOURCES:
{_format_sources(research_result['retrieved_documents'])}
"""

    return {
        "question": research_question,
        "answer": research_result["answer"],
        "citations": citations["output"]["citations"],
        "memo": memo,
        "retrieved_documents": research_result["retrieved_documents"]
    }
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goal**: Setup infrastructure and base agent framework

- [ ] Create project structure (`src/agents/`, `src/models/`, etc.)
- [ ] Implement `BaseAgent` class
- [ ] Setup Gemma 3 1B integration with transformers
- [ ] Create agent communication protocol (Pydantic models)
- [ ] Setup LangGraph workflow orchestration
- [ ] Write unit tests for base classes

**Deliverables**:
- Working `BaseAgent` class
- Gemma 3 1B model wrapper
- Basic LangGraph workflow example

### Phase 2: Core Agents (Week 2)

**Goal**: Implement essential document processing agents

- [ ] PDFParsingAgent (PyPDF2 + pdfplumber)
- [ ] EntityExtractionAgent (spaCy + Gemma 3 1B)
- [ ] ClauseExtractionAgent (CUAD dataset + Gemma 3 1B)
- [ ] CitationAnalysisAgent (regex + Bluebook validation)
- [ ] Integration tests for each agent

**Deliverables**:
- 4 working agents with tests
- Example workflows (contract review, citation analysis)

### Phase 3: RAG & Fine-tuning (Week 2-3)

**Goal**: Build RAG system and fine-tune Gemma 3 1B

- [ ] Setup Chroma vector database
- [ ] Curate legal corpus (CUAD, CaseHOLD, LexGLUE)
- [ ] Implement RAG retrieval system
- [ ] Fine-tune Gemma 3 1B on Kaggle TPU with LoRA
- [ ] Test Gemma 2.5/3 cross-compatibility
- [ ] Benchmark performance

**Deliverables**:
- Fine-tuned Gemma 3 1B model (legal domain)
- RAG system with legal corpus
- Cross-compatibility validation report

### Phase 4: Advanced Agents (Week 3)

**Goal**: Implement reasoning and output agents

- [ ] LegalQAAgent (RAG + Gemma 3 1B)
- [ ] SummarizationAgent
- [ ] ComplianceAgent
- [ ] DraftingAgent
- [ ] ReportingAgent
- [ ] End-to-end workflow tests

**Deliverables**:
- 5 additional agents
- Complete contract review workflow
- Legal research workflow

### Phase 5: UI & Documentation (Week 3-4)

**Goal**: Build user interface and hackathon materials

- [ ] FastAPI backend with agent endpoints
- [ ] React frontend (document upload, results display)
- [ ] Kaggle notebook with fine-tuning demo
- [ ] Comprehensive README.md
- [ ] TUNIX_SETUP.md guide
- [ ] 2-minute demo video
- [ ] Upload model weights to Hugging Face

**Deliverables**:
- Working web application
- Complete hackathon submission package

---

## Success Criteria

### Technical Metrics
- ✅ Gemma 3 1B inference <1s per query on TPU
- ✅ Cross-compatibility: <5% accuracy loss (Gemma 2.5 vs 3)
- ✅ Entity extraction F1 >0.85
- ✅ Clause extraction F1 >0.80
- ✅ Citation accuracy >0.90
- ✅ Test coverage >80%

### Hackathon Metrics
- ✅ All agents documented with examples
- ✅ Clear reasoning traces for all outputs
- ✅ Explainable AI with citation mapping
- ✅ Working Kaggle notebook
- ✅ Video demonstration
- ✅ Model weights published

---

## Conclusion

This multi-agent architecture provides a scalable, maintainable foundation for JudicAIta. By leveraging:
- **Hierarchical orchestration** with LangGraph
- **Specialized agents** with single responsibilities
- **Gemma 3 1B** optimized for legal domain
- **Cross-compatibility** with Gemma 2.5/3
- **RAG** for grounded generation

JudicAIta can deliver transparent, explainable legal automation that meets the needs of legal professionals and the Google Tunix Hackathon requirements.

---

**Next Steps**: Implement foundation (Phase 1) and begin agent development.
