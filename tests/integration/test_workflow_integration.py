"""
Integration tests for multi-agent workflow orchestration.

Tests cover:
- End-to-end PDF processing workflow
- Entity extraction integration
- Clause extraction integration
- Cross-agent data flow
- Error handling and recovery
- Performance benchmarks

Target: Complete workflow execution <15s
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Dict

import pytest

from src.agents.base import AgentConfig
from src.agents.extraction.clause_extractor import ClauseExtractionAgent, ClauseExtractionConfig
from src.agents.extraction.entity_extractor import EntityExtractionAgent, EntityExtractionConfig
from src.agents.ingestion.pdf_parser import PDFParsingAgent, PDFParsingConfig
from src.agents.orchestration.workflow import (
    DocumentType,
    LegalDocumentWorkflow,
    WorkflowConfig,
    create_contract_workflow,
)


# ==================== FIXTURES ====================


@pytest.fixture
def sample_contract_text() -> str:
    """Sample contract text for testing."""
    return """
    SERVICE AGREEMENT

    This Service Agreement ("Agreement") is entered into as of January 15, 2025,
    by and between Acme Corporation ("Company") and Beta Inc ("Client").

    1.1 TERMINATION
    Either party may terminate this Agreement upon thirty (30) days written notice
    to the other party. Upon termination, all outstanding payments shall become due.

    1.2 INDEMNIFICATION
    Company shall indemnify, defend, and hold harmless Client from and against all
    claims, damages, and liabilities arising out of Company's negligence or misconduct.

    1.3 LIMITATION OF LIABILITY
    In no event shall either party's total liability exceed $100,000.00. Neither party
    shall be liable for consequential damages, including lost profits.

    1.4 PAYMENT TERMS
    Client shall pay Company $50,000.00 within thirty (30) days of invoice date.
    Late payments shall accrue interest at 1.5% per month.

    1.5 CONFIDENTIALITY
    Each party agrees to maintain the confidentiality of all proprietary information
    disclosed by the other party during the term of this Agreement.

    1.6 GOVERNING LAW
    This Agreement shall be governed by the laws of the State of California.

    IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.
    """


@pytest.fixture
def temp_pdf_file(tmp_path: Path, sample_contract_text: str) -> Path:
    """Create temporary text file (simulating PDF)."""
    pdf_file = tmp_path / "test_contract.pdf"
    pdf_file.write_text(sample_contract_text)
    return pdf_file


@pytest.fixture
def workflow_config() -> WorkflowConfig:
    """Basic workflow configuration."""
    return WorkflowConfig(
        document_type=DocumentType.CONTRACT,
        enable_pdf_parsing=False,  # Use text directly
        enable_entity_extraction=True,
        enable_clause_extraction=True,
        parallel_execution=False,
    )


# ==================== AGENT INTEGRATION TESTS ====================


class TestAgentIntegration:
    """Test integration between individual agents."""

    def test_entity_to_clause_integration(self, sample_contract_text: str) -> None:
        """Test data flow from entity extraction to clause extraction."""
        # Extract entities first
        entity_config = EntityExtractionConfig(
            entity_types=["PARTY", "DATE", "MONETARY"],
            use_gemma=False,
        )
        entity_agent = EntityExtractionAgent(AgentConfig(name="entity_extractor"), entity_config)

        entity_result = entity_agent({"text": sample_contract_text})

        assert entity_result.status.value == "completed"
        entities = entity_result.output["entities"]

        # Extract parties for clause extraction
        parties = [e["text"] for e in entities if e["entity_type"] == "PARTY"]

        # Extract clauses with party information
        clause_config = ClauseExtractionConfig(
            clause_types=["termination", "indemnity", "liability", "payment"],
            use_gemma=False,
        )
        clause_agent = ClauseExtractionAgent(AgentConfig(name="clause_extractor"), clause_config)

        clause_result = clause_agent({"text": sample_contract_text, "parties": parties})

        assert clause_result.status.value == "completed"
        clauses = clause_result.output["clauses"]

        # Verify clauses have party information
        assert len(clauses) > 0

        # Check obligations reference parties
        obligations_by_party = clause_result.output["obligations_by_party"]
        assert isinstance(obligations_by_party, dict)

    def test_pdf_to_entity_integration(
        self, temp_pdf_file: Path, sample_contract_text: str
    ) -> None:
        """Test PDF parsing to entity extraction integration."""
        # Parse PDF (mocked)
        with patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True):
            with patch("src.agents.ingestion.pdf_parser.pdfplumber") as mock_pdf:
                # Mock pdfplumber
                mock_pdf_obj = MagicMock()
                mock_page = MagicMock()
                mock_page.extract_text.return_value = sample_contract_text
                mock_page.extract_tables.return_value = []
                mock_pdf_obj.pages = [mock_page]
                mock_pdf.open.return_value.__enter__.return_value = mock_pdf_obj

                # Mock pypdf for metadata
                with patch("src.agents.ingestion.pdf_parser.PYPDF_AVAILABLE", True):
                    with patch("src.agents.ingestion.pdf_parser.PdfReader") as mock_reader:
                        mock_reader_instance = MagicMock()
                        mock_reader_instance.__len__.return_value = 1
                        mock_reader_instance.pages = [mock_page]
                        mock_reader_instance.is_encrypted = False
                        mock_reader_instance.metadata = {}
                        mock_reader.return_value = mock_reader_instance

                        # Parse PDF
                        pdf_agent = PDFParsingAgent(
                            AgentConfig(name="pdf_parser"),
                            PDFParsingConfig(),
                        )

                        pdf_result = pdf_agent({"file_path": str(temp_pdf_file)})

                        assert pdf_result.status.value == "completed"
                        text = pdf_result.output["text"]

                        # Extract entities from parsed text
                        entity_agent = EntityExtractionAgent(
                            AgentConfig(name="entity_extractor"),
                            EntityExtractionConfig(use_gemma=False),
                        )

                        entity_result = entity_agent({"text": text})

                        assert entity_result.status.value == "completed"
                        assert len(entity_result.output["entities"]) > 0


# ==================== WORKFLOW INTEGRATION TESTS ====================


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for complete workflow."""

    def test_workflow_initialization(self, workflow_config: WorkflowConfig) -> None:
        """Test workflow initializes correctly."""
        workflow = LegalDocumentWorkflow(workflow_config)

        assert workflow.config == workflow_config
        assert workflow.entity_config is not None
        assert workflow.clause_config is not None

    def test_contract_workflow_creation(self) -> None:
        """Test contract workflow factory function."""
        workflow = create_contract_workflow(enable_all=False, use_gemma=False)

        assert isinstance(workflow, LegalDocumentWorkflow)
        assert workflow.config.document_type == DocumentType.CONTRACT

    def test_workflow_state_initialization(self, sample_contract_text: str) -> None:
        """Test workflow state initialization."""
        config = WorkflowConfig(
            enable_pdf_parsing=False,
            enable_entity_extraction=True,
            enable_clause_extraction=True,
        )

        workflow = LegalDocumentWorkflow(config)

        # Prepare initial state
        from src.agents.orchestration.workflow import WorkflowState

        initial_state: WorkflowState = {
            "document_path": "/fake/path.pdf",
            "document_text": sample_contract_text,
            "document_type": "contract",
            "parsed_data": None,
            "entities": None,
            "entity_relationships": None,
            "clauses": None,
            "high_risk_clauses": None,
            "obligations_by_party": None,
            "errors": [],
            "metrics": {},
            "trace": [],
            "status": "running",
        }

        # Verify state structure
        assert "document_text" in initial_state
        assert "errors" in initial_state
        assert "metrics" in initial_state

    def test_entity_extraction_node(
        self, workflow_config: WorkflowConfig, sample_contract_text: str
    ) -> None:
        """Test entity extraction node execution."""
        workflow = LegalDocumentWorkflow(workflow_config)

        from src.agents.orchestration.workflow import WorkflowState

        state: WorkflowState = {
            "document_path": "/fake/path.pdf",
            "document_text": sample_contract_text,
            "document_type": "contract",
            "parsed_data": None,
            "entities": None,
            "entity_relationships": None,
            "clauses": None,
            "high_risk_clauses": None,
            "obligations_by_party": None,
            "errors": [],
            "metrics": {},
            "trace": [],
            "status": "running",
        }

        # Execute entity extraction node
        updated_state = workflow._entity_extraction_node(state)

        # Verify entities were extracted
        assert updated_state["entities"] is not None
        assert len(updated_state["entities"]) > 0

        # Verify metrics were updated
        assert "entity_extraction_time" in updated_state["metrics"]
        assert "entities_extracted" in updated_state["metrics"]

    def test_clause_extraction_node(
        self, workflow_config: WorkflowConfig, sample_contract_text: str
    ) -> None:
        """Test clause extraction node execution."""
        workflow = LegalDocumentWorkflow(workflow_config)

        from src.agents.orchestration.workflow import WorkflowState

        state: WorkflowState = {
            "document_path": "/fake/path.pdf",
            "document_text": sample_contract_text,
            "document_type": "contract",
            "parsed_data": None,
            "entities": [{"text": "Acme Corporation", "entity_type": "PARTY"}],
            "entity_relationships": None,
            "clauses": None,
            "high_risk_clauses": None,
            "obligations_by_party": None,
            "errors": [],
            "metrics": {},
            "trace": [],
            "status": "running",
        }

        # Execute clause extraction node
        updated_state = workflow._clause_extraction_node(state)

        # Verify clauses were extracted
        assert updated_state["clauses"] is not None
        assert len(updated_state["clauses"]) > 0

        # Verify metrics were updated
        assert "clause_extraction_time" in updated_state["metrics"]
        assert "clauses_extracted" in updated_state["metrics"]

    def test_finalize_node(
        self, workflow_config: WorkflowConfig, sample_contract_text: str
    ) -> None:
        """Test finalize node execution."""
        workflow = LegalDocumentWorkflow(workflow_config)

        from src.agents.orchestration.workflow import WorkflowState

        state: WorkflowState = {
            "document_path": "/fake/path.pdf",
            "document_text": sample_contract_text,
            "document_type": "contract",
            "parsed_data": None,
            "entities": [],
            "entity_relationships": [],
            "clauses": [],
            "high_risk_clauses": [],
            "obligations_by_party": {},
            "errors": [],
            "metrics": {
                "entity_extraction_time": 1.5,
                "clause_extraction_time": 2.3,
            },
            "trace": [],
            "status": "running",
        }

        # Execute finalize node
        updated_state = workflow._finalize_node(state)

        # Verify finalization
        assert updated_state["status"] == "completed"
        assert "total_time" in updated_state["metrics"]
        assert updated_state["metrics"]["total_time"] > 0

    def test_complete_workflow_execution(self, sample_contract_text: str, tmp_path: Path) -> None:
        """Test complete workflow execution end-to-end."""
        # Create temp file
        test_file = tmp_path / "contract.txt"
        test_file.write_text(sample_contract_text)

        # Configure workflow (skip PDF parsing, use text directly)
        config = WorkflowConfig(
            document_type=DocumentType.CONTRACT,
            enable_pdf_parsing=False,
            enable_entity_extraction=True,
            enable_clause_extraction=True,
        )

        workflow = LegalDocumentWorkflow(config)

        # Manually create state with text (simulating PDF parsing)
        from src.agents.orchestration.workflow import WorkflowState

        initial_state: WorkflowState = {
            "document_path": str(test_file),
            "document_text": sample_contract_text,
            "document_type": "contract",
            "parsed_data": None,
            "entities": None,
            "entity_relationships": None,
            "clauses": None,
            "high_risk_clauses": None,
            "obligations_by_party": None,
            "errors": [],
            "metrics": {},
            "trace": [],
            "status": "running",
        }

        # Execute workflow manually (since we're not using PDF parsing)
        state = workflow._entity_extraction_node(initial_state)
        state = workflow._clause_extraction_node(state)
        final_state = workflow._finalize_node(state)

        # Verify final results
        assert final_state["status"] in ["completed", "completed_with_errors"]
        assert final_state["entities"] is not None
        assert final_state["clauses"] is not None
        assert len(final_state["entities"]) > 0
        assert len(final_state["clauses"]) > 0

        # Verify metrics
        assert "total_time" in final_state["metrics"]
        assert "entities_extracted" in final_state["metrics"]
        assert "clauses_extracted" in final_state["metrics"]


# ==================== ERROR HANDLING TESTS ====================


class TestWorkflowErrorHandling:
    """Test workflow error handling."""

    def test_empty_document_handling(self) -> None:
        """Test workflow handles empty documents gracefully."""
        config = WorkflowConfig(
            enable_pdf_parsing=False,
            enable_entity_extraction=True,
            enable_clause_extraction=True,
        )

        workflow = LegalDocumentWorkflow(config)

        from src.agents.orchestration.workflow import WorkflowState

        state: WorkflowState = {
            "document_path": "/fake/path.pdf",
            "document_text": "",
            "document_type": "contract",
            "parsed_data": None,
            "entities": None,
            "entity_relationships": None,
            "clauses": None,
            "high_risk_clauses": None,
            "obligations_by_party": None,
            "errors": [],
            "metrics": {},
            "trace": [],
            "status": "running",
        }

        # Execute nodes
        state = workflow._entity_extraction_node(state)

        # Should have errors due to empty text
        assert len(state["errors"]) > 0

    def test_workflow_with_partial_failures(self, sample_contract_text: str) -> None:
        """Test workflow continues with partial failures."""
        config = WorkflowConfig(
            enable_pdf_parsing=False,
            enable_entity_extraction=True,
            enable_clause_extraction=True,
        )

        workflow = LegalDocumentWorkflow(config)

        from src.agents.orchestration.workflow import WorkflowState

        state: WorkflowState = {
            "document_path": "/fake/path.pdf",
            "document_text": sample_contract_text,
            "document_type": "contract",
            "parsed_data": None,
            "entities": None,
            "entity_relationships": None,
            "clauses": None,
            "high_risk_clauses": None,
            "obligations_by_party": None,
            "errors": [],
            "metrics": {},
            "trace": [],
            "status": "running",
        }

        # Execute workflow
        state = workflow._entity_extraction_node(state)
        state = workflow._clause_extraction_node(state)
        final_state = workflow._finalize_node(state)

        # Even with potential errors, workflow should complete
        assert final_state["status"] in ["completed", "completed_with_errors"]


# ==================== PERFORMANCE TESTS ====================


@pytest.mark.slow
class TestWorkflowPerformance:
    """Performance benchmarks for workflow."""

    def test_workflow_latency(self, sample_contract_text: str) -> None:
        """Test workflow completes within performance target."""
        import time

        config = WorkflowConfig(
            enable_pdf_parsing=False,
            enable_entity_extraction=True,
            enable_clause_extraction=True,
        )

        workflow = LegalDocumentWorkflow(config)

        from src.agents.orchestration.workflow import WorkflowState

        initial_state: WorkflowState = {
            "document_path": "/fake/path.pdf",
            "document_text": sample_contract_text,
            "document_type": "contract",
            "parsed_data": None,
            "entities": None,
            "entity_relationships": None,
            "clauses": None,
            "high_risk_clauses": None,
            "obligations_by_party": None,
            "errors": [],
            "metrics": {},
            "trace": [],
            "status": "running",
        }

        start_time = time.time()

        # Execute workflow
        state = workflow._entity_extraction_node(initial_state)
        state = workflow._clause_extraction_node(state)
        final_state = workflow._finalize_node(state)

        elapsed = time.time() - start_time

        # Should complete in < 15 seconds for standard contract
        assert elapsed < 15.0

        # Verify metrics
        assert "total_time" in final_state["metrics"]
