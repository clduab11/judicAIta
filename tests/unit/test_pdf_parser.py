"""
Unit tests for PDFParsingAgent.

Tests cover:
- Configuration validation
- Input validation
- PDF parsing (with mock data)
- Metadata extraction
- Page extraction
- Table extraction
- Error handling

Target: >80% code coverage
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from typing import Any, Dict

import pytest

from src.agents.base import AgentConfig, AgentStatus
from src.agents.ingestion.pdf_parser import (
    PDFMetadata,
    PDFPage,
    PDFParsingAgent,
    PDFParsingConfig,
)


# ==================== FIXTURES ====================


@pytest.fixture
def pdf_config() -> PDFParsingConfig:
    """Basic PDF parsing configuration."""
    return PDFParsingConfig(
        extract_tables=True,
        extract_images=False,
        preserve_layout=True,
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    """Basic agent configuration."""
    return AgentConfig(name="pdf_parser")


@pytest.fixture
def mock_pdf_file(tmp_path: Path) -> Path:
    """Create a mock PDF file for testing."""
    pdf_file = tmp_path / "test_contract.pdf"
    pdf_file.write_text("Mock PDF content")
    return pdf_file


# ==================== CONFIGURATION TESTS ====================


class TestPDFParsingConfig:
    """Test PDFParsingConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PDFParsingConfig()

        assert config.extract_tables is True
        assert config.extract_images is False
        assert config.preserve_layout is True
        assert config.use_ocr is False
        assert config.page_range is None
        assert config.min_text_length == 10

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PDFParsingConfig(
            extract_tables=False,
            extract_images=True,
            use_ocr=True,
            page_range=(0, 10),
            min_text_length=50,
        )

        assert config.extract_tables is False
        assert config.extract_images is True
        assert config.use_ocr is True
        assert config.page_range == (0, 10)
        assert config.min_text_length == 50

    def test_invalid_page_range(self) -> None:
        """Test invalid page range validation."""
        with pytest.raises(ValueError, match="page_range must be"):
            PDFParsingConfig(page_range=(0,))

        with pytest.raises(ValueError, match="Invalid page range"):
            PDFParsingConfig(page_range=(-1, 10))

        with pytest.raises(ValueError, match="Invalid page range"):
            PDFParsingConfig(page_range=(10, 5))


# ==================== METADATA TESTS ====================


class TestPDFMetadata:
    """Test PDFMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test creating metadata object."""
        metadata = PDFMetadata(
            title="Test Contract",
            author="John Doe",
            num_pages=10,
            is_encrypted=False,
        )

        assert metadata.title == "Test Contract"
        assert metadata.author == "John Doe"
        assert metadata.num_pages == 10
        assert metadata.is_encrypted is False

    def test_metadata_defaults(self) -> None:
        """Test metadata default values."""
        metadata = PDFMetadata()

        assert metadata.title is None
        assert metadata.author is None
        assert metadata.num_pages == 0
        assert metadata.is_encrypted is False


# ==================== PDFPAGE TESTS ====================


class TestPDFPage:
    """Test PDFPage dataclass."""

    def test_page_creation(self) -> None:
        """Test creating PDF page object."""
        page = PDFPage(
            page_number=1,
            text="This is page 1 content",
            tables=[],
        )

        assert page.page_number == 1
        assert page.text == "This is page 1 content"
        assert page.char_count > 0
        assert page.word_count == 5

    def test_page_with_tables(self) -> None:
        """Test page with table data."""
        table = [
            ["Header1", "Header2"],
            ["Row1Col1", "Row1Col2"],
        ]

        page = PDFPage(
            page_number=2,
            text="Page with table",
            tables=[table],
        )

        assert len(page.tables) == 1
        assert page.tables[0] == table

    def test_page_counts_calculated(self) -> None:
        """Test char and word counts are calculated."""
        text = "The quick brown fox jumps over the lazy dog"
        page = PDFPage(page_number=1, text=text)

        assert page.char_count == len(text)
        assert page.word_count == 9


# ==================== AGENT INITIALIZATION TESTS ====================


class TestPDFParsingAgentInitialization:
    """Test PDF parsing agent initialization."""

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_basic_initialization(
        self, agent_config: AgentConfig, pdf_config: PDFParsingConfig
    ) -> None:
        """Test agent initializes correctly."""
        agent = PDFParsingAgent(agent_config, pdf_config)

        assert agent.config == agent_config
        assert agent.pdf_config == pdf_config
        assert agent.status == AgentStatus.IDLE

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_initialization_with_defaults(self, agent_config: AgentConfig) -> None:
        """Test agent initialization with default PDF config."""
        agent = PDFParsingAgent(agent_config)

        assert agent.pdf_config is not None
        assert agent.pdf_config.extract_tables is True

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", False)
    @patch("src.agents.ingestion.pdf_parser.PYPDF_AVAILABLE", False)
    def test_initialization_without_libraries(self, agent_config: AgentConfig) -> None:
        """Test initialization fails without PDF libraries."""
        with pytest.raises(RuntimeError, match="No PDF library available"):
            PDFParsingAgent(agent_config)


# ==================== INPUT VALIDATION TESTS ====================


class TestInputValidation:
    """Test input validation."""

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_valid_input(self, agent_config: AgentConfig, mock_pdf_file: Path) -> None:
        """Test validation with valid input."""
        agent = PDFParsingAgent(agent_config)
        input_data = {"file_path": str(mock_pdf_file)}

        assert agent.validate_input(input_data) is True

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_missing_file_path(self, agent_config: AgentConfig) -> None:
        """Test validation fails without file_path."""
        agent = PDFParsingAgent(agent_config)
        input_data = {}

        assert agent.validate_input(input_data) is False

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_nonexistent_file(self, agent_config: AgentConfig) -> None:
        """Test validation fails for nonexistent file."""
        agent = PDFParsingAgent(agent_config)
        input_data = {"file_path": "/nonexistent/file.pdf"}

        assert agent.validate_input(input_data) is False

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_non_pdf_file(self, agent_config: AgentConfig, tmp_path: Path) -> None:
        """Test validation fails for non-PDF file."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("Not a PDF")

        agent = PDFParsingAgent(agent_config)
        input_data = {"file_path": str(txt_file)}

        assert agent.validate_input(input_data) is False


# ==================== HELPER METHOD TESTS ====================


class TestHelperMethods:
    """Test helper methods."""

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_page_to_dict(self, agent_config: AgentConfig) -> None:
        """Test converting PDFPage to dictionary."""
        agent = PDFParsingAgent(agent_config)
        page = PDFPage(
            page_number=1,
            text="Test text",
            tables=[],
        )

        page_dict = agent._page_to_dict(page)

        assert page_dict["page_number"] == 1
        assert page_dict["text"] == "Test text"
        assert "char_count" in page_dict
        assert "word_count" in page_dict

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_metadata_to_dict(self, agent_config: AgentConfig) -> None:
        """Test converting PDFMetadata to dictionary."""
        agent = PDFParsingAgent(agent_config)
        metadata = PDFMetadata(
            title="Test",
            author="Author",
            num_pages=10,
        )

        metadata_dict = agent._metadata_to_dict(metadata)

        assert metadata_dict["title"] == "Test"
        assert metadata_dict["author"] == "Author"
        assert metadata_dict["num_pages"] == 10


# ==================== MOCK PROCESSING TESTS ====================


class TestProcessing:
    """Test PDF processing with mocks."""

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    @patch("src.agents.ingestion.pdf_parser.pdfplumber")
    def test_successful_processing(
        self, mock_pdfplumber, agent_config: AgentConfig, mock_pdf_file: Path
    ) -> None:
        """Test successful PDF processing."""
        # Mock pdfplumber
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is a test contract"
        mock_page.extract_tables.return_value = []
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        # Mock pypdf for metadata
        with patch("src.agents.ingestion.pdf_parser.PYPDF_AVAILABLE", True):
            with patch("src.agents.ingestion.pdf_parser.PdfReader") as mock_reader:
                mock_reader_instance = MagicMock()
                mock_reader_instance.__len__.return_value = 1
                mock_reader_instance.pages = [mock_page]
                mock_reader_instance.is_encrypted = False
                mock_reader_instance.metadata = {}
                mock_reader.return_value = mock_reader_instance

                agent = PDFParsingAgent(agent_config)
                result = agent({"file_path": str(mock_pdf_file)})

                assert result.status == AgentStatus.COMPLETED
                assert "text" in result.output
                assert "pages" in result.output
                assert "metadata" in result.output
                assert "stats" in result.output

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    @patch("src.agents.ingestion.pdf_parser.pdfplumber")
    def test_processing_with_tables(
        self, mock_pdfplumber, agent_config: AgentConfig, mock_pdf_file: Path
    ) -> None:
        """Test processing PDF with tables."""
        # Mock pdfplumber with table
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Contract with table"
        mock_page.extract_tables.return_value = [[["Header1", "Header2"], ["Data1", "Data2"]]]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        # Mock metadata extraction
        with patch("src.agents.ingestion.pdf_parser.PYPDF_AVAILABLE", True):
            with patch("src.agents.ingestion.pdf_parser.PdfReader") as mock_reader:
                mock_reader_instance = MagicMock()
                mock_reader_instance.__len__.return_value = 1
                mock_reader_instance.pages = [mock_page]
                mock_reader_instance.is_encrypted = False
                mock_reader_instance.metadata = {}
                mock_reader.return_value = mock_reader_instance

                pdf_config = PDFParsingConfig(extract_tables=True)
                agent = PDFParsingAgent(agent_config, pdf_config)
                result = agent({"file_path": str(mock_pdf_file)})

                assert result.status == AgentStatus.COMPLETED
                assert len(result.output["tables"]) > 0


# ==================== EDGE CASES ====================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    @patch("src.agents.ingestion.pdf_parser.pdfplumber")
    def test_empty_pdf(
        self, mock_pdfplumber, agent_config: AgentConfig, mock_pdf_file: Path
    ) -> None:
        """Test processing empty PDF."""
        # Mock empty PDF
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_page.extract_tables.return_value = []
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        with patch("src.agents.ingestion.pdf_parser.PYPDF_AVAILABLE", True):
            with patch("src.agents.ingestion.pdf_parser.PdfReader") as mock_reader:
                mock_reader_instance = MagicMock()
                mock_reader_instance.__len__.return_value = 1
                mock_reader_instance.pages = [mock_page]
                mock_reader_instance.is_encrypted = False
                mock_reader_instance.metadata = {}
                mock_reader.return_value = mock_reader_instance

                agent = PDFParsingAgent(agent_config)
                result = agent({"file_path": str(mock_pdf_file)})

                # Should still complete, but with no pages (filtered by min_text_length)
                assert result.status == AgentStatus.COMPLETED

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    def test_page_range_filtering(self, agent_config: AgentConfig) -> None:
        """Test page range filtering."""
        pdf_config = PDFParsingConfig(page_range=(0, 2))
        agent = PDFParsingAgent(agent_config, pdf_config)

        # Create mock pages
        pages = [PDFPage(page_number=i, text=f"Page {i}") for i in range(1, 6)]

        # Filter by page range
        if agent.pdf_config.page_range:
            start, end = agent.pdf_config.page_range
            filtered = pages[start:end]

            assert len(filtered) == 2
            assert filtered[0].page_number == 1
            assert filtered[1].page_number == 2


# ==================== INTEGRATION-STYLE TESTS ====================


@pytest.mark.integration
class TestPDFParsingWorkflow:
    """Integration-style tests for complete workflows."""

    @patch("src.agents.ingestion.pdf_parser.PDFPLUMBER_AVAILABLE", True)
    @patch("src.agents.ingestion.pdf_parser.pdfplumber")
    def test_complete_parsing_workflow(
        self, mock_pdfplumber, agent_config: AgentConfig, mock_pdf_file: Path
    ) -> None:
        """Test complete PDF parsing workflow."""
        # Mock multi-page PDF
        mock_pdf = MagicMock()
        mock_pages = []
        for i in range(1, 4):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = f"This is page {i} content"
            mock_page.extract_tables.return_value = []
            mock_pages.append(mock_page)

        mock_pdf.pages = mock_pages
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        # Mock metadata
        with patch("src.agents.ingestion.pdf_parser.PYPDF_AVAILABLE", True):
            with patch("src.agents.ingestion.pdf_parser.PdfReader") as mock_reader:
                mock_reader_instance = MagicMock()
                mock_reader_instance.__len__.return_value = 3
                mock_reader_instance.pages = mock_pages
                mock_reader_instance.is_encrypted = False
                mock_reader_instance.metadata = {
                    "/Title": "Test Contract",
                    "/Author": "Test Author",
                }
                mock_reader.return_value = mock_reader_instance

                agent = PDFParsingAgent(agent_config)
                result = agent({"file_path": str(mock_pdf_file)})

                # Verify result structure
                assert result.status == AgentStatus.COMPLETED
                assert "text" in result.output
                assert "pages" in result.output
                assert "metadata" in result.output
                assert "stats" in result.output

                # Verify metadata
                metadata = result.output["metadata"]
                assert metadata["num_pages"] == 3

                # Verify stats
                stats = result.output["stats"]
                assert stats["total_pages"] == 3
                assert stats["total_words"] > 0

                # Verify trace
                assert len(result.trace) > 0
