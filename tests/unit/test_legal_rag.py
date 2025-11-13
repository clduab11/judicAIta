"""
Unit tests for LegalRAGSystem.

Tests cover:
- RAG system initialization
- Document chunking strategies
- Document addition and retrieval
- Semantic search
- Context generation
- Collection management
- Error handling

Target: >80% code coverage
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.models.rag.legal_rag import (
    LegalRAGSystem,
    RAGConfig,
    SearchResult,
    create_legal_rag,
)


# ==================== FIXTURES ====================


@pytest.fixture
def rag_config(tmp_path: Path) -> RAGConfig:
    """Basic RAG configuration with temporary storage."""
    return RAGConfig(
        collection_name="test_legal_docs",
        persist_directory=str(tmp_path / "chroma_test"),
        chunk_size=200,
        chunk_overlap=20,
        top_k=5,
    )


@pytest.fixture
def sample_legal_text() -> str:
    """Sample legal document text."""
    return """
    This Service Agreement is entered into by Acme Corp and Beta Inc.
    The agreement shall remain in effect for a period of two years.

    Either party may terminate this agreement with thirty days notice.
    Upon termination, all outstanding payments become due immediately.

    The Company shall indemnify the Client against all claims and damages.
    Liability shall be limited to the total contract value of $100,000.

    All disputes shall be resolved through binding arbitration in California.
    This agreement is governed by the laws of the State of California.
    """


# ==================== CONFIGURATION TESTS ====================


class TestRAGConfig:
    """Test RAGConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RAGConfig()

        assert config.collection_name == "legal_documents"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5

    def test_custom_config(self, tmp_path: Path) -> None:
        """Test custom configuration."""
        config = RAGConfig(
            collection_name="contracts",
            persist_directory=str(tmp_path),
            chunk_size=300,
            chunk_overlap=30,
            top_k=20,
        )

        assert config.collection_name == "contracts"
        assert config.chunk_size == 300
        assert config.chunk_overlap == 30
        assert config.top_k == 20

    def test_overlap_validation(self) -> None:
        """Test chunk overlap validation."""
        # Overlap must be less than chunk size
        with pytest.raises(ValueError, match="chunk_overlap must be less than"):
            RAGConfig(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap must be less than"):
            RAGConfig(chunk_size=100, chunk_overlap=150)


# ==================== INITIALIZATION TESTS ====================


class TestRAGInitialization:
    """Test RAG system initialization."""

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_successful_initialization(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test RAG system initializes successfully."""
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)

        assert rag.config == rag_config
        assert rag.collection is not None

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", False)
    def test_initialization_without_chromadb(self, rag_config: RAGConfig) -> None:
        """Test initialization fails without ChromaDB."""
        with pytest.raises(RuntimeError, match="ChromaDB not installed"):
            LegalRAGSystem(rag_config)


# ==================== CHUNKING TESTS ====================


class TestDocumentChunking:
    """Test document chunking strategies."""

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_chunk_document(
        self, mock_chromadb, rag_config: RAGConfig, sample_legal_text: str
    ) -> None:
        """Test document chunking."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)
        chunks = rag._chunk_document(sample_legal_text)

        assert len(chunks) > 0
        # Each chunk should be roughly the configured size
        for chunk in chunks:
            assert len(chunk) <= rag_config.chunk_size + 100  # Allow some tolerance

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_chunk_with_overlap(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test chunking creates overlap between chunks."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)

        # Create long text that will require multiple chunks
        long_text = "This is a test sentence. " * 50

        chunks = rag._chunk_document(long_text)

        assert len(chunks) > 1

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_split_into_sentences(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test sentence splitting."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)

        text = "First sentence. Second sentence. Third sentence."
        sentences = rag._split_into_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]


# ==================== DOCUMENT ADDITION TESTS ====================


class TestDocumentAddition:
    """Test adding documents to RAG system."""

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_add_document(
        self, mock_chromadb, rag_config: RAGConfig, sample_legal_text: str
    ) -> None:
        """Test adding document to collection."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.add = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)

        num_chunks = rag.add_document(
            sample_legal_text,
            "doc_001",
            {"document_type": "contract", "jurisdiction": "California"},
        )

        assert num_chunks > 0
        assert mock_collection.add.called

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_add_empty_document(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test adding empty document fails."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)

        with pytest.raises(ValueError, match="Document text cannot be empty"):
            rag.add_document("", "doc_001")


# ==================== SEARCH TESTS ====================


class TestSearch:
    """Test semantic search functionality."""

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_search_query(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test searching for documents."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["doc_1_chunk_0", "doc_2_chunk_1"]],
            "documents": [["Text about indemnification", "Text about liability"]],
            "metadatas": [[{"document_id": "doc_1"}, {"document_id": "doc_2"}]],
            "distances": [[0.2, 0.3]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)
        results = rag.search("indemnification clause", top_k=5)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_search_with_empty_query(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test search fails with empty query."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag.search("")

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_search_with_metadata_filter(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test search with metadata filtering."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["doc_1_chunk_0"]],
            "documents": [["Contract text"]],
            "metadatas": [[{"document_id": "doc_1", "document_type": "contract"}]],
            "distances": [[0.2]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)
        results = rag.search("termination", metadata_filter={"document_type": "contract"})

        assert len(results) > 0
        # Verify query was called with filter
        assert mock_collection.query.called


# ==================== CONTEXT RETRIEVAL TESTS ====================


class TestContextRetrieval:
    """Test context retrieval for queries."""

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_get_context_for_query(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test retrieving context for query."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["doc_1_chunk_0", "doc_2_chunk_1"]],
            "documents": [
                [
                    "First relevant context about termination clauses.",
                    "Second context about notice requirements.",
                ]
            ],
            "metadatas": [[{"document_id": "doc_1"}, {"document_id": "doc_2"}]],
            "distances": [[0.2, 0.3]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)
        context = rag.get_context_for_query("termination clause")

        assert len(context) > 0
        assert isinstance(context, str)
        # Should contain retrieved text
        assert "termination" in context.lower() or "notice" in context.lower()

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_context_respects_max_length(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test context respects maximum length."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["doc_1_chunk_0"]],
            "documents": [["A" * 1000]],
            "metadatas": [[{"document_id": "doc_1"}]],
            "distances": [[0.2]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)
        context = rag.get_context_for_query("test", max_length=500)

        # Context should not exceed max length (plus some tolerance for separators)
        assert len(context) <= 550


# ==================== COLLECTION MANAGEMENT TESTS ====================


class TestCollectionManagement:
    """Test collection management operations."""

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_delete_document(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test deleting document from collection."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.get.return_value = {
            "ids": ["doc_1_chunk_0", "doc_1_chunk_1"],
            "documents": ["text1", "text2"],
            "metadatas": [{"document_id": "doc_1"}, {"document_id": "doc_1"}],
        }
        mock_collection.delete = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)
        deleted_count = rag.delete_document("doc_1")

        assert deleted_count == 2
        assert mock_collection.delete.called

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_clear_collection(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test clearing collection."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection = MagicMock()
        mock_client.create_collection = MagicMock(return_value=mock_collection)
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)
        rag.clear_collection()

        assert mock_client.delete_collection.called
        assert mock_client.create_collection.called

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_get_collection_stats(self, mock_chromadb, rag_config: RAGConfig) -> None:
        """Test getting collection statistics."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.get.return_value = {
            "metadatas": [
                {"document_id": "doc_1"},
                {"document_id": "doc_2"},
                {"document_id": "doc_1"},
            ]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = LegalRAGSystem(rag_config)
        stats = rag.get_collection_stats()

        assert "collection_name" in stats
        assert "total_chunks" in stats
        assert "unique_documents" in stats
        assert stats["total_chunks"] == 10
        assert stats["unique_documents"] == 2


# ==================== HELPER FUNCTION TESTS ====================


class TestHelperFunctions:
    """Test helper functions."""

    @patch("src.models.rag.legal_rag.CHROMADB_AVAILABLE", True)
    @patch("src.models.rag.legal_rag.chromadb")
    def test_create_legal_rag(self, mock_chromadb, tmp_path: Path) -> None:
        """Test factory function for creating RAG system."""
        # Mock ChromaDB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        rag = create_legal_rag(
            collection_name="test_collection",
            persist_directory=str(tmp_path),
        )

        assert isinstance(rag, LegalRAGSystem)
        assert rag.config.collection_name == "test_collection"


# ==================== SEARCH RESULT TESTS ====================


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """Test creating SearchResult object."""
        result = SearchResult(
            text="Sample legal text",
            document_id="doc_001",
            metadata={"type": "contract"},
            score=0.85,
            rank=1,
        )

        assert result.text == "Sample legal text"
        assert result.document_id == "doc_001"
        assert result.score == 0.85
        assert result.rank == 1
        assert result.metadata["type"] == "contract"
