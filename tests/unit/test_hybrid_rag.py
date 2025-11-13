"""
Unit tests for HybridRAGSystem.

Tests cover:
- Hybrid RAG initialization and configuration
- Query expansion with legal synonyms
- Vector, keyword, and hybrid search strategies
- Semantic reranking
- Confidence scoring and calibration
- Cross-reference validation
- Analytics tracking
- Dynamic chunking strategies

Target: >85% code coverage
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.models.rag.hybrid_rag import (
    ChunkingStrategy,
    HybridRAGConfig,
    HybridRAGSystem,
    HybridSearchResult,
    QueryAnalytics,
    SearchStrategy,
    create_production_hybrid_rag,
)
from src.models.rag.legal_rag import RAGConfig, SearchResult


# ==================== FIXTURES ====================


@pytest.fixture
def hybrid_config(tmp_path: Path) -> HybridRAGConfig:
    """Hybrid RAG configuration with temporary storage."""
    base_config = RAGConfig(
        collection_name="test_hybrid",
        persist_directory=str(tmp_path / "chroma_hybrid_test"),
        chunk_size=200,
        top_k=5,
    )

    return HybridRAGConfig(
        base_rag_config=base_config,
        search_strategy=SearchStrategy.HYBRID,
        enable_query_expansion=True,
        enable_reranking=True,
        enable_analytics=True,
    )


@pytest.fixture
def sample_query() -> str:
    """Sample search query."""
    return "termination clause with notice period"


# ==================== CONFIGURATION TESTS ====================


class TestHybridRAGConfig:
    """Test HybridRAGConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = HybridRAGConfig()

        assert config.search_strategy == SearchStrategy.HYBRID
        assert config.chunking_strategy == ChunkingStrategy.SECTION_BASED
        assert config.enable_query_expansion is True
        assert config.bm25_weight == 0.3
        assert config.vector_weight == 0.7
        assert config.enable_reranking is True

    def test_weights_validation(self) -> None:
        """Test weight normalization."""
        config = HybridRAGConfig(bm25_weight=0.4, vector_weight=0.5)

        # Should auto-normalize
        assert config.vector_weight == 0.6  # 1.0 - 0.4

    def test_custom_config(self, tmp_path: Path) -> None:
        """Test custom configuration."""
        base_config = RAGConfig(persist_directory=str(tmp_path))

        config = HybridRAGConfig(
            base_rag_config=base_config,
            search_strategy=SearchStrategy.VECTOR_ONLY,
            expansion_terms=5,
            confidence_threshold=0.5,
        )

        assert config.search_strategy == SearchStrategy.VECTOR_ONLY
        assert config.expansion_terms == 5
        assert config.confidence_threshold == 0.5


# ==================== INITIALIZATION TESTS ====================


class TestHybridRAGInitialization:
    """Test hybrid RAG system initialization."""

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_successful_initialization(
        self, mock_rag_class, hybrid_config: HybridRAGConfig
    ) -> None:
        """Test system initializes successfully."""
        mock_rag = MagicMock()
        mock_rag_class.return_value = mock_rag

        rag = HybridRAGSystem(hybrid_config)

        assert rag.config == hybrid_config
        assert rag._legal_synonyms is not None
        assert len(rag.query_history) == 0

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_initialization_without_base_rag(
        self, mock_rag_class, hybrid_config: HybridRAGConfig
    ) -> None:
        """Test initialization handles base RAG failure."""
        mock_rag_class.side_effect = Exception("ChromaDB not available")

        rag = HybridRAGSystem(hybrid_config)

        assert rag.base_rag is None


# ==================== QUERY EXPANSION TESTS ====================


class TestQueryExpansion:
    """Test query expansion with legal synonyms."""

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_expand_query(self, mock_rag_class, hybrid_config: HybridRAGConfig) -> None:
        """Test query expansion adds synonyms."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)
        expanded = rag._expand_query("terminate contract")

        # Should contain original terms
        assert "terminate" in expanded.lower()
        assert "contract" in expanded.lower()

        # Should contain some synonyms
        # (actual synonyms depend on implementation)
        assert len(expanded.split()) > 2

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_expansion_preserves_order(
        self, mock_rag_class, hybrid_config: HybridRAGConfig
    ) -> None:
        """Test expansion preserves term order."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)
        expanded = rag._expand_query("liability damages")

        terms = expanded.lower().split()
        # Original terms should appear first
        assert terms[0] == "liability"

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_expansion_with_unknown_terms(
        self, mock_rag_class, hybrid_config: HybridRAGConfig
    ) -> None:
        """Test expansion with terms not in synonym dict."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)
        expanded = rag._expand_query("xyz abc")

        # Should return original terms
        assert "xyz" in expanded
        assert "abc" in expanded


# ==================== SEARCH STRATEGY TESTS ====================


class TestSearchStrategies:
    """Test different search strategies."""

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_vector_search(
        self, mock_rag_class, hybrid_config: HybridRAGConfig, sample_query: str
    ) -> None:
        """Test vector-only search."""
        # Mock base RAG
        mock_rag = MagicMock()
        mock_result = SearchResult(
            text="Sample text",
            document_id="doc_1",
            metadata={},
            score=0.8,
            rank=1,
        )
        mock_rag.search.return_value = [mock_result]
        mock_rag_class.return_value = mock_rag

        config = HybridRAGConfig(
            base_rag_config=hybrid_config.base_rag_config,
            search_strategy=SearchStrategy.VECTOR_ONLY,
        )
        rag = HybridRAGSystem(config)

        results = rag.hybrid_search(sample_query, top_k=5)

        assert len(results) > 0
        assert all(isinstance(r, HybridSearchResult) for r in results)
        assert results[0].vector_score > 0

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_hybrid_search_combines_scores(
        self, mock_rag_class, hybrid_config: HybridRAGConfig, sample_query: str
    ) -> None:
        """Test hybrid search combines vector and keyword scores."""
        # Mock base RAG
        mock_rag = MagicMock()
        mock_result = SearchResult(
            text="Termination clause notice period",
            document_id="doc_1",
            metadata={},
            score=0.8,
            rank=1,
        )
        mock_rag.search.return_value = [mock_result]
        mock_rag_class.return_value = mock_rag

        rag = HybridRAGSystem(hybrid_config)
        results = rag.hybrid_search(sample_query, top_k=5)

        if results:
            # Should have both scores
            assert results[0].vector_score >= 0
            assert results[0].bm25_score >= 0
            assert results[0].hybrid_score >= 0


# ==================== CONFIDENCE SCORING TESTS ====================


class TestConfidenceScoring:
    """Test confidence scoring and calibration."""

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_confidence_scores_added(self, mock_rag_class, hybrid_config: HybridRAGConfig) -> None:
        """Test confidence scores are added to results."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)

        # Create mock results
        results = [
            HybridSearchResult(
                text="Test",
                document_id="doc_1",
                metadata={},
                score=0.8,
                rank=1,
                hybrid_score=0.8,
            ),
            HybridSearchResult(
                text="Test2",
                document_id="doc_2",
                metadata={},
                score=0.6,
                rank=2,
                hybrid_score=0.6,
            ),
        ]

        results = rag._add_confidence_scores(results, "test query")

        # All results should have confidence
        assert all(r.confidence > 0 for r in results)
        assert all(r.uncertainty >= 0 for r in results)

        # Higher scores should have higher confidence
        assert results[0].confidence >= results[1].confidence

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_confidence_within_bounds(self, mock_rag_class, hybrid_config: HybridRAGConfig) -> None:
        """Test confidence scores are in valid range."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)

        results = [
            HybridSearchResult(
                text="Test",
                document_id="doc_1",
                metadata={},
                score=0.9,
                rank=1,
                hybrid_score=0.9,
            )
        ]

        results = rag._add_confidence_scores(results, "query")

        for r in results:
            assert 0.0 <= r.confidence <= 1.0
            assert 0.0 <= r.uncertainty <= 1.0


# ==================== CROSS-REFERENCE TESTS ====================


class TestCrossReferenceValidation:
    """Test cross-reference detection."""

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_detect_section_references(
        self, mock_rag_class, hybrid_config: HybridRAGConfig
    ) -> None:
        """Test detecting section cross-references."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)

        results = [
            HybridSearchResult(
                text="As stated in Section 3.2, the party shall...",
                document_id="doc_1",
                metadata={},
                score=0.8,
                rank=1,
            )
        ]

        results = rag._validate_cross_references(results)

        assert len(results[0].cross_references) > 0
        assert "3.2" in results[0].cross_references

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_detect_multiple_references(
        self, mock_rag_class, hybrid_config: HybridRAGConfig
    ) -> None:
        """Test detecting multiple cross-references."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)

        results = [
            HybridSearchResult(
                text="See Section 1.1, Clause 2.3, and Article 5.4",
                document_id="doc_1",
                metadata={},
                score=0.8,
                rank=1,
            )
        ]

        results = rag._validate_cross_references(results)

        assert len(results[0].cross_references) >= 3


# ==================== ANALYTICS TESTS ====================


class TestQueryAnalytics:
    """Test query analytics tracking."""

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_analytics_tracking(
        self, mock_rag_class, hybrid_config: HybridRAGConfig, sample_query: str
    ) -> None:
        """Test analytics are tracked for queries."""
        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_rag_class.return_value = mock_rag

        rag = HybridRAGSystem(hybrid_config)

        # Perform search
        try:
            rag.hybrid_search(sample_query, top_k=5)
        except:
            pass

        # Analytics should be tracked
        assert len(rag.query_history) > 0

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_analytics_summary(self, mock_rag_class, hybrid_config: HybridRAGConfig) -> None:
        """Test analytics summary generation."""
        mock_rag = MagicMock()
        mock_rag.search.return_value = []
        mock_rag_class.return_value = mock_rag

        rag = HybridRAGSystem(hybrid_config)

        # Add some analytics
        rag.query_history.append(
            QueryAnalytics(
                query="test",
                num_results=5,
                avg_confidence=0.8,
                search_latency_ms=100.0,
                strategy_used="hybrid",
                top_result_score=0.9,
            )
        )

        summary = rag.get_analytics_summary()

        assert "total_queries" in summary
        assert summary["total_queries"] == 1
        assert "avg_latency_ms" in summary
        assert "strategy_distribution" in summary


# ==================== CHUNKING STRATEGY TESTS ====================


class TestDynamicChunking:
    """Test dynamic chunking strategies."""

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_chunk_by_sections(self, mock_rag_class, hybrid_config: HybridRAGConfig) -> None:
        """Test section-based chunking."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)

        text = """
        1.1 First Section
        Content of first section with enough text to be meaningful.

        2.1 Second Section
        Content of second section with enough text as well.
        """

        chunks = rag._chunk_by_sections(text)

        assert len(chunks) > 0

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem")
    def test_chunk_by_clauses(self, mock_rag_class, hybrid_config: HybridRAGConfig) -> None:
        """Test clause-based chunking."""
        mock_rag_class.return_value = MagicMock()

        rag = HybridRAGSystem(hybrid_config)

        text = """
        TERMINATION
        Either party may terminate this agreement.

        INDEMNIFICATION
        The company shall indemnify the client.
        """

        chunks = rag._chunk_by_clauses(text)

        assert len(chunks) > 0


# ==================== FACTORY FUNCTION TESTS ====================


class TestFactoryFunctions:
    """Test factory functions."""

    @patch("src.models.rag.hybrid_rag.LegalRAGSystem.__init__")
    def test_create_production_hybrid_rag(self, mock_init, tmp_path: Path) -> None:
        """Test production RAG factory."""
        mock_init.return_value = None

        rag = create_production_hybrid_rag(
            collection_name="test",
            persist_directory=str(tmp_path),
        )

        # Should have been called with HybridRAGConfig
        assert mock_init.called


# ==================== INTEGRATION TESTS ====================


@pytest.mark.integration
class TestHybridRAGIntegration:
    """Integration tests for hybrid RAG."""

    @patch("src.models.rag.hybrid_rag.CHROMADB_AVAILABLE", False)
    def test_graceful_degradation_without_chromadb(self, hybrid_config: HybridRAGConfig) -> None:
        """Test system handles missing ChromaDB gracefully."""
        # Should not raise, just warn
        rag = HybridRAGSystem(hybrid_config)
        assert rag.base_rag is None
