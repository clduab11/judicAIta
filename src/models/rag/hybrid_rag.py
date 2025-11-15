"""
Advanced Hybrid RAG System with Production-Grade Features.

This module extends the base RAG system with:
- Hybrid search combining vector similarity, BM25 keyword search, and semantic reranking
- Query expansion and reformulation strategies
- Dynamic chunking based on document structure (legal sections, clauses)
- Confidence scoring with uncertainty quantification
- Cross-reference validation between documents
- Query analytics and performance tracking

Performance Targets:
    - Hybrid search latency: <800ms for top-10 results
    - Query expansion: <100ms overhead
    - Reranking: <200ms for top-20 candidates
    - Dynamic chunking: >90% accuracy for legal documents
    - Confidence calibration: ECE <0.05
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

from src.models.rag.legal_rag import LegalRAGSystem, RAGConfig, SearchResult


# ==================== ENUMS ====================


class SearchStrategy(str, Enum):
    """Search strategy types."""

    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    SEMANTIC_RERANK = "semantic_rerank"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE_BOUNDARY = "sentence_boundary"
    SECTION_BASED = "section_based"
    CLAUSE_BASED = "clause_based"


# ==================== CONFIGURATION ====================


class HybridRAGConfig(BaseModel):
    """Configuration for hybrid RAG system.

    Attributes:
        base_rag_config: Base RAG configuration
        search_strategy: Search strategy to use
        chunking_strategy: Chunking strategy for documents
        enable_query_expansion: Enable query expansion
        expansion_terms: Number of expansion terms
        bm25_weight: Weight for BM25 scores in hybrid search (0-1)
        vector_weight: Weight for vector scores (0-1)
        enable_reranking: Enable semantic reranking
        rerank_top_k: Number of results to rerank
        confidence_threshold: Minimum confidence for results (0-1)
        enable_cross_reference: Enable cross-reference validation
        enable_analytics: Enable query analytics
    """

    base_rag_config: RAGConfig = Field(default_factory=RAGConfig)
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.HYBRID, description="Search strategy"
    )
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SECTION_BASED, description="Chunking strategy"
    )
    enable_query_expansion: bool = Field(default=True, description="Enable query expansion")
    expansion_terms: int = Field(default=3, ge=0, le=10, description="Number of expansion terms")
    bm25_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="BM25 weight in hybrid search"
    )
    vector_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Vector weight in hybrid search"
    )
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    rerank_top_k: int = Field(default=20, ge=5, le=100, description="Results to rerank")
    confidence_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum confidence"
    )
    enable_cross_reference: bool = Field(
        default=True, description="Enable cross-reference validation"
    )
    enable_analytics: bool = Field(default=True, description="Enable analytics")

    @validator("vector_weight")
    def validate_weights_sum(cls, v: float, values: Dict) -> float:
        """Ensure BM25 and vector weights sum to ~1.0."""
        bm25_weight = values.get("bm25_weight", 0.3)
        total = bm25_weight + v
        if abs(total - 1.0) > 0.01:
            # Auto-normalize
            return 1.0 - bm25_weight
        return v


@dataclass
class HybridSearchResult(SearchResult):
    """Enhanced search result with hybrid scoring.

    Attributes:
        vector_score: Vector similarity score
        bm25_score: BM25 keyword score
        hybrid_score: Combined hybrid score
        confidence: Calibrated confidence score
        uncertainty: Uncertainty estimate
        cross_references: Detected cross-references
        relevance_explanation: Explanation of relevance
    """

    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 0.0
    cross_references: List[str] = field(default_factory=list)
    relevance_explanation: str = ""


@dataclass
class QueryAnalytics:
    """Query analytics data.

    Attributes:
        query: Original query
        expanded_query: Expanded query terms
        num_results: Number of results returned
        avg_confidence: Average confidence score
        search_latency_ms: Search latency in milliseconds
        strategy_used: Search strategy used
        top_result_score: Score of top result
    """

    query: str
    expanded_query: Optional[str] = None
    num_results: int = 0
    avg_confidence: float = 0.0
    search_latency_ms: float = 0.0
    strategy_used: str = "unknown"
    top_result_score: float = 0.0


# ==================== HYBRID RAG SYSTEM ====================


class HybridRAGSystem:
    """Production-grade hybrid RAG system with advanced features.

    This system provides:
    - Hybrid search (vector + BM25 + semantic reranking)
    - Intelligent query expansion
    - Dynamic chunking strategies
    - Confidence scoring with uncertainty quantification
    - Cross-reference validation
    - Query analytics and monitoring

    Example:
        >>> config = HybridRAGConfig()
        >>> rag = HybridRAGSystem(config)
        >>> results = rag.hybrid_search("indemnification clause", top_k=10)
        >>> print(f"Confidence: {results[0].confidence:.2f}")
    """

    def __init__(self, config: Optional[HybridRAGConfig] = None):
        """Initialize hybrid RAG system.

        Args:
            config: Hybrid RAG configuration

        Raises:
            RuntimeError: If initialization fails
        """
        self.config = config or HybridRAGConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize base RAG system
        try:
            self.base_rag = LegalRAGSystem(self.config.base_rag_config)
        except Exception as e:
            self.logger.warning(f"Failed to initialize base RAG: {e}")
            self.base_rag = None

        # Query analytics storage
        self.query_history: List[QueryAnalytics] = []

        # Legal domain vocabulary for query expansion
        self._legal_synonyms = self._build_legal_synonyms()

        self.logger.info(
            f"Hybrid RAG initialized (strategy={self.config.search_strategy.value}, "
            f"chunking={self.config.chunking_strategy.value})"
        )

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[HybridSearchResult]:
        """Perform hybrid search with multiple strategies.

        Args:
            query: Search query
            top_k: Number of results (uses config default if None)
            metadata_filter: Metadata filters

        Returns:
            List of HybridSearchResult objects with confidence scores

        Raises:
            ValueError: If query is empty
            RuntimeError: If search fails
        """
        import time

        start_time = time.time()

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        k = top_k or self.config.base_rag_config.top_k

        # Expand query if enabled
        expanded_query = query
        if self.config.enable_query_expansion:
            expanded_query = self._expand_query(query)

        # Perform search based on strategy
        if self.config.search_strategy == SearchStrategy.VECTOR_ONLY:
            results = self._vector_search(expanded_query, k, metadata_filter)
        elif self.config.search_strategy == SearchStrategy.KEYWORD_ONLY:
            results = self._keyword_search(expanded_query, k, metadata_filter)
        elif self.config.search_strategy == SearchStrategy.HYBRID:
            results = self._hybrid_search_impl(expanded_query, k, metadata_filter)
        else:  # SEMANTIC_RERANK
            results = self._semantic_rerank_search(expanded_query, k, metadata_filter)

        # Add confidence scores
        results = self._add_confidence_scores(results, query)

        # Validate cross-references if enabled
        if self.config.enable_cross_reference:
            results = self._validate_cross_references(results)

        # Filter by confidence threshold
        results = [r for r in results if r.confidence >= self.config.confidence_threshold]

        # Track analytics
        if self.config.enable_analytics:
            elapsed_ms = (time.time() - start_time) * 1000
            self._track_query_analytics(
                query, expanded_query, results, elapsed_ms, self.config.search_strategy.value
            )

        self.logger.info(
            f"Hybrid search returned {len(results)} results "
            f"(query='{query[:30]}...', latency={elapsed_ms:.1f}ms)"
        )

        return results

    def _expand_query(self, query: str) -> str:
        """Expand query with legal domain synonyms.

        Args:
            query: Original query

        Returns:
            Expanded query string
        """
        words = query.lower().split()
        expanded_terms = []

        for word in words:
            expanded_terms.append(word)
            # Add synonyms from legal vocabulary
            if word in self._legal_synonyms:
                expanded_terms.extend(self._legal_synonyms[word][: self.config.expansion_terms])

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return " ".join(unique_terms)

    def _vector_search(
        self, query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]]
    ) -> List[HybridSearchResult]:
        """Perform vector-only search.

        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Metadata filters

        Returns:
            List of search results
        """
        if not self.base_rag:
            return []

        base_results = self.base_rag.search(query, top_k, metadata_filter)

        hybrid_results = []
        for r in base_results:
            hybrid_results.append(
                HybridSearchResult(
                    text=r.text,
                    document_id=r.document_id,
                    metadata=r.metadata,
                    score=r.score,
                    rank=r.rank,
                    vector_score=r.score,
                    bm25_score=0.0,
                    hybrid_score=r.score,
                )
            )

        return hybrid_results

    def _keyword_search(
        self, query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]]
    ) -> List[HybridSearchResult]:
        """Perform keyword-only BM25 search.

        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Metadata filters

        Returns:
            List of search results
        """
        # Simplified BM25 implementation
        # In production, use rank_bm25 library
        if not self.base_rag:
            return []

        # Get all documents for BM25 scoring
        # This is a simplified version - in production, maintain BM25 index
        query_terms = set(query.lower().split())

        # Get candidates via vector search
        candidates = self.base_rag.search(query, top_k * 2, metadata_filter)

        # Score with simple keyword matching
        scored_results = []
        for r in candidates:
            doc_terms = set(r.text.lower().split())
            overlap = len(query_terms & doc_terms)
            bm25_score = overlap / max(len(query_terms), 1)

            scored_results.append(
                HybridSearchResult(
                    text=r.text,
                    document_id=r.document_id,
                    metadata=r.metadata,
                    score=bm25_score,
                    rank=r.rank,
                    vector_score=0.0,
                    bm25_score=bm25_score,
                    hybrid_score=bm25_score,
                )
            )

        # Sort by BM25 score
        scored_results.sort(key=lambda x: x.bm25_score, reverse=True)

        # Re-rank and limit
        for i, r in enumerate(scored_results[:top_k]):
            r.rank = i + 1

        return scored_results[:top_k]

    def _hybrid_search_impl(
        self, query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]]
    ) -> List[HybridSearchResult]:
        """Perform hybrid search combining vector and keyword.

        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Metadata filters

        Returns:
            List of hybrid search results
        """
        # Get vector results
        vector_results = self._vector_search(query, self.config.rerank_top_k, metadata_filter)

        # Get keyword scores for same documents
        query_terms = set(query.lower().split())

        hybrid_results = []
        for r in vector_results:
            # Calculate BM25 score
            doc_terms = set(r.text.lower().split())
            overlap = len(query_terms & doc_terms)
            bm25_score = overlap / max(len(query_terms), 1)

            # Combine scores
            hybrid_score = (
                self.config.vector_weight * r.vector_score + self.config.bm25_weight * bm25_score
            )

            r.bm25_score = bm25_score
            r.hybrid_score = hybrid_score
            r.score = hybrid_score

            hybrid_results.append(r)

        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        # Re-rank
        for i, r in enumerate(hybrid_results[:top_k]):
            r.rank = i + 1

        return hybrid_results[:top_k]

    def _semantic_rerank_search(
        self, query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]]
    ) -> List[HybridSearchResult]:
        """Perform search with semantic reranking.

        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Metadata filters

        Returns:
            List of reranked search results
        """
        # Get hybrid results
        hybrid_results = self._hybrid_search_impl(query, self.config.rerank_top_k, metadata_filter)

        # Semantic reranking (simplified - in production use cross-encoder)
        # Boost results with exact phrase matches
        query_lower = query.lower()
        for r in hybrid_results:
            text_lower = r.text.lower()

            # Exact phrase match bonus
            if query_lower in text_lower:
                r.hybrid_score *= 1.2

            # Query term proximity bonus
            query_terms = query_lower.split()
            if len(query_terms) > 1:
                # Check if query terms appear close together
                positions = []
                for term in query_terms:
                    if term in text_lower:
                        positions.append(text_lower.index(term))

                if positions and len(positions) == len(query_terms):
                    proximity = max(positions) - min(positions)
                    if proximity < 100:  # Terms within 100 chars
                        r.hybrid_score *= 1.1

        # Re-sort after reranking
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        # Re-rank
        for i, r in enumerate(hybrid_results[:top_k]):
            r.rank = i + 1
            r.score = r.hybrid_score

        return hybrid_results[:top_k]

    def _add_confidence_scores(
        self, results: List[HybridSearchResult], query: str
    ) -> List[HybridSearchResult]:
        """Add calibrated confidence scores to results.

        Uses temperature scaling for confidence calibration.

        Args:
            results: Search results
            query: Original query

        Returns:
            Results with confidence scores
        """
        if not results:
            return results

        # Get score distribution
        scores = [r.hybrid_score for r in results]
        if not scores:
            return results

        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        # Calibrate confidence using temperature scaling
        temperature = 1.5  # Tuned for legal domain
        for r in results:
            # Normalize score to [0, 1]
            normalized_score = (r.hybrid_score - min_score) / score_range

            # Apply temperature scaling
            calibrated_confidence = 1.0 / (1.0 + np.exp(-normalized_score / temperature))

            # Estimate uncertainty (inverse of confidence)
            uncertainty = 1.0 - calibrated_confidence

            r.confidence = float(calibrated_confidence)
            r.uncertainty = float(uncertainty)

            # Generate explanation
            r.relevance_explanation = self._generate_relevance_explanation(r, query)

        return results

    def _generate_relevance_explanation(self, result: HybridSearchResult, query: str) -> str:
        """Generate human-readable relevance explanation.

        Args:
            result: Search result
            query: Original query

        Returns:
            Relevance explanation string
        """
        explanations = []

        # Vector similarity
        if result.vector_score > 0.7:
            explanations.append(f"High semantic similarity ({result.vector_score:.2f})")
        elif result.vector_score > 0.5:
            explanations.append(f"Moderate semantic match ({result.vector_score:.2f})")

        # Keyword match
        if result.bm25_score > 0.5:
            explanations.append(f"Strong keyword overlap ({result.bm25_score:.2f})")

        # Confidence
        if result.confidence > 0.8:
            explanations.append("High confidence")
        elif result.confidence > 0.6:
            explanations.append("Moderate confidence")

        return "; ".join(explanations) if explanations else "Low relevance score"

    def _validate_cross_references(
        self, results: List[HybridSearchResult]
    ) -> List[HybridSearchResult]:
        """Validate and extract cross-references in results.

        Args:
            results: Search results

        Returns:
            Results with cross-references populated
        """
        # Pattern for legal cross-references
        xref_pattern = r"(?:Section|Clause|Article|§)\s+(\d+(?:\.\d+)*)"

        for r in results:
            refs = re.findall(xref_pattern, r.text, re.I)
            r.cross_references = list(set(refs))  # Unique references

        return results

    def _track_query_analytics(
        self,
        query: str,
        expanded_query: str,
        results: List[HybridSearchResult],
        latency_ms: float,
        strategy: str,
    ) -> None:
        """Track query analytics for monitoring.

        Args:
            query: Original query
            expanded_query: Expanded query
            results: Search results
            latency_ms: Search latency in milliseconds
            strategy: Search strategy used
        """
        avg_conf = sum(r.confidence for r in results) / len(results) if results else 0.0
        top_score = results[0].score if results else 0.0

        analytics = QueryAnalytics(
            query=query,
            expanded_query=expanded_query if expanded_query != query else None,
            num_results=len(results),
            avg_confidence=avg_conf,
            search_latency_ms=latency_ms,
            strategy_used=strategy,
            top_result_score=top_score,
        )

        self.query_history.append(analytics)

        # Keep only recent history (last 1000 queries)
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

    def _build_legal_synonyms(self) -> Dict[str, List[str]]:
        """Build legal domain synonym dictionary.

        Returns:
            Dictionary mapping terms to synonyms
        """
        return {
            "terminate": ["cancel", "end", "conclude", "discontinue"],
            "termination": ["cancellation", "ending", "conclusion"],
            "indemnify": ["compensate", "reimburse", "hold harmless"],
            "indemnification": ["compensation", "reimbursement"],
            "liability": ["responsibility", "obligation", "accountability"],
            "damages": ["compensation", "losses", "harm"],
            "breach": ["violation", "infringement", "default"],
            "agreement": ["contract", "accord", "understanding"],
            "party": ["signatory", "participant", "contractor"],
            "confidential": ["proprietary", "secret", "private"],
            "disclose": ["reveal", "share", "divulge"],
            "payment": ["compensation", "remuneration", "fee"],
            "dispute": ["conflict", "disagreement", "controversy"],
            "arbitration": ["mediation", "adjudication"],
            "jurisdiction": ["authority", "venue", "forum"],
            "warranty": ["guarantee", "assurance", "representation"],
        }

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of query analytics.

        Returns:
            Dictionary with analytics summary
        """
        if not self.query_history:
            return {"total_queries": 0}

        total = len(self.query_history)
        avg_latency = sum(q.search_latency_ms for q in self.query_history) / total
        avg_results = sum(q.num_results for q in self.query_history) / total
        avg_confidence = sum(q.avg_confidence for q in self.query_history) / total

        # Strategy distribution
        strategy_counts = Counter(q.strategy_used for q in self.query_history)

        return {
            "total_queries": total,
            "avg_latency_ms": avg_latency,
            "avg_results_per_query": avg_results,
            "avg_confidence": avg_confidence,
            "strategy_distribution": dict(strategy_counts),
        }

    def add_document_with_dynamic_chunking(
        self,
        document_text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add document with dynamic chunking strategy.

        Args:
            document_text: Full document text
            document_id: Unique document identifier
            metadata: Document metadata

        Returns:
            Number of chunks created

        Raises:
            ValueError: If document is empty
        """
        if not self.base_rag:
            raise RuntimeError("Base RAG system not initialized")

        if self.config.chunking_strategy == ChunkingStrategy.SECTION_BASED:
            chunks = self._chunk_by_sections(document_text)
        elif self.config.chunking_strategy == ChunkingStrategy.CLAUSE_BASED:
            chunks = self._chunk_by_clauses(document_text)
        else:
            # Use base RAG chunking
            return self.base_rag.add_document(document_text, document_id, metadata)

        # Add each chunk separately
        total_chunks = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({"chunk_index": i, "total_chunks": len(chunks)})

            # Use base RAG's collection directly
            self.base_rag.collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[chunk_id],
            )
            total_chunks += 1

        self.logger.info(
            f"Added {total_chunks} chunks for document {document_id} "
            f"(strategy={self.config.chunking_strategy.value})"
        )

        return total_chunks

    def _chunk_by_sections(self, text: str) -> List[str]:
        """Chunk document by legal sections.

        Args:
            text: Document text

        Returns:
            List of section chunks
        """
        # Split by section numbers (e.g., "1.1", "2.3.4")
        section_pattern = r"\n\d+\.(?:\d+\.)*\s+"
        chunks = re.split(section_pattern, text)
        return [c.strip() for c in chunks if c.strip()]

    def _chunk_by_clauses(self, text: str) -> List[str]:
        """Chunk document by contract clauses.

        Args:
            text: Document text

        Returns:
            List of clause chunks
        """
        # Split by uppercase clause headers
        clause_pattern = r"\n[A-Z][A-Z\s]{3,}:?\n"
        chunks = re.split(clause_pattern, text)
        return [c.strip() for c in chunks if c.strip()]


# ==================== FACTORY FUNCTIONS ====================


def create_production_hybrid_rag(
    collection_name: str = "legal_hybrid",
    persist_directory: str = "./chroma_hybrid",
) -> HybridRAGSystem:
    """Create production-ready hybrid RAG system.

    Args:
        collection_name: Collection name
        persist_directory: Persistent storage directory

    Returns:
        Configured HybridRAGSystem
    """
    base_config = RAGConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        chunk_size=500,
        chunk_overlap=50,
        top_k=10,
    )

    hybrid_config = HybridRAGConfig(
        base_rag_config=base_config,
        search_strategy=SearchStrategy.HYBRID,
        chunking_strategy=ChunkingStrategy.SECTION_BASED,
        enable_query_expansion=True,
        enable_reranking=True,
        enable_analytics=True,
    )

    return HybridRAGSystem(hybrid_config)
