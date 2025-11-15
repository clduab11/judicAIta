"""
Unit tests for ClauseExtractionAgent.

Tests cover:
- Configuration validation
- Data model creation
- Input validation
- Clause splitting strategies
- Clause type classification
- Risk assessment
- Obligation extraction
- Key term extraction
- Cross-reference detection
- Complete workflow testing

Target: >80% code coverage
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from typing import Any, Dict

import pytest

from src.agents.base import AgentConfig, AgentStatus
from src.agents.extraction.clause_extractor import (
    Clause,
    ClauseExtractionAgent,
    ClauseExtractionConfig,
    ClauseType,
    Obligation,
    RiskLevel,
)


# ==================== FIXTURES ====================


@pytest.fixture
def clause_config() -> ClauseExtractionConfig:
    """Basic clause extraction configuration."""
    return ClauseExtractionConfig(
        clause_types=[
            ClauseType.TERMINATION,
            ClauseType.INDEMNITY,
            ClauseType.LIABILITY,
        ],
        use_gemma=False,
        assess_risk=True,
        extract_obligations=True,
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    """Basic agent configuration."""
    return AgentConfig(name="clause_extractor")


@pytest.fixture
def sample_contract() -> str:
    """Sample contract text for testing."""
    return """
    1.1 TERMINATION
    Either party may terminate this agreement with 30 days written notice.

    1.2 INDEMNIFICATION
    The Company shall indemnify and hold harmless the Client against all claims.

    1.3 LIABILITY LIMITATION
    In no event shall either party be liable for consequential damages exceeding $100,000.
    """


# ==================== CONFIGURATION TESTS ====================


class TestClauseExtractionConfig:
    """Test ClauseExtractionConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ClauseExtractionConfig()

        assert ClauseType.TERMINATION in config.clause_types
        assert ClauseType.INDEMNITY in config.clause_types
        assert config.use_gemma is True
        assert config.assess_risk is True
        assert config.extract_obligations is True
        assert config.min_clause_length == 50
        assert config.max_risk_score == 0.8
        assert config.include_reasoning is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ClauseExtractionConfig(
            clause_types=[ClauseType.PAYMENT, ClauseType.WARRANTY],
            use_gemma=False,
            assess_risk=False,
            min_clause_length=100,
            max_risk_score=0.5,
        )

        assert config.clause_types == [ClauseType.PAYMENT, ClauseType.WARRANTY]
        assert config.use_gemma is False
        assert config.assess_risk is False
        assert config.min_clause_length == 100
        assert config.max_risk_score == 0.5

    def test_empty_clause_types_validation(self) -> None:
        """Test validation fails with empty clause types."""
        with pytest.raises(ValueError, match="At least one clause type"):
            ClauseExtractionConfig(clause_types=[])


# ==================== DATA MODEL TESTS ====================


class TestDataModels:
    """Test data model classes."""

    def test_obligation_creation(self) -> None:
        """Test creating Obligation object."""
        obligation = Obligation(
            text="Provide monthly reports",
            responsible_party="Client",
            deadline="30 days",
        )

        assert obligation.text == "Provide monthly reports"
        assert obligation.responsible_party == "Client"
        assert obligation.deadline == "30 days"
        assert obligation.recurring is False
        assert obligation.conditions == []

    def test_obligation_defaults(self) -> None:
        """Test Obligation default values."""
        obligation = Obligation(text="Do something")

        assert obligation.text == "Do something"
        assert obligation.responsible_party is None
        assert obligation.deadline is None
        assert obligation.recurring is False
        assert obligation.conditions == []

    def test_clause_creation(self) -> None:
        """Test creating Clause object."""
        clause = Clause(
            text="This is a termination clause",
            clause_type=ClauseType.TERMINATION,
            section_number="1.1",
            risk_score=0.4,
            risk_level=RiskLevel.MEDIUM,
        )

        assert clause.text == "This is a termination clause"
        assert clause.clause_type == ClauseType.TERMINATION
        assert clause.section_number == "1.1"
        assert clause.risk_score == 0.4
        assert clause.risk_level == RiskLevel.MEDIUM

    def test_clause_defaults(self) -> None:
        """Test Clause default values."""
        clause = Clause(
            text="Test clause",
            clause_type=ClauseType.OTHER,
        )

        assert clause.section_number is None
        assert clause.risk_score == 0.0
        assert clause.risk_level == RiskLevel.LOW
        assert clause.obligations == []
        assert clause.parties_mentioned == set()
        assert clause.key_terms == []
        assert clause.cross_references == []
        assert clause.confidence == 1.0
        assert clause.reasoning == ""


# ==================== ENUM TESTS ====================


class TestEnums:
    """Test enum definitions."""

    def test_clause_type_enum(self) -> None:
        """Test ClauseType enum values."""
        assert ClauseType.TERMINATION.value == "termination"
        assert ClauseType.INDEMNITY.value == "indemnity"
        assert ClauseType.LIABILITY.value == "liability"
        assert ClauseType.CONFIDENTIALITY.value == "confidentiality"
        assert ClauseType.PAYMENT.value == "payment"
        assert ClauseType.OTHER.value == "other"

    def test_risk_level_enum(self) -> None:
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


# ==================== AGENT INITIALIZATION TESTS ====================


class TestClauseExtractionAgentInitialization:
    """Test clause extraction agent initialization."""

    def test_basic_initialization(
        self,
        agent_config: AgentConfig,
        clause_config: ClauseExtractionConfig,
    ) -> None:
        """Test agent initializes correctly."""
        agent = ClauseExtractionAgent(agent_config, clause_config)

        assert agent.config == agent_config
        assert agent.clause_config == clause_config
        assert agent.status == AgentStatus.IDLE
        assert agent.model is None  # use_gemma=False

    def test_initialization_with_defaults(self, agent_config: AgentConfig) -> None:
        """Test agent initialization with default config."""
        # Mock model initialization to avoid loading real model
        with patch.object(ClauseExtractionAgent, "_compile_clause_patterns"):
            with patch.object(ClauseExtractionAgent, "_compile_risk_keywords"):
                agent = ClauseExtractionAgent(agent_config)

                assert agent.clause_config is not None
                assert agent.clause_config.use_gemma is True


# ==================== INPUT VALIDATION TESTS ====================


class TestInputValidation:
    """Test input validation."""

    def test_valid_input(self, agent_config: AgentConfig, sample_contract: str) -> None:
        """Test validation with valid input."""
        agent = ClauseExtractionAgent(agent_config)
        input_data = {"text": sample_contract}

        assert agent.validate_input(input_data) is True

    def test_missing_text(self, agent_config: AgentConfig) -> None:
        """Test validation fails without text."""
        agent = ClauseExtractionAgent(agent_config)
        input_data = {}

        assert agent.validate_input(input_data) is False

    def test_invalid_text_type(self, agent_config: AgentConfig) -> None:
        """Test validation fails with non-string text."""
        agent = ClauseExtractionAgent(agent_config)
        input_data = {"text": 123}

        assert agent.validate_input(input_data) is False

    def test_text_too_short(self, agent_config: AgentConfig) -> None:
        """Test validation fails with text too short."""
        agent = ClauseExtractionAgent(agent_config)
        input_data = {"text": "Short"}

        assert agent.validate_input(input_data) is False


# ==================== CLAUSE SPLITTING TESTS ====================


class TestClauseSplitting:
    """Test clause splitting strategies."""

    def test_split_by_section_numbers(self, agent_config: AgentConfig) -> None:
        """Test splitting by section numbers."""
        text = """
        1.1 First clause with enough text to meet minimum length requirement for testing purposes
        2.1 Second clause with enough text to meet minimum length requirement for testing purposes
        3.1 Third clause with enough text to meet minimum length requirement for testing purposes
        """

        agent = ClauseExtractionAgent(agent_config)
        clauses = agent._split_into_clauses(text)

        assert len(clauses) >= 3

    def test_split_by_headers(self, agent_config: AgentConfig) -> None:
        """Test splitting by uppercase headers."""
        text = """
        TERMINATION
        Either party may terminate this agreement with thirty days written notice to the other party.

        INDEMNIFICATION
        The company shall indemnify and hold harmless the client against all claims and damages.
        """

        agent = ClauseExtractionAgent(agent_config)
        clauses = agent._split_into_clauses(text)

        assert len(clauses) >= 1

    def test_split_by_paragraphs(self, agent_config: AgentConfig) -> None:
        """Test splitting by double newlines."""
        text = """
        This is the first paragraph with enough text to meet minimum length requirements for testing purposes.

        This is the second paragraph with enough text to meet minimum length requirements for testing purposes.

        This is the third paragraph with enough text to meet minimum length requirements for testing purposes.
        """

        agent = ClauseExtractionAgent(agent_config)
        clauses = agent._split_into_clauses(text)

        assert len(clauses) >= 3

    def test_min_clause_length_filtering(self, agent_config: AgentConfig) -> None:
        """Test that short clauses are filtered out."""
        config = ClauseExtractionConfig(min_clause_length=100)
        agent = ClauseExtractionAgent(agent_config, config)

        text = "Short.\n\nThis is a much longer clause that should definitely meet the minimum length requirement of one hundred characters for testing purposes."

        clauses = agent._split_into_clauses(text)

        # Only the long clause should remain
        assert all(len(c) >= 100 for c in clauses)


# ==================== CLAUSE CLASSIFICATION TESTS ====================


class TestClauseClassification:
    """Test clause type classification."""

    def test_classify_termination_clause(self, agent_config: AgentConfig) -> None:
        """Test classifying termination clause."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Either party may terminate this agreement with 30 days notice."

        clause_type, confidence = agent._classify_clause_type(clause_text)

        assert clause_type == ClauseType.TERMINATION
        assert confidence > 0.5

    def test_classify_indemnity_clause(self, agent_config: AgentConfig) -> None:
        """Test classifying indemnity clause."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "The Company shall indemnify and hold harmless the Client."

        clause_type, confidence = agent._classify_clause_type(clause_text)

        assert clause_type == ClauseType.INDEMNITY
        assert confidence > 0.5

    def test_classify_liability_clause(self, agent_config: AgentConfig) -> None:
        """Test classifying liability clause."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Limitation of liability shall not exceed $100,000 in damages."

        clause_type, confidence = agent._classify_clause_type(clause_text)

        assert clause_type == ClauseType.LIABILITY
        assert confidence > 0.5

    def test_classify_confidentiality_clause(self, agent_config: AgentConfig) -> None:
        """Test classifying confidentiality clause."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "All confidential information must be kept secret."

        clause_type, confidence = agent._classify_clause_type(clause_text)

        assert clause_type == ClauseType.CONFIDENTIALITY
        assert confidence > 0.5

    def test_classify_payment_clause(self, agent_config: AgentConfig) -> None:
        """Test classifying payment clause."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Payment shall be made within 30 days of invoice date."

        clause_type, confidence = agent._classify_clause_type(clause_text)

        assert clause_type == ClauseType.PAYMENT
        assert confidence > 0.5

    def test_classify_unknown_clause(self, agent_config: AgentConfig) -> None:
        """Test classifying unknown clause defaults to OTHER."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "This is some random text that doesn't match any patterns."

        clause_type, confidence = agent._classify_clause_type(clause_text)

        assert clause_type == ClauseType.OTHER


# ==================== RISK ASSESSMENT TESTS ====================


class TestRiskAssessment:
    """Test risk assessment."""

    def test_assess_indemnity_risk(self, agent_config: AgentConfig) -> None:
        """Test risk assessment for indemnity clause."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "The Company shall indemnify the Client."

        risk_score, reasoning = agent._assess_risk(clause_text, ClauseType.INDEMNITY)

        assert risk_score >= 0.7  # Base risk for indemnity
        assert "indemnity" in reasoning.lower()

    def test_assess_liability_risk(self, agent_config: AgentConfig) -> None:
        """Test risk assessment for liability clause."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Liability is limited to direct damages only."

        risk_score, reasoning = agent._assess_risk(clause_text, ClauseType.LIABILITY)

        assert risk_score >= 0.6  # Base risk for liability
        assert "liability" in reasoning.lower()

    def test_assess_risk_with_high_risk_keywords(self, agent_config: AgentConfig) -> None:
        """Test risk assessment increases with high-risk keywords."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "The Company has unlimited liability without limitation."

        risk_score, reasoning = agent._assess_risk(clause_text, ClauseType.LIABILITY)

        # Should have base risk + keyword risk
        assert risk_score > 0.6
        assert "unlimited" in reasoning.lower() or "keyword" in reasoning.lower()

    def test_risk_score_clamping(self, agent_config: AgentConfig) -> None:
        """Test risk score is clamped to [0, 1]."""
        agent = ClauseExtractionAgent(agent_config)
        # Create clause with many high-risk keywords
        clause_text = (
            "unlimited perpetual irrevocable sole discretion without limitation all damages"
        )

        risk_score, _ = agent._assess_risk(clause_text, ClauseType.INDEMNITY)

        assert 0.0 <= risk_score <= 1.0

    def test_score_to_risk_level_conversion(self, agent_config: AgentConfig) -> None:
        """Test converting risk scores to risk levels."""
        agent = ClauseExtractionAgent(agent_config)

        assert agent._score_to_risk_level(0.9) == RiskLevel.CRITICAL
        assert agent._score_to_risk_level(0.7) == RiskLevel.HIGH
        assert agent._score_to_risk_level(0.5) == RiskLevel.MEDIUM
        assert agent._score_to_risk_level(0.2) == RiskLevel.LOW


# ==================== OBLIGATION EXTRACTION TESTS ====================


class TestObligationExtraction:
    """Test obligation extraction."""

    def test_extract_obligation_with_shall(self, agent_config: AgentConfig) -> None:
        """Test extracting obligation with 'shall'."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "The Client shall provide written notice within 30 days."

        obligations = agent._extract_obligations(clause_text, ["Client", "Company"])

        assert len(obligations) > 0
        assert any("provide written notice" in o.text.lower() for o in obligations)

    def test_extract_obligation_with_must(self, agent_config: AgentConfig) -> None:
        """Test extracting obligation with 'must'."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "The Company must deliver the goods by December 1, 2025."

        obligations = agent._extract_obligations(clause_text, ["Client", "Company"])

        assert len(obligations) > 0
        assert any("deliver" in o.text.lower() for o in obligations)

    def test_extract_obligation_with_will(self, agent_config: AgentConfig) -> None:
        """Test extracting obligation with 'will'."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "The Vendor will maintain insurance coverage."

        obligations = agent._extract_obligations(clause_text, ["Vendor"])

        assert len(obligations) > 0
        assert any("maintain insurance" in o.text.lower() for o in obligations)

    def test_extract_multiple_obligations(self, agent_config: AgentConfig) -> None:
        """Test extracting multiple obligations."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = """
        The Client shall provide documents and must review the report.
        The Company will deliver results.
        """

        obligations = agent._extract_obligations(clause_text, ["Client", "Company"])

        assert len(obligations) >= 3

    def test_match_responsible_party(self, agent_config: AgentConfig) -> None:
        """Test matching responsible party to known parties."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Acme Corp shall provide services."

        obligations = agent._extract_obligations(clause_text, ["Acme Corp", "Beta Inc"])

        assert len(obligations) > 0
        # Check if any obligation has the responsible party
        assert any(o.responsible_party == "Acme Corp" for o in obligations)


# ==================== KEY TERM EXTRACTION TESTS ====================


class TestKeyTermExtraction:
    """Test key term extraction."""

    def test_extract_monetary_amounts(self, agent_config: AgentConfig) -> None:
        """Test extracting monetary amounts."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Payment of $50,000.00 is due upon completion."

        key_terms = agent._extract_key_terms(clause_text, ClauseType.PAYMENT)

        assert any("$50,000.00" in term for term in key_terms)

    def test_extract_time_periods(self, agent_config: AgentConfig) -> None:
        """Test extracting time periods."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Notice must be given within 30 days and 2 months."

        key_terms = agent._extract_key_terms(clause_text, ClauseType.TERMINATION)

        assert any("30 days" in term.lower() for term in key_terms)
        assert any("2 months" in term.lower() for term in key_terms)

    def test_extract_dates(self, agent_config: AgentConfig) -> None:
        """Test extracting dates."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Effective date is January 15, 2025 and ends December 31, 2026."

        key_terms = agent._extract_key_terms(clause_text, ClauseType.PAYMENT)

        assert any("January 15, 2025" in term for term in key_terms)

    def test_key_terms_limit(self, agent_config: AgentConfig) -> None:
        """Test key terms are limited to 10."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = """
        Payment terms: $1,000, $2,000, $3,000, $4,000, $5,000,
        $6,000, $7,000, $8,000, $9,000, $10,000, $11,000, $12,000
        """

        key_terms = agent._extract_key_terms(clause_text, ClauseType.PAYMENT)

        assert len(key_terms) <= 10


# ==================== CROSS-REFERENCE DETECTION TESTS ====================


class TestCrossReferenceDetection:
    """Test cross-reference detection."""

    def test_find_section_references(self, agent_config: AgentConfig) -> None:
        """Test finding section references."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "As described in Section 3.2, the terms shall apply."

        refs = agent._find_cross_references(clause_text)

        assert "3.2" in refs

    def test_find_clause_references(self, agent_config: AgentConfig) -> None:
        """Test finding clause references."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Subject to Clause 4.5, the obligations continue."

        refs = agent._find_cross_references(clause_text)

        assert "4.5" in refs

    def test_find_article_references(self, agent_config: AgentConfig) -> None:
        """Test finding article references."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "Pursuant to Article 7.1.2, indemnification applies."

        refs = agent._find_cross_references(clause_text)

        assert "7.1.2" in refs

    def test_find_multiple_references(self, agent_config: AgentConfig) -> None:
        """Test finding multiple references."""
        agent = ClauseExtractionAgent(agent_config)
        clause_text = "See Section 1.1, Clause 2.3, and Article 5.4."

        refs = agent._find_cross_references(clause_text)

        assert len(refs) >= 3
        assert "1.1" in refs
        assert "2.3" in refs
        assert "5.4" in refs


# ==================== HELPER METHOD TESTS ====================


class TestHelperMethods:
    """Test helper methods."""

    def test_extract_section_number(self, agent_config: AgentConfig) -> None:
        """Test extracting section number."""
        agent = ClauseExtractionAgent(agent_config)

        assert agent._extract_section_number("1.2 Payment Terms") == "1.2"
        assert agent._extract_section_number("3.4.5 Obligations") == "3.4.5"
        assert agent._extract_section_number("No section here") is None

    def test_clause_to_dict(self, agent_config: AgentConfig) -> None:
        """Test converting clause to dictionary."""
        agent = ClauseExtractionAgent(agent_config)

        clause = Clause(
            text="Test clause text",
            clause_type=ClauseType.PAYMENT,
            section_number="1.1",
            risk_score=0.5,
            risk_level=RiskLevel.MEDIUM,
        )

        clause_dict = agent._clause_to_dict(clause)

        assert clause_dict["clause_type"] == "payment"
        assert clause_dict["section_number"] == "1.1"
        assert clause_dict["risk_score"] == 0.5
        assert clause_dict["risk_level"] == "medium"

    def test_group_obligations_by_party(self, agent_config: AgentConfig) -> None:
        """Test grouping obligations by party."""
        agent = ClauseExtractionAgent(agent_config)

        clauses = [
            Clause(
                text="Clause 1",
                clause_type=ClauseType.PAYMENT,
                obligations=[
                    Obligation(text="Pay invoice", responsible_party="Client"),
                    Obligation(text="Provide service", responsible_party="Company"),
                ],
            ),
            Clause(
                text="Clause 2",
                clause_type=ClauseType.TERMINATION,
                obligations=[
                    Obligation(text="Give notice", responsible_party="Client"),
                ],
            ),
        ]

        grouped = agent._group_obligations_by_party(clauses)

        assert "Client" in grouped
        assert "Company" in grouped
        assert len(grouped["Client"]) == 2
        assert len(grouped["Company"]) == 1


# ==================== COMPLETE WORKFLOW TESTS ====================


class TestCompleteWorkflow:
    """Test complete clause extraction workflow."""

    def test_successful_extraction(
        self,
        agent_config: AgentConfig,
        clause_config: ClauseExtractionConfig,
        sample_contract: str,
    ) -> None:
        """Test successful clause extraction."""
        agent = ClauseExtractionAgent(agent_config, clause_config)
        result = agent({"text": sample_contract})

        assert result.status == AgentStatus.COMPLETED
        assert "clauses" in result.output
        assert "stats" in result.output
        assert isinstance(result.output["clauses"], list)

    def test_high_risk_clause_identification(
        self,
        agent_config: AgentConfig,
        sample_contract: str,
    ) -> None:
        """Test identifying high-risk clauses."""
        config = ClauseExtractionConfig(
            clause_types=[ClauseType.INDEMNITY, ClauseType.LIABILITY],
            max_risk_score=0.5,
        )
        agent = ClauseExtractionAgent(agent_config, config)

        contract_with_high_risk = """
        INDEMNIFICATION
        The Company shall have unlimited liability and shall indemnify without limitation
        against all damages and claims in perpetuity.
        """

        result = agent({"text": contract_with_high_risk})

        # Should identify high-risk clauses
        assert result.status == AgentStatus.COMPLETED
        assert "high_risk_clauses" in result.output

    def test_statistics_calculation(
        self,
        agent_config: AgentConfig,
        clause_config: ClauseExtractionConfig,
        sample_contract: str,
    ) -> None:
        """Test statistics are calculated correctly."""
        agent = ClauseExtractionAgent(agent_config, clause_config)
        result = agent({"text": sample_contract})

        assert result.status == AgentStatus.COMPLETED
        stats = result.output["stats"]

        assert "total_clauses" in stats
        assert "clause_types" in stats
        assert "high_risk_count" in stats
        assert "total_obligations" in stats
        assert "avg_risk_score" in stats

    def test_with_known_parties(
        self,
        agent_config: AgentConfig,
        clause_config: ClauseExtractionConfig,
    ) -> None:
        """Test extraction with known parties."""
        contract = """
        OBLIGATIONS
        Acme Corp shall deliver goods within 30 days.
        Beta Inc shall pay within 15 days of delivery.
        """

        agent = ClauseExtractionAgent(agent_config, clause_config)
        result = agent({"text": contract, "parties": ["Acme Corp", "Beta Inc"]})

        assert result.status == AgentStatus.COMPLETED
        assert "obligations_by_party" in result.output

        obligations = result.output["obligations_by_party"]
        # Should have obligations grouped by party
        assert isinstance(obligations, dict)

    def test_empty_contract(
        self, agent_config: AgentConfig, clause_config: ClauseExtractionConfig
    ) -> None:
        """Test handling empty contract."""
        agent = ClauseExtractionAgent(agent_config, clause_config)

        # Should fail validation
        result = agent({"text": ""})
        assert result.status == AgentStatus.FAILED

    def test_trace_collection(
        self,
        agent_config: AgentConfig,
        clause_config: ClauseExtractionConfig,
        sample_contract: str,
    ) -> None:
        """Test trace steps are collected."""
        agent = ClauseExtractionAgent(agent_config, clause_config)
        result = agent({"text": sample_contract})

        assert result.status == AgentStatus.COMPLETED
        assert len(result.trace) > 0

        # Check for expected trace steps
        step_names = [step.get("step") for step in result.trace]
        assert "start_extraction" in step_names
        assert "candidates_identified" in step_names
        assert "clauses_analyzed" in step_names
