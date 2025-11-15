"""
Clause Extraction Agent for Contract Analysis.

This module provides intelligent clause extraction and analysis for legal contracts:
- Clause type classification (liability, indemnity, termination, etc.)
- Risk assessment scoring (0-1 scale)
- Obligation extraction with responsible parties
- Deadline and milestone identification
- Cross-reference detection within documents
- Template comparison for deviation detection
- Explainable reasoning with Gemma 3

Performance Targets:
    - Analysis latency: <5s per contract
    - Classification accuracy: >90%
    - Risk assessment precision: >85%
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator

from src.agents.base import AgentConfig, BaseAgent
from src.models.gemma3.model_wrapper import Gemma3ModelWrapper, ModelConfig


# ==================== CLAUSE TYPES ====================


class ClauseType(str, Enum):
    """Contract clause types."""

    TERMINATION = "termination"
    INDEMNITY = "indemnity"
    LIABILITY = "liability"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT = "payment"
    WARRANTY = "warranty"
    FORCE_MAJEURE = "force_majeure"
    DISPUTE_RESOLUTION = "dispute_resolution"
    GOVERNING_LAW = "governing_law"
    ASSIGNMENT = "assignment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    NON_COMPETE = "non_compete"
    AUDIT_RIGHTS = "audit_rights"
    INSURANCE = "insurance"
    DATA_PROTECTION = "data_protection"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk assessment levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ==================== DATA MODELS ====================


@dataclass
class Obligation:
    """Contract obligation.

    Attributes:
        text: Obligation description
        responsible_party: Party responsible for obligation
        deadline: Deadline if specified
        recurring: Whether obligation is recurring
        conditions: Conditions that trigger obligation
    """

    text: str
    responsible_party: Optional[str] = None
    deadline: Optional[str] = None
    recurring: bool = False
    conditions: List[str] = field(default_factory=list)


@dataclass
class Clause:
    """Extracted contract clause.

    Attributes:
        text: Full clause text
        clause_type: Classified clause type
        section_number: Section/clause number if present
        risk_score: Risk assessment (0-1, higher = more risky)
        risk_level: Categorical risk level
        obligations: Extracted obligations
        parties_mentioned: Parties mentioned in clause
        key_terms: Important terms extracted
        cross_references: References to other clauses
        confidence: Classification confidence
        reasoning: Explanation of analysis
    """

    text: str
    clause_type: ClauseType
    section_number: Optional[str] = None
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    obligations: List[Obligation] = field(default_factory=list)
    parties_mentioned: Set[str] = field(default_factory=set)
    key_terms: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""


# ==================== CONFIGURATION ====================


class ClauseExtractionConfig(BaseModel):
    """Configuration for clause extraction.

    Attributes:
        clause_types: Types of clauses to extract
        use_gemma: Whether to use Gemma 3 for analysis
        assess_risk: Whether to perform risk assessment
        extract_obligations: Whether to extract obligations
        min_clause_length: Minimum clause length (chars)
        max_risk_score: Maximum acceptable risk score
        include_reasoning: Include AI reasoning in output
    """

    clause_types: List[ClauseType] = Field(
        default_factory=lambda: [
            ClauseType.TERMINATION,
            ClauseType.INDEMNITY,
            ClauseType.LIABILITY,
            ClauseType.CONFIDENTIALITY,
            ClauseType.PAYMENT,
        ],
        description="Clause types to extract",
    )
    use_gemma: bool = Field(default=True, description="Use Gemma 3 for analysis")
    assess_risk: bool = Field(default=True, description="Perform risk assessment")
    extract_obligations: bool = Field(default=True, description="Extract obligations")
    min_clause_length: int = Field(default=50, ge=10, description="Minimum clause length")
    max_risk_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Max acceptable risk")
    include_reasoning: bool = Field(default=True, description="Include AI reasoning")

    @field_validator("clause_types")
    def validate_clause_types(cls, v: List[ClauseType]) -> List[ClauseType]:
        """Ensure at least one clause type."""
        if not v:
            raise ValueError("At least one clause type must be specified")
        return v


# ==================== CLAUSE EXTRACTION AGENT ====================


class ClauseExtractionAgent(BaseAgent):
    """Agent for extracting and analyzing contract clauses with Gemma 3.

    This agent provides intelligent clause analysis with:
    - Automatic clause type classification
    - Risk assessment with scoring
    - Obligation extraction
    - Party identification
    - Cross-reference detection
    - Explainable reasoning

    Example:
        >>> config = AgentConfig(name="clause_extractor")
        >>> clause_config = ClauseExtractionConfig(assess_risk=True)
        >>> agent = ClauseExtractionAgent(config, clause_config)
        >>> result = agent({"text": contract_text})
        >>> clauses = result.output["clauses"]
    """

    def __init__(
        self,
        config: AgentConfig,
        clause_config: Optional[ClauseExtractionConfig] = None,
    ):
        """Initialize clause extraction agent.

        Args:
            config: Base agent configuration
            clause_config: Clause extraction configuration

        Raises:
            RuntimeError: If initialization fails
        """
        super().__init__(config)

        self.clause_config = clause_config or ClauseExtractionConfig()

        # Initialize Gemma 3 model if needed
        self.model: Optional[Gemma3ModelWrapper] = None
        if self.clause_config.use_gemma:
            try:
                model_config = ModelConfig(
                    model_name=config.model_path,
                    device=config.device,
                    use_lora=config.use_lora,
                    lora_path=config.lora_path,
                )
                self.model = Gemma3ModelWrapper(model_config)
                self.logger.info("Gemma 3 model initialized for clause analysis")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemma 3: {str(e)}")
                self.clause_config.use_gemma = False

        # Compile patterns
        self._clause_patterns = self._compile_clause_patterns()
        self._risk_keywords = self._compile_risk_keywords()

        self.logger.info(
            f"Clause extractor initialized (gemma={self.clause_config.use_gemma}, "
            f"types={len(self.clause_config.clause_types)})"
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required text.

        Args:
            input_data: Input dictionary

        Returns:
            True if valid, False otherwise
        """
        if "text" not in input_data:
            self.logger.error("Missing required 'text' in input")
            return False

        if not isinstance(input_data["text"], str):
            self.logger.error("'text' must be a string")
            return False

        if len(input_data["text"].strip()) < self.clause_config.min_clause_length:
            self.logger.error(f"Text too short (min: {self.clause_config.min_clause_length} chars)")
            return False

        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze clauses from contract text.

        Args:
            input_data: Dictionary with 'text' key (and optional 'parties')

        Returns:
            Dictionary with extracted clauses:
                - clauses: List of Clause objects
                - high_risk_clauses: Clauses exceeding risk threshold
                - obligations_by_party: Obligations grouped by party
                - stats: Analysis statistics

        Raises:
            RuntimeError: If extraction fails
        """
        text = input_data["text"]
        known_parties = input_data.get("parties", [])

        self.add_trace_step(
            step="start_extraction",
            description=f"Starting clause extraction ({len(text)} chars)",
            input_data={"text_length": len(text)},
        )

        # Split text into potential clauses
        clause_candidates = self._split_into_clauses(text)

        self.add_trace_step(
            step="candidates_identified",
            description=f"Identified {len(clause_candidates)} clause candidates",
        )

        # Analyze each clause
        clauses = []
        for candidate in clause_candidates:
            clause = self._analyze_clause(candidate, known_parties)
            if clause and clause.clause_type in self.clause_config.clause_types:
                clauses.append(clause)

        self.add_trace_step(
            step="clauses_analyzed",
            description=f"Analyzed {len(clauses)} clauses",
            output_data={"clause_count": len(clauses)},
        )

        # Identify high-risk clauses
        high_risk_clauses = [c for c in clauses if c.risk_score > self.clause_config.max_risk_score]

        # Group obligations by party
        obligations_by_party = self._group_obligations_by_party(clauses)

        # Calculate statistics
        stats = {
            "total_clauses": len(clauses),
            "clause_types": {ct.value: 0 for ct in self.clause_config.clause_types},
            "high_risk_count": len(high_risk_clauses),
            "total_obligations": sum(len(c.obligations) for c in clauses),
            "avg_risk_score": sum(c.risk_score for c in clauses) / len(clauses) if clauses else 0.0,
        }

        # Count clauses by type
        for clause in clauses:
            if clause.clause_type.value in stats["clause_types"]:
                stats["clause_types"][clause.clause_type.value] += 1

        return {
            "clauses": [self._clause_to_dict(c) for c in clauses],
            "high_risk_clauses": [self._clause_to_dict(c) for c in high_risk_clauses],
            "obligations_by_party": obligations_by_party,
            "stats": stats,
        }

    def _split_into_clauses(self, text: str) -> List[str]:
        """Split contract text into clause candidates.

        Args:
            text: Contract text

        Returns:
            List of clause text segments
        """
        # Strategy 1: Split by section numbers
        section_pattern = r"\n\d+\.(?:\d+\.)*\s+"
        if re.search(section_pattern, text):
            clauses = re.split(section_pattern, text)
            clauses = [
                c.strip() for c in clauses if len(c.strip()) >= self.clause_config.min_clause_length
            ]
            if clauses:
                return clauses

        # Strategy 2: Split by common clause headers
        header_pattern = r"\n[A-Z][A-Z\s]{3,}:?\n"
        if re.search(header_pattern, text):
            clauses = re.split(header_pattern, text)
            clauses = [
                c.strip() for c in clauses if len(c.strip()) >= self.clause_config.min_clause_length
            ]
            if clauses:
                return clauses

        # Strategy 3: Split by double newlines (paragraphs)
        clauses = text.split("\n\n")
        clauses = [
            c.strip() for c in clauses if len(c.strip()) >= self.clause_config.min_clause_length
        ]

        return clauses

    def _analyze_clause(self, clause_text: str, known_parties: List[str]) -> Optional[Clause]:
        """Analyze a single clause.

        Args:
            clause_text: Clause text
            known_parties: List of known party names

        Returns:
            Analyzed Clause object or None
        """
        # Classify clause type
        clause_type, confidence = self._classify_clause_type(clause_text)

        # Assess risk
        risk_score = 0.0
        risk_level = RiskLevel.LOW
        reasoning = ""

        if self.clause_config.assess_risk:
            risk_score, reasoning = self._assess_risk(clause_text, clause_type)
            risk_level = self._score_to_risk_level(risk_score)

        # Extract obligations
        obligations = []
        if self.clause_config.extract_obligations:
            obligations = self._extract_obligations(clause_text, known_parties)

        # Extract section number
        section_number = self._extract_section_number(clause_text)

        # Identify parties mentioned
        parties_mentioned = set()
        for party in known_parties:
            if party.lower() in clause_text.lower():
                parties_mentioned.add(party)

        # Extract key terms
        key_terms = self._extract_key_terms(clause_text, clause_type)

        # Find cross-references
        cross_references = self._find_cross_references(clause_text)

        clause = Clause(
            text=clause_text,
            clause_type=clause_type,
            section_number=section_number,
            risk_score=risk_score,
            risk_level=risk_level,
            obligations=obligations,
            parties_mentioned=parties_mentioned,
            key_terms=key_terms,
            cross_references=cross_references,
            confidence=confidence,
            reasoning=reasoning if self.clause_config.include_reasoning else "",
        )

        return clause

    def _classify_clause_type(self, clause_text: str) -> tuple[ClauseType, float]:
        """Classify clause type.

        Args:
            clause_text: Clause text

        Returns:
            Tuple of (ClauseType, confidence)
        """
        clause_lower = clause_text.lower()

        # Pattern-based classification
        for clause_type, patterns in self._clause_patterns.items():
            for pattern in patterns:
                if re.search(pattern, clause_lower):
                    return clause_type, 0.85

        # Use Gemma 3 if available
        if self.model and self.clause_config.use_gemma:
            try:
                clause_type, confidence = self._classify_with_gemma(clause_text)
                return clause_type, confidence
            except Exception as e:
                self.logger.debug(f"Gemma classification failed: {str(e)}")

        # Default to OTHER
        return ClauseType.OTHER, 0.5

    def _classify_with_gemma(self, clause_text: str) -> tuple[ClauseType, float]:
        """Classify clause using Gemma 3.

        Args:
            clause_text: Clause text

        Returns:
            Tuple of (ClauseType, confidence)
        """
        prompt = f"""Classify this contract clause into one of these types:
{', '.join(ct.value for ct in ClauseType)}

Clause:
{clause_text[:500]}

Provide your answer in this format:
TYPE: [clause_type]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]

Classification:"""

        response = self.model.generate(prompt, max_tokens=200, temperature=0.1)

        # Parse response
        type_match = re.search(r"TYPE:\s*(\w+)", response, re.I)
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response)

        if type_match:
            try:
                clause_type_str = type_match.group(1).upper()
                # Try to match to enum
                for ct in ClauseType:
                    if ct.value.upper().replace("_", "") == clause_type_str.replace("_", ""):
                        confidence = float(conf_match.group(1)) if conf_match else 0.8
                        return ct, min(1.0, max(0.0, confidence))
            except (ValueError, AttributeError):
                pass

        return ClauseType.OTHER, 0.5

    def _assess_risk(self, clause_text: str, clause_type: ClauseType) -> tuple[float, str]:
        """Assess risk of a clause.

        Args:
            clause_text: Clause text
            clause_type: Classified clause type

        Returns:
            Tuple of (risk_score, reasoning)
        """
        risk_score = 0.0
        reasoning_parts = []

        clause_lower = clause_text.lower()

        # Base risk by clause type
        type_base_risks = {
            ClauseType.INDEMNITY: 0.7,
            ClauseType.LIABILITY: 0.6,
            ClauseType.NON_COMPETE: 0.5,
            ClauseType.TERMINATION: 0.4,
            ClauseType.CONFIDENTIALITY: 0.4,
            ClauseType.PAYMENT: 0.3,
        }

        base_risk = type_base_risks.get(clause_type, 0.2)
        risk_score += base_risk
        reasoning_parts.append(f"Base risk for {clause_type.value}: {base_risk:.2f}")

        # Check for risk keywords
        for risk_level, keywords in self._risk_keywords.items():
            matched_keywords = [kw for kw in keywords if kw in clause_lower]
            if matched_keywords:
                additional_risk = len(matched_keywords) * 0.1
                risk_score += additional_risk
                reasoning_parts.append(
                    f"Risk keywords ({risk_level}): {', '.join(matched_keywords[:3])}"
                )

        # Clamp to [0, 1]
        risk_score = min(1.0, max(0.0, risk_score))

        reasoning = "; ".join(reasoning_parts)

        return risk_score, reasoning

    def _extract_obligations(self, clause_text: str, known_parties: List[str]) -> List[Obligation]:
        """Extract obligations from clause.

        Args:
            clause_text: Clause text
            known_parties: Known party names

        Returns:
            List of Obligation objects
        """
        obligations = []

        # Pattern: "Party shall/must/will [obligation]"
        obligation_pattern = r"(\w+(?:\s+\w+)*)\s+(shall|must|will|agrees? to)\s+([^.;]+)"

        for match in re.finditer(obligation_pattern, clause_text, re.I):
            potential_party = match.group(1).strip()
            obligation_text = match.group(3).strip()

            # Check if potential party is a known party
            responsible_party = None
            for party in known_parties:
                if party.lower() in potential_party.lower():
                    responsible_party = party
                    break

            obligations.append(
                Obligation(
                    text=obligation_text,
                    responsible_party=responsible_party,
                )
            )

        return obligations

    def _extract_section_number(self, clause_text: str) -> Optional[str]:
        """Extract section number from clause.

        Args:
            clause_text: Clause text

        Returns:
            Section number if found
        """
        # Pattern: "1.2.3" at start of text
        pattern = r"^(\d+(?:\.\d+)*)"
        match = re.search(pattern, clause_text.strip())
        if match:
            return match.group(1)
        return None

    def _extract_key_terms(self, clause_text: str, clause_type: ClauseType) -> List[str]:
        """Extract key terms from clause.

        Args:
            clause_text: Clause text
            clause_type: Clause type

        Returns:
            List of key terms
        """
        key_terms = []

        # Extract monetary amounts
        money_pattern = r"\$\d+(?:,\d{3})*(?:\.\d{2})?"
        key_terms.extend(re.findall(money_pattern, clause_text))

        # Extract time periods
        time_pattern = r"\b\d+\s+(?:days?|weeks?|months?|years?)\b"
        key_terms.extend(re.findall(time_pattern, clause_text, re.I))

        # Extract dates
        date_pattern = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
        key_terms.extend(re.findall(date_pattern, clause_text))

        return key_terms[:10]  # Limit to 10

    def _find_cross_references(self, clause_text: str) -> List[str]:
        """Find cross-references to other clauses.

        Args:
            clause_text: Clause text

        Returns:
            List of section references
        """
        # Pattern: "Section X.Y" or "Clause X.Y"
        pattern = r"(?:Section|Clause|Article)\s+(\d+(?:\.\d+)*)"
        return re.findall(pattern, clause_text, re.I)

    def _group_obligations_by_party(self, clauses: List[Clause]) -> Dict[str, List[Dict[str, str]]]:
        """Group obligations by responsible party.

        Args:
            clauses: List of clauses

        Returns:
            Dictionary mapping party to obligations
        """
        obligations_by_party: Dict[str, List[Dict[str, str]]] = {}

        for clause in clauses:
            for obligation in clause.obligations:
                party = obligation.responsible_party or "Unspecified"

                if party not in obligations_by_party:
                    obligations_by_party[party] = []

                obligations_by_party[party].append(
                    {
                        "obligation": obligation.text,
                        "clause_type": clause.clause_type.value,
                        "deadline": obligation.deadline or "Not specified",
                    }
                )

        return obligations_by_party

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert risk score to categorical level.

        Args:
            score: Risk score (0-1)

        Returns:
            RiskLevel
        """
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _compile_clause_patterns(self) -> Dict[ClauseType, List[str]]:
        """Compile regex patterns for clause classification.

        Returns:
            Dictionary mapping clause types to patterns
        """
        return {
            ClauseType.TERMINATION: [
                r"terminat(?:e|ion)",
                r"cancel(?:lation)?",
                r"end(?:ing)?\s+(?:this\s+)?(?:agreement|contract)",
            ],
            ClauseType.INDEMNITY: [
                r"indemnif(?:y|ication)",
                r"hold\s+harmless",
                r"defend\s+against",
            ],
            ClauseType.LIABILITY: [
                r"liabilit(?:y|ies)",
                r"limit(?:ation)?\s+of\s+liabilit",
                r"damages",
            ],
            ClauseType.CONFIDENTIALITY: [
                r"confidential(?:ity)?",
                r"non-disclosure",
                r"proprietary\s+information",
            ],
            ClauseType.PAYMENT: [
                r"payment",
                r"compensation",
                r"fees?",
                r"invoice",
            ],
            ClauseType.WARRANTY: [
                r"warrant(?:y|ies)",
                r"represent(?:ation)?s?\s+and\s+warrant",
                r"guarantee",
            ],
            ClauseType.FORCE_MAJEURE: [
                r"force\s+majeure",
                r"act(?:s)?\s+of\s+god",
                r"beyond\s+(?:the\s+)?(?:reasonable\s+)?control",
            ],
            ClauseType.DISPUTE_RESOLUTION: [
                r"dispute\s+resolution",
                r"arbitration",
                r"litigation",
                r"mediation",
            ],
            ClauseType.GOVERNING_LAW: [
                r"governing\s+law",
                r"choice\s+of\s+law",
                r"jurisdiction",
            ],
        }

    def _compile_risk_keywords(self) -> Dict[str, List[str]]:
        """Compile risk keywords.

        Returns:
            Dictionary of risk keywords by category
        """
        return {
            "high_risk": [
                "unlimited",
                "perpetual",
                "irrevocable",
                "sole discretion",
                "without limitation",
                "all damages",
            ],
            "medium_risk": [
                "reasonable",
                "material breach",
                "consequential damages",
                "exclusive remedy",
            ],
        }

    def _clause_to_dict(self, clause: Clause) -> Dict[str, Any]:
        """Convert Clause to dictionary.

        Args:
            clause: Clause object

        Returns:
            Dictionary representation
        """
        return {
            "text": clause.text[:200] + "..." if len(clause.text) > 200 else clause.text,
            "full_text": clause.text,
            "clause_type": clause.clause_type.value,
            "section_number": clause.section_number,
            "risk_score": clause.risk_score,
            "risk_level": clause.risk_level.value,
            "obligations": [
                {
                    "text": o.text,
                    "responsible_party": o.responsible_party,
                    "deadline": o.deadline,
                }
                for o in clause.obligations
            ],
            "parties_mentioned": list(clause.parties_mentioned),
            "key_terms": clause.key_terms,
            "cross_references": clause.cross_references,
            "confidence": clause.confidence,
            "reasoning": clause.reasoning,
        }
