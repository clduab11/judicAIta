"""
Entity Extraction Agent for Legal Document Processing.

This module provides entity extraction capabilities specialized for legal documents:
- Named Entity Recognition (NER) with legal context
- Party identification (plaintiffs, defendants, witnesses)
- Organization extraction (companies, courts, agencies)
- Legal professional detection (lawyers, judges)
- Relationship mapping between entities
- Cross-document entity resolution
- Confidence scoring with reasoning traces

Uses Gemma 3 1B for intelligent entity extraction with explainable reasoning.

Performance Targets:
    - Extraction latency: <2s per document
    - Accuracy: >95% for standard legal entities
    - F1 Score: >0.85 on legal NER benchmarks
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator

from src.agents.base import AgentConfig, BaseAgent
from src.models.gemma3.model_wrapper import Gemma3ModelWrapper, ModelConfig


# ==================== ENTITY TYPES ====================


class EntityType(str, Enum):
    """Legal entity types."""

    PARTY = "PARTY"  # Plaintiffs, defendants, parties to a contract
    JUDGE = "JUDGE"  # Judges, magistrates
    ATTORNEY = "ATTORNEY"  # Lawyers, counsels
    COURT = "COURT"  # Courts, tribunals
    ORGANIZATION = "ORGANIZATION"  # Companies, agencies, institutions
    STATUTE = "STATUTE"  # Laws, regulations, statutes
    CASE_CITATION = "CASE_CITATION"  # Legal case citations
    DATE = "DATE"  # Important dates
    LOCATION = "LOCATION"  # Jurisdictions, venues
    MONETARY = "MONETARY"  # Amounts, damages
    PERSON = "PERSON"  # General persons not in other categories


class RelationType(str, Enum):
    """Types of relationships between entities."""

    REPRESENTS = "represents"  # Attorney represents party
    EMPLOYED_BY = "employed_by"  # Person employed by organization
    OPPOSED_TO = "opposed_to"  # Parties in opposition
    AFFILIATED_WITH = "affiliated_with"  # Affiliation relationship
    LOCATED_IN = "located_in"  # Location relationship
    CITES = "cites"  # Citation relationship


# ==================== DATA MODELS ====================


@dataclass
class Entity:
    """Extracted legal entity.

    Attributes:
        text: Entity text as it appears in document
        type: Entity type classification
        start_char: Starting character position
        end_char: Ending character position
        confidence: Confidence score (0-1)
        context: Surrounding context (optional)
        metadata: Additional entity metadata
        aliases: Known aliases for this entity
    """

    text: str
    type: EntityType
    start_char: int
    end_char: int
    confidence: float = 1.0
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    aliases: Set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        """Make entity hashable for set operations."""
        return hash((self.text.lower(), self.type))

    def __eq__(self, other: object) -> bool:
        """Entity equality based on text and type."""
        if not isinstance(other, Entity):
            return False
        return self.text.lower() == other.text.lower() and self.type == other.type


@dataclass
class EntityRelationship:
    """Relationship between two entities.

    Attributes:
        source: Source entity
        target: Target entity
        relation_type: Type of relationship
        confidence: Confidence score (0-1)
        evidence: Text evidence for relationship
    """

    source: Entity
    target: Entity
    relation_type: RelationType
    confidence: float = 1.0
    evidence: Optional[str] = None


@dataclass
class ResolvedEntity:
    """Entity resolved across multiple mentions/documents.

    Attributes:
        canonical_name: Canonical form of entity name
        entity_type: Entity type
        mentions: All mentions of this entity
        confidence: Overall confidence score
        documents: Document IDs where entity appears
    """

    canonical_name: str
    entity_type: EntityType
    mentions: List[Entity] = field(default_factory=list)
    confidence: float = 1.0
    documents: Set[str] = field(default_factory=set)


# ==================== CONFIGURATION ====================


class EntityExtractionConfig(BaseModel):
    """Configuration for entity extraction.

    Attributes:
        entity_types: Types of entities to extract
        use_gemma: Whether to use Gemma 3 for extraction
        use_patterns: Whether to use regex patterns
        min_confidence: Minimum confidence threshold
        extract_relationships: Whether to extract relationships
        resolve_entities: Whether to resolve entity references
        context_window: Context characters around entity
    """

    entity_types: List[EntityType] = Field(
        default_factory=lambda: [
            EntityType.PARTY,
            EntityType.JUDGE,
            EntityType.ATTORNEY,
            EntityType.COURT,
            EntityType.ORGANIZATION,
        ],
        description="Entity types to extract",
    )
    use_gemma: bool = Field(default=True, description="Use Gemma 3 for extraction")
    use_patterns: bool = Field(default=True, description="Use regex patterns")
    min_confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    extract_relationships: bool = Field(default=True, description="Extract relationships")
    resolve_entities: bool = Field(default=True, description="Resolve entity references")
    context_window: int = Field(
        default=100, ge=0, le=500, description="Context characters around entity"
    )

    @validator("entity_types")
    def validate_entity_types(cls, v: List[EntityType]) -> List[EntityType]:
        """Ensure at least one entity type."""
        if not v:
            raise ValueError("At least one entity type must be specified")
        return v


# ==================== ENTITY EXTRACTION AGENT ====================


class EntityExtractionAgent(BaseAgent):
    """Agent for extracting legal entities with Gemma 3 reasoning.

    This agent provides intelligent entity extraction with:
    - Multi-strategy extraction (Gemma 3 + patterns)
    - Legal-specific entity types
    - Confidence scoring
    - Relationship extraction
    - Cross-document entity resolution
    - Explainable reasoning traces

    Example:
        >>> config = AgentConfig(name="entity_extractor")
        >>> entity_config = EntityExtractionConfig(use_gemma=True)
        >>> agent = EntityExtractionAgent(config, entity_config)
        >>> result = agent({"text": "John Doe v. Acme Corp..."})
        >>> entities = result.output["entities"]
    """

    def __init__(self, config: AgentConfig, entity_config: Optional[EntityExtractionConfig] = None):
        """Initialize entity extraction agent.

        Args:
            config: Base agent configuration
            entity_config: Entity extraction configuration

        Raises:
            RuntimeError: If initialization fails
        """
        super().__init__(config)

        self.entity_config = entity_config or EntityExtractionConfig()

        # Initialize Gemma 3 model if needed
        self.model: Optional[Gemma3ModelWrapper] = None
        if self.entity_config.use_gemma:
            try:
                model_config = ModelConfig(
                    model_name=config.model_path,
                    device=config.device,
                    use_lora=config.use_lora,
                    lora_path=config.lora_path,
                )
                self.model = Gemma3ModelWrapper(model_config)
                self.logger.info("Gemma 3 model initialized for entity extraction")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemma 3: {str(e)}")
                self.logger.warning("Falling back to pattern-based extraction only")
                self.entity_config.use_gemma = False

        # Compile regex patterns
        self._patterns = self._compile_patterns()

        # Entity resolution cache
        self._entity_cache: Dict[str, ResolvedEntity] = {}

        self.logger.info(
            f"Entity extractor initialized (gemma={self.entity_config.use_gemma}, "
            f"types={len(self.entity_config.entity_types)})"
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

        if len(input_data["text"].strip()) == 0:
            self.logger.error("'text' cannot be empty")
            return False

        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from text.

        Args:
            input_data: Dictionary with 'text' key

        Returns:
            Dictionary with extracted entities:
                - entities: List of Entity objects
                - relationships: List of EntityRelationship objects (if enabled)
                - resolved_entities: List of ResolvedEntity objects (if enabled)
                - stats: Extraction statistics

        Raises:
            RuntimeError: If extraction fails
        """
        text = input_data["text"]
        document_id = input_data.get("document_id", "default")

        self.add_trace_step(
            step="start_extraction",
            description=f"Starting entity extraction ({len(text)} chars)",
            input_data={
                "text_length": len(text),
                "entity_types": len(self.entity_config.entity_types),
            },
        )

        # Extract entities using multiple strategies
        entities = self._extract_entities(text)

        self.add_trace_step(
            step="entities_extracted",
            description=f"Extracted {len(entities)} entities",
            output_data={"entity_count": len(entities)},
        )

        # Extract relationships if enabled
        relationships = []
        if self.entity_config.extract_relationships and len(entities) > 1:
            relationships = self._extract_relationships(text, entities)
            self.add_trace_step(
                step="relationships_extracted",
                description=f"Extracted {len(relationships)} relationships",
                output_data={"relationship_count": len(relationships)},
            )

        # Resolve entities if enabled
        resolved_entities = []
        if self.entity_config.resolve_entities:
            resolved_entities = self._resolve_entities(entities, document_id)
            self.add_trace_step(
                step="entities_resolved",
                description=f"Resolved {len(resolved_entities)} unique entities",
                output_data={"resolved_count": len(resolved_entities)},
            )

        # Calculate statistics
        stats = {
            "total_entities": len(entities),
            "entity_types": {et.value: 0 for et in self.entity_config.entity_types},
            "total_relationships": len(relationships),
            "resolved_entities": len(resolved_entities),
            "avg_confidence": (
                sum(e.confidence for e in entities) / len(entities) if entities else 0.0
            ),
        }

        # Count entities by type
        for entity in entities:
            if entity.type.value in stats["entity_types"]:
                stats["entity_types"][entity.type.value] += 1

        return {
            "entities": [self._entity_to_dict(e) for e in entities],
            "relationships": [self._relationship_to_dict(r) for r in relationships],
            "resolved_entities": [self._resolved_entity_to_dict(re) for re in resolved_entities],
            "stats": stats,
        }

    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using multiple strategies.

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        entities = []

        # Strategy 1: Pattern-based extraction
        if self.entity_config.use_patterns:
            pattern_entities = self._extract_with_patterns(text)
            entities.extend(pattern_entities)
            self.add_trace_step(
                step="pattern_extraction",
                description=f"Pattern-based extraction: {len(pattern_entities)} entities",
            )

        # Strategy 2: Gemma 3-based extraction
        if self.entity_config.use_gemma and self.model:
            gemma_entities = self._extract_with_gemma(text)
            entities.extend(gemma_entities)
            self.add_trace_step(
                step="gemma_extraction",
                description=f"Gemma 3 extraction: {len(gemma_entities)} entities",
            )

        # Merge and deduplicate entities
        entities = self._merge_entities(entities)

        # Filter by confidence threshold
        entities = [e for e in entities if e.confidence >= self.entity_config.min_confidence]

        return entities

    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns.

        Args:
            text: Input text

        Returns:
            List of entities
        """
        entities = []

        for entity_type, patterns in self._patterns.items():
            if entity_type not in self.entity_config.entity_types:
                continue

            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = Entity(
                        text=match.group(),
                        type=entity_type,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.8,  # Pattern-based confidence
                        context=self._get_context(text, match.start(), match.end()),
                    )
                    entities.append(entity)

        return entities

    def _extract_with_gemma(self, text: str) -> List[Entity]:
        """Extract entities using Gemma 3.

        Args:
            text: Input text

        Returns:
            List of entities
        """
        if not self.model:
            return []

        # Build prompt for entity extraction
        entity_types_str = ", ".join(et.value for et in self.entity_config.entity_types)

        prompt = f"""Extract legal entities from the following text. Identify these entity types: {entity_types_str}

Text:
{text[:2000]}  # Limit to 2000 chars for efficiency

For each entity, provide:
- Entity text
- Entity type
- Confidence (0-1)

Format your response as:
[ENTITY: text | TYPE: type | CONFIDENCE: score]

Entities:"""

        try:
            # Generate response from Gemma 3
            response = self.model.generate(prompt, max_tokens=512, temperature=0.1)

            # Parse response
            entities = self._parse_gemma_response(response, text)

            return entities

        except Exception as e:
            self.logger.error(f"Gemma extraction failed: {str(e)}")
            return []

    def _parse_gemma_response(self, response: str, original_text: str) -> List[Entity]:
        """Parse Gemma 3 response into entities.

        Args:
            response: Gemma 3 response text
            original_text: Original input text

        Returns:
            List of entities
        """
        entities = []

        # Pattern: [ENTITY: text | TYPE: type | CONFIDENCE: score]
        pattern = r"\[ENTITY:\s*(.+?)\s*\|\s*TYPE:\s*(\w+)\s*\|\s*CONFIDENCE:\s*([\d.]+)\]"

        for match in re.finditer(pattern, response):
            entity_text = match.group(1).strip()
            entity_type_str = match.group(2).strip().upper()
            confidence_str = match.group(3).strip()

            try:
                # Validate entity type
                entity_type = EntityType[entity_type_str]
                if entity_type not in self.entity_config.entity_types:
                    continue

                # Parse confidence
                confidence = float(confidence_str)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

                # Find entity position in original text using re.finditer for duplicate handling
                # Use context to disambiguate if multiple matches exist
                matches = list(
                    re.finditer(re.escape(entity_text.lower()), original_text.lower())
                )
                if not matches:
                    continue  # Entity not found in original text

                # If multiple matches, try to use context for disambiguation
                # For now, use the first match (can be improved with context analysis)
                match = matches[0]
                start_char = match.start()
                end_char = start_char + len(entity_text)

                entity = Entity(
                    text=entity_text,
                    type=entity_type,
                    start_char=start_char,
                    end_char=end_char,
                    confidence=confidence,
                    context=self._get_context(original_text, start_char, end_char),
                )

                entities.append(entity)

            except (KeyError, ValueError) as e:
                self.logger.debug(f"Failed to parse entity: {str(e)}")
                continue

        return entities

    def _extract_relationships(self, text: str, entities: List[Entity]) -> List[EntityRelationship]:
        """Extract relationships between entities.

        Args:
            text: Input text
            entities: Extracted entities

        Returns:
            List of relationships
        """
        relationships = []

        # Simple heuristic-based relationship extraction
        # In production, this would use more sophisticated NLP

        for i, source in enumerate(entities):
            for target in entities[i + 1 :]:
                # Skip same entity
                if source == target:
                    continue

                # Check for representative relationship (Attorney represents Party)
                if source.type == EntityType.ATTORNEY and target.type == EntityType.PARTY:
                    if self._check_represents_pattern(text, source, target):
                        relationships.append(
                            EntityRelationship(
                                source=source,
                                target=target,
                                relation_type=RelationType.REPRESENTS,
                                confidence=0.8,
                            )
                        )

                # Check for opposition relationship (Party v. Party)
                if source.type == EntityType.PARTY and target.type == EntityType.PARTY:
                    if self._check_opposition_pattern(text, source, target):
                        relationships.append(
                            EntityRelationship(
                                source=source,
                                target=target,
                                relation_type=RelationType.OPPOSED_TO,
                                confidence=0.9,
                            )
                        )

        return relationships

    def _resolve_entities(self, entities: List[Entity], document_id: str) -> List[ResolvedEntity]:
        """Resolve entity references.

        Args:
            entities: Extracted entities
            document_id: Document identifier

        Returns:
            List of resolved entities
        """
        resolved = []

        for entity in entities:
            # Create canonical key
            canonical_key = f"{entity.type.value}:{entity.text.lower()}"

            if canonical_key in self._entity_cache:
                # Update existing resolved entity
                resolved_entity = self._entity_cache[canonical_key]
                resolved_entity.mentions.append(entity)
                resolved_entity.documents.add(document_id)
                # Update confidence (average)
                all_confidences = [e.confidence for e in resolved_entity.mentions]
                resolved_entity.confidence = sum(all_confidences) / len(all_confidences)
            else:
                # Create new resolved entity
                resolved_entity = ResolvedEntity(
                    canonical_name=entity.text,
                    entity_type=entity.type,
                    mentions=[entity],
                    confidence=entity.confidence,
                    documents={document_id},
                )
                self._entity_cache[canonical_key] = resolved_entity

            resolved.append(resolved_entity)

        return list(set(resolved))  # Deduplicate

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities.

        Args:
            entities: List of entities

        Returns:
            Merged list
        """
        if not entities:
            return []

        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: e.start_char)

        merged = [sorted_entities[0]]

        for current in sorted_entities[1:]:
            last = merged[-1]

            # Check for overlap
            if current.start_char < last.end_char:
                # Overlapping - keep entity with higher confidence
                if current.confidence > last.confidence:
                    merged[-1] = current
            else:
                # No overlap - add new entity
                merged.append(current)

        return merged

    def _get_context(self, text: str, start: int, end: int) -> str:
        """Get context around entity.

        Args:
            text: Full text
            start: Entity start position
            end: Entity end position

        Returns:
            Context string
        """
        window = self.entity_config.context_window
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _check_represents_pattern(self, text: str, attorney: Entity, party: Entity) -> bool:
        """Check if attorney represents party.

        Args:
            text: Full text
            attorney: Attorney entity
            party: Party entity

        Returns:
            True if representation pattern found
        """
        # Simple pattern check
        # In production, use more sophisticated NLP
        representation_keywords = ["represents", "representing", "counsel for", "attorney for"]

        # Get text between entities
        start = min(attorney.start_char, party.start_char)
        end = max(attorney.end_char, party.end_char)
        between_text = text[start:end].lower()

        return any(keyword in between_text for keyword in representation_keywords)

    def _check_opposition_pattern(self, text: str, party1: Entity, party2: Entity) -> bool:
        """Check if parties are in opposition.

        Args:
            text: Full text
            party1: First party
            party2: Second party

        Returns:
            True if opposition pattern found
        """
        # Check for "v." or "vs." between parties
        start = min(party1.start_char, party2.start_char)
        end = max(party1.end_char, party2.end_char)
        between_text = text[start:end].lower()

        return " v. " in between_text or " vs. " in between_text or " versus " in between_text

    def _compile_patterns(self) -> Dict[EntityType, List[re.Pattern]]:
        """Compile regex patterns for entity extraction.

        Returns:
            Dictionary mapping entity types to patterns
        """
        patterns = {
            EntityType.COURT: [
                re.compile(r"\b(?:Supreme|District|Circuit|Court of Appeals?)\s+Court\b", re.I),
                re.compile(r"\bU\.S\.\s+(?:Supreme\s+)?Court\b", re.I),
            ],
            EntityType.JUDGE: [
                re.compile(r"\b(?:Judge|Justice|Magistrate)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"),
                re.compile(r"\b(?:Hon\.|Honorable)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"),
            ],
            EntityType.ATTORNEY: [
                re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+Esq\.?\b"),
                re.compile(r"\bAttorney\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"),
            ],
            EntityType.CASE_CITATION: [
                re.compile(
                    r"\b[A-Z][a-z]+(?:\s+v\.\s+|\s+vs\.\s+)[A-Z][a-z]+,\s+\d+\s+[A-Z][a-z.]+\s+\d+\b"
                ),
            ],
            EntityType.STATUTE: [
                re.compile(r"\b(?:\d+\s+)?U\.S\.C\.(?:\s+§\s*\d+)?", re.I),
                re.compile(r"\bTitle\s+\d+,\s+Section\s+\d+", re.I),
            ],
            EntityType.DATE: [
                re.compile(
                    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
                ),
                re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
            ],
            EntityType.MONETARY: [
                re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s+(?:million|billion))?", re.I),
            ],
        }

        return patterns

    def _entity_to_dict(self, entity: Entity) -> Dict[str, Any]:
        """Convert Entity to dictionary.

        Args:
            entity: Entity object

        Returns:
            Dictionary representation
        """
        return {
            "text": entity.text,
            "type": entity.type.value,
            "start_char": entity.start_char,
            "end_char": entity.end_char,
            "confidence": entity.confidence,
            "context": entity.context,
            "metadata": entity.metadata,
            "aliases": list(entity.aliases),
        }

    def _relationship_to_dict(self, relationship: EntityRelationship) -> Dict[str, Any]:
        """Convert EntityRelationship to dictionary.

        Args:
            relationship: Relationship object

        Returns:
            Dictionary representation
        """
        return {
            "source": self._entity_to_dict(relationship.source),
            "target": self._entity_to_dict(relationship.target),
            "relation_type": relationship.relation_type.value,
            "confidence": relationship.confidence,
            "evidence": relationship.evidence,
        }

    def _resolved_entity_to_dict(self, resolved: ResolvedEntity) -> Dict[str, Any]:
        """Convert ResolvedEntity to dictionary.

        Args:
            resolved: Resolved entity object

        Returns:
            Dictionary representation
        """
        return {
            "canonical_name": resolved.canonical_name,
            "entity_type": resolved.entity_type.value,
            "mention_count": len(resolved.mentions),
            "confidence": resolved.confidence,
            "documents": list(resolved.documents),
        }
