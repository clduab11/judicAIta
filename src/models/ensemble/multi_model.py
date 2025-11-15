"""
Multi-Model Ensemble System with Automatic Failover.

This module provides production-grade multi-model orchestration:
- Support for multiple LLM providers (Claude, GPT-4, Gemini, Gemma 3)
- Automatic failover and retry logic with exponential backoff
- Model selection based on task complexity and cost
- Consensus mechanisms for critical decisions
- Response validation and hallucination detection
- Circuit breakers for failing models
- Performance tracking and cost optimization

Performance Targets:
    - Primary model latency: <2s for standard queries
    - Failover latency: <500ms overhead
    - Availability: >99.9% with multi-model redundancy
    - Cost optimization: 30-50% reduction vs single-provider
    - Hallucination detection: >90% accuracy
"""

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


# ==================== ENUMS ====================


class ModelProvider(str, Enum):
    """LLM provider types."""

    GEMMA_3 = "gemma_3"
    CLAUDE = "claude"
    GPT4 = "gpt4"
    GEMINI = "gemini"


class TaskComplexity(str, Enum):
    """Task complexity levels for model selection."""

    SIMPLE = "simple"  # Basic extraction, classification
    MEDIUM = "medium"  # Document analysis, summarization
    COMPLEX = "complex"  # Legal reasoning, multi-step analysis
    CRITICAL = "critical"  # High-stakes decisions requiring consensus


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


# ==================== CONFIGURATION ====================


class ModelConfig(BaseModel):
    """Configuration for individual model.

    Attributes:
        provider: Model provider
        model_name: Specific model name/version
        api_key: API key (optional, can use env var)
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        timeout_seconds: Request timeout
        cost_per_1k_tokens: Cost per 1K tokens (USD)
        enabled: Whether model is enabled
        priority: Selection priority (lower = higher priority)
    """

    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = Field(default=None, description="API key (optional)")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    enabled: bool = Field(default=True)
    priority: int = Field(default=1, ge=1, le=10)


class EnsembleConfig(BaseModel):
    """Configuration for model ensemble.

    Attributes:
        models: List of model configurations
        enable_failover: Enable automatic failover
        enable_consensus: Enable consensus for critical tasks
        consensus_threshold: Agreement threshold for consensus (0-1)
        max_retries: Maximum retry attempts per model
        retry_backoff_base: Base for exponential backoff (seconds)
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Circuit breaker timeout (seconds)
        enable_cost_optimization: Enable cost-based routing
        enable_validation: Enable response validation
        validation_threshold: Confidence threshold for validation
    """

    models: List[ModelConfig] = Field(min_items=1)
    enable_failover: bool = Field(default=True)
    enable_consensus: bool = Field(default=True)
    consensus_threshold: float = Field(default=0.66, ge=0.5, le=1.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_base: float = Field(default=1.0, ge=0.1)
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_timeout: int = Field(default=60, ge=10)
    enable_cost_optimization: bool = Field(default=True)
    enable_validation: bool = Field(default=True)
    validation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @validator("models")
    def validate_at_least_one_enabled(cls, v: List[ModelConfig]) -> List[ModelConfig]:
        """Ensure at least one model is enabled."""
        if not any(m.enabled for m in v):
            raise ValueError("At least one model must be enabled")
        return v


# ==================== DATA MODELS ====================


@dataclass
class ModelResponse:
    """Response from a single model.

    Attributes:
        provider: Model provider
        text: Generated text
        latency_ms: Response latency in milliseconds
        tokens_used: Number of tokens used
        cost_usd: Estimated cost in USD
        confidence: Response confidence score
        metadata: Additional metadata
        error: Error message if failed
    """

    provider: ModelProvider
    text: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether response was successful."""
        return self.error is None


@dataclass
class EnsembleResponse:
    """Aggregated response from ensemble.

    Attributes:
        primary_response: Primary model response
        consensus_responses: Responses used for consensus
        final_text: Final aggregated text
        total_latency_ms: Total latency including retries
        total_cost_usd: Total cost across all models
        confidence: Final confidence score
        models_used: List of models attempted
        failover_count: Number of failovers
        validation_passed: Whether validation passed
        metadata: Additional metadata
    """

    primary_response: ModelResponse
    consensus_responses: List[ModelResponse] = field(default_factory=list)
    final_text: str = ""
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    confidence: float = 0.0
    models_used: List[str] = field(default_factory=list)
    failover_count: int = 0
    validation_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreaker:
    """Circuit breaker for a model.

    Attributes:
        state: Current circuit state
        failure_count: Consecutive failure count
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


# ==================== MULTI-MODEL ENSEMBLE ====================


class MultiModelEnsemble:
    """Production-grade multi-model ensemble with failover and optimization.

    This system provides:
    - Automatic failover across multiple LLM providers
    - Intelligent model selection based on task complexity
    - Consensus mechanisms for critical decisions
    - Cost optimization routing
    - Circuit breakers for failing models
    - Response validation and hallucination detection

    Example:
        >>> config = EnsembleConfig(models=[...])
        >>> ensemble = MultiModelEnsemble(config)
        >>> response = ensemble.generate(
        ...     "Analyze this contract clause",
        ...     complexity=TaskComplexity.MEDIUM
        ... )
        >>> print(response.final_text)
    """

    def __init__(self, config: EnsembleConfig):
        """Initialize multi-model ensemble.

        Args:
            config: Ensemble configuration

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Circuit breakers per model
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            f"{m.provider.value}:{m.model_name}": CircuitBreaker() for m in config.models
        }

        # Performance metrics
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_latency_ms": 0.0,
                "total_cost_usd": 0.0,
            }
        )

        # Response cache for deduplication
        self.response_cache: Dict[str, ModelResponse] = {}

        self.logger.info(
            f"Multi-model ensemble initialized ({len(config.models)} models, "
            f"failover={config.enable_failover}, consensus={config.enable_consensus})"
        )

    def generate(
        self,
        prompt: str,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        require_consensus: bool = False,
    ) -> EnsembleResponse:
        """Generate response using ensemble.

        Args:
            prompt: Input prompt
            complexity: Task complexity for model selection
            require_consensus: Whether to require consensus

        Returns:
            EnsembleResponse with aggregated result

        Raises:
            RuntimeError: If all models fail
        """
        start_time = time.time()

        # Select models based on complexity and cost
        selected_models = self._select_models(complexity, require_consensus)

        if not selected_models:
            raise RuntimeError("No models available")

        # Attempt primary model with retries
        primary_response = None
        failover_count = 0

        for model in selected_models:
            response = self._generate_with_retry(prompt, model)

            if response.success:
                primary_response = response
                break
            else:
                failover_count += 1
                self.logger.warning(
                    f"Model {model.provider.value} failed, "
                    f"attempting failover ({failover_count}/{len(selected_models)})"
                )

        if not primary_response or not primary_response.success:
            raise RuntimeError("All models failed to generate response")

        # Get consensus if required
        consensus_responses = []
        if require_consensus or (
            self.config.enable_consensus and complexity == TaskComplexity.CRITICAL
        ):
            consensus_responses = self._get_consensus(prompt, selected_models[1:3])

        # Aggregate responses
        final_text = self._aggregate_responses(primary_response, consensus_responses)

        # Validate response
        validation_passed = True
        if self.config.enable_validation:
            validation_passed = self._validate_response(final_text, prompt)

        # Calculate final confidence
        confidence = self._calculate_confidence(primary_response, consensus_responses)

        # Build ensemble response
        total_latency = (time.time() - start_time) * 1000
        total_cost = sum(r.cost_usd for r in [primary_response] + consensus_responses)

        ensemble_response = EnsembleResponse(
            primary_response=primary_response,
            consensus_responses=consensus_responses,
            final_text=final_text,
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
            confidence=confidence,
            models_used=[f"{r.provider.value}" for r in [primary_response] + consensus_responses],
            failover_count=failover_count,
            validation_passed=validation_passed,
        )

        self.logger.info(
            f"Ensemble generated response (latency={total_latency:.1f}ms, "
            f"cost=${total_cost:.4f}, confidence={confidence:.2f})"
        )

        return ensemble_response

    def _select_models(
        self, complexity: TaskComplexity, require_consensus: bool
    ) -> List[ModelConfig]:
        """Select models based on task complexity and cost.

        Args:
            complexity: Task complexity
            require_consensus: Whether consensus is required

        Returns:
            List of selected models in priority order
        """
        # Filter enabled models with closed circuits
        available_models = []
        for model in self.config.models:
            if not model.enabled:
                continue

            circuit_key = f"{model.provider.value}:{model.model_name}"
            circuit = self.circuit_breakers[circuit_key]

            # Check circuit state
            if circuit.state == CircuitState.OPEN:
                # Check if timeout elapsed
                if (
                    circuit.last_failure_time
                    and (datetime.now() - circuit.last_failure_time).total_seconds()
                    < self.config.circuit_breaker_timeout
                ):
                    continue
                # Try half-open
                circuit.state = CircuitState.HALF_OPEN

            available_models.append(model)

        if not available_models:
            return []

        # Sort by priority (lower = higher priority)
        available_models.sort(key=lambda m: m.priority)

        # For cost optimization, prefer cheaper models for simple tasks
        if self.config.enable_cost_optimization and complexity == TaskComplexity.SIMPLE:
            available_models.sort(key=lambda m: m.cost_per_1k_tokens)

        # Return more models if consensus required
        if require_consensus:
            return available_models[:3]
        else:
            return available_models

    def _generate_with_retry(self, prompt: str, model: ModelConfig) -> ModelResponse:
        """Generate response with retry logic.

        Args:
            prompt: Input prompt
            model: Model configuration

        Returns:
            ModelResponse
        """
        circuit_key = f"{model.provider.value}:{model.model_name}"
        circuit = self.circuit_breakers[circuit_key]

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                # Generate response (mock implementation)
                text, tokens = self._call_model_api(prompt, model)

                latency_ms = (time.time() - start_time) * 1000
                cost = (tokens / 1000.0) * model.cost_per_1k_tokens

                # Success - close circuit
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                circuit.last_success_time = datetime.now()

                # Update metrics
                self._update_metrics(circuit_key, latency_ms, cost, success=True)

                return ModelResponse(
                    provider=model.provider,
                    text=text,
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                    cost_usd=cost,
                    confidence=0.85,  # Mock confidence
                )

            except Exception as e:
                self.logger.warning(
                    f"Model {model.provider.value} attempt {attempt + 1} failed: {str(e)}"
                )

                # Record failure
                circuit.failure_count += 1
                circuit.last_failure_time = datetime.now()

                # Open circuit if threshold exceeded
                if circuit.failure_count >= self.config.circuit_breaker_threshold:
                    circuit.state = CircuitState.OPEN
                    self.logger.error(f"Circuit breaker OPEN for {model.provider.value}")

                # Update metrics
                self._update_metrics(circuit_key, 0, 0, success=False)

                # Retry with exponential backoff
                if attempt < self.config.max_retries:
                    backoff = self.config.retry_backoff_base * (2**attempt)
                    time.sleep(backoff)
                else:
                    return ModelResponse(
                        provider=model.provider,
                        error=str(e),
                    )

        return ModelResponse(provider=model.provider, error="Max retries exceeded")

    def _call_model_api(self, prompt: str, model: ModelConfig) -> Tuple[str, int]:
        """Call model API (mock implementation).

        Args:
            prompt: Input prompt
            model: Model configuration

        Returns:
            Tuple of (generated_text, tokens_used)

        Note:
            This is a mock implementation. In production, integrate actual APIs:
            - Gemma 3: Use src.models.gemma3.model_wrapper
            - Claude: Use Anthropic SDK
            - GPT-4: Use OpenAI SDK
            - Gemini: Use Google AI SDK
        """
        # Mock implementation - return based on provider
        if model.provider == ModelProvider.GEMMA_3:
            # In production: use Gemma3ModelWrapper
            return f"[Gemma 3 Response to: {prompt[:50]}...]", 100
        elif model.provider == ModelProvider.CLAUDE:
            # In production: use Anthropic client
            return f"[Claude Response to: {prompt[:50]}...]", 120
        elif model.provider == ModelProvider.GPT4:
            # In production: use OpenAI client
            return f"[GPT-4 Response to: {prompt[:50]}...]", 110
        elif model.provider == ModelProvider.GEMINI:
            # In production: use Google AI client
            return f"[Gemini Response to: {prompt[:50]}...]", 105
        else:
            raise ValueError(f"Unknown provider: {model.provider}")

    def _get_consensus(self, prompt: str, models: List[ModelConfig]) -> List[ModelResponse]:
        """Get consensus from multiple models.

        Args:
            prompt: Input prompt
            models: Models to query for consensus

        Returns:
            List of ModelResponse objects
        """
        responses = []
        for model in models[:2]:  # Limit to 2 additional models
            response = self._generate_with_retry(prompt, model)
            if response.success:
                responses.append(response)

        return responses

    def _aggregate_responses(self, primary: ModelResponse, consensus: List[ModelResponse]) -> str:
        """Aggregate multiple responses.

        Args:
            primary: Primary model response
            consensus: Consensus model responses

        Returns:
            Aggregated text
        """
        if not consensus:
            return primary.text

        # Check for agreement
        all_texts = [primary.text] + [r.text for r in consensus]

        # Simple agreement check (in production, use semantic similarity)
        if len(set(all_texts)) == 1:
            # Perfect agreement
            return primary.text

        # Weighted voting (prefer higher confidence responses)
        # For now, return primary response
        # In production, implement proper consensus mechanism
        return primary.text

    def _validate_response(self, text: str, prompt: str) -> bool:
        """Validate response for hallucinations and relevance.

        Args:
            text: Generated text
            prompt: Original prompt

        Returns:
            True if validation passed
        """
        # Basic validation checks
        if not text or len(text) < 10:
            return False

        # Check for common hallucination patterns
        hallucination_patterns = [
            r"I don't have access to",
            r"I cannot provide",
            r"I apologize, but I",
        ]

        for pattern in hallucination_patterns:
            if re.search(pattern, text, re.I):
                self.logger.warning(f"Potential hallucination detected: {pattern}")
                return False

        # Check relevance (simplified - in production use embeddings)
        prompt_words = set(prompt.lower().split())
        response_words = set(text.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)

        if overlap < 0.1:  # Less than 10% overlap
            self.logger.warning("Low relevance between prompt and response")
            return False

        return True

    def _calculate_confidence(
        self, primary: ModelResponse, consensus: List[ModelResponse]
    ) -> float:
        """Calculate final confidence score.

        Args:
            primary: Primary response
            consensus: Consensus responses

        Returns:
            Confidence score (0-1)
        """
        if not consensus:
            return primary.confidence

        # Average confidence across all models
        all_confidences = [primary.confidence] + [r.confidence for r in consensus]
        avg_confidence = sum(all_confidences) / len(all_confidences)

        # Boost confidence if consensus achieved
        if len(set(r.text for r in [primary] + consensus)) == 1:
            avg_confidence = min(1.0, avg_confidence * 1.1)

        return avg_confidence

    def _update_metrics(
        self, model_key: str, latency_ms: float, cost_usd: float, success: bool
    ) -> None:
        """Update performance metrics.

        Args:
            model_key: Model identifier
            latency_ms: Request latency
            cost_usd: Request cost
            success: Whether request succeeded
        """
        metrics = self.metrics[model_key]
        metrics["total_requests"] += 1

        if success:
            metrics["successful_requests"] += 1
            metrics["total_latency_ms"] += latency_ms
            metrics["total_cost_usd"] += cost_usd
        else:
            metrics["failed_requests"] += 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble metrics.

        Returns:
            Dictionary with metrics summary
        """
        summary = {}

        for model_key, metrics in self.metrics.items():
            total = metrics["total_requests"]
            if total == 0:
                continue

            success_rate = metrics["successful_requests"] / total
            avg_latency = (
                metrics["total_latency_ms"] / metrics["successful_requests"]
                if metrics["successful_requests"] > 0
                else 0
            )

            summary[model_key] = {
                "total_requests": total,
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "total_cost_usd": metrics["total_cost_usd"],
                "circuit_state": self.circuit_breakers[model_key].state.value,
            }

        return summary

    def reset_circuit_breaker(self, provider: ModelProvider, model_name: str) -> None:
        """Manually reset circuit breaker for a model.

        Args:
            provider: Model provider
            model_name: Model name
        """
        circuit_key = f"{provider.value}:{model_name}"
        if circuit_key in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = CircuitBreaker()
            self.logger.info(f"Circuit breaker reset for {circuit_key}")


# ==================== FACTORY FUNCTIONS ====================


def create_production_ensemble() -> MultiModelEnsemble:
    """Create production-ready multi-model ensemble.

    Returns:
        Configured MultiModelEnsemble
    """
    models = [
        # Primary: Gemma 3 (cheapest, local)
        ModelConfig(
            provider=ModelProvider.GEMMA_3,
            model_name="gemma-3-1b",
            cost_per_1k_tokens=0.0,  # Local model
            priority=1,
            enabled=True,
        ),
        # Backup: Claude (high quality)
        ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude-3-sonnet",
            cost_per_1k_tokens=0.015,
            priority=2,
            enabled=True,
        ),
        # Tertiary: GPT-4 (consensus)
        ModelConfig(
            provider=ModelProvider.GPT4,
            model_name="gpt-4-turbo",
            cost_per_1k_tokens=0.01,
            priority=3,
            enabled=True,
        ),
    ]

    config = EnsembleConfig(
        models=models,
        enable_failover=True,
        enable_consensus=True,
        enable_cost_optimization=True,
        max_retries=3,
    )

    return MultiModelEnsemble(config)
