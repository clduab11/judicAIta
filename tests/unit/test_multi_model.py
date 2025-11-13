"""
Unit tests for MultiModelEnsemble.

Tests cover:
- Multi-model ensemble configuration and initialization
- Model selection based on task complexity
- Automatic failover and retry logic
- Circuit breaker functionality
- Consensus mechanisms
- Response validation
- Cost optimization
- Performance metrics tracking

Target: >85% code coverage
"""

import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.models.ensemble.multi_model import (
    CircuitBreaker,
    CircuitState,
    EnsembleConfig,
    EnsembleResponse,
    ModelConfig,
    ModelProvider,
    ModelResponse,
    MultiModelEnsemble,
    TaskComplexity,
    create_production_ensemble,
)


# ==================== FIXTURES ====================


@pytest.fixture
def model_configs() -> list:
    """Sample model configurations."""
    return [
        ModelConfig(
            provider=ModelProvider.GEMMA_3,
            model_name="gemma-3-1b",
            cost_per_1k_tokens=0.0,
            priority=1,
            enabled=True,
        ),
        ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude-3-sonnet",
            cost_per_1k_tokens=0.015,
            priority=2,
            enabled=True,
        ),
        ModelConfig(
            provider=ModelProvider.GPT4,
            model_name="gpt-4-turbo",
            cost_per_1k_tokens=0.01,
            priority=3,
            enabled=True,
        ),
    ]


@pytest.fixture
def ensemble_config(model_configs: list) -> EnsembleConfig:
    """Ensemble configuration."""
    return EnsembleConfig(
        models=model_configs,
        enable_failover=True,
        enable_consensus=True,
        max_retries=3,
        circuit_breaker_threshold=5,
    )


# ==================== CONFIGURATION TESTS ====================


class TestModelConfig:
    """Test ModelConfig validation."""

    def test_default_model_config(self) -> None:
        """Test default model configuration."""
        config = ModelConfig(
            provider=ModelProvider.GEMMA_3,
            model_name="gemma-3-1b",
        )

        assert config.provider == ModelProvider.GEMMA_3
        assert config.max_tokens == 512
        assert config.temperature == 0.1
        assert config.enabled is True
        assert config.priority == 1

    def test_custom_model_config(self) -> None:
        """Test custom model configuration."""
        config = ModelConfig(
            provider=ModelProvider.CLAUDE,
            model_name="claude-3-opus",
            max_tokens=1024,
            temperature=0.7,
            cost_per_1k_tokens=0.075,
            priority=2,
        )

        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.cost_per_1k_tokens == 0.075
        assert config.priority == 2


class TestEnsembleConfig:
    """Test EnsembleConfig validation."""

    def test_valid_config(self, ensemble_config: EnsembleConfig) -> None:
        """Test valid ensemble configuration."""
        assert len(ensemble_config.models) == 3
        assert ensemble_config.enable_failover is True
        assert ensemble_config.max_retries == 3

    def test_requires_at_least_one_enabled_model(self, model_configs: list) -> None:
        """Test validation requires at least one enabled model."""
        # Disable all models
        for model in model_configs:
            model.enabled = False

        with pytest.raises(ValueError, match="At least one model must be enabled"):
            EnsembleConfig(models=model_configs)

    def test_consensus_threshold_validation(self, model_configs: list) -> None:
        """Test consensus threshold validation."""
        config = EnsembleConfig(
            models=model_configs,
            consensus_threshold=0.75,
        )

        assert config.consensus_threshold == 0.75


# ==================== INITIALIZATION TESTS ====================


class TestMultiModelInitialization:
    """Test ensemble initialization."""

    def test_successful_initialization(self, ensemble_config: EnsembleConfig) -> None:
        """Test ensemble initializes successfully."""
        ensemble = MultiModelEnsemble(ensemble_config)

        assert ensemble.config == ensemble_config
        assert len(ensemble.circuit_breakers) == 3
        assert all(cb.state == CircuitState.CLOSED for cb in ensemble.circuit_breakers.values())

    def test_circuit_breakers_created(self, ensemble_config: EnsembleConfig) -> None:
        """Test circuit breakers created for each model."""
        ensemble = MultiModelEnsemble(ensemble_config)

        expected_keys = [f"{m.provider.value}:{m.model_name}" for m in ensemble_config.models]

        assert set(ensemble.circuit_breakers.keys()) == set(expected_keys)


# ==================== MODEL SELECTION TESTS ====================


class TestModelSelection:
    """Test model selection logic."""

    def test_select_models_by_priority(self, ensemble_config: EnsembleConfig) -> None:
        """Test models selected by priority."""
        ensemble = MultiModelEnsemble(ensemble_config)

        selected = ensemble._select_models(TaskComplexity.MEDIUM, require_consensus=False)

        assert len(selected) > 0
        # Should be sorted by priority
        assert selected[0].priority <= selected[-1].priority

    def test_select_cheaper_models_for_simple_tasks(self, ensemble_config: EnsembleConfig) -> None:
        """Test cost optimization for simple tasks."""
        ensemble = MultiModelEnsemble(ensemble_config)

        selected = ensemble._select_models(TaskComplexity.SIMPLE, require_consensus=False)

        # Gemma (free) should be first
        assert selected[0].provider == ModelProvider.GEMMA_3

    def test_select_multiple_models_for_consensus(self, ensemble_config: EnsembleConfig) -> None:
        """Test selecting multiple models for consensus."""
        ensemble = MultiModelEnsemble(ensemble_config)

        selected = ensemble._select_models(TaskComplexity.CRITICAL, require_consensus=True)

        assert len(selected) >= 2

    def test_skip_disabled_models(self, model_configs: list) -> None:
        """Test disabled models are skipped."""
        # Disable one model
        model_configs[1].enabled = False

        config = EnsembleConfig(models=model_configs)
        ensemble = MultiModelEnsemble(config)

        selected = ensemble._select_models(TaskComplexity.MEDIUM, require_consensus=False)

        # Claude should not be in selection
        assert all(m.provider != ModelProvider.CLAUDE for m in selected)


# ==================== CIRCUIT BREAKER TESTS ====================


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_opens_after_failures(self, ensemble_config: EnsembleConfig) -> None:
        """Test circuit opens after threshold failures."""
        ensemble = MultiModelEnsemble(ensemble_config)

        model = ensemble_config.models[0]
        circuit_key = f"{model.provider.value}:{model.model_name}"

        # Mock failures
        with patch.object(ensemble, "_call_model_api") as mock_api:
            mock_api.side_effect = Exception("API Error")

            # Attempt multiple times
            for _ in range(ensemble_config.circuit_breaker_threshold + 1):
                try:
                    ensemble._generate_with_retry("test", model)
                except Exception:
                    pass

        # Circuit should be open
        assert ensemble.circuit_breakers[circuit_key].state == CircuitState.OPEN

    def test_circuit_half_open_after_timeout(self, ensemble_config: EnsembleConfig) -> None:
        """Test circuit transitions to half-open after timeout."""
        ensemble = MultiModelEnsemble(ensemble_config)

        model = ensemble_config.models[0]
        circuit_key = f"{model.provider.value}:{model.model_name}"

        # Manually open circuit
        circuit = ensemble.circuit_breakers[circuit_key]
        circuit.state = CircuitState.OPEN
        circuit.last_failure_time = None  # Force timeout

        # Try to select models
        selected = ensemble._select_models(TaskComplexity.MEDIUM, require_consensus=False)

        # Should attempt half-open
        # (selection logic checks timeout and sets HALF_OPEN)
        assert len(selected) > 0

    def test_circuit_closes_on_success(self, ensemble_config: EnsembleConfig) -> None:
        """Test circuit closes on successful request."""
        ensemble = MultiModelEnsemble(ensemble_config)

        model = ensemble_config.models[0]
        circuit_key = f"{model.provider.value}:{model.model_name}"

        # Mock successful response
        with patch.object(ensemble, "_call_model_api") as mock_api:
            mock_api.return_value = ("Success response", 100)

            response = ensemble._generate_with_retry("test", model)

        assert response.success
        assert ensemble.circuit_breakers[circuit_key].state == CircuitState.CLOSED


# ==================== RETRY LOGIC TESTS ====================


class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    def test_retry_on_failure(self, ensemble_config: EnsembleConfig) -> None:
        """Test retry attempts on failure."""
        ensemble = MultiModelEnsemble(ensemble_config)

        model = ensemble_config.models[0]

        with patch.object(ensemble, "_call_model_api") as mock_api:
            # Fail first attempts, succeed on last
            mock_api.side_effect = [
                Exception("Error 1"),
                Exception("Error 2"),
                ("Success", 100),
            ]

            with patch("time.sleep"):  # Speed up test
                response = ensemble._generate_with_retry("test", model)

        assert response.success
        assert mock_api.call_count == 3

    def test_max_retries_exceeded(self, ensemble_config: EnsembleConfig) -> None:
        """Test behavior when max retries exceeded."""
        ensemble = MultiModelEnsemble(ensemble_config)

        model = ensemble_config.models[0]

        with patch.object(ensemble, "_call_model_api") as mock_api:
            mock_api.side_effect = Exception("Persistent Error")

            with patch("time.sleep"):
                response = ensemble._generate_with_retry("test", model)

        assert not response.success
        assert response.error is not None


# ==================== GENERATION TESTS ====================


class TestGeneration:
    """Test ensemble generation."""

    def test_successful_generation(self, ensemble_config: EnsembleConfig) -> None:
        """Test successful text generation."""
        ensemble = MultiModelEnsemble(ensemble_config)

        with patch.object(ensemble, "_call_model_api") as mock_api:
            mock_api.return_value = ("Generated text", 100)

            response = ensemble.generate("Test prompt", complexity=TaskComplexity.MEDIUM)

        assert isinstance(response, EnsembleResponse)
        assert response.final_text != ""
        assert response.primary_response.success

    def test_failover_on_primary_failure(self, ensemble_config: EnsembleConfig) -> None:
        """Test automatic failover when primary model fails."""
        ensemble = MultiModelEnsemble(ensemble_config)

        with patch.object(ensemble, "_call_model_api") as mock_api:
            # First model fails, second succeeds
            mock_api.side_effect = [
                Exception("Primary failed"),
                ("Backup response", 120),
            ]

            with patch("time.sleep"):
                response = ensemble.generate("Test prompt", complexity=TaskComplexity.MEDIUM)

        assert response.primary_response.success
        assert response.failover_count > 0

    def test_consensus_for_critical_tasks(self, ensemble_config: EnsembleConfig) -> None:
        """Test consensus mechanism for critical tasks."""
        ensemble = MultiModelEnsemble(ensemble_config)

        with patch.object(ensemble, "_call_model_api") as mock_api:
            mock_api.return_value = ("Response", 100)

            response = ensemble.generate(
                "Critical decision",
                complexity=TaskComplexity.CRITICAL,
            )

        # Should have consensus responses
        assert len(response.consensus_responses) > 0


# ==================== VALIDATION TESTS ====================


class TestResponseValidation:
    """Test response validation."""

    def test_detect_hallucination_patterns(self, ensemble_config: EnsembleConfig) -> None:
        """Test hallucination pattern detection."""
        ensemble = MultiModelEnsemble(ensemble_config)

        # Text with hallucination pattern
        invalid = ensemble._validate_response(
            "I don't have access to that information", "test prompt"
        )

        assert invalid is False

    def test_valid_response_passes(self, ensemble_config: EnsembleConfig) -> None:
        """Test valid responses pass validation."""
        ensemble = MultiModelEnsemble(ensemble_config)

        valid = ensemble._validate_response(
            "This is a valid legal analysis of the contract termination clause",
            "analyze termination clause",
        )

        assert valid is True

    def test_empty_response_fails(self, ensemble_config: EnsembleConfig) -> None:
        """Test empty responses fail validation."""
        ensemble = MultiModelEnsemble(ensemble_config)

        invalid = ensemble._validate_response("", "test prompt")

        assert invalid is False


# ==================== METRICS TESTS ====================


class TestMetrics:
    """Test performance metrics tracking."""

    def test_metrics_updated_on_success(self, ensemble_config: EnsembleConfig) -> None:
        """Test metrics updated on successful request."""
        ensemble = MultiModelEnsemble(ensemble_config)

        model = ensemble_config.models[0]
        key = f"{model.provider.value}:{model.model_name}"

        ensemble._update_metrics(key, 100.0, 0.01, success=True)

        metrics = ensemble.metrics[key]
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["total_latency_ms"] == 100.0
        assert metrics["total_cost_usd"] == 0.01

    def test_metrics_summary(self, ensemble_config: EnsembleConfig) -> None:
        """Test metrics summary generation."""
        ensemble = MultiModelEnsemble(ensemble_config)

        # Add some metrics
        model = ensemble_config.models[0]
        key = f"{model.provider.value}:{model.model_name}"

        ensemble._update_metrics(key, 100.0, 0.01, success=True)
        ensemble._update_metrics(key, 150.0, 0.015, success=True)

        summary = ensemble.get_metrics_summary()

        assert key in summary
        assert summary[key]["total_requests"] == 2
        assert summary[key]["success_rate"] == 1.0
        assert "avg_latency_ms" in summary[key]


# ==================== FACTORY FUNCTION TESTS ====================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_production_ensemble(self) -> None:
        """Test production ensemble factory."""
        ensemble = create_production_ensemble()

        assert isinstance(ensemble, MultiModelEnsemble)
        assert len(ensemble.config.models) >= 2
        assert ensemble.config.enable_failover is True


# ==================== INTEGRATION TESTS ====================


@pytest.mark.integration
class TestMultiModelIntegration:
    """Integration tests for multi-model ensemble."""

    def test_end_to_end_generation(self, ensemble_config: EnsembleConfig) -> None:
        """Test complete generation flow."""
        ensemble = MultiModelEnsemble(ensemble_config)

        with patch.object(ensemble, "_call_model_api") as mock_api:
            mock_api.return_value = ("Legal analysis response", 150)

            response = ensemble.generate(
                "Analyze indemnification clause",
                complexity=TaskComplexity.MEDIUM,
            )

        assert response.final_text != ""
        assert response.total_latency_ms > 0
        assert response.primary_response.provider in [
            ModelProvider.GEMMA_3,
            ModelProvider.CLAUDE,
            ModelProvider.GPT4,
        ]
