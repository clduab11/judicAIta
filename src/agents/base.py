"""
Base Agent Architecture for JudicAIta Multi-Agent System.

This module provides the abstract base class for all legal document processing agents,
with integrated support for Gemma 3 1B models, LoRA fine-tuning, and cross-model
compatibility (Gemma 2.5/3).

Design Principles:
    - Single responsibility per agent
    - Composable and chainable
    - Explainable with reasoning traces
    - Model-agnostic core logic
    - TPU-optimized for Kaggle deployment

Performance Targets:
    - Inference: <100ms per request
    - Cross-model delta: <5%
    - Test coverage: >80%
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import logging

from pydantic import BaseModel, Field, validator


# ==================== ENUMS ====================


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ModelFamily(str, Enum):
    """Supported Gemma model families."""

    GEMMA_2_5 = "gemma-2.5"
    GEMMA_3 = "gemma-3"
    GEMMA_3N = "gemma-3n"  # Generic 3n family


# ==================== CONFIGURATION MODELS ====================


class AgentConfig(BaseModel):
    """Configuration for agent initialization.

    Attributes:
        name: Unique agent identifier
        model_path: HuggingFace model path (e.g., "google/gemma-3-1b")
        use_lora: Whether to use LoRA adapters
        lora_path: Path to LoRA weights (optional)
        device: Computation device ("tpu", "cuda", "cpu")
        max_tokens: Maximum output tokens
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Nucleus sampling parameter
        enable_logging: Enable detailed logging
        enable_tracing: Enable reasoning trace collection
        timeout_seconds: Maximum execution time
    """

    name: str = Field(..., description="Agent unique identifier")
    model_path: str = Field(
        default="google/gemma-3-1b", description="HuggingFace model path"
    )
    use_lora: bool = Field(default=False, description="Use LoRA adapters")
    lora_path: Optional[str] = Field(None, description="Path to LoRA weights")
    device: str = Field(default="cpu", description="Computation device")
    max_tokens: int = Field(default=512, ge=1, le=8192, description="Max output tokens")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling")
    enable_logging: bool = Field(default=True, description="Enable logging")
    enable_tracing: bool = Field(default=True, description="Enable reasoning traces")
    timeout_seconds: int = Field(default=300, ge=1, description="Execution timeout")

    @field_validator("model_path")
    def validate_model_path(cls, v: str) -> str:
        """Validate model path is Gemma 3n family."""
        if "gemma" not in v.lower():
            raise ValueError(f"Model must be from Gemma family, got: {v}")
        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class AgentResponse(BaseModel):
    """Standardized agent response format.

    Attributes:
        status: Execution status
        output: Agent output data
        metadata: Additional metadata
        trace: Reasoning trace steps
        error: Error message if failed
        model_version: Model used for inference
        compatibility_score: Cross-model compatibility (0-1)
        execution_time_ms: Execution duration
        timestamp: Completion timestamp
    """

    status: AgentStatus = Field(..., description="Execution status")
    output: Dict[str, Any] = Field(default_factory=dict, description="Agent output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    trace: List[Dict[str, Any]] = Field(default_factory=list, description="Reasoning trace")
    error: Optional[str] = Field(None, description="Error message")
    model_version: str = Field(..., description="Model used")
    compatibility_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Cross-model compatibility"
    )
    execution_time_ms: float = Field(..., ge=0.0, description="Execution time")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


# ==================== TRACE MODELS ====================


class TraceStep(TypedDict):
    """Single step in reasoning trace."""

    step: str
    description: str
    input: Optional[Any]
    output: Optional[Any]
    timestamp: str
    duration_ms: float


# ==================== BASE AGENT ====================


class BaseAgent(ABC):
    """Abstract base class for all JudicAIta agents.

    This class provides:
    - Standard interface for agent execution
    - Gemma 3 1B model integration
    - LoRA adapter support
    - Cross-model compatibility checking
    - Reasoning trace collection
    - Error handling and logging
    - Performance monitoring

    Example:
        >>> class MyAgent(BaseAgent):
        ...     def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        ...         return {"result": "processed"}
        ...
        ...     def validate_input(self, input_data: Dict[str, Any]) -> bool:
        ...         return "text" in input_data
        ...
        >>> config = AgentConfig(name="my_agent")
        >>> agent = MyAgent(config)
        >>> result = agent({"text": "sample"})
    """

    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration.

        Args:
            config: Agent configuration

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If model initialization fails
        """
        self.config = config
        self.status = AgentStatus.IDLE
        self._trace: List[TraceStep] = []
        self._start_time: Optional[float] = None

        # Setup logging
        self.logger = self._setup_logger()

        # Initialize model (lazy loading - subclasses handle actual model init)
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

        self.logger.info(f"Initialized {self.config.name} agent")

    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logger.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"judicaita.agents.{self.config.name}")

        if not self.config.enable_logging:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # Add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output.

        This is the core method that each agent must implement.

        Args:
            input_data: Agent-specific input data

        Returns:
            Agent-specific output data

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    def add_trace_step(
        self,
        step: str,
        description: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Add a step to the reasoning trace.

        Args:
            step: Step identifier (e.g., "parse_pdf", "extract_entities")
            description: Human-readable description
            input_data: Input to this step (optional)
            output_data: Output from this step (optional)
            duration_ms: Step execution time in milliseconds
        """
        if not self.config.enable_tracing:
            return

        trace_step: TraceStep = {
            "step": step,
            "description": description,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
        }

        self._trace.append(trace_step)
        self.logger.debug(f"Trace step: {step} - {description}")

    def clear_trace(self) -> None:
        """Clear the reasoning trace."""
        self._trace = []

    def get_trace(self) -> List[TraceStep]:
        """Get the current reasoning trace.

        Returns:
            List of trace steps
        """
        return self._trace.copy()

    def _check_compatibility(self) -> float:
        """Check cross-model compatibility score.

        This method should be overridden by subclasses to implement
        actual compatibility checking against Gemma 2.5/3 models.

        Returns:
            Compatibility score (0.0-1.0), where 1.0 means 100% compatible
        """
        # Default: assume full compatibility
        # Subclasses with model-specific logic should override this
        return 1.0

    def _get_execution_time_ms(self) -> float:
        """Calculate execution time in milliseconds.

        Returns:
            Execution time in milliseconds
        """
        if self._start_time is None:
            return 0.0

        import time

        return (time.time() - self._start_time) * 1000

    def __call__(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Execute agent with input data.

        This is the main entry point for agent execution with:
        - Input validation
        - Status management
        - Error handling
        - Trace collection
        - Performance monitoring

        Args:
            input_data: Agent-specific input data

        Returns:
            Standardized agent response

        Raises:
            ValueError: If input validation fails
            TimeoutError: If execution exceeds timeout
            RuntimeError: If agent execution fails
        """
        import time

        self._start_time = time.time()
        self.status = AgentStatus.PROCESSING
        self.clear_trace()

        try:
            # Validate input
            self.add_trace_step(
                step="validate_input",
                description="Validating input data",
                input_data={"keys": list(input_data.keys())},
            )

            if not self.validate_input(input_data):
                raise ValueError(f"Invalid input for {self.config.name}")

            # Process
            self.logger.info(f"Processing with {self.config.name}")
            self.add_trace_step(
                step="start_processing",
                description=f"Starting {self.config.name} processing",
            )

            output = self.process(input_data)

            # Check compatibility
            compatibility_score = self._check_compatibility()

            self.add_trace_step(
                step="processing_complete",
                description="Processing completed successfully",
                output_data={"output_keys": list(output.keys())},
            )

            # Success
            self.status = AgentStatus.COMPLETED
            execution_time = self._get_execution_time_ms()

            self.logger.info(
                f"{self.config.name} completed in {execution_time:.2f}ms "
                f"(compatibility: {compatibility_score:.2%})"
            )

            return AgentResponse(
                status=AgentStatus.COMPLETED,
                output=output,
                metadata={
                    "agent_name": self.config.name,
                    "model_path": self.config.model_path,
                },
                trace=self.get_trace(),
                model_version=self.config.model_path,
                compatibility_score=compatibility_score,
                execution_time_ms=execution_time,
            )

        except ValueError as e:
            self.status = AgentStatus.FAILED
            self.logger.error(f"Validation error in {self.config.name}: {str(e)}")

            return AgentResponse(
                status=AgentStatus.FAILED,
                output={},
                error=f"Validation error: {str(e)}",
                trace=self.get_trace(),
                model_version=self.config.model_path,
                execution_time_ms=self._get_execution_time_ms(),
            )

        except TimeoutError as e:
            self.status = AgentStatus.TIMEOUT
            self.logger.error(f"Timeout in {self.config.name}: {str(e)}")

            return AgentResponse(
                status=AgentStatus.TIMEOUT,
                output={},
                error=f"Execution timeout: {str(e)}",
                trace=self.get_trace(),
                model_version=self.config.model_path,
                execution_time_ms=self._get_execution_time_ms(),
            )

        except Exception as e:
            self.status = AgentStatus.FAILED
            self.logger.error(f"Error in {self.config.name}: {str(e)}", exc_info=True)

            return AgentResponse(
                status=AgentStatus.FAILED,
                output={},
                error=str(e),
                trace=self.get_trace(),
                model_version=self.config.model_path,
                execution_time_ms=self._get_execution_time_ms(),
            )

    def __repr__(self) -> str:
        """String representation of agent.

        Returns:
            Agent representation string
        """
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name}, "
            f"model={self.config.model_path}, "
            f"status={self.status.value})"
        )
