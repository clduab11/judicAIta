"""
Production Observability and Monitoring Framework.

This module provides comprehensive observability for the JudicAIta platform:
- Structured logging with contextual enrichment
- Performance metrics collection and aggregation
- Distributed tracing with OpenTelemetry (when available)
- Anomaly detection for workflow degradation
- Real-time dashboard metrics endpoints
- Error tracking and alerting

Performance Targets:
    - Logging overhead: <5ms per log entry
    - Metrics aggregation: <100ms per endpoint
    - Trace sampling: 10% of requests in production
    - Anomaly detection latency: <1s
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


# ==================== ENUMS ====================


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ==================== CONFIGURATION ====================


class ObservabilityConfig(BaseModel):
    """Configuration for observability system.

    Attributes:
        enable_logging: Enable structured logging
        log_level: Minimum log level
        enable_metrics: Enable metrics collection
        enable_tracing: Enable distributed tracing
        trace_sampling_rate: Sampling rate for traces (0-1)
        enable_anomaly_detection: Enable anomaly detection
        anomaly_window_size: Window size for anomaly detection
        anomaly_std_threshold: Standard deviation threshold
        metrics_retention_seconds: Metrics retention period
    """

    enable_logging: bool = Field(default=True)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=False)  # Requires OpenTelemetry
    trace_sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    enable_anomaly_detection: bool = Field(default=True)
    anomaly_window_size: int = Field(default=100, ge=10, le=1000)
    anomaly_std_threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    metrics_retention_seconds: int = Field(default=3600, ge=60)


# ==================== DATA MODELS ====================


@dataclass
class LogEntry:
    """Structured log entry.

    Attributes:
        timestamp: Log timestamp
        level: Log level
        message: Log message
        logger_name: Logger name
        context: Additional context data
        trace_id: Distributed trace ID
        span_id: Span ID within trace
    """

    timestamp: datetime
    level: str
    message: str
    logger_name: str
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(
            {
                "timestamp": self.timestamp.isoformat(),
                "level": self.level,
                "message": self.message,
                "logger": self.logger_name,
                "context": self.context,
                "trace_id": self.trace_id,
                "span_id": self.span_id,
            }
        )


@dataclass
class Metric:
    """Performance metric.

    Attributes:
        name: Metric name
        type: Metric type
        value: Current value
        timestamp: Metric timestamp
        labels: Metric labels
    """

    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert.

    Attributes:
        severity: Alert severity
        message: Alert message
        metric_name: Related metric name
        current_value: Current metric value
        threshold: Threshold value
        timestamp: Alert timestamp
    """

    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime


# ==================== STRUCTURED LOGGER ====================


class StructuredLogger:
    """Structured logger with context enrichment.

    Example:
        >>> logger = StructuredLogger("my_agent")
        >>> logger.info("Processing document", doc_id="123", pages=10)
    """

    def __init__(self, name: str, config: Optional[ObservabilityConfig] = None):
        """Initialize structured logger.

        Args:
            name: Logger name
            config: Observability configuration
        """
        self.name = name
        self.config = config or ObservabilityConfig()
        self.context: Dict[str, Any] = {}

        # Standard Python logger
        self.logger = logging.getLogger(name)
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure underlying Python logger."""
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        self.logger.setLevel(level_map[self.config.log_level])

    def set_context(self, **kwargs: Any) -> None:
        """Set persistent context for all log entries.

        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear persistent context."""
        self.context = {}

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, kwargs)

    def _log(self, level: LogLevel, message: str, context: Dict[str, Any]) -> None:
        """Internal log method.

        Args:
            level: Log level
            message: Log message
            context: Additional context
        """
        if not self.config.enable_logging:
            return

        # Merge persistent context with log context
        full_context = {**self.context, **context}

        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level.value,
            message=message,
            logger_name=self.name,
            context=full_context,
        )

        # Log to standard logger
        level_method = getattr(self.logger, level.value)
        level_method(entry.to_json())


# ==================== METRICS COLLECTOR ====================


class MetricsCollector:
    """Metrics collection and aggregation.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.increment("requests_total", labels={"method": "POST"})
        >>> collector.record_timer("request_duration_ms", 123.45)
        >>> summary = collector.get_summary()
    """

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """Initialize metrics collector.

        Args:
            config: Observability configuration
        """
        self.config = config or ObservabilityConfig()

        # Metric storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.anomaly_window_size)
        )

        # Metric metadata
        self.metric_timestamps: Dict[str, datetime] = {}

    def increment(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment counter metric.

        Args:
            name: Metric name
            value: Increment value
            labels: Metric labels
        """
        if not self.config.enable_metrics:
            return

        metric_key = self._make_key(name, labels)
        self.counters[metric_key] += value
        self.metric_timestamps[metric_key] = datetime.now()

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
        """
        if not self.config.enable_metrics:
            return

        metric_key = self._make_key(name, labels)
        self.gauges[metric_key] = value
        self.metric_timestamps[metric_key] = datetime.now()

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record histogram value.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        if not self.config.enable_metrics:
            return

        metric_key = self._make_key(name, labels)
        self.histograms[metric_key].append(value)
        self.metric_timestamps[metric_key] = datetime.now()

        # Prune old values
        self._prune_histogram(metric_key)

    def record_timer(
        self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record timer metric.

        Args:
            name: Metric name
            duration_ms: Duration in milliseconds
            labels: Metric labels
        """
        if not self.config.enable_metrics:
            return

        metric_key = self._make_key(name, labels)
        self.timers[metric_key].append(duration_ms)
        self.metric_timestamps[metric_key] = datetime.now()

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and labels.

        Args:
            name: Metric name
            labels: Metric labels

        Returns:
            Metric key string
        """
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _prune_histogram(self, key: str) -> None:
        """Prune old histogram values.

        Args:
            key: Histogram key
        """
        # Keep only recent values (within retention period)
        # Simplified - in production, track timestamps per value
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Dictionary with metrics summary
        """
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {},
            "timers": {},
        }

        # Aggregate histograms
        for key, values in self.histograms.items():
            if values:
                summary["histograms"][key] = {
                    "count": len(values),
                    "sum": sum(values),
                    "mean": np.mean(values),
                    "p50": np.percentile(values, 50),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99),
                }

        # Aggregate timers
        for key, values in self.timers.items():
            if values:
                summary["timers"][key] = {
                    "count": len(values),
                    "mean_ms": np.mean(values),
                    "p50_ms": np.percentile(values, 50),
                    "p95_ms": np.percentile(values, 95),
                    "p99_ms": np.percentile(values, 99),
                }

        return summary


# ==================== ANOMALY DETECTOR ====================


class AnomalyDetector:
    """Real-time anomaly detection for metrics.

    Example:
        >>> detector = AnomalyDetector()
        >>> is_anomaly = detector.detect_anomaly("latency_ms", 1500.0)
    """

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """Initialize anomaly detector.

        Args:
            config: Observability configuration
        """
        self.config = config or ObservabilityConfig()

        # Time series storage
        self.time_series: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.anomaly_window_size)
        )

        # Detected anomalies
        self.anomalies: List[Alert] = []

    def detect_anomaly(self, metric_name: str, value: float) -> bool:
        """Detect if value is anomalous.

        Uses standard deviation-based anomaly detection.

        Args:
            metric_name: Metric name
            value: New metric value

        Returns:
            True if anomaly detected
        """
        if not self.config.enable_anomaly_detection:
            return False

        # Add to time series
        series = self.time_series[metric_name]
        series.append(value)

        # Need enough data points
        if len(series) < 30:
            return False

        # Calculate statistics
        mean = np.mean(series)
        std = np.std(series)

        # Check if value is anomalous
        if std > 0:
            z_score = abs(value - mean) / std

            if z_score > self.config.anomaly_std_threshold:
                # Anomaly detected
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    message=f"Anomaly detected in {metric_name}",
                    metric_name=metric_name,
                    current_value=value,
                    threshold=mean + self.config.anomaly_std_threshold * std,
                    timestamp=datetime.now(),
                )
                self.anomalies.append(alert)

                # Keep only recent anomalies
                if len(self.anomalies) > 100:
                    self.anomalies = self.anomalies[-100:]

                return True

        return False

    def get_recent_anomalies(self, minutes: int = 60) -> List[Alert]:
        """Get recent anomalies.

        Args:
            minutes: Time window in minutes

        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self.anomalies if a.timestamp >= cutoff]


# ==================== PERFORMANCE PROFILER ====================


class PerformanceProfiler:
    """Performance profiling with context manager.

    Example:
        >>> profiler = PerformanceProfiler(collector)
        >>> with profiler.profile("my_operation"):
        ...     # Do work
        ...     pass
    """

    def __init__(self, collector: MetricsCollector):
        """Initialize profiler.

        Args:
            collector: Metrics collector
        """
        self.collector = collector
        self.start_times: Dict[str, float] = {}

    def profile(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Profile operation context manager.

        Args:
            operation_name: Operation name
            labels: Metric labels

        Returns:
            Context manager
        """
        return ProfileContext(self, operation_name, labels)


class ProfileContext:
    """Context manager for profiling."""

    def __init__(
        self,
        profiler: PerformanceProfiler,
        operation: str,
        labels: Optional[Dict[str, str]],
    ):
        """Initialize context.

        Args:
            profiler: Performance profiler
            operation: Operation name
            labels: Metric labels
        """
        self.profiler = profiler
        self.operation = operation
        self.labels = labels
        self.start_time: float = 0.0

    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record duration."""
        duration_ms = (time.time() - self.start_time) * 1000
        self.profiler.collector.record_timer(
            f"{self.operation}_duration_ms", duration_ms, self.labels
        )


# ==================== OBSERVABILITY MANAGER ====================


class ObservabilityManager:
    """Central observability manager.

    Example:
        >>> obs = ObservabilityManager()
        >>> logger = obs.get_logger("my_component")
        >>> logger.info("Started processing")
        >>> obs.collector.increment("requests_total")
    """

    _instance = None

    def __new__(cls, config: Optional[ObservabilityConfig] = None):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """Initialize observability manager."""
        if self._initialized:
            return

        self.config = config or ObservabilityConfig()

        # Components
        self.collector = MetricsCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.profiler = PerformanceProfiler(self.collector)

        # Loggers cache
        self._loggers: Dict[str, StructuredLogger] = {}

        self._initialized = True

    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create logger.

        Args:
            name: Logger name

        Returns:
            StructuredLogger instance
        """
        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(name, self.config)

        return self._loggers[name]

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard.

        Returns:
            Dictionary with dashboard metrics
        """
        metrics_summary = self.collector.get_summary()
        recent_anomalies = self.anomaly_detector.get_recent_anomalies(minutes=60)

        return {
            "metrics": metrics_summary,
            "anomalies": [
                {
                    "severity": a.severity.value,
                    "message": a.message,
                    "metric": a.metric_name,
                    "value": a.current_value,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in recent_anomalies
            ],
            "timestamp": datetime.now().isoformat(),
        }


# ==================== GLOBAL INSTANCE ====================


# Global observability manager
_obs_manager: Optional[ObservabilityManager] = None


def get_observability() -> ObservabilityManager:
    """Get global observability manager.

    Returns:
        ObservabilityManager singleton
    """
    global _obs_manager
    if _obs_manager is None:
        _obs_manager = ObservabilityManager()
    return _obs_manager
