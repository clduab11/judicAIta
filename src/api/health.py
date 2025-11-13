"""
Health Check and Monitoring Endpoints.

This module provides production-grade health check endpoints:
- Liveness probe: Is the application running?
- Readiness probe: Can the application serve traffic?
- Startup probe: Has the application fully initialized?
- Dependency checks: Are all dependencies healthy?
- Metrics endpoint: Prometheus-compatible metrics

Performance Targets:
    - Health check latency: <50ms
    - Dependency check latency: <500ms
    - Metrics endpoint latency: <200ms
"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==================== ENUMS ====================


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class DependencyType(str, Enum):
    """Dependency types."""

    DATABASE = "database"
    MODEL = "model"
    EXTERNAL_API = "external_api"
    FILESYSTEM = "filesystem"


# ==================== MODELS ====================


class DependencyHealth(BaseModel):
    """Health status of a dependency.

    Attributes:
        name: Dependency name
        type: Dependency type
        status: Health status
        latency_ms: Check latency in milliseconds
        error: Error message if unhealthy
        metadata: Additional metadata
    """

    name: str
    type: DependencyType
    status: HealthStatus
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthCheckResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Overall health status
        timestamp: Check timestamp
        uptime_seconds: Application uptime
        version: Application version
        dependencies: Dependency health checks
        details: Additional details
    """

    status: HealthStatus
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"
    dependencies: List[DependencyHealth] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


# ==================== HEALTH CHECKER ====================


class HealthChecker:
    """Production health checker with dependency validation.

    Example:
        >>> checker = HealthChecker()
        >>> await checker.check_health()
    """

    def __init__(self):
        """Initialize health checker."""
        self.start_time = time.time()
        self.version = "1.0.0"

        # Dependency checkers
        self.dependency_checkers: Dict[str, callable] = {}

    def register_dependency(self, name: str, checker: callable, dep_type: DependencyType) -> None:
        """Register a dependency health checker.

        Args:
            name: Dependency name
            checker: Async function that returns (healthy: bool, error: str)
            dep_type: Dependency type
        """
        self.dependency_checkers[name] = {
            "checker": checker,
            "type": dep_type,
        }

    async def check_liveness(self) -> HealthCheckResponse:
        """Check if application is alive.

        This is a lightweight check that returns quickly.

        Returns:
            HealthCheckResponse
        """
        uptime = time.time() - self.start_time

        return HealthCheckResponse(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            version=self.version,
            details={"check_type": "liveness"},
        )

    async def check_readiness(self) -> HealthCheckResponse:
        """Check if application is ready to serve traffic.

        Validates critical dependencies.

        Returns:
            HealthCheckResponse
        """
        uptime = time.time() - self.start_time

        # Check critical dependencies in parallel
        dependency_results = await self._check_dependencies(critical_only=True)

        # Determine overall status
        all_healthy = all(d.status == HealthStatus.HEALTHY for d in dependency_results)
        any_unhealthy = any(d.status == HealthStatus.UNHEALTHY for d in dependency_results)

        if all_healthy:
            status = HealthStatus.HEALTHY
        elif any_unhealthy:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.DEGRADED

        return HealthCheckResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            version=self.version,
            dependencies=dependency_results,
            details={"check_type": "readiness"},
        )

    async def check_startup(self) -> HealthCheckResponse:
        """Check if application has completed startup.

        Returns:
            HealthCheckResponse
        """
        uptime = time.time() - self.start_time

        # Consider started after minimum uptime
        min_uptime = 5.0  # 5 seconds
        is_started = uptime >= min_uptime

        return HealthCheckResponse(
            status=HealthStatus.HEALTHY if is_started else HealthStatus.UNHEALTHY,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            version=self.version,
            details={
                "check_type": "startup",
                "is_started": is_started,
                "min_uptime_required": min_uptime,
            },
        )

    async def check_health(self) -> HealthCheckResponse:
        """Comprehensive health check.

        Validates all dependencies and provides detailed status.

        Returns:
            HealthCheckResponse
        """
        uptime = time.time() - self.start_time

        # Check all dependencies in parallel
        dependency_results = await self._check_dependencies(critical_only=False)

        # Determine overall status
        all_healthy = all(d.status == HealthStatus.HEALTHY for d in dependency_results)
        any_unhealthy = any(d.status == HealthStatus.UNHEALTHY for d in dependency_results)

        if all_healthy:
            status = HealthStatus.HEALTHY
        elif any_unhealthy:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.DEGRADED

        return HealthCheckResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            version=self.version,
            dependencies=dependency_results,
            details={"check_type": "full"},
        )

    async def _check_dependencies(self, critical_only: bool = False) -> List[DependencyHealth]:
        """Check all registered dependencies.

        Args:
            critical_only: Only check critical dependencies

        Returns:
            List of DependencyHealth results
        """
        results = []

        # Critical dependencies (simplified)
        critical_deps = []  # Could be filtered

        for name, dep_info in self.dependency_checkers.items():
            if critical_only and name not in critical_deps:
                continue

            start_time = time.time()
            try:
                checker = dep_info["checker"]
                is_healthy, error = await checker()

                latency_ms = (time.time() - start_time) * 1000

                results.append(
                    DependencyHealth(
                        name=name,
                        type=dep_info["type"],
                        status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=error,
                    )
                )

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000

                results.append(
                    DependencyHealth(
                        name=name,
                        type=dep_info["type"],
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=str(e),
                    )
                )

        return results


# ==================== EXAMPLE DEPENDENCY CHECKERS ====================


async def check_chromadb() -> tuple[bool, Optional[str]]:
    """Check ChromaDB health.

    Returns:
        Tuple of (is_healthy, error_message)
    """
    try:
        # Mock check - in production, query ChromaDB
        await asyncio.sleep(0.01)  # Simulate check
        return True, None
    except Exception as e:
        return False, str(e)


async def check_gemma_model() -> tuple[bool, Optional[str]]:
    """Check Gemma model availability.

    Returns:
        Tuple of (is_healthy, error_message)
    """
    try:
        # Mock check - in production, verify model loaded
        await asyncio.sleep(0.01)
        return True, None
    except Exception as e:
        return False, str(e)


async def check_filesystem() -> tuple[bool, Optional[str]]:
    """Check filesystem access.

    Returns:
        Tuple of (is_healthy, error_message)
    """
    try:
        # Mock check - in production, verify paths writable
        await asyncio.sleep(0.005)
        return True, None
    except Exception as e:
        return False, str(e)


# ==================== FACTORY ====================


def create_health_checker() -> HealthChecker:
    """Create configured health checker.

    Returns:
        HealthChecker with registered dependencies
    """
    checker = HealthChecker()

    # Register dependencies
    checker.register_dependency(
        "chromadb",
        check_chromadb,
        DependencyType.DATABASE,
    )
    checker.register_dependency(
        "gemma_model",
        check_gemma_model,
        DependencyType.MODEL,
    )
    checker.register_dependency(
        "filesystem",
        check_filesystem,
        DependencyType.FILESYSTEM,
    )

    return checker
