"""
Streaming utilities for SSE and WebSocket support.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any


def format_sse_event(event: str, data: Any) -> str:
    """
    Format data as a Server-Sent Event (SSE).

    Args:
        event: Event type name
        data: Event data (will be JSON serialized)

    Returns:
        Formatted SSE string with event and data fields
    """
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = str(data)

    return f"event: {event}\ndata: {data_str}\n\n"


def format_sse_data(data: Any) -> str:
    """
    Format data as an SSE data-only event.

    Args:
        data: Event data (will be JSON serialized)

    Returns:
        Formatted SSE string with data field only
    """
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = str(data)

    return f"data: {data_str}\n\n"


def format_sse_comment(comment: str) -> str:
    """
    Format a comment in SSE format (used for keep-alive).

    Args:
        comment: Comment text

    Returns:
        Formatted SSE comment
    """
    return f": {comment}\n\n"


async def heartbeat_generator(
    interval_seconds: float = 15.0,
) -> AsyncGenerator[str, None]:
    """
    Generate periodic heartbeat comments for SSE keep-alive.

    Args:
        interval_seconds: Interval between heartbeats

    Yields:
        SSE comment strings
    """
    import asyncio

    while True:
        await asyncio.sleep(interval_seconds)
        yield format_sse_comment("heartbeat")


class SSEStream:
    """
    Helper class for building SSE streams.

    Example:
        async def generate():
            stream = SSEStream()
            yield stream.start("generation")
            for i in range(10):
                yield stream.event("progress", {"step": i})
            yield stream.complete({"result": "done"})
    """

    def __init__(self) -> None:
        """Initialize the SSE stream helper."""
        self._event_count = 0

    def event(self, event_type: str, data: Any) -> str:
        """
        Create an SSE event.

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            Formatted SSE string
        """
        self._event_count += 1
        return format_sse_event(event_type, data)

    def data(self, data: Any) -> str:
        """
        Create a data-only SSE event.

        Args:
            data: Event data

        Returns:
            Formatted SSE string
        """
        self._event_count += 1
        return format_sse_data(data)

    def start(self, operation: str) -> str:
        """
        Create a start event.

        Args:
            operation: Name of the operation starting

        Returns:
            Formatted SSE start event
        """
        return self.event("start", {"operation": operation, "status": "started"})

    def progress(self, current: int, total: int, message: str = "") -> str:
        """
        Create a progress event.

        Args:
            current: Current progress value
            total: Total value
            message: Optional progress message

        Returns:
            Formatted SSE progress event
        """
        return self.event(
            "progress",
            {
                "current": current,
                "total": total,
                "percentage": round(current / total * 100, 1) if total > 0 else 0,
                "message": message,
            },
        )

    def complete(self, result: Any = None) -> str:
        """
        Create a completion event.

        Args:
            result: Optional result data

        Returns:
            Formatted SSE complete event
        """
        return self.event(
            "complete",
            {"status": "completed", "events_sent": self._event_count, "result": result},
        )

    def error(self, message: str, details: dict | None = None) -> str:
        """
        Create an error event.

        Args:
            message: Error message
            details: Optional error details

        Returns:
            Formatted SSE error event
        """
        return self.event(
            "error",
            {"status": "error", "message": message, "details": details or {}},
        )

    def heartbeat(self) -> str:
        """
        Create a heartbeat comment.

        Returns:
            Formatted SSE comment for keep-alive
        """
        return format_sse_comment("heartbeat")
