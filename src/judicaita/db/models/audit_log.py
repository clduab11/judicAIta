"""
Audit Log model for compliance tracking.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from judicaita.db.base import Base


class AuditLog(Base):
    """
    Audit Log model for compliance and activity tracking.

    Records all significant actions for transparency and regulatory compliance.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB().with_variant(Text, "sqlite"), nullable=True
    )
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="success", index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action={self.action}, timestamp={self.timestamp})>"

    @classmethod
    def create_log(
        cls,
        action: str,
        user_id: int | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        status: str = "success",
        error_message: str | None = None,
        duration_ms: int | None = None,
    ) -> "AuditLog":
        """
        Factory method to create an audit log entry.

        Args:
            action: The action being logged
            user_id: ID of the user performing the action
            resource_type: Type of resource affected
            resource_id: ID of the resource affected
            details: Additional details as JSON
            ip_address: Client IP address
            user_agent: Client user agent string
            status: Status of the action (success, error, etc.)
            error_message: Error message if status is error
            duration_ms: Duration of the action in milliseconds

        Returns:
            New AuditLog instance
        """
        return cls(
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            error_message=error_message,
            duration_ms=duration_ms,
        )
