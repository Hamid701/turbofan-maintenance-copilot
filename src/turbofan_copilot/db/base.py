"""Shared SQLAlchemy metadata for application database models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class whose metadata is the source of truth for migrations."""
