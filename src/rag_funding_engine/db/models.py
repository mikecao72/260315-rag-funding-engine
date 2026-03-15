from __future__ import annotations

from pathlib import Path
from sqlalchemy import String, Float, Integer, Date, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ScheduleVersion(Base):
    __tablename__ = "schedule_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schedule_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effective_date: Mapped[Date | None] = mapped_column(Date, nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class ScheduleCode(Base):
    """Generic schedule code - works for any fee schedule."""
    __tablename__ = "schedule_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schedule_id: Mapped[str] = mapped_column(String(64), index=True)
    code: Mapped[str] = mapped_column(String(32), index=True)
    description: Mapped[str] = mapped_column(Text)
    fee_excl_gst: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee_incl_gst: Mapped[float | None] = mapped_column(Float, nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BillingRule(Base):
    __tablename__ = "billing_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schedule_id: Mapped[str] = mapped_column(String(64), index=True)
    code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rule_type: Mapped[str] = mapped_column(String(64), index=True)
    rule_text: Mapped[str] = mapped_column(Text)
    related_code: Mapped[str | None] = mapped_column(String(32), nullable=True)


def get_db_path(schedule_id: str, base_dir: Path | None = None) -> Path:
    """Get database path for a given schedule_id.
    
    Structure: data/processed/{schedule_id}/{schedule_id}.sqlite3
    """
    if base_dir is None:
        # Find project root
        import rag_funding_engine
        pkg_dir = Path(rag_funding_engine.__file__).resolve().parent
        base_dir = pkg_dir.parents[1] / "data" / "processed"
    
    return base_dir / schedule_id / f"{schedule_id}.sqlite3"
