from sqlalchemy import String, Float, Integer, Date, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ScheduleVersion(Base):
    __tablename__ = "schedule_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    effective_date: Mapped[Date | None] = mapped_column(Date, nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class AccCode(Base):
    __tablename__ = "acc_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schedule_version: Mapped[str] = mapped_column(String(64), index=True)
    code: Mapped[str] = mapped_column(String(32), index=True)
    description: Mapped[str] = mapped_column(Text)
    fee: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class BillingRule(Base):
    __tablename__ = "billing_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    schedule_version: Mapped[str] = mapped_column(String(64), index=True)
    code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rule_type: Mapped[str] = mapped_column(String(64), index=True)
    rule_text: Mapped[str] = mapped_column(Text)
    related_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
