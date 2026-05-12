"""Tests for the DBMetadataHints model.

Phase A captures these structural hints at extraction time; Phase B
consumes them (FK as relationship prior, PK as key-candidate hint).
This test pins the schema + ORM contract.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dataraum.storage.models import DBMetadataHints, Source


def _make_source(session: Session, name: str = "erp") -> Source:
    src = Source(name=name, source_type="db_recipe", connection_config={})
    session.add(src)
    session.flush()
    return src


def test_default_empty_lists(session):
    src = _make_source(session)
    hints = DBMetadataHints(source_id=src.source_id)
    session.add(hints)
    session.flush()

    fetched = session.execute(
        select(DBMetadataHints).where(DBMetadataHints.source_id == src.source_id)
    ).scalar_one()
    assert fetched.primary_keys == []
    assert fetched.foreign_keys == []
    assert fetched.indexes == []
    assert fetched.check_constraints == []
    assert fetched.harvested_at is not None


def test_roundtrip_real_payloads(session):
    src = _make_source(session)
    hints = DBMetadataHints(
        source_id=src.source_id,
        primary_keys=[{"table": "Invoices", "columns": ["invoice_id"]}],
        foreign_keys=[
            {
                "from_table": "Orders",
                "from_columns": ["customer_id"],
                "to_table": "Customers",
                "to_columns": ["customer_id"],
                "constraint_name": "fk_orders_customers",
            }
        ],
        indexes=[
            {
                "table": "Invoices",
                "name": "ix_invoices_date",
                "columns": ["invoice_date"],
                "unique": False,
            }
        ],
        check_constraints=[
            {
                "table": "Invoices",
                "name": "ck_total_positive",
                "expression": "total > 0",
            }
        ],
    )
    session.add(hints)
    session.flush()
    session.expire_all()

    fetched = session.execute(
        select(DBMetadataHints).where(DBMetadataHints.source_id == src.source_id)
    ).scalar_one()
    assert fetched.primary_keys[0]["columns"] == ["invoice_id"]
    assert fetched.foreign_keys[0]["to_table"] == "Customers"
    assert fetched.indexes[0]["unique"] is False
    assert fetched.check_constraints[0]["expression"] == "total > 0"


def test_unique_per_source(session):
    src = _make_source(session)
    session.add(DBMetadataHints(source_id=src.source_id))
    session.flush()
    session.add(DBMetadataHints(source_id=src.source_id))
    with pytest.raises(IntegrityError):
        session.flush()


def test_cascade_delete_on_source_removal(session):
    src = _make_source(session)
    session.add(DBMetadataHints(source_id=src.source_id))
    session.flush()
    session.delete(src)
    session.flush()

    remaining = session.execute(select(DBMetadataHints)).scalars().all()
    assert remaining == []
