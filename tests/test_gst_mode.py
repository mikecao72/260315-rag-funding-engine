"""Tests for gst_mode parameter in recommend endpoint."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_funding_engine.pipeline.recommend import recommend_codes


def _create_test_db(tmpdir: str, schedule_id: str = "test-schedule"):
    """Create a minimal test schedule directory with DB and manifest."""
    out_dir = Path(tmpdir)
    schedule_dir = out_dir / schedule_id
    schedule_dir.mkdir(parents=True)

    manifest = {
        "schedule_id": schedule_id,
        "profile": {
            "schedule_type": "medical",
            "key_dimensions": ["procedure", "complexity"],
        },
        "schedule_guidance": {
            "document_scope": "Test schedule",
            "code_groupings": [],
            "selection_rules": [],
        },
    }
    (schedule_dir / "manifest.json").write_text(json.dumps(manifest))

    import sqlite3
    db_path = schedule_dir / f"{schedule_id}.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE schedule_codes (
            code TEXT, description TEXT, fee_excl_gst REAL,
            fee_incl_gst REAL, page INTEGER, schedule_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE policy_chunks (
            id INTEGER PRIMARY KEY, schedule_id TEXT, page INTEGER, chunk_text TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE policy_chunk_embeddings (
            chunk_id INTEGER, schedule_id TEXT, embedding_json TEXT
        )
    """)
    conn.execute(
        "INSERT INTO schedule_codes VALUES (?, ?, ?, ?, ?, ?)",
        ("GP1", "GP Consultation Standard", 45.00, 51.75, 5, schedule_id)
    )
    conn.commit()
    conn.close()

    return out_dir, schedule_id


# Mock LLM response that selects GP1
MOCK_LLM_SELECTION = [{"code": "GP1", "confidence": 0.9, "reason": "Standard GP consultation"}]


class TestGstMode:
    """Tests for gst_mode parameter."""

    @patch("rag_funding_engine.pipeline.recommend._llm_select_codes", return_value=MOCK_LLM_SELECTION)
    @patch("rag_funding_engine.pipeline.recommend._extract_consult_facts", return_value={"mode": "test", "facts": {}})
    def test_recommendation_includes_primary_fee_excl_mode(self, mock_facts, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, schedule_id = _create_test_db(tmpdir)

            result = recommend_codes(
                schedule_id=schedule_id,
                consult_text="GP consultation standard",
                consult_template=None,
                base_dir=out_dir,
                top_n=5,
                gst_mode="excl",
            )

            assert "recommendations" in result
            assert len(result["recommendations"]) > 0

            rec = result["recommendations"][0]
            assert "fee" in rec
            assert "fee_gst" in rec
            assert "fee_excl_gst" in rec
            assert "fee_incl_gst" in rec

            # In excl mode, fee should be fee_excl_gst
            assert rec["fee"] == rec["fee_excl_gst"]
            assert rec["fee_gst"] == rec["fee_incl_gst"]
            assert rec["fee_excl_gst"] == 45.00
            assert rec["fee_incl_gst"] == 51.75

            assert result["gst_mode"] == "excl"

    @patch("rag_funding_engine.pipeline.recommend._llm_select_codes", return_value=MOCK_LLM_SELECTION)
    @patch("rag_funding_engine.pipeline.recommend._extract_consult_facts", return_value={"mode": "test", "facts": {}})
    def test_recommendation_includes_primary_fee_incl_mode(self, mock_facts, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, schedule_id = _create_test_db(tmpdir)

            result = recommend_codes(
                schedule_id=schedule_id,
                consult_text="GP consultation standard",
                consult_template=None,
                base_dir=out_dir,
                top_n=5,
                gst_mode="incl",
            )

            assert "recommendations" in result
            assert len(result["recommendations"]) > 0

            rec = result["recommendations"][0]

            # In incl mode, fee should be fee_incl_gst
            assert rec["fee"] == rec["fee_incl_gst"]
            assert rec["fee_gst"] == rec["fee_excl_gst"]
            assert rec["fee_excl_gst"] == 45.00
            assert rec["fee_incl_gst"] == 51.75

            assert result["gst_mode"] == "incl"

    @patch("rag_funding_engine.pipeline.recommend._llm_select_codes", return_value=MOCK_LLM_SELECTION)
    @patch("rag_funding_engine.pipeline.recommend._extract_consult_facts", return_value={"mode": "test", "facts": {}})
    def test_default_gst_mode_is_excl(self, mock_facts, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir, schedule_id = _create_test_db(tmpdir)

            result = recommend_codes(
                schedule_id=schedule_id,
                consult_text="GP consultation standard",
                consult_template=None,
                base_dir=out_dir,
                top_n=5,
            )

            assert result["gst_mode"] == "excl"

            rec = result["recommendations"][0]
            assert rec["fee"] == rec["fee_excl_gst"]
