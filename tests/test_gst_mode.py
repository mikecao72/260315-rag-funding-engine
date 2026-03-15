"""Tests for gst_mode parameter in recommend endpoint."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rag_funding_engine.pipeline.recommend import recommend_codes


class TestGstMode:
    """Tests for gst_mode parameter."""

    def test_recommendation_includes_primary_fee_excl_mode(self):
        """Test that gst_mode='excl' returns fee_excl_gst as primary fee."""
        # Mock the database and related functions
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            "code": "GP1",
            "description": "GP Consultation Standard",
            "fee_excl_gst": 45.00,
            "fee_incl_gst": 51.75,
            "page": 5,
        }.get(key)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            schedule_id = "test-schedule"
            
            # Create schedule directory structure
            schedule_dir = out_dir / schedule_id
            schedule_dir.mkdir(parents=True)
            
            # Create manifest
            manifest = {
                "schedule_id": schedule_id,
                "profile": {
                    "schedule_type": "medical",
                    "key_dimensions": ["procedure", "complexity"],
                }
            }
            (schedule_dir / "manifest.json").write_text(json.dumps(manifest))
            
            # Create mock DB
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
            
            # Test with gst_mode="excl"
            result = recommend_codes(
                schedule_id=schedule_id,
                consult_text="GP consultation standard",
                consult_template=None,
                base_dir=out_dir,
                top_n=5,
                gst_mode="excl"
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
            
            # Verify gst_mode is returned in response
            assert result["gst_mode"] == "excl"

    def test_recommendation_includes_primary_fee_incl_mode(self):
        """Test that gst_mode='incl' returns fee_incl_gst as primary fee."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            schedule_id = "test-schedule"
            
            # Create schedule directory structure
            schedule_dir = out_dir / schedule_id
            schedule_dir.mkdir(parents=True)
            
            # Create manifest
            manifest = {
                "schedule_id": schedule_id,
                "profile": {
                    "schedule_type": "medical",
                    "key_dimensions": ["procedure", "complexity"],
                }
            }
            (schedule_dir / "manifest.json").write_text(json.dumps(manifest))
            
            # Create mock DB
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
            
            # Test with gst_mode="incl"
            result = recommend_codes(
                schedule_id=schedule_id,
                consult_text="GP consultation standard",
                consult_template=None,
                base_dir=out_dir,
                top_n=5,
                gst_mode="incl"
            )
            
            assert "recommendations" in result
            assert len(result["recommendations"]) > 0
            
            rec = result["recommendations"][0]
            
            # In incl mode, fee should be fee_incl_gst
            assert rec["fee"] == rec["fee_incl_gst"]
            assert rec["fee_gst"] == rec["fee_excl_gst"]
            assert rec["fee_excl_gst"] == 45.00
            assert rec["fee_incl_gst"] == 51.75
            
            # Verify gst_mode is returned in response
            assert result["gst_mode"] == "incl"

    def test_default_gst_mode_is_excl(self):
        """Test that default gst_mode is 'excl'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            schedule_id = "test-schedule"
            
            # Create schedule directory structure
            schedule_dir = out_dir / schedule_id
            schedule_dir.mkdir(parents=True)
            
            # Create manifest
            manifest = {
                "schedule_id": schedule_id,
                "profile": {
                    "schedule_type": "medical",
                    "key_dimensions": ["procedure", "complexity"],
                }
            }
            (schedule_dir / "manifest.json").write_text(json.dumps(manifest))
            
            # Create mock DB
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
            
            # Test without specifying gst_mode (should default to "excl")
            result = recommend_codes(
                schedule_id=schedule_id,
                consult_text="GP consultation standard",
                consult_template=None,
                base_dir=out_dir,
                top_n=5
            )
            
            assert result["gst_mode"] == "excl"
            
            rec = result["recommendations"][0]
            # Default should be excl mode
            assert rec["fee"] == rec["fee_excl_gst"]
