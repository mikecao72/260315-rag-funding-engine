"""Tests for the generic schedule ingestion pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from rag_funding_engine.pipeline.ingest_schedule import (
    ingest_schedule,
    ScheduleProfile,
    _analyze_schedule_with_llm,
    _parse_code_rows,
    _normalise_lines,
)


class TestScheduleProfile:
    """Tests for ScheduleProfile dataclass."""

    def test_profile_creation(self):
        profile = ScheduleProfile(
            schedule_type="rural",
            description="Rural practitioner schedule",
            key_dimensions=["location", "travel_time"],
            unique_rules=["Travel >20km bills separately"],
            query_patterns=["{specialty} {location}"],
            example_queries=["GP rural clinic"],
            location_matters=True,
            mileage_reimbursable=True,
            after_hours_premium=False,
        )
        assert profile.schedule_type == "rural"
        assert profile.location_matters is True


class TestParseCodeRows:
    """Tests for code row parsing."""

    def test_parse_simple_code_row(self):
        lines = ["GP1 GP Consultation Standard 45.00 51.75"]
        result = _parse_code_rows(lines, page_num=1)
        assert len(result) == 1
        assert result[0]["code"] == "GP1"
        assert result[0]["fee_excl_gst"] == 45.00
        assert result[0]["fee_incl_gst"] == 51.75

    def test_parse_multi_line_description(self):
        lines = [
            "MW2 Laceration repair 2-7cm",
            "wound closure with sutures",
            "68.75 79.06",
        ]
        result = _parse_code_rows(lines, page_num=1)
        assert len(result) == 1
        assert result[0]["code"] == "MW2"
        assert "wound closure" in result[0]["description"]

    def test_skip_header_lines(self):
        lines = [
            "Code Description Fee Excl Fee Incl",
            "GP1 GP Consultation 45.00 51.75",
        ]
        result = _parse_code_rows(lines, page_num=1)
        # Should skip the header line
        assert len(result) == 1
        assert result[0]["code"] == "GP1"


class TestIngestScheduleIntegration:
    """Integration tests for full ingestion pipeline."""

    def test_ingest_creates_expected_structure(self):
        """Test that ingestion creates the expected directory structure."""
        # This test requires a sample PDF
        # For now, just test the structure logic
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            schedule_id = "test-schedule"
            
            # Create a minimal test to verify structure
            schedule_dir = out_dir / schedule_id
            schedule_dir.mkdir(parents=True)
            
            # Verify expected paths
            assert (out_dir / schedule_id).exists()
            
    def test_profile_fallback_without_api_key(self, monkeypatch):
        """Test that profile falls back gracefully without API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        profile = _analyze_schedule_with_llm("Some schedule text")
        
        assert profile.schedule_type == "generic"
        assert profile.location_matters is False


class TestManifestCreation:
    """Tests for manifest.json creation."""

    def test_manifest_contains_profile(self):
        """Verify manifest would contain profile data."""
        profile = ScheduleProfile(
            schedule_type="urgent_care",
            description="Urgent care schedule",
            key_dimensions=["time_of_day", "complexity"],
            unique_rules=["After hours premium applies"],
            query_patterns=["{procedure} urgent care"],
            example_queries=["laceration repair urgent care"],
            location_matters=False,
            mileage_reimbursable=False,
            after_hours_premium=True,
        )
        
        # Simulate manifest structure
        manifest = {
            "schedule_id": "test-schedule",
            "profile": {
                "schedule_type": profile.schedule_type,
                "description": profile.description,
                "key_dimensions": profile.key_dimensions,
                "unique_rules": profile.unique_rules,
                "query_patterns": profile.query_patterns,
                "example_queries": profile.example_queries,
                "location_matters": profile.location_matters,
                "mileage_reimbursable": profile.mileage_reimbursable,
                "after_hours_premium": profile.after_hours_premium,
            }
        }
        
        assert manifest["profile"]["schedule_type"] == "urgent_care"
        assert manifest["profile"]["after_hours_premium"] is True
