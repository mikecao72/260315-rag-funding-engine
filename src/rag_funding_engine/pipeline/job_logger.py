"""Per-job logging for the recommendation pipeline.

Creates a log folder per job under logs/YYMMDD/YYMMDD-HHMMSS/job.log
with a full step-by-step trace of the pipeline execution.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class JobLogger:
    """Writes a human-readable trace of a single recommendation job."""

    def __init__(self, base_dir: Path | None = None):
        now = datetime.now()
        self._start_time = time.monotonic()
        self._timestamp = now

        date_str = now.strftime("%y%m%d")
        job_id = now.strftime("%y%m%d-%H%M%S")
        self.job_id = job_id

        if base_dir is None:
            # Default: project_root/logs/
            root = Path(__file__).resolve().parents[3]
            base_dir = root / "logs"

        self._job_dir = base_dir / date_str / job_id
        self._job_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._job_dir / "job.log"

        # Open the log file
        self._file = open(self._log_path, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        self._write("=" * 80)
        self._write(f"JOB: {self.job_id}")
        self._write(f"Timestamp: {self._timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write("=" * 80)
        self._write("")

    def step(self, step_num: int, title: str):
        """Write a step header."""
        label = f"STEP {step_num}: {title}"
        self._write(f"── {label} {'─' * max(1, 76 - len(label))}")
        self._write("")

    def log(self, message: str):
        """Write a plain text line."""
        self._write(message)

    def log_kv(self, key: str, value: Any):
        """Write a key-value pair."""
        self._write(f"  {key}: {value}")

    def log_json(self, label: str, data: Any, max_length: int = 0):
        """Write a JSON-formatted block."""
        formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        if max_length and len(formatted) > max_length:
            formatted = formatted[:max_length] + f"\n  ... (truncated, {len(json.dumps(data, ensure_ascii=False, default=str))} chars total)"
        self._write(f"{label}:")
        for line in formatted.split("\n"):
            self._write(f"  {line}")
        self._write("")

    def log_table(self, label: str, rows: list[dict[str, Any]], columns: list[str]):
        """Write a formatted table."""
        self._write(f"{label}:")
        if not rows:
            self._write("  (empty)")
            self._write("")
            return

        # Calculate column widths
        widths = {col: len(col) for col in columns}
        str_rows = []
        for row in rows:
            str_row = {}
            for col in columns:
                val = str(row.get(col, ""))
                if len(val) > 80:
                    val = val[:77] + "..."
                str_row[col] = val
                widths[col] = max(widths[col], len(val))
            str_rows.append(str_row)

        # Header
        header = " | ".join(col.ljust(widths[col]) for col in columns)
        self._write(f"  {header}")
        self._write(f"  {' | '.join('-' * widths[col] for col in columns)}")

        # Rows
        for sr in str_rows:
            line = " | ".join(sr[col].ljust(widths[col]) for col in columns)
            self._write(f"  {line}")
        self._write("")

    def log_prompt(self, label: str, prompt: str):
        """Write an LLM prompt block."""
        self._write(f"{label}:")
        self._write("  ┌" + "─" * 76 + "┐")
        for line in prompt.split("\n"):
            truncated = line[:74] if len(line) > 74 else line
            self._write(f"  │ {truncated:<74s} │")
        self._write("  └" + "─" * 76 + "┘")
        self._write("")

    def elapsed(self) -> float:
        """Seconds since job started."""
        return round(time.monotonic() - self._start_time, 2)

    def save_job_data(self, input_data: dict[str, Any], response: dict[str, Any]):
        """Save the job input and output as job.json for later review."""
        job_data = {
            "job_id": self.job_id,
            "timestamp": self._timestamp.isoformat(),
            "duration_seconds": self.elapsed(),
            "input": input_data,
            "output": response,
        }
        data_path = self._job_dir / "job.json"
        data_path.write_text(json.dumps(job_data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def finish(self):
        """Write the footer and close the log file."""
        self._write("")
        self._write("=" * 80)
        self._write(f"JOB COMPLETE: {self.job_id} | Duration: {self.elapsed()}s")
        self._write(f"Log: {self._log_path}")
        self._write("=" * 80)
        self._file.close()

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def job_dir(self) -> Path:
        return self._job_dir

    def _write(self, line: str):
        self._file.write(line + "\n")
        self._file.flush()
