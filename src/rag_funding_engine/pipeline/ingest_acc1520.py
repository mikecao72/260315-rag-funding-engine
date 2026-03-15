from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import sqlite3
from typing import Iterable

from pypdf import PdfReader
from rag_funding_engine.pipeline.semantic import index_policy_chunks


MONEY_RE = re.compile(r"\b\d{1,4}\.\d{2}\b")
CODE_LINE_RE = re.compile(r"^([A-Z]{2,6}\d{0,3})\s+(.+)$")


@dataclass
class IngestResult:
    source_path: str
    source_hash: str
    text_output_path: str
    manifest_path: str
    db_path: str
    code_count: int
    indexed_chunks: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append({"page": i, "text": page.extract_text() or ""})
    return pages


def _normalise_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _parse_code_rows(lines: Iterable[str], page_num: int) -> list[dict]:
    rows: list[dict] = []
    lines = list(lines)
    i = 0
    while i < len(lines):
        m = CODE_LINE_RE.match(lines[i])
        if not m:
            i += 1
            continue

        code = m.group(1)
        first_part = m.group(2)
        desc_parts = [first_part]
        j = i + 1
        prices: list[float] = [float(v) for v in MONEY_RE.findall(first_part)]

        while j < len(lines):
            # New code row starts.
            if CODE_LINE_RE.match(lines[j]):
                break

            money = [float(v) for v in MONEY_RE.findall(lines[j])]
            if money:
                prices.extend(money)
            else:
                # Most rows have wrapped descriptions before price line appears.
                # Keep short, non-header text as description continuation.
                lower = lines[j].lower()
                if not lower.startswith(("code ", "acc1520", "page ", "per unit", "per hour", "flat rate")):
                    desc_parts.append(lines[j])
            j += 1

        fee_excl = prices[0] if prices else None
        fee_incl = prices[1] if len(prices) > 1 else None

        description = " ".join(desc_parts).strip()
        description = re.sub(r"(?:\s+\d{1,4}\.\d{2}){1,4}\s*$", "", description).strip()

        rows.append(
            {
                "code": code,
                "description": description,
                "fee_excl_gst": fee_excl,
                "fee_incl_gst": fee_incl,
                "page": page_num,
            }
        )
        i = j
    return rows


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE,
            source_hash TEXT,
            source_path TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS acc_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_version TEXT,
            code TEXT,
            description TEXT,
            fee_excl_gst REAL,
            fee_incl_gst REAL,
            page INTEGER,
            UNIQUE(schedule_version, code)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_version TEXT,
            page INTEGER,
            chunk_text TEXT
        )
        """
    )
    conn.commit()
    return conn


def _chunk_text(text: str, max_words: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return []
    out = []
    for i in range(0, len(words), max_words):
        out.append(" ".join(words[i : i + max_words]))
    return out


def ingest_schedule(pdf_path: Path, out_dir: Path, schedule_version: str = "ACC1520-v2") -> IngestResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(pdf_path)
    pages = _read_pages(pdf_path)

    text_output = out_dir / "acc1520_text_pages.json"
    text_output.write_text(json.dumps(pages, ensure_ascii=False, indent=2))

    code_rows: list[dict] = []
    for page in pages:
        lines = _normalise_lines(page["text"])
        code_rows.extend(_parse_code_rows(lines, page_num=page["page"]))

    # Deduplicate by code, keep latest occurrence.
    dedup = {row["code"]: row for row in code_rows}
    code_rows = list(dedup.values())

    db_path = out_dir / "acc1520.sqlite3"
    conn = _init_db(db_path)
    cur = conn.cursor()

    cur.execute(
        "INSERT OR REPLACE INTO schedule_versions(version, source_hash, source_path) VALUES (?, ?, ?)",
        (schedule_version, source_hash, str(pdf_path)),
    )

    cur.execute("DELETE FROM acc_codes WHERE schedule_version = ?", (schedule_version,))
    for row in code_rows:
        cur.execute(
            """
            INSERT INTO acc_codes(schedule_version, code, description, fee_excl_gst, fee_incl_gst, page)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                schedule_version,
                row["code"],
                row["description"],
                row["fee_excl_gst"],
                row["fee_incl_gst"],
                row["page"],
            ),
        )

    cur.execute("DELETE FROM policy_chunks WHERE schedule_version = ?", (schedule_version,))
    for page in pages:
        for chunk in _chunk_text(page["text"], max_words=120):
            cur.execute(
                "INSERT INTO policy_chunks(schedule_version, page, chunk_text) VALUES (?, ?, ?)",
                (schedule_version, page["page"], chunk),
            )

    conn.commit()
    conn.close()

    indexed_chunks = index_policy_chunks(str(db_path), schedule_version=schedule_version)

    parsed_json = out_dir / "acc1520_codes.json"
    parsed_json.write_text(json.dumps(code_rows, ensure_ascii=False, indent=2))

    manifest = {
        "source": str(pdf_path),
        "source_hash": source_hash,
        "pages": len(pages),
        "codes_parsed": len(code_rows),
        "chunks_indexed": indexed_chunks,
        "artifacts": {
            "text_pages": str(text_output),
            "codes": str(parsed_json),
            "db": str(db_path),
        },
    }
    manifest_path = out_dir / "ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return IngestResult(
        source_path=str(pdf_path),
        source_hash=source_hash,
        text_output_path=str(text_output),
        manifest_path=str(manifest_path),
        db_path=str(db_path),
        code_count=len(code_rows),
        indexed_chunks=indexed_chunks,
    )
