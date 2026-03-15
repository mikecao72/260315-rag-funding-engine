from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception:
        return None


def embed_texts(texts: Iterable[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    client = _get_openai_client()
    texts = list(texts)
    if not texts:
        return []

    if client is None:
        # Deterministic fallback pseudo-embedding for offline development.
        out = []
        for t in texts:
            vec = np.zeros(128, dtype=float)
            for token in t.lower().split():
                vec[hash(token) % 128] += 1.0
            n = np.linalg.norm(vec)
            out.append((vec / n).tolist() if n else vec.tolist())
        return out

    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=float)
    bv = np.array(b, dtype=float)

    # Be tolerant to mixed embedding sources (e.g. cached 1536-d OpenAI vectors
    # with local 128-d fallback vectors during offline development).
    if av.size != bv.size:
        min_len = min(av.size, bv.size)
        if min_len == 0:
            return 0.0
        av = av[:min_len]
        bv = bv[:min_len]

    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def ensure_embedding_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_chunk_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id TEXT,
            chunk_id INTEGER,
            embedding_json TEXT,
            UNIQUE(schedule_id, chunk_id)
        )
        """
    )
    conn.commit()


def index_policy_chunks(db_path: str, schedule_id: str, model: str = "text-embedding-3-small") -> int:
    """
    Generate embeddings for policy chunks in the given schedule database.
    
    Args:
        db_path: Path to SQLite database
        schedule_id: Schedule identifier
        model: OpenAI embedding model to use
        
    Returns:
        Number of chunks indexed
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_embedding_table(conn)

    cur = conn.cursor()
    cur.execute(
        "SELECT id, chunk_text FROM policy_chunks WHERE schedule_id = ?",
        (schedule_id,),
    )
    rows = cur.fetchall()
    texts = [r["chunk_text"] for r in rows]
    embeddings = embed_texts(texts, model=model)

    cur.execute("DELETE FROM policy_chunk_embeddings WHERE schedule_id = ?", (schedule_id,))
    for r, emb in zip(rows, embeddings):
        cur.execute(
            "INSERT INTO policy_chunk_embeddings(schedule_id, chunk_id, embedding_json) VALUES (?, ?, ?)",
            (schedule_id, r["id"], json.dumps(emb)),
        )

    conn.commit()
    conn.close()
    return len(rows)


def get_vector_store_path(schedule_id: str, base_dir: Path | None = None) -> Path:
    """Get the path to the vector store/embeddings directory for a schedule."""
    if base_dir is None:
        # Default to data/processed/{schedule_id}/embeddings/
        from rag_funding_engine.api.main import ROOT
        base_dir = ROOT / "data" / "processed"
    return base_dir / schedule_id / "embeddings"
