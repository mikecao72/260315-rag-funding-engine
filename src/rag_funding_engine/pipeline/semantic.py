from __future__ import annotations

import json
import os
import sqlite3
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
            schedule_version TEXT,
            chunk_id INTEGER,
            embedding_json TEXT,
            UNIQUE(schedule_version, chunk_id)
        )
        """
    )
    conn.commit()


def index_policy_chunks(db_path: str, schedule_version: str = "ACC1520-v2", model: str = "text-embedding-3-small") -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_embedding_table(conn)

    cur = conn.cursor()
    cur.execute(
        "SELECT id, chunk_text FROM policy_chunks WHERE schedule_version = ?",
        (schedule_version,),
    )
    rows = cur.fetchall()
    texts = [r["chunk_text"] for r in rows]
    embeddings = embed_texts(texts, model=model)

    cur.execute("DELETE FROM policy_chunk_embeddings WHERE schedule_version = ?", (schedule_version,))
    for r, emb in zip(rows, embeddings):
        cur.execute(
            "INSERT INTO policy_chunk_embeddings(schedule_version, chunk_id, embedding_json) VALUES (?, ?, ?)",
            (schedule_version, r["id"], json.dumps(emb)),
        )

    conn.commit()
    conn.close()
    return len(rows)
