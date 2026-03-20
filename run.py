"""
Start the RAG Funding Engine server.

Usage:
    python run.py              # starts on http://localhost:8010
    python run.py --port 9000  # custom port
"""
import argparse
import sys
from pathlib import Path

# Ensure the src/ directory is importable without pip install or PYTHONPATH tricks.
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="RAG Funding Engine server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8010, help="Port (default: 8010)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"Starting RAG Funding Engine on http://localhost:{args.port}")
    uvicorn.run(
        "rag_funding_engine.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
