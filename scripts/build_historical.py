"""scripts/build_historical.py — One-time historical DB setup."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.chunk4 import chunk_directory
from rag.vector import build_collection

if __name__ == "__main__":
    print("Chunking historical folder...")
    chunks = chunk_directory("./data/RFP collection")
    if not chunks:
        print("ERROR: No chunks found.")
    else:
        print(f"Extracted {len(chunks)} chunks. Building DB...")
        build_collection("historical_rfps", "./chroma_data/historical_rfps", chunks)
        print("SUCCESS! Historical database is ready.")
