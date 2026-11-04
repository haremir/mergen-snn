"""Test RAG vector search using multilingual-e5-small and sqlite-vec.

Default query: "OOP kalıtım nedir?"
Returns: Top 3 chunks with similarity scores, sources, and text.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer


def search_rag(query: str, db_path: Path, top_k: int = 3, device: str = "cpu"):
    if not db_path.exists():
        print(f"Hata: Veritabanı bulunamadı: {db_path}")
        sys.exit(1)

    print(f"Sorgu: '{query}'")
    print(f"Veritabanı: {db_path.resolve()}")
    print("Embedding modeli yükleniyor ('intfloat/multilingual-e5-small')...")
    model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)

    # Encode query with e5 prefix
    query_text = "query: " + query
    query_emb = model.encode([query_text], normalize_embeddings=True)
    query_bytes = np.asarray(query_emb[0], dtype=np.float32).tobytes()

    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Search via vec0 virtual table
    sql = """
    SELECT c.id, c.text, c.source, v.distance
    FROM vec_chunks v
    JOIN chunks c ON c.id = v.id
    WHERE v.embedding MATCH ? AND k = ?
    ORDER BY v.distance
    """
    rows = db.execute(sql, (query_bytes, top_k)).fetchall()

    print("\n==========================================")
    print(f"RAG ARAMA SONUÇLARI (Top {top_k})")
    print(f"Sorgu: {query}")
    print("==========================================")

    for rank, (chunk_id, text, source, dist) in enumerate(rows, 1):
        # Convert cosine distance to cosine similarity
        sim = max(0.0, 1.0 - dist)
        print(f"\n--- Sonuç {rank} [Benzerlik: {sim:.4f} | Mesafe: {dist:.4f}] ---")
        print(f"ID: {chunk_id}")
        print(f"Kaynak: {source}")
        print("Metin:")
        print(text)
        print("-" * 50)

    db.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Test RAG search")
    parser.add_argument("--query", type=str, default="OOP kalıtım nedir?", help="Search query")
    parser.add_argument("--db-path", type=str, default="data/rag.db", help="Path to SQLite DB")
    parser.add_argument("--top-k", type=int, default=3, help="Top results count")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    search_rag(args.query, Path(args.db_path), top_k=args.top_k, device=args.device)


if __name__ == "__main__":
    main()
