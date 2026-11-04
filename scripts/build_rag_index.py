"""Build vector index for Mergen RAG using multilingual-e5-small and sqlite-vec.

Database: data/rag.db
Schema: id INTEGER PRIMARY KEY, text TEXT, source TEXT, embedding BLOB
Virtual Table: vec_chunks (id INTEGER PRIMARY KEY, embedding float[384])
Chunk Size: 500 chars
Overlap: 50 chars
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import sqlite_vec
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import torch_directml
    HAS_DIRECTML = torch_directml.is_available()
except Exception:
    HAS_DIRECTML = False


def extract_chunks(corpus_dir: Path, chunk_size: int = 500, overlap: int = 50):
    stride = chunk_size - overlap
    
    # Priority folders to ensure core documents (SE Handbook, OpenLib, Papers We Love, TR Docs) come first
    priority_order = [
        "se_handbook",
        "openlib_cs",
        "papers_we_love",
        "tr_docs",
        "ostep",
        "lecture_notes",
        "algorithms",
        "theory",
        "systems",
        "databases",
        "compilers",
    ]

    all_files = list(corpus_dir.rglob("*.md")) + list(corpus_dir.rglob("*.txt"))

    def file_sort_key(f: Path):
        rel_str = f.relative_to(corpus_dir).as_posix()
        is_thesis = 1 if "theses" in rel_str.lower() else 0
        is_wiki = 0 if "wiki_" in f.name.lower() else 1
        for idx, p in enumerate(priority_order):
            if p in rel_str:
                return (idx, is_thesis, is_wiki, rel_str)
        return (len(priority_order), is_thesis, is_wiki, rel_str)

    sorted_files = sorted(all_files, key=file_sort_key)
    chunks = []

    print(f"Reading corpus files from {corpus_dir.resolve()} ({len(sorted_files)} files)...")
    for f in sorted_files:
        try:
            rel_path = f.relative_to(corpus_dir).as_posix()
        except ValueError:
            rel_path = f.name

        try:
            content = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue

            for i in range(0, len(content), stride):
                chunk_text = content[i : i + chunk_size].strip()
                if len(chunk_text) >= 50:
                    chunks.append((chunk_text, rel_path))
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    return chunks


def init_db(db_path: Path):
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Required table schema
    db.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        embedding BLOB NOT NULL
    );
    """)

    # sqlite-vec virtual table for accelerated KNN search
    db.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
        id INTEGER PRIMARY KEY,
        embedding float[384]
    );
    """)

    db.commit()
    return db


def main():
    parser = argparse.ArgumentParser(description="Build RAG vector index")
    parser.add_argument("--corpus-dir", type=str, default="data/rag_corpus", help="Directory with corpus files")
    parser.add_argument("--db-path", type=str, default="data/rag.db", help="Path to SQLite database")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embeddings")
    parser.add_argument("--max-chunks", type=int, default=15000, help="Optional limit for chunk count (default: 15,000 high-priority chunks)")
    parser.add_argument("--all-chunks", action="store_true", help="Index all chunks in the corpus without limit")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, dml, or directml:0)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Mergen-SNN — RAG Vector Index Builder")
    print(f"Corpus: {corpus_dir.resolve()}")
    print(f"Database: {db_path.resolve()}")
    print("==========================================")

    # 1. Chunk documents
    chunks = extract_chunks(corpus_dir, chunk_size=500, overlap=50)
    total_found = len(chunks)
    print(f"Total chunks extracted: {total_found:,}")

    if not args.all_chunks and args.max_chunks and args.max_chunks < total_found:
        chunks = chunks[: args.max_chunks]
        print(f"Selected first {len(chunks):,} high-priority chunks for index.")
    else:
        print(f"Indexing all {len(chunks):,} chunks.")

    # 2. Setup Device and Model
    if args.device:
        device = args.device
    elif HAS_DIRECTML:
        device = str(torch_directml.device())
    else:
        device = "cpu"
        torch.set_num_threads(8)

    print(f"Loading embedding model 'intfloat/multilingual-e5-small' on device: {device}...")
    model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
    model.max_seq_length = 192

    # 3. Initialize SQLite DB
    db = init_db(db_path)

    # 4. Batch encode and insert
    batch_size = args.batch_size
    total_indexed = 0
    t0 = time.time()

    print(f"Indexing {len(chunks):,} chunks (batch_size={batch_size})...")
    with tqdm(total=len(chunks), desc="Indexing chunks", unit="chunk") as pbar:
        for idx in range(0, len(chunks), batch_size):
            batch = chunks[idx : idx + batch_size]
            passages = ["passage: " + item[0] for item in batch]

            embs = model.encode(passages, batch_size=batch_size, normalize_embeddings=True)
            embs_np = np.asarray(embs, dtype=np.float32)

            cursor = db.cursor()
            for (text, src), emb in zip(batch, embs_np):
                emb_bytes = emb.tobytes()
                cursor.execute(
                    "INSERT INTO chunks(text, source, embedding) VALUES (?, ?, ?)",
                    (text, src, emb_bytes),
                )
                row_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO vec_chunks(id, embedding) VALUES (?, ?)",
                    (row_id, emb_bytes),
                )

            db.commit()
            total_indexed += len(batch)
            pbar.update(len(batch))

    elapsed = time.time() - t0
    db_size = db_path.stat().st_size / (1024 * 1024)

    print("\n=== Indexing Tamamlandı ===")
    print(f"İndekslenen Chunk: {total_indexed:,}")
    print(f"Geçen Süre: {elapsed:.2f} saniye ({total_indexed / max(1e-5, elapsed):.1f} chunks/sec)")
    print(f"Veritabanı Boyutu: {db_size:.2f} MB")
    print(f"Hedef Dosya: {db_path.resolve()}")
    print("==========================================")


if __name__ == "__main__":
    main()
