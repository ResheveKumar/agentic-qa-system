"""
rag_engine.py
--------------
Local RAG pipeline: ingests source files into a LanceDB vector table using
sentence-transformers embeddings, and exposes semantic search over the
ingested codebase.

Standalone test:
    python rag_engine.py
"""

import os
import uuid
from typing import List, Dict, Optional

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb_store")
TABLE_NAME = "codebase_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
ALLOWED_EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "lancedb_store"}


class RAGEngine:
    """Handles ingestion and semantic retrieval for the local codebase knowledge base."""

    def __init__(self, db_path: str = DB_PATH, table_name: str = TABLE_NAME):
        self.db_path = db_path
        self.table_name = table_name
        self._model: Optional[SentenceTransformer] = None
        self._db = None
        self._table = None

        try:
            os.makedirs(self.db_path, exist_ok=True)
            self._db = lancedb.connect(self.db_path)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to LanceDB at '{self.db_path}': {e}")

        self._load_existing_table()

    def _load_existing_table(self):
        try:
            if self.table_name in self._db.table_names():
                self._table = self._db.open_table(self.table_name)
        except Exception:
            self._table = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        return self._model

    def is_ready(self) -> bool:
        """True if the vector table exists and has at least one row."""
        try:
            return self._table is not None and self._table.count_rows() > 0
        except Exception:
            return False

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start = end - overlap
        return chunks

    def _collect_files(self, root_dir: str) -> List[str]:
        matches = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            for fname in filenames:
                if os.path.splitext(fname)[1] in ALLOWED_EXTENSIONS:
                    matches.append(os.path.join(dirpath, fname))
        return matches

    def ingest_directory(self, root_dir: str) -> Dict:
        """Recursively ingest all supported source files under root_dir into LanceDB."""
        if not os.path.isdir(root_dir):
            return {"status": "error", "message": f"Directory not found: {root_dir}", "chunks_ingested": 0}

        files = self._collect_files(root_dir)
        if not files:
            return {
                "status": "error",
                "message": f"No supported files (.py, .md, .txt, .json, .yaml) found under '{root_dir}'",
                "chunks_ingested": 0,
            }

        records = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue

            for idx, chunk in enumerate(self._chunk_text(content)):
                if not chunk.strip():
                    continue
                records.append({
                    "id": str(uuid.uuid4()),
                    "path": file_path,
                    "chunk_id": idx,
                    "text": chunk,
                })

        if not records:
            return {"status": "error", "message": "No valid content chunks extracted.", "chunks_ingested": 0}

        try:
            texts = [r["text"] for r in records]
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            for r, emb in zip(records, embeddings):
                r["vector"] = emb.astype(np.float32).tolist()
        except Exception as e:
            return {"status": "error", "message": f"Embedding generation failed: {e}", "chunks_ingested": 0}

        try:
            if self.table_name in self._db.table_names():
                self._table = self._db.open_table(self.table_name)
                self._table.add(records)
            else:
                self._table = self._db.create_table(self.table_name, data=records)
        except Exception as e:
            return {"status": "error", "message": f"Failed to write to LanceDB: {e}", "chunks_ingested": 0}

        return {
            "status": "success",
            "message": f"Ingested {len(records)} chunks from {len(files)} files.",
            "chunks_ingested": len(records),
            "files_processed": len(files),
        }

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Semantic search over ingested code chunks. Returns a list of match dicts."""
        if not query or not query.strip():
            return []

        if self._table is None:
            self._load_existing_table()
        if self._table is None or self._table.count_rows() == 0:
            return []

        try:
            query_vector = self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32).tolist()
            results = self._table.search(query_vector).limit(k).to_list()
            return [
                {
                    "path": r.get("path", "unknown"),
                    "chunk_id": r.get("chunk_id", -1),
                    "text": r.get("text", ""),
                    "distance": r.get("_distance", None),
                }
                for r in results
            ]
        except Exception as e:
            return [{"path": "ERROR", "chunk_id": -1, "text": f"Search failed: {e}", "distance": None}]

    def clear(self):
        """Drop the existing vector table (useful before re-ingestion)."""
        try:
            if self.table_name in self._db.table_names():
                self._db.drop_table(self.table_name)
            self._table = None
        except Exception as e:
            raise RuntimeError(f"Failed to clear table: {e}")


if __name__ == "__main__":
    engine = RAGEngine()
    result = engine.ingest_directory(".")
    print(result)
    if engine.is_ready():
        hits = engine.search("error handling", k=3)
        for h in hits:
            print(h["path"], h["chunk_id"], h["distance"])
