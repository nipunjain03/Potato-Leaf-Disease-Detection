"""
Vector RAG: Semantic document retrieval from agricultural texts and PDFs.
ChromaDB stores embeddings; queries return relevant chunks for context.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_STORE_PATH, DOCUMENTS_DIR

# Optional: use Chroma's default embedding. If chromadb is not installed, we fall back gracefully.
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorRAG:
    """
    Semantic retrieval over ingested documents.
    Ingestion: chunk documents from DOCUMENTS_DIR, embed, store in Chroma.
    Query: embed query, return top-k chunks as context string.
    """

    def __init__(self, persist_directory: str = None, collection_name: str = "potato_docs"):
        self.persist_directory = persist_directory or VECTOR_STORE_PATH
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._init_client()

    def _init_client(self):
        if not CHROMA_AVAILABLE:
            self.client = None
            return
        # Use Chroma's default configuration; no need for chromadb.config.Settings (simpler and avoids import issues)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Potato disease and agriculture documents"},
        )

    def add_documents(self, documents: list, ids: list = None, metadatas: list = None):
        """Add text chunks to the vector store. Uses Chroma's default embedding if available."""
        if not CHROMA_AVAILABLE or not self.collection:
            return
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{}] * len(documents)

        # Strict sanitization: Chroma tokenizer requires plain Python str only.
        safe_docs = []
        safe_ids = []
        safe_meta = []
        for i, doc in enumerate(documents):
            if doc is None:
                continue
            if not isinstance(doc, str):
                doc = str(doc)
            doc = doc.encode("utf-8", errors="ignore").decode("utf-8").strip()
            if not doc:
                continue
            safe_docs.append(doc)
            safe_ids.append(ids[i] if i < len(ids) else f"doc_{i}")
            safe_meta.append(metadatas[i] if i < len(metadatas) else {})

        if not safe_docs:
            return

        # Add in smaller batches so one bad record doesn't fail the entire ingest run.
        batch_size = 64
        for i in range(0, len(safe_docs), batch_size):
            batch_docs = safe_docs[i : i + batch_size]
            batch_ids = safe_ids[i : i + batch_size]
            batch_meta = safe_meta[i : i + batch_size]
            self.collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_meta)

    def query(self, query_text: str, top_k: int = 5) -> str:
        """
        Retrieve top-k relevant chunks and return as a single context string.
        If Chroma is not available, returns empty string (graceful fallback).
        """
        if not CHROMA_AVAILABLE or not self.collection:
            return ""
        try:
            results = self.collection.query(query_texts=[query_text], n_results=min(top_k, 20))
            if not results or not results.get("documents") or not results["documents"][0]:
                return ""
            chunks = results["documents"][0]
            return "\n\n".join(chunks)
        except Exception:
            return ""

    def query_with_sources(self, query_text: str, top_k: int = 5):
        """
        Retrieve top-k chunks and include source metadata for UI citations.
        Returns list of dicts: [{"source": "...", "excerpt": "..."}].
        """
        if not CHROMA_AVAILABLE or not self.collection:
            return []
        try:
            results = self.collection.query(query_texts=[query_text], n_results=min(top_k, 20))
            docs = results.get("documents", [[]])[0] if results else []
            metas = results.get("metadatas", [[]])[0] if results else []
            out = []
            for i, doc in enumerate(docs):
                if not isinstance(doc, str):
                    continue
                source = "Unknown"
                if i < len(metas) and isinstance(metas[i], dict):
                    source = metas[i].get("source", "Unknown")
                out.append({"source": source, "excerpt": doc.strip()[:300]})
            return out
        except Exception:
            return []

    def ingest_directory(self, directory: str = None, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Load .txt and .pdf files from directory, chunk text, and add to vector store.
        """
        """
        Load .txt and .pdf files from directory, chunk text, and add to vector store.
        """
        """
        Load .txt and .pdf files from directory, chunk text, and add to vector store.
        """
        """
        Load .txt and .pdf files from directory, chunk text, and add to vector store.
        """
        """
        Load .txt and .pdf files from directory, chunk text, and add to vector store.
        """
        directory = directory or DOCUMENTS_DIR
        if not os.path.isdir(directory):
            return 0
        texts = []
        for fname in os.listdir(directory):
            path = os.path.join(directory, fname)
            if fname.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append((fname, f.read()))
            elif fname.lower().endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(path)
                    content = " ".join(p.extract_text() or "" for p in reader.pages)
                    texts.append((fname, content))
                except Exception:
                    pass
        if not texts:
            return 0
        # Simple chunking
        all_chunks = []
        chunk_ids = []
        meta = []
        for fname, text in texts:
            # Guarantee plain str — pypdf can return None or bytes on scanned/corrupt pages
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size].strip()
                if not isinstance(chunk, str) or len(chunk) < 50:
                    continue
                all_chunks.append(chunk)
                chunk_ids.append(f"{fname}_{i}")
                meta.append({"source": fname})
        if all_chunks and CHROMA_AVAILABLE:
            self.add_documents(all_chunks, ids=chunk_ids, metadatas=meta)
        return len(all_chunks)
