"""
Ingest documents from data/documents into the Vector RAG (ChromaDB).
Run once after adding PDFs or .txt files to populate the vector store.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.vector_rag import VectorRAG
from config import DOCUMENTS_DIR

if __name__ == "__main__":
    rag = VectorRAG()
    n = rag.ingest_directory(DOCUMENTS_DIR)
    print(f"Ingested {n} chunks from {DOCUMENTS_DIR}")
