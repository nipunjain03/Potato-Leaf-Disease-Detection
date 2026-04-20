"""
Disease Advisory Chatbot: combines Vector RAG + Graph RAG and injects image prediction context.
Answers are grounded in retrieved content to reduce hallucinations.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

from chatbot.ollama_client import OllamaClient
from rag.vector_rag import VectorRAG
from rag.graph_rag import GraphRAG


# System prompt to keep answers grounded and disease-focused
SYSTEM_PROMPT = """You are an agricultural assistant for potato leaf diseases. 
Answer only based on the provided context (image prediction result, knowledge graph facts, and retrieved documents). 
If the context does not contain enough information, say so. Do not invent symptoms, causes, or treatments.
Be concise and helpful. For prevention and treatment, stick to the given facts."""


class DiseaseAdvisoryChatbot:
    """
    Hybrid RAG chatbot: image prediction (disease + confidence) is injected as context;
    Vector RAG and Graph RAG provide additional grounding. All context is passed to Ollama.
    """

    def __init__(
        self,
        ollama_base_url: str = None,
        ollama_model: str = None,
        vector_rag: VectorRAG = None,
        graph_rag: GraphRAG = None,
    ):
        self.ollama = OllamaClient(ollama_base_url or OLLAMA_BASE_URL, ollama_model or OLLAMA_MODEL)
        self.vector_rag = vector_rag or VectorRAG()
        self.graph_rag = graph_rag or GraphRAG()

    def _build_context(self, disease_label: str = None, confidence: float = None, user_query: str = "") -> str:
        """Combine image result + Graph RAG + Vector RAG into one context block."""
        parts = []
        if disease_label is not None:
            pred_text = f"Image prediction: {disease_label}"
            if confidence is not None:
                pred_text += f" (confidence: {confidence:.2f})"
            parts.append(pred_text)
        graph_ctx = self.graph_rag.get_context_for_disease(disease_label or "")
        if graph_ctx:
            parts.append("Structured knowledge (symptoms, causes, treatment, prevention):\n" + graph_ctx)
        vector_ctx = self.vector_rag.query(user_query or disease_label or "potato leaf disease", top_k=5)
        if vector_ctx:
            parts.append("Relevant document excerpts:\n" + vector_ctx)
        return "\n\n".join(parts) if parts else "No additional context."

    def chat(
        self,
        user_message: str,
        disease_label: str = None,
        confidence: float = None,
    ) -> str:
        """
        Answer the user using image prediction (if any), Graph RAG, and Vector RAG context.
        disease_label and confidence typically come from the classifier (e.g. Early_Blight, 0.92).
        """
        context = self._build_context(disease_label, confidence, user_message)
        prompt = f"""Context:\n{context}\n\nUser question: {user_message}\n\nAnswer (based only on the context above):"""
        return self.ollama.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)

    def prepare_context(self, user_message: str, disease_label: str = None, confidence: float = None, top_k: int = 5):
        """
        Build prompt context and return document sources for UI citations.
        """
        vector_sources = self.vector_rag.query_with_sources(
            user_message or disease_label or "potato leaf disease", top_k=top_k
        )
        context = self._build_context(disease_label, confidence, user_message)
        prompt = f"""Context:\n{context}\n\nUser question: {user_message}\n\nAnswer (based only on the context above):"""
        return prompt, vector_sources

    def chat_stream(self, user_message: str, disease_label: str = None, confidence: float = None):
        """
        Stream answer text chunks from Ollama with the same grounded context.
        """
        prompt, vector_sources = self.prepare_context(user_message, disease_label, confidence)
        return self.ollama.generate_stream(prompt, system=SYSTEM_PROMPT, max_tokens=512), vector_sources
