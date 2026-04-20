"""
Knowledge Chatbot Module: Disease-related Q&A using Ollama (local LLM).
Combines Vector RAG + Graph RAG context and optional image prediction context.
"""

from .ollama_client import OllamaClient
from .chatbot import DiseaseAdvisoryChatbot

__all__ = ["OllamaClient", "DiseaseAdvisoryChatbot"]
