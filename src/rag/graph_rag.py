"""
Graph RAG: Domain-specific knowledge graph for potato diseases.
Entities: Disease, Symptom, Cause, Treatment, Prevention.
Supports structured reasoning: cause -> symptom -> treatment.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KNOWLEDGE_GRAPH_PATH, CLASS_NAMES


def _default_knowledge_graph():
    """Default potato disease knowledge graph (Early Blight, Late Blight, Healthy)."""
    return {
        "nodes": [
            {"id": "Early_Blight", "type": "Disease", "name": "Early Blight"},
            {"id": "Late_Blight", "type": "Disease", "name": "Late Blight"},
            {"id": "Healthy", "type": "Status", "name": "Healthy"},
            {"id": "eb_s1", "type": "Symptom", "name": "Dark brown spots with concentric rings on leaves"},
            {"id": "eb_s2", "type": "Symptom", "name": "Yellow halos around lesions; older leaves affected first"},
            {"id": "eb_c1", "type": "Cause", "name": "Fungus Alternaria solani; warm humid weather"},
            {"id": "eb_t1", "type": "Treatment", "name": "Fungicides (chlorothalonil, mancozeb); remove infected leaves"},
            {"id": "eb_p1", "type": "Prevention", "name": "Crop rotation; resistant varieties; avoid overhead irrigation"},
            {"id": "lb_s1", "type": "Symptom", "name": "Water-soaked lesions that turn brown; white mold in humidity"},
            {"id": "lb_s2", "type": "Symptom", "name": "Rapid spread; stems and tubers can be affected"},
            {"id": "lb_c1", "type": "Cause", "name": "Oomycete Phytophthora infestans; cool wet conditions"},
            {"id": "lb_t1", "type": "Treatment", "name": "Fungicides (copper, mancozeb); destroy severely infected plants"},
            {"id": "lb_p1", "type": "Prevention", "name": "Certified seed; fungicide sprays; avoid planting in low areas"},
        ],
        "edges": [
            {"source": "Early_Blight", "target": "eb_s1", "relation": "has_symptom"},
            {"source": "Early_Blight", "target": "eb_s2", "relation": "has_symptom"},
            {"source": "Early_Blight", "target": "eb_c1", "relation": "has_cause"},
            {"source": "Early_Blight", "target": "eb_t1", "relation": "has_treatment"},
            {"source": "Early_Blight", "target": "eb_p1", "relation": "has_prevention"},
            {"source": "Late_Blight", "target": "lb_s1", "relation": "has_symptom"},
            {"source": "Late_Blight", "target": "lb_s2", "relation": "has_symptom"},
            {"source": "Late_Blight", "target": "lb_c1", "relation": "has_cause"},
            {"source": "Late_Blight", "target": "lb_t1", "relation": "has_treatment"},
            {"source": "Late_Blight", "target": "lb_p1", "relation": "has_prevention"},
        ],
    }


class GraphRAG:
    """
    Knowledge graph retrieval: given a disease name, return related symptoms, causes,
    treatments, and prevention in a structured format for the LLM.
    """

    def __init__(self, graph_path: str = None):
        self.graph_path = graph_path or KNOWLEDGE_GRAPH_PATH
        self.nodes = {}
        self.edges = []
        self._load_graph()

    def _load_graph(self):
        if os.path.isfile(self.graph_path):
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = _default_knowledge_graph()
            os.makedirs(os.path.dirname(self.graph_path) or ".", exist_ok=True)
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        for n in data.get("nodes", []):
            self.nodes[n["id"]] = n
        self.edges = data.get("edges", [])

    def _normalize_disease(self, label: str) -> str:
        """Map prediction label to graph disease id."""
        if not label:
            return ""
        label = label.strip().replace(" ", "_")
        if label in self.nodes:
            return label
        for key in ("Early_Blight", "Late_Blight", "Healthy"):
            if key.lower() == label.lower():
                return key
        return label

    def get_context_for_disease(self, disease_label: str, include_healthy: bool = True) -> str:
        """
        Retrieve structured context for a disease: symptoms, causes, treatment, prevention.
        Supports cause -> symptom -> treatment reasoning when formatted for the LLM.
        """
        disease_id = self._normalize_disease(disease_label)
        if not disease_id:
            return ""
        if disease_id == "Healthy" and not include_healthy:
            return "The plant is healthy; no disease-specific advice."
        if disease_id == "Healthy":
            return (
                "The potato leaf is classified as Healthy. "
                "No disease detected. General advice: maintain good crop hygiene and monitor for pests."
            )
        out = []
        for e in self.edges:
            if e["source"] != disease_id:
                continue
            target = self.nodes.get(e["target"])
            if not target:
                continue
            rel = e["relation"]
            name = target.get("name", target.get("id", ""))
            if rel == "has_symptom":
                out.append(f"Symptom: {name}")
            elif rel == "has_cause":
                out.append(f"Cause: {name}")
            elif rel == "has_treatment":
                out.append(f"Treatment: {name}")
            elif rel == "has_prevention":
                out.append(f"Prevention: {name}")
        if not out:
            return f"Disease: {disease_id}. No structured knowledge in graph."
        return f"Disease: {disease_id}\n" + "\n".join(out)
