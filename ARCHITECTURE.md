# Potato Leaf Disease Detection & Advisory System — Architecture

## High-Level Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│   User      │────▶│  Streamlit App    │────▶│   Classifier     │────▶│  Prediction  │
│  (browser)  │     │  (Web UI)         │     │  (EfficientNet)  │     │  + Chat      │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                            │                         │
                            │                         │ label + confidence
                            ▼                         ▼
                    ┌───────────────────────────────────────────────────────┐
                    │              Disease Advisory Chatbot                  │
                    │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
                    │  │ Vector RAG  │  │ Graph RAG   │  │ Ollama (LLM)  │  │
                    │  │ (ChromaDB)  │  │ (JSON KG)   │  │ (Mistral)     │  │
                    │  └─────────────┘  └─────────────┘  └───────────────┘  │
                    └───────────────────────────────────────────────────────┘
```

## Mermaid Diagram

```mermaid
flowchart TB
    subgraph User["👤 User"]
        Upload[Upload Leaf Image]
        Chat[Ask Questions]
    end

    subgraph Streamlit["Streamlit Web App"]
        UI[Streamlit UI]
        Session[Session State]
    end

    subgraph Classification["Image Classification"]
        Preprocess[Preprocess Image<br/>OpenCV, resize 224×224]
        Model[Transfer Learning Model<br/>EfficientNetB0 / MobileNet / ResNet]
        Predict[Prediction: label + confidence]
    end

    subgraph RAG["RAG Pipeline"]
        VectorRAG[Vector RAG<br/>ChromaDB + embeddings]
        GraphRAG[Graph RAG<br/>Knowledge Graph JSON]
        Context[Combined Context]
    end

    subgraph LLM["LLM Inference"]
        Ollama[Ollama<br/>Mistral / Llama]
    end

    subgraph Data["Data & Models"]
        Dataset[(Dataset<br/>train/val/test)]
        Docs[(Documents<br/>PDF, TXT)]
        KG[(Knowledge Graph)]
        ClassifierModel[(classifier_tl.h5)]
        ChromaDB[(ChromaDB)]
    end

    Upload --> UI
    UI --> Preprocess
    Preprocess --> Model
    Model --> Predict
    ClassifierModel --> Model
    Dataset --> Model

    Predict --> Session
    Session --> Context
    Chat --> UI
    UI --> Context

    GraphRAG --> Context
    VectorRAG --> Context
    KG --> GraphRAG
    Docs --> VectorRAG
    ChromaDB --> VectorRAG

    Context --> Ollama
    Ollama --> UI
```

## Component Diagram (Simplified)

```mermaid
C4Component
    title Component Diagram - Potato Disease Advisory System

    Container_Boundary(app, "Streamlit App") {
        Component(ui, "Streamlit UI", "Python", "Image upload, chat interface")
        Component(classifier, "Classifier", "TensorFlow/Keras", "Transfer learning on EfficientNetB0")
        Component(chatbot, "Chatbot", "Python", "Orchestrates RAG + LLM")
    }

    Container_Boundary(rag, "RAG Layer") {
        Component(vector, "Vector RAG", "ChromaDB", "Semantic document retrieval")
        Component(graph, "Graph RAG", "Custom", "Structured disease knowledge")
    }

    Container(ollama, "Ollama", "Local LLM", "Mistral/Llama inference")
    ContainerDb(chroma, "ChromaDB", "Vector Store", "Document embeddings")
    ContainerDb(kg, "Knowledge Graph", "JSON", "Symptoms, causes, treatments")

    Rel(ui, classifier, "Image")
    Rel(ui, chatbot, "User query + prediction")
    Rel(chatbot, vector, "Query")
    Rel(chatbot, graph, "Disease label")
    Rel(chatbot, ollama, "Context + prompt")
    Rel(vector, chroma, "Query/Store")
    Rel(graph, kg, "Load")
```

## Data Flow (Sequence)

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit
    participant C as Classifier
    participant R as Chatbot (RAG)
    participant V as Vector RAG
    participant G as Graph RAG
    participant O as Ollama

    U->>S: Upload leaf image
    S->>C: Image path
    C->>C: Preprocess (resize, normalize)
    C->>C: EfficientNet inference
    C->>S: label, confidence
    S->>S: Store in session_state

    U->>S: Ask "How to treat this?"
    S->>R: prompt + disease_label + confidence

    R->>G: get_context_for_disease(label)
    G->>R: symptoms, causes, treatment, prevention
    R->>V: query(user question)
    V->>R: relevant document chunks

    R->>R: Build combined context
    R->>O: context + prompt + system_prompt
    O->>R: Generated answer
    R->>S: Response
    S->>U: Display answer
```

## File Structure (Modules)

```
src/
├── app.py              # Streamlit entry point
├── config.py           # Paths, hyperparameters, class names
├── classifier/
│   ├── model.py        # Transfer learning model (EfficientNet, etc.)
│   ├── predict.py      # Inference API
│   ├── train_classifier.py   # Train classifier
│   ├── evaluate_classifier.py
│   └── plot_confusion_matrix_demo.py
├── rag/
│   ├── vector_rag.py   # ChromaDB-based retrieval
│   ├── graph_rag.py    # Knowledge graph retrieval
│   ├── ingest_documents.py
│   └── visualize_graph.py   # Render knowledge graph as PNG (requires networkx)
└── chatbot/
    ├── chatbot.py      # Hybrid RAG + Ollama orchestration
    └── ollama_client.py # HTTP client for Ollama API
```

### Data & Model Paths

- `dataset/train/`, `val/`, `test/` — Real images by class
- `model/classifier_tl.h5` — Trained classifier
- `data/knowledge_graph.json` — Graph RAG nodes & edges
