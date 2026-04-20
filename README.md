# Potato Leaf Disease Detection and Advisory System

A Python machine learning project that classifies potato leaf diseases from images and provides treatment guidance through a hybrid chatbot.

## Overview

This project combines:
- An image classifier (transfer learning with TensorFlow/Keras)
- A Streamlit web app for image upload and chat
- Hybrid retrieval (Vector RAG + Graph RAG)
- Local LLM responses via Ollama

Main user flow:
1. Upload a potato leaf image.
2. Get predicted disease label and confidence.
3. Ask follow-up questions about symptoms, causes, treatment, and prevention.
4. Receive grounded answers from project documents + knowledge graph + LLM.

## Features

- Potato disease classification (`Early_Blight`, `Healthy`, `Late_Blight`)
- Streamlit UI for end-to-end usage
- Vector RAG over project documents (`data/documents/`)
- Graph RAG over structured knowledge (`data/knowledge_graph.json`)
- Ollama integration for local LLM inference

## Project Structure

```text
src/
  app.py
  config.py
  classifier/
    model.py
    train_classifier.py
    evaluate_classifier.py
    predict.py
  rag/
    vector_rag.py
    graph_rag.py
    ingest_documents.py
  chatbot/
    chatbot.py
    ollama_client.py

data/
  documents/
  chroma_db/
  chat_sessions/
  knowledge_graph.json

dataset/
  train/
  val/
  test/

model/
  classifier_tl.h5
  classifier_tl_best.weights.h5
```

## Requirements

- Python 3.9+
- Pip
- Ollama installed and running locally

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create and activate virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Verify dataset layout

Expected class folders:
- `dataset/train/Early_Blight/`, `dataset/train/Healthy/`, `dataset/train/Late_Blight/`
- `dataset/val/Early_Blight/`, `dataset/val/Healthy/`, `dataset/val/Late_Blight/`
- `dataset/test/Early_Blight/`, `dataset/test/Healthy/`, `dataset/test/Late_Blight/`

### 3. Train classifier

```bash
python -m src.classifier.train_classifier
```

### 4. (Optional) Evaluate classifier

```bash
python -m src.classifier.evaluate_classifier
```

### 5. Ingest RAG documents

Put text/PDF files in `data/documents/`, then run:

```bash
python -m src.rag.ingest_documents
```

### 6. Prepare Ollama

```bash
ollama pull llama3.2
```

Set the model name in `src/config.py` (`OLLAMA_MODEL`) to match what you pulled.

### 7. Run the web app

```bash
streamlit run src/app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

## CLI Prediction (Single Image)

```bash
python -m src.classifier.predict "path\to\image.jpg"
```

## Notes

- Runtime artifacts such as vector DB and chat sessions are generated under:
  - `data/chroma_db/`
  - `data/chat_sessions/`
- Large datasets and model artifacts are ignored via `.gitignore`.

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- ChromaDB
- Ollama

## License

Add your preferred license here.
