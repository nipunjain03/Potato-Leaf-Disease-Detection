# Potato Leaf Disease System – Steps from Scratch

Follow these steps in order. Use the same terminal (and keep your virtual environment activated after Step 1).

---

## Step 1: Environment setup

1. Open a terminal (e.g. Command Prompt or PowerShell).
2. Go to the project folder:
   ```bash
   cd "C:\Minor Potato"
   ```
3. Create and activate a virtual environment (if you don’t already have one):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   You should see `(venv)` in the prompt.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 2: Dataset layout

Ensure your dataset is organized as:

- `dataset/train/Early_Blight/`  (leaf images)
- `dataset/train/Healthy/`
- `dataset/train/Late_Blight/`
- `dataset/val/Early_Blight/`, `Healthy/`, `Late_Blight/`
- `dataset/test/` (same class subfolders)

Class folder names must match exactly: `Early_Blight`, `Healthy`, `Late_Blight`.

---

## Step 3: Train the classifier (required)

1. From the project root:
   ```bash
   python -m src.classifier.train_classifier
   ```
2. Wait until it finishes. You should see:
   - `Classifier saved to ...\model\classifier_tl.h5`

## Step 4: (Optional) Check classifier accuracy

1. Run:
   ```bash
   python -m src.classifier.evaluate_classifier
   ```
2. In the terminal you’ll see:
   - Test accuracy and loss
   - Classification report (precision, recall, F1 per class)
   - Confusion matrix

---

## Step 5: RAG – add documents and ingest

1. Put your text/PDF files in:
   ```text
   data/documents/
   ```
   (e.g. `potato_diseases.txt`, agriculture PDFs).

2. Ingest them into the vector store (run once, or after adding new files):
   ```bash
   python -m src.rag.ingest_documents
   ```
   You should see something like: `Ingested N chunks from ...\data\documents`.

---

## Step 6: Ollama (for the chatbot)

1. Install Ollama from [ollama.ai](https://ollama.ai) and start it.
2. Pull a model (e.g. llama3.2 or mistral):
   ```bash
   ollama pull llama3.2
   ```
3. In `src/config.py`, set `OLLAMA_MODEL` to the model you pulled (e.g. `"llama3.2"` or `"mistral"`).

---

## Step 7: Run the web app

1. From the project root:
   ```bash
   streamlit run src/app.py
   ```
2. Open the URL shown (e.g. `http://localhost:8501`).
3. In the app:
   - **Sidebar:** Upload a potato leaf image → the classifier shows the predicted disease and confidence.
   - **Main area:** Ask questions (e.g. “What causes this disease?”, “How can I treat it?”). The chatbot uses the prediction + Vector RAG + Graph RAG and answers via Ollama.

---

## Step 8: (Optional) Test a single image from the terminal

To check the classifier on one file:

```bash
python -m src.classifier.predict "path\to\your_image.jpg"
```

You’ll see the predicted label and confidence.

---

## Step 9: (Optional) Visualize the knowledge graph

```bash
pip install networkx
python -m src.rag.visualize_graph
```

Output: `model/knowledge_graph.png`

---

## Quick reference – order of steps

| Step | What                         | Command / action |
|------|------------------------------|-------------------|
| 1    | Setup env                    | `cd "C:\Minor Potato"`, activate venv, `pip install -r requirements.txt` |
| 2    | Check dataset layout         | `dataset/train`, `val`, `test` with class subfolders |
| 3    | Train classifier (required)  | `python -m src.classifier.train_classifier` |
| 4    | Evaluate (optional)          | `python -m src.classifier.evaluate_classifier` |
| 5    | Ingest RAG documents          | Put files in `data/documents/`, run `python -m src.rag.ingest_documents` |
| 6    | Ollama                        | Install, run Ollama, `ollama pull <model>`, set `OLLAMA_MODEL` in config |
| 7    | Run app                       | `streamlit run src/app.py` |
| 8    | Single-image test (optional)  | `python -m src.classifier.predict "image.jpg"` |
| 9    | Visualize knowledge graph (optional) | `pip install networkx` then `python -m src.rag.visualize_graph` |

---

## If the sidebar image causes an error

If you see an error like `image() got an unexpected keyword argument 'use_container_width'`, your Streamlit version may use a different parameter. The app code can use `use_column_width=True` instead for older Streamlit. If the error persists, update Streamlit:

```bash
pip install --upgrade streamlit
```
