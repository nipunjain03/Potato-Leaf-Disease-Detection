"""
Main application: Potato Leaf Disease Detection and Advisory System.
Flow: Upload image -> Classifier (transfer learning) -> Disease + confidence ->
      Injected as context into chatbot -> User asks follow-ups (Vector RAG + Graph RAG + Ollama).
"""

import os
import sys
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from classifier.predict import predict_image_with_probs
from chatbot.chatbot import DiseaseAdvisoryChatbot
from rag.vector_rag import VectorRAG
from rag.graph_rag import GraphRAG
from config import DATA_DIR


SESSIONS_DIR = os.path.join(DATA_DIR, "chat_sessions")


def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def list_session_ids():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    ids = []
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            ids.append(fname[:-5])
    return sorted(ids, reverse=True)


def save_session(session_id: str, messages: list, prediction):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    payload = {
        "session_id": session_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "prediction": prediction,
        "messages": messages,
    }
    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_session(session_id: str):
    path = _session_path(session_id)
    if not os.path.isfile(path):
        return [], None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("messages", []), payload.get("prediction")
    except Exception:
        return [], None


def main():
    st.set_page_config(page_title="Potato Leaf Disease Advisory", layout="wide")
    st.title("Potato Leaf Disease Detection & Advisory")
    st.caption("Transfer learning classifier + Vector RAG + Graph RAG + Ollama")

    # Session state: current prediction and chat history
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "prediction" not in st.session_state:
        st.session_state.prediction = None  # (label, confidence)
    if "prediction_probs" not in st.session_state:
        st.session_state.prediction_probs = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "loaded_session_id" not in st.session_state:
        st.session_state.loaded_session_id = None

    # Load persisted session once per selected session id
    if st.session_state.loaded_session_id != st.session_state.session_id:
        msgs, pred = load_session(st.session_state.session_id)
        st.session_state.messages = msgs
        st.session_state.prediction = pred
        st.session_state.prediction_probs = None
        st.session_state.loaded_session_id = st.session_state.session_id

    # Sidebar: image upload and prediction
    with st.sidebar:
        st.header("0. Chat sessions")
        existing_ids = list_session_ids()
        all_ids = existing_ids if existing_ids else [st.session_state.session_id]
        selected_session = st.selectbox("Session", options=all_ids, index=all_ids.index(st.session_state.session_id) if st.session_state.session_id in all_ids else 0)
        if selected_session != st.session_state.session_id:
            st.session_state.session_id = selected_session
            st.rerun()
        if st.button("New session"):
            new_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            st.session_state.session_id = new_id
            st.session_state.messages = []
            st.session_state.prediction = None
            save_session(new_id, st.session_state.messages, st.session_state.prediction)
            st.session_state.loaded_session_id = new_id
            st.rerun()

        st.header("1. Chat controls")
        st.session_state.show_sources = st.toggle("Show source citations", value=st.session_state.show_sources)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear chat"):
                st.session_state.messages = []
                save_session(st.session_state.session_id, st.session_state.messages, st.session_state.prediction)
                st.rerun()
        with c2:
            export_payload = {
                "session_id": st.session_state.session_id,
                "prediction": st.session_state.prediction,
                "messages": st.session_state.messages,
            }
            st.download_button(
                "Export chat",
                data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                file_name=f"{st.session_state.session_id}.json",
                mime="application/json",
            )

        st.header("1. Image analysis")
        uploaded = st.file_uploader("Upload potato leaf image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            path = os.path.join(DATA_DIR, "upload_temp.jpg")
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(uploaded.getvalue())
            try:
                label, confidence, probs = predict_image_with_probs(path)
                st.session_state.prediction = (label, confidence)
                st.session_state.prediction_probs = probs
                save_session(st.session_state.session_id, st.session_state.messages, st.session_state.prediction)
                st.success(f"**{label}** ({confidence:.2%})")
                if confidence < 0.70:
                    st.warning("Low confidence prediction. Try a clearer, close-up leaf image in good light.")
                top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                st.markdown("**Top-3 class probabilities**")
                for name, prob in top3:
                    st.write(f"- {name}: {prob:.2%}")
                st.image(uploaded, width="stretch")
            except FileNotFoundError as e:
                st.error("Classifier model not found. Train it first: run `python -m src.classifier.train_classifier`")
            except Exception as e:
                st.error(str(e))
        else:
            if st.session_state.prediction:
                label, conf = st.session_state.prediction
                st.info(f"Current: **{label}** ({conf:.2%})")
                if st.session_state.prediction_probs:
                    top3 = sorted(st.session_state.prediction_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    st.caption("Top-3 probabilities:")
                    for name, prob in top3:
                        st.caption(f"{name}: {prob:.2%}")
            else:
                st.info("Upload an image to get a prediction.")

    # Main: chat
    st.header("2. Ask about the disease")
    if st.session_state.prediction:
        label, conf = st.session_state.prediction
        st.caption(f"Context: Image predicted **{label}** (confidence {conf:.2%}). Ask: causes, treatment, prevention.")
    else:
        st.caption("Upload a leaf image first so the chatbot can use the prediction as context.")

    chatbot = DiseaseAdvisoryChatbot(vector_rag=VectorRAG(), graph_rag=GraphRAG())
    disease_label, confidence = st.session_state.prediction or (None, None)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg.get("role") == "assistant"
                and st.session_state.show_sources
                and msg.get("sources")
            ):
                with st.expander("Sources"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"{i}. **{src.get('source', 'Unknown')}**")
                        st.caption(src.get("excerpt", ""))

    if prompt := st.chat_input("e.g. What causes this disease? How can I treat it?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream_iter, sources = chatbot.chat_stream(prompt, disease_label=disease_label, confidence=confidence)
            placeholder = st.empty()
            parts = []
            for chunk in stream_iter:
                parts.append(chunk)
                placeholder.markdown("".join(parts) + "▌")
            response = "".join(parts).strip()
            placeholder.markdown(response)
            if st.session_state.show_sources and sources:
                with st.expander("Sources"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(f"{i}. **{src.get('source', 'Unknown')}**")
                        st.caption(src.get("excerpt", ""))
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
        save_session(st.session_state.session_id, st.session_state.messages, st.session_state.prediction)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Architecture**")
    st.sidebar.markdown("- Classifier: Transfer learning (EfficientNet)")
    st.sidebar.markdown("- RAG: Vector + Graph")
    st.sidebar.markdown("- LLM: Ollama (local)")


if __name__ == "__main__":
    main()
