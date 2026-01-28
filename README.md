# PEARC Local RAG Workshop (Ollama + Chroma + Streamlit)

This project is a **local-first Retrieval-Augmented Generation (RAG)** workshop demo designed for research computing audiences.

- **LLM**: Ollama (runs locally, no API keys)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Database**: Chroma
- **Application**: Streamlit (upload documents at runtime, chat with citations)

The system is intentionally designed so that **no documents persist between sessions**.

---

## Prerequisites

- Python **3.12+**
- Ollama installed locally

---

## Project layout
---
pearc-local-rag-workshop/

├── app.py # Streamlit RAG application

├── scripts/

│ └── run_app.sh # Starts app with a fresh vector DB each run

├── rag_chroma_db/ # Local vector DB (auto-managed, wiped on start)

├── notebooks/

│ └── pearc_rag_workshop.ipynb # Optional workshop notebook

├── requirements.txt # Minimal runtime dependencies

├── requirements-dev.txt # Runtime + JupyterLab (optional)

├── requirements-lock.txt # Exact frozen environment (pip freeze)

└── README.md
---

## Ollama setup

Verify Ollama:

ollama --version
Pull a model (choose one):


ollama pull llama3.1:8b
# or smaller/faster:
# ollama pull llama3.2:3b
If needed, start the Ollama server:

ollama serve
Verify it is running:

curl http://127.0.0.1:11434/api/tags
Python environment setup
Create and activate a virtual environment:

---

Python environment setup

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
Install runtime dependencies:

pip install -r requirements.txt
Optional (only if you want to use the notebook):

pip install -r requirements-dev.txt
Run the app (fresh session every time)

---

Running the application (fresh session each time)

Start the app using the provided script:

./scripts/run_app.sh
Open in your browser:

http://localhost:8501

---

What “fresh session” means
Each time the app is started via ./scripts/run_app.sh:

The vector database (rag_chroma_db/) is deleted and recreated

No previously uploaded documents persist

All documents must be uploaded again through the UI

This guarantees:

No cross-session data leakage

Clean, reproducible workshop runs

Privacy-safe defaults

---

Typical workflow
Start the app

Upload one or more PDF / TXT / MD / DOCX files

Click Ingest / Update Vector DB

Ask questions in the chat interface

Inspect citations and retrieved context if desired

---

Troubleshooting

Streamlit port already in use
streamlit run app.py --server.port 8502

Ollama not reachable
ollama serve
curl http://127.0.0.1:11434/api/tags

---

First run downloads embeddings
The first run may download the embedding model once and cache it locally.
No account or token is required.

Notes
This project is intended for hands-on workshops and training

The design prioritizes transparency, reproducibility, and offline operation

The notebook is optional; the Streamlit app is the primary interface
