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

## Ollama setup

Verify Ollama:
```bash
ollama --version
Pull a model (choose one):

bash
Copy code
ollama pull llama3.1:8b
# or smaller/faster:
# ollama pull llama3.2:3b
If needed, start the Ollama server:

bash
Copy code
ollama serve
Verify it is running:

bash
Copy code
curl http://127.0.0.1:11434/api/tags
Python environment setup
Create and activate a virtual environment:

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
Install runtime dependencies:

bash
Copy code
pip install -r requirements.txt
Optional (only if you want to use the notebook):

bash
Copy code
pip install -r requirements-dev.txt
Run the app (fresh session every time)
bash
Copy code
./scripts/run_app.sh
Open in your browser:

arduino
Copy code
http://localhost:8501
What “fresh session” means
Each time the app is started via ./scripts/run_app.sh:

The vector database (rag_chroma_db/) is deleted and recreated

No previously uploaded documents persist

All documents must be uploaded again through the UI

This guarantees:

No cross-session data leakage

Clean, reproducible workshop runs

Privacy-safe defaults

Typical workflow
Start the app

Upload one or more PDF / TXT / MD / DOCX files

Click Ingest / Update Vector DB

Ask questions in the chat interface

Inspect citations and retrieved context if desired

Troubleshooting
Streamlit port already in use
bash
Copy code
streamlit run app.py --server.port 8502
Ollama not reachable
bash
Copy code
ollama serve
curl http://127.0.0.1:11434/api/tags
First run downloads embeddings
The first run may download the embedding model once and cache it locally.
No account or token is required.

Notes
This project is intended for hands-on workshops and training

The design prioritizes transparency, reproducibility, and offline operation

The notebook is optional; the Streamlit app is the primary interface
