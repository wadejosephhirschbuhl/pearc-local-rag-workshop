------------------------------------------------------------
PEARC LOCAL RAG WORKSHOP
(Ollama + Chroma + Streamlit)
------------------------------------------------------------

This project is a local-first Retrieval-Augmented Generation (RAG)
workshop demo designed for research computing audiences.

The workshop emphasizes transparency, reproducibility, and hands-on
understanding of modern RAG systems using only local, open-source tools.

The system is intentionally designed so that NO DOCUMENTS persist
between sessions.


------------------------------------------------------------
CORE COMPONENTS
------------------------------------------------------------

LLM
  - Ollama (runs locally, no API keys required)

Embeddings
  - SentenceTransformers (model: all-MiniLM-L6-v2)

Vector Database
  - Chroma (local, on-disk)

Application
  - Streamlit (runtime document upload + chat with citations)


------------------------------------------------------------
PREREQUISITES
------------------------------------------------------------

- Python 3.12.x recommended (Python 3.11 also works)
- Ollama installed locally

Notes:
  - A .python-version file is included for users of pyenv
  - Avoid Python 3.13 (some binary dependencies may not be available)


------------------------------------------------------------
PROJECT LAYOUT
------------------------------------------------------------

pearc-local-rag-workshop/

  app.py
    Streamlit RAG application

  scripts/
    run_app.sh
      Starts the app and clears the vector database on every run

  rag_chroma_db/
    Local vector database directory
    (auto-managed and wiped at app start)

  notebooks/
    pearc_rag_workshop.ipynb
      Optional workshop notebook

  requirements.txt
    Minimal runtime dependencies

  requirements-dev.txt
    Runtime dependencies plus JupyterLab (optional)

  requirements-lock.txt
    Exact frozen environment snapshot (pip freeze)

  README.txt
    This file

  .python-version
    Recommended Python version (3.12.12) for pyenv users


------------------------------------------------------------
OLLAMA SETUP
------------------------------------------------------------

Verify Ollama is installed:
  ollama --version

Pull a model (choose one):

  Recommended:
    ollama pull llama3.1:8b

  Smaller / faster option:
    ollama pull llama3.2:3b

Start the Ollama server


------------------------------------------------------------
PYTHON ENVIRONMENT SETUP
------------------------------------------------------------

If you use pyenv:
  - The included .python-version file will automatically select Python 3.12.12

Create and activate a virtual environment:

  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel

Install runtime dependencies:

  pip install -r requirements.txt

Optional (only if you want to run the notebook):

  pip install -r requirements-dev.txt


------------------------------------------------------------
RUNNING THE APPLICATION
(FRESH SESSION EVERY TIME)
------------------------------------------------------------

Start the app using the provided script:

  ./scripts/run_app.sh

Open your browser to:

  http://localhost:8501


------------------------------------------------------------
OPTIONAL: CROSS-ENCODER RERANKER (BETTER RELEVANCE)
------------------------------------------------------------

The app includes an optional cross-encoder reranker that can improve
retrieval quality when multiple documents are indexed.

What it does:
  - Retrieves a larger candidate pool from the vector database
  - Reranks chunks by query-to-chunk relevance
  - Selects the best top-k chunks for the model prompt

How to use it:
  - In the sidebar, enable:
      "Use cross-encoder reranker (slower, often better)"
  - Adjust the candidate pool slider if needed

Notes:
  - The reranker is slower than vector-only retrieval
  - The reranker model may download once on first use (then caches locally)
  - To pre-cache the reranker before a workshop, run:

      python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('reranker cached')"


------------------------------------------------------------
WHAT "FRESH SESSION" MEANS
------------------------------------------------------------

Each time the app is started via ./scripts/run_app.sh:

  - The vector database (rag_chroma_db/) is deleted and recreated
  - No previously uploaded documents persist
  - All documents must be uploaded again through the UI

This guarantees:

  - No cross-session data leakage
  - Clean, reproducible workshop runs
  - Privacy-safe defaults


------------------------------------------------------------
TYPICAL WORKSHOP WORKFLOW
------------------------------------------------------------

1. Start the app
2. Upload one or more PDF, TXT, MD, or DOCX files
3. Click "Ingest / Update Vector DB"
4. Ask questions in the chat interface
5. Inspect citations and retrieved context if desired


------------------------------------------------------------
TROUBLESHOOTING
------------------------------------------------------------

Streamlit port already in use:

  streamlit run app.py --server.port 8502


Ollama not reachable:

  ollama serve
  curl http://127.0.0.1:11434/api/tags


First run downloads embeddings:

  The first run may download the embedding model once and cache it
  locally. No account or token is required.


------------------------------------------------------------
NOTES
------------------------------------------------------------

- This project is intended for hands-on workshops and training
- The design prioritizes transparency, reproducibility, and offline operation
- The notebook demonstrates the baseline RAG pipeline
- The Streamlit app extends that baseline with production-style upgrades
  such as optional reranking
