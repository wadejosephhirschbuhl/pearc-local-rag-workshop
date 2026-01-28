#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Fresh session: clear vector DB every run (no carryover)
rm -rf rag_chroma_db
mkdir rag_chroma_db

streamlit run app.py
